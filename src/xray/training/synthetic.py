# ============================================================
# EcoVision AI
# Atividade 4.1
# Classificador Raio-X com augmentation sintética COVID
#
# Comparação experimental:
#
# Baseline:
#     apenas imagens reais
#
# Novo experimento:
#     imagens reais
#     +
#     84 imagens COVID geradas pela cGAN
#
# IMPORTANTE:
#
# Validação e teste continuam contendo
# SOMENTE imagens reais.
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)

from torch.optim import AdamW

from torch.utils.data import (
    ConcatDataset,
    DataLoader
)

from torchvision import datasets

from src.xray.data_pipeline import (
    localizar_dataset,
    criar_datasets,
    criar_transforms,
    BATCH_SIZE
)

from src.xray.models.classifier import (
    XRayClassifier
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================
#
# Mantemos as MESMAS configurações do baseline.
#
# Isso é importante para a comparação ser justa.
# ============================================================

SEED = 42

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 3


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT


SYNTHETIC_DATASET_PATH = (
    PROJECT_DIR
    / "data"
    / "xray_synthetic"
)


MODELS_DIR = (
    PROJECT_DIR
    / "models"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xray"
)


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHECKPOINT_PATH = (
    MODELS_DIR
    / "xray_synthetic_best.pth"
)


HISTORY_PATH = (
    RESULTS_DIR
    / "xray_synthetic_history.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# SEED
# ============================================================

def configurar_seed():
    """
    Define seeds para aumentar a reprodutibilidade
    e manter o experimento comparável ao baseline.
    """

    random.seed(
        SEED
    )


    np.random.seed(
        SEED
    )


    torch.manual_seed(
        SEED
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


# ============================================================
# CRIAR DATASET COM AUGMENTATION SINTÉTICA
# ============================================================

def criar_datasets_com_sinteticas():
    """
    Cria:

        train_real
        +
        COVID sintético

    enquanto mantém:

        validation = real
        test       = real

    Isso evita data leakage.

    As imagens sintéticas nunca entram
    na validação nem no teste.
    """

    dataset_path = localizar_dataset()


    # ========================================================
    # SPLITS REAIS
    # ========================================================

    (
        dataset_base,
        train_real,
        val_dataset,
        test_dataset
    ) = criar_datasets(
        dataset_path
    )


    # ========================================================
    # TRANSFORM DE TREINO
    # ========================================================

    (
        train_transform,
        _
    ) = criar_transforms()


    # ========================================================
    # VERIFICAR IMAGENS SINTÉTICAS
    # ========================================================

    if not SYNTHETIC_DATASET_PATH.exists():

        raise FileNotFoundError(

            "Dataset sintético não encontrado:\n"

            f"{SYNTHETIC_DATASET_PATH}"
        )


    # ========================================================
    # DATASET SINTÉTICO
    # ========================================================
    #
    # Estrutura:
    #
    # xray_synthetic/
    #     COVID/
    #         covid_synthetic_0001.png
    #         ...
    #
    # ImageFolder atribui:
    #
    # COVID -> label 0
    #
    # exatamente como no dataset real.
    # ========================================================

    synthetic_dataset = datasets.ImageFolder(

        SYNTHETIC_DATASET_PATH,

        transform=train_transform
    )


    # ========================================================
    # VALIDAR LABEL
    # ========================================================

    print()
    print(
        "Classes sintéticas:",
        synthetic_dataset.class_to_idx
    )


    assert (
        synthetic_dataset.class_to_idx[
            "COVID"
        ]
        == dataset_base.class_to_idx[
            "COVID"
        ]
    )


    assert (
        len(synthetic_dataset)
        == 84
    )


    # ========================================================
    # CONCATENAR
    # ========================================================

    train_augmented = ConcatDataset([

        train_real,

        synthetic_dataset
    ])


    return (

        dataset_base,

        train_real,

        synthetic_dataset,

        train_augmented,

        val_dataset,

        test_dataset
    )


# ============================================================
# DATALOADERS
# ============================================================

def criar_dataloaders_augmented(

    train_dataset,
    val_dataset,
    test_dataset
):
    """
    Cria os DataLoaders.

    O conjunto de treino contém imagens reais + sintéticas.

    Validação/teste continuam reais.
    """

    usar_pin_memory = (
        torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # Generator para controlar o shuffle.
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=usar_pin_memory,

        generator=generator
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    return (

        train_loader,

        val_loader,

        test_loader
    )


# ============================================================
# TREINAR UMA ÉPOCA
# ============================================================

def treinar_epoca(

    model,
    loader,
    criterion,
    optimizer
):
    """
    Executa uma época completa de treinamento.
    """

    model.train()


    total_loss = 0.0

    predictions = []

    targets = []


    for (
        images,
        labels
    ) in loader:

        images = images.to(
            DEVICE
        )


        labels = labels.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Limpar gradientes.
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward.
        # ----------------------------------------------------

        logits = model(
            images
        )


        # ----------------------------------------------------
        # Loss.
        # ----------------------------------------------------

        loss = criterion(

            logits,

            labels
        )


        # ----------------------------------------------------
        # Backpropagation.
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Atualização dos pesos.
        # ----------------------------------------------------

        optimizer.step()


        total_loss += (

            loss.item()

            * images.size(0)
        )


        predicted = torch.argmax(

            logits,

            dim=1
        )


        predictions.extend(

            predicted
            .detach()
            .cpu()
            .tolist()
        )


        targets.extend(

            labels
            .detach()
            .cpu()
            .tolist()
        )


    loss_medio = (

        total_loss

        / len(
            loader.dataset
        )
    )


    accuracy = accuracy_score(

        targets,

        predictions
    )


    return (

        loss_medio,

        accuracy
    )


# ============================================================
# VALIDAÇÃO
# ============================================================

def validar(

    model,
    loader,
    criterion,
    covid_class_index
):
    """
    Avalia utilizando apenas imagens reais.

    Métricas:

        loss
        accuracy
        macro F1
        recall COVID
    """

    model.eval()


    total_loss = 0.0

    predictions = []

    targets = []


    with torch.inference_mode():

        for (
            images,
            labels
        ) in loader:

            images = images.to(
                DEVICE
            )


            labels = labels.to(
                DEVICE
            )


            logits = model(
                images
            )


            loss = criterion(

                logits,

                labels
            )


            total_loss += (

                loss.item()

                * images.size(0)
            )


            predicted = torch.argmax(

                logits,

                dim=1
            )


            predictions.extend(

                predicted
                .cpu()
                .tolist()
            )


            targets.extend(

                labels
                .cpu()
                .tolist()
            )


    # ========================================================
    # LOSS
    # ========================================================

    loss_medio = (

        total_loss

        / len(
            loader.dataset
        )
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(

        targets,

        predictions
    )


    # ========================================================
    # MACRO F1
    # ========================================================

    macro_f1 = f1_score(

        targets,

        predictions,

        average="macro",

        zero_division=0
    )


    # ========================================================
    # COVID RECALL
    # ========================================================

    covid_recall = recall_score(

        targets,

        predictions,

        labels=[
            covid_class_index
        ],

        average="macro",

        zero_division=0
    )


    return (

        loss_medio,

        accuracy,

        macro_f1,

        covid_recall
    )


# ============================================================
# SALVAR CHECKPOINT
# ============================================================

def salvar_checkpoint(

    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
    val_macro_f1,
    val_covid_recall,
    class_to_idx
):
    """
    Salva o melhor modelo por validation loss.

    Mantemos o MESMO critério utilizado no baseline.

    Isso torna a comparação mais justa.
    """

    torch.save({

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_macro_f1":
            val_macro_f1,

        "val_covid_recall":
            val_covid_recall,

        "class_to_idx":
            class_to_idx,

        "experiment":
            "real_plus_synthetic_covid",

        "synthetic_covid_count":
            84

    }, CHECKPOINT_PATH)


# ============================================================
# GRÁFICOS
# ============================================================

def salvar_graficos(
    history
):
    """
    Gera as curvas de treinamento.
    """

    epochs = list(

        range(

            1,

            len(
                history[
                    "train_loss"
                ]
            )
            + 1
        )
    )


    # ========================================================
    # LOSS
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5
        )
    )


    plt.plot(

        epochs,

        history[
            "train_loss"
        ],

        label="Train"
    )


    plt.plot(

        epochs,

        history[
            "val_loss"
        ],

        label="Validation"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Raio-X + COVID Sintético - Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "xray_synthetic_loss.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # ACCURACY
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5
        )
    )


    plt.plot(

        epochs,

        history[
            "train_accuracy"
        ],

        label="Train"
    )


    plt.plot(

        epochs,

        history[
            "val_accuracy"
        ],

        label="Validation"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "Raio-X + COVID Sintético - Accuracy"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "xray_synthetic_accuracy.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # COVID RECALL
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5
        )
    )


    plt.plot(

        epochs,

        history[
            "val_covid_recall"
        ]
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Recall COVID"
    )


    plt.title(
        "Raio-X + COVID Sintético - Validation COVID Recall"
    )


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "xray_synthetic_covid_recall.png",

        dpi=150
    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    configurar_seed()


    print()
    print("=" * 70)

    print(
        "EcoVision AI - Raio-X com COVID Sintético"
    )

    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        DEVICE
    )


    # ========================================================
    # DATASETS
    # ========================================================

    (
        dataset_base,

        train_real,

        synthetic_dataset,

        train_augmented,

        val_dataset,

        test_dataset

    ) = criar_datasets_com_sinteticas()


    # ========================================================
    # DISTRIBUIÇÃO
    # ========================================================

    print()
    print("=" * 70)
    print("DISTRIBUIÇÃO DO EXPERIMENTO")
    print("=" * 70)


    print()
    print(
        "Treino real:",
        len(
            train_real
        )
    )


    print(
        "COVID sintético:",
        len(
            synthetic_dataset
        )
    )


    print(
        "Treino total:",
        len(
            train_augmented
        )
    )


    print(
        "Validação real:",
        len(
            val_dataset
        )
    )


    print(
        "Teste real:",
        len(
            test_dataset
        )
    )


    print()

    print(
        "COVID treino:"
    )


    print(
        "  Reais:      84"
    )


    print(
        "  Sintéticos: 84"
    )


    print(
        "  Total:      168"
    )


    print()

    print(
        "Normal treino:     588"
    )


    print(
        "Pneumonia treino:  168"
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    (
        train_loader,
        val_loader,
        _
    ) = criar_dataloaders_augmented(

        train_augmented,

        val_dataset,

        test_dataset
    )


    # ========================================================
    # CLASSES
    # ========================================================

    class_to_idx = (
        dataset_base.class_to_idx
    )


    covid_class_index = (

        class_to_idx[
            "COVID"
        ]
    )


    print()
    print(
        "Classes:",
        class_to_idx
    )


    # ========================================================
    # MODELO
    # ========================================================
    #
    # MESMA arquitetura do baseline.
    # ========================================================

    model = XRayClassifier(

        num_classes=len(
            dataset_base.classes
        ),

        pretrained=True
    )


    model = model.to(
        DEVICE
    )


    # ========================================================
    # LOSS
    # ========================================================
    #
    # Mantemos CrossEntropyLoss sem class weights.
    #
    # Queremos testar especificamente o impacto
    # da augmentation sintética.
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": [],

        "val_macro_f1": [],

        "val_covid_recall": []
    }


    melhor_val_loss = float(
        "inf"
    )


    epochs_sem_melhora = 0


    # ========================================================
    # TREINAMENTO
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("=" * 70)


        inicio = time.time()


        # ====================================================
        # TRAIN
        # ====================================================

        (
            train_loss,
            train_accuracy
        ) = treinar_epoca(

            model,

            train_loader,

            criterion,

            optimizer
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_covid_recall
        ) = validar(

            model,

            val_loader,

            criterion,

            covid_class_index
        )


        tempo = (
            time.time()
            - inicio
        )


        # ====================================================
        # HISTORY
        # ====================================================

        history[
            "train_loss"
        ].append(
            train_loss
        )


        history[
            "val_loss"
        ].append(
            val_loss
        )


        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )


        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )


        history[
            "val_macro_f1"
        ].append(
            val_macro_f1
        )


        history[
            "val_covid_recall"
        ].append(
            val_covid_recall
        )


        # ====================================================
        # LOG
        # ====================================================

        print(
            f"Train Loss:       "
            f"{train_loss:.4f}"
        )


        print(
            f"Validation Loss:  "
            f"{val_loss:.4f}"
        )


        print()


        print(
            f"Train Accuracy:   "
            f"{train_accuracy * 100:.2f}%"
        )


        print(
            f"Val Accuracy:     "
            f"{val_accuracy * 100:.2f}%"
        )


        print(
            f"Val Macro F1:     "
            f"{val_macro_f1 * 100:.2f}%"
        )


        print(
            f"Val COVID Recall: "
            f"{val_covid_recall * 100:.2f}%"
        )


        print()
        print(
            f"Tempo: "
            f"{tempo / 60:.2f} minutos"
        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        if val_loss < melhor_val_loss:

            melhor_val_loss = (
                val_loss
            )


            epochs_sem_melhora = 0


            salvar_checkpoint(

                model,

                optimizer,

                epoch,

                val_loss,

                val_accuracy,

                val_macro_f1,

                val_covid_recall,

                class_to_idx
            )


            print()
            print(
                "Novo melhor modelo."
            )


            print(
                "Checkpoint salvo."
            )


        else:

            epochs_sem_melhora += 1


            print()
            print(

                "Sem melhoria por "

                f"{epochs_sem_melhora} "

                "epoch(s)."
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_sem_melhora
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early Stopping."
            )

            break


    # ========================================================
    # SALVAR HISTÓRICO
    # ========================================================

    with open(

        HISTORY_PATH,

        "w",

        encoding="utf-8"

    ) as arquivo:

        json.dump(

            history,

            arquivo,

            indent=4
        )


    # ========================================================
    # GRÁFICOS
    # ========================================================

    salvar_graficos(
        history
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)

    print(
        "TREINAMENTO COM COVID SINTÉTICO FINALIZADO"
    )

    print("=" * 70)


    print()
    print(
        "Checkpoint:"
    )


    print(
        CHECKPOINT_PATH
    )


    print()
    print(
        "Histórico:"
    )


    print(
        HISTORY_PATH
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()