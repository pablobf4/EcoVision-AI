# ============================================================
# EcoVision AI
# Atividade 4.1
# Treinamento baseline - Raio-X / COVID-19
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

from src.xray.data_pipeline import (
    localizar_dataset,
    criar_datasets,
    criar_dataloaders
)

from src.xray.models.classifier import (
    XRayClassifier
)


# ============================================================
# CONFIGURAÇÕES
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
    / "xray_baseline_best.pth"
)


HISTORY_PATH = (
    RESULTS_DIR
    / "xray_baseline_history.json"
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
# REPRODUTIBILIDADE
# ============================================================

def configurar_seed():
    """
    Define seeds para tornar os experimentos
    mais reprodutíveis.
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


    for batch_index, (
        images,
        labels
    ) in enumerate(
        loader,
        start=1
    ):

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
        # Atualizar pesos.
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
        / len(loader.dataset)
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
# VALIDAR
# ============================================================

def validar(
    model,
    loader,
    criterion,
    covid_class_index
):
    """
    Avalia o modelo sem atualizar pesos.

    Além de accuracy, calculamos:

        Macro F1
        Recall COVID

    Recall COVID será a métrica central do estudo.
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


    loss_medio = (
        total_loss
        / len(loader.dataset)
    )


    accuracy = accuracy_score(
        targets,
        predictions
    )


    macro_f1 = f1_score(

        targets,
        predictions,

        average="macro",

        zero_division=0
    )


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
    Salva o melhor modelo encontrado.
    """

    torch.save({

        "epoch": epoch,

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
            "baseline_real_only"

    }, CHECKPOINT_PATH)


# ============================================================
# GRÁFICOS
# ============================================================

def salvar_graficos(
    history
):
    """
    Salva curvas para análise posterior.
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
        figsize=(8, 5)
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
        "Raio-X Baseline - Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "xray_baseline_loss.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # ACCURACY
    # ========================================================

    plt.figure(
        figsize=(8, 5)
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
        "Raio-X Baseline - Accuracy"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "xray_baseline_accuracy.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # COVID RECALL
    # ========================================================

    plt.figure(
        figsize=(8, 5)
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
        "Raio-X Baseline - Validation COVID Recall"
    )


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "xray_baseline_covid_recall.png",

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
    print("EcoVision AI - Baseline Raio-X")
    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        DEVICE
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset_path = localizar_dataset()


    (
        dataset_base,
        train_dataset,
        val_dataset,
        test_dataset
    ) = criar_datasets(
        dataset_path
    )


    (
        train_loader,
        val_loader,
        _
    ) = criar_dataloaders(

        train_dataset,
        val_dataset,
        test_dataset
    )


    class_to_idx = (
        dataset_base.class_to_idx
    )


    print()
    print(
        "Classes:",
        class_to_idx
    )


    covid_class_index = (
        class_to_idx[
            "COVID"
        ]
    )


    print(
        "Índice COVID:",
        covid_class_index
    )


    print()
    print(
        "Treino:",
        len(train_dataset)
    )


    print(
        "Validação:",
        len(val_dataset)
    )


    # ========================================================
    # MODELO
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
    # Não usamos class weights neste experimento.
    #
    # Isso é proposital:
    #
    # queremos usar a geração sintética como variável
    # experimental principal para tratamento da classe COVID.
    #
    # Depois treinaremos o MESMO classificador acrescentando
    # imagens COVID geradas pela GAN.
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
    # HISTÓRICO
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
        # TREINO
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
        # VALIDAÇÃO
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
        # HISTÓRICO
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
        # RESULTADOS
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


    salvar_graficos(
        history
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("TREINAMENTO BASELINE FINALIZADO")
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


if __name__ == "__main__":

    main()