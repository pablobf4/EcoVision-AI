# ============================================================
# EcoVision AI
# Atividade 3 - Treinamento EfficientNet-B0
#
# Fase 1:
#   Feature Extraction
#   Backbone congelado
#
# Fase 2:
#   Fine-Tuning
#   Modelo completo descongelado
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .data_pipeline import (
    localizar_dataset,
    criar_datasets,
    criar_dataloaders
)

from .models.pretrained_cnn import (
    EcoVisionPretrainedCNN,
    contar_parametros
)


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_MODEL_PATH = (
    MODEL_DIR
    / "ecovision_cnn_pretrained_best.pth"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "cnn_pretrained_history.json"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

NUM_CLASSES = 7

HEAD_EPOCHS = 2
HEAD_LR = 1e-3

FINETUNE_EPOCHS = 8
FINETUNE_LR = 1e-4

WEIGHT_DECAY = 0.01

EARLY_STOPPING_PATIENCE = 3


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TREINAR UMA ÉPOCA
# ============================================================

def treinar_epoca(
    model,
    dataloader,
    criterion,
    optimizer,
    somente_head=False
):

    model.train()

    # --------------------------------------------------------
    # Durante feature extraction:
    #
    # Mantemos o backbone em modo eval.
    #
    # Isso evita alteração das estatísticas internas
    # das BatchNorms enquanto o backbone está congelado.
    # --------------------------------------------------------

    if somente_head:

        model.model.features.eval()

        model.model.classifier.train()


    loss_total = 0.0

    total_acertos = 0

    total_amostras = 0


    for batch_index, (imagens, labels) in enumerate(
        dataloader
    ):

        imagens = imagens.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        # Forward
        logits = model(
            imagens
        )


        # Loss
        loss = criterion(
            logits,
            labels
        )


        # Backpropagation
        loss.backward()


        # Atualização dos parâmetros treináveis
        optimizer.step()


        batch_size = imagens.size(0)


        loss_total += (
            loss.item()
            * batch_size
        )


        predicoes = torch.argmax(
            logits,
            dim=1
        )


        total_acertos += (
            predicoes == labels
        ).sum().item()


        total_amostras += batch_size


        if (
            batch_index == 0
            or (batch_index + 1) % 20 == 0
        ):

            print(
                f"    Batch "
                f"{batch_index + 1:03d}/"
                f"{len(dataloader):03d}"
                f" | Loss: {loss.item():.4f}"
            )


    loss_media = (
        loss_total
        / total_amostras
    )


    accuracy = (
        total_acertos
        / total_amostras
    )


    return (
        loss_media,
        accuracy
    )


# ============================================================
# VALIDAÇÃO
# ============================================================

def avaliar(
    model,
    dataloader,
    criterion
):

    model.eval()

    loss_total = 0.0

    total_acertos = 0

    total_amostras = 0


    with torch.no_grad():

        for imagens, labels in dataloader:

            imagens = imagens.to(device)

            labels = labels.to(device)


            logits = model(
                imagens
            )


            loss = criterion(
                logits,
                labels
            )


            batch_size = imagens.size(0)


            loss_total += (
                loss.item()
                * batch_size
            )


            predicoes = torch.argmax(
                logits,
                dim=1
            )


            total_acertos += (
                predicoes == labels
            ).sum().item()


            total_amostras += batch_size


    return (
        loss_total / total_amostras,
        total_acertos / total_amostras
    )


# ============================================================
# SALVAR CHECKPOINT
# ============================================================

def salvar_checkpoint(
    model,
    epoch,
    fase,
    val_loss,
    val_accuracy
):

    checkpoint = {

        "epoch": epoch,

        "fase": fase,

        "model_state_dict": model.state_dict(),

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "num_classes": NUM_CLASSES
    }


    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )


# ============================================================
# EXECUTAR FASE
# ============================================================

def executar_fase(
    nome_fase,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    history,
    melhor_val_loss,
    somente_head=False
):

    epochs_sem_melhora = 0


    for epoch in range(
        1,
        epochs + 1
    ):

        inicio = time.time()


        print()
        print("=" * 70)

        print(
            f"{nome_fase} - Epoch "
            f"{epoch}/{epochs}"
        )

        print("=" * 70)


        train_loss, train_accuracy = treinar_epoca(

            model=model,

            dataloader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            somente_head=somente_head
        )


        val_loss, val_accuracy = avaliar(

            model,

            val_loader,

            criterion
        )


        tempo = (
            time.time()
            - inicio
        )


        # ====================================================
        # HISTÓRICO
        # ====================================================

        history["fase"].append(
            nome_fase
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["epoch_time_seconds"].append(
            tempo
        )


        print()

        print(
            f"Train Loss:      "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print()

        print(
            f"Train Accuracy:  "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Accuracy:    "
            f"{val_accuracy * 100:.2f}%"
        )

        print()

        print(
            f"Tempo: "
            f"{tempo / 60:.2f} minutos"
        )


        # ====================================================
        # MELHOR MODELO
        # ====================================================

        if val_loss < melhor_val_loss:

            melhor_val_loss = val_loss

            epochs_sem_melhora = 0


            print()
            print(
                "Novo melhor modelo."
            )

            print(
                "Salvando checkpoint..."
            )


            salvar_checkpoint(

                model=model,

                epoch=epoch,

                fase=nome_fase,

                val_loss=val_loss,

                val_accuracy=val_accuracy
            )


        else:

            epochs_sem_melhora += 1


            print()

            print(
                f"Sem melhoria por "
                f"{epochs_sem_melhora} epoch(s)."
            )


        # ----------------------------------------------------
        # Salvar histórico a cada epoch
        # ----------------------------------------------------

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


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_sem_melhora
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early Stopping nesta fase."
            )

            break


    return melhor_val_loss


# ============================================================
# GRÁFICOS
# ============================================================

def gerar_graficos(
    history
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    # ========================================================
    # LOSS
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        epochs,
        history["train_loss"],
        label="Train"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation"
    )


    if HEAD_EPOCHS < len(
        history["train_loss"]
    ):

        plt.axvline(
            HEAD_EPOCHS,
            linestyle="--",
            label="Início Fine-Tuning"
        )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "EfficientNet-B0 - Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        RESULTS_DIR
        / "cnn_pretrained_loss.png"
    )


    plt.close()


    # ========================================================
    # ACCURACY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation"
    )


    if HEAD_EPOCHS < len(
        history["train_accuracy"]
    ):

        plt.axvline(
            HEAD_EPOCHS,
            linestyle="--",
            label="Início Fine-Tuning"
        )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "EfficientNet-B0 - Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        RESULTS_DIR
        / "cnn_pretrained_accuracy.png"
    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Treinamento CNN")
    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        device
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
        test_loader
    ) = criar_dataloaders(

        train_dataset,

        val_dataset,

        test_dataset
    )


    print()
    print(
        f"Treino:    "
        f"{len(train_dataset)}"
    )

    print(
        f"Validação: "
        f"{len(val_dataset)}"
    )

    print(
        f"Teste:     "
        f"{len(test_dataset)}"
    )


    # ========================================================
    # MODELO
    # ========================================================

    model = EcoVisionPretrainedCNN(

        num_classes=NUM_CLASSES,

        pretrained=True
    )


    model = model.to(
        device
    )


    criterion = nn.CrossEntropyLoss()


    history = {

        "fase": [],

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": [],

        "epoch_time_seconds": []
    }


    melhor_val_loss = float(
        "inf"
    )


    # ========================================================
    # FASE 1
    # FEATURE EXTRACTION
    # ========================================================

    print()
    print("=" * 70)
    print("FASE 1 - FEATURE EXTRACTION")
    print("=" * 70)


    model.congelar_backbone()


    total, treinaveis = contar_parametros(
        model
    )


    print()
    print(
        f"Parâmetros totais: "
        f"{total:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{treinaveis:,}"
    )


    optimizer_head = torch.optim.AdamW(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=HEAD_LR,

        weight_decay=WEIGHT_DECAY
    )


    melhor_val_loss = executar_fase(

        nome_fase="Feature Extraction",

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer_head,

        criterion=criterion,

        epochs=HEAD_EPOCHS,

        history=history,

        melhor_val_loss=melhor_val_loss,

        somente_head=True
    )


    # ========================================================
    # FASE 2
    # FINE-TUNING
    # ========================================================

    print()
    print("=" * 70)
    print("FASE 2 - FINE-TUNING COMPLETO")
    print("=" * 70)


    model.descongelar_modelo()


    total, treinaveis = contar_parametros(
        model
    )


    print()
    print(
        f"Parâmetros totais: "
        f"{total:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{treinaveis:,}"
    )


    optimizer_finetune = torch.optim.AdamW(

        model.parameters(),

        lr=FINETUNE_LR,

        weight_decay=WEIGHT_DECAY
    )


    executar_fase(

        nome_fase="Fine-Tuning",

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer_finetune,

        criterion=criterion,

        epochs=FINETUNE_EPOCHS,

        history=history,

        melhor_val_loss=melhor_val_loss,

        somente_head=False
    )


    # ========================================================
    # GRÁFICOS
    # ========================================================

    gerar_graficos(
        history
    )


    print()
    print("=" * 70)
    print("TREINAMENTO FINALIZADO")
    print("=" * 70)


    print()
    print(
        "Melhor modelo:"
    )

    print(
        BEST_MODEL_PATH
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