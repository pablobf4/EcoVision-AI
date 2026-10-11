# ============================================================
# EcoVision AI
# Treinamento completo do Vision Transformer do zero
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

from .models.vit_model import EcoVisionViT


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
    MODEL_DIR / "ecovision_vit_scratch_best.pth"
)

LAST_MODEL_PATH = (
    MODEL_DIR / "ecovision_vit_scratch_last.pth"
)

HISTORY_PATH = (
    RESULTS_DIR / "vit_scratch_history.json"
)

CURVES_PATH = (
    RESULTS_DIR / "vit_scratch_training_curves.png"
)


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16

NUM_CLASSES = 9

EMBED_DIM = 192
NUM_HEADS = 3
NUM_LAYERS = 4
MLP_DIM = 768

DROPOUT = 0.1


# ============================================================
# CONFIGURAÇÃO DO TREINAMENTO
# ============================================================

# Começaremos com 15 épocas.
#
# O Early Stopping pode interromper antes.
EPOCHS = 15

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.05

# Número de épocas sem melhoria antes de interromper.
EARLY_STOPPING_PATIENCE = 4


# ============================================================
# TREINAR UMA ÉPOCA
# ============================================================

def treinar_epoca(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Executa uma época completa de treinamento.
    """

    model.train()

    loss_total = 0.0
    total_acertos = 0
    total_amostras = 0


    for batch_index, (imagens, labels) in enumerate(
        dataloader
    ):

        imagens = imagens.to(
            device
        )

        labels = labels.to(
            device
        )


        # ----------------------------------------------------
        # Zerar os gradientes calculados no batch anterior.
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            imagens
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Atualização dos pesos
        # ----------------------------------------------------

        optimizer.step()


        # ====================================================
        # MÉTRICAS
        # ====================================================

        batch_size = imagens.size(0)


        loss_total += (
            loss.item()
            * batch_size
        )


        classes_previstas = torch.argmax(
            logits,
            dim=1
        )


        total_acertos += (
            classes_previstas
            == labels
        ).sum().item()


        total_amostras += batch_size


        # Mostra progresso de tempos em tempos.
        if (
            (batch_index + 1) % 20 == 0
            or batch_index == 0
        ):

            print(
                f"    Batch "
                f"{batch_index + 1:03d}/"
                f"{len(dataloader):03d} "
                f"| Loss: {loss.item():.4f}"
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
# VALIDAR UMA ÉPOCA
# ============================================================

def avaliar(
    model,
    dataloader,
    criterion,
    device
):
    """
    Executa validação sem atualizar os pesos.
    """

    model.eval()

    loss_total = 0.0
    total_acertos = 0
    total_amostras = 0


    with torch.no_grad():

        for imagens, labels in dataloader:

            imagens = imagens.to(
                device
            )

            labels = labels.to(
                device
            )


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


            classes_previstas = torch.argmax(
                logits,
                dim=1
            )


            total_acertos += (
                classes_previstas
                == labels
            ).sum().item()


            total_amostras += batch_size


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
# SALVAR CHECKPOINT
# ============================================================

def salvar_checkpoint(
    caminho,
    epoch,
    model,
    optimizer,
    val_loss,
    val_accuracy
):
    """
    Salva o estado necessário para recuperar o treinamento.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,

        "config": {
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "num_classes": NUM_CLASSES,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "mlp_dim": MLP_DIM,
            "dropout": DROPOUT
        }
    }


    torch.save(
        checkpoint,
        caminho
    )


# ============================================================
# SALVAR HISTÓRICO
# ============================================================

def salvar_historico(
    history
):
    """
    Salva as métricas em JSON.
    """

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


# ============================================================
# GRÁFICOS
# ============================================================

def gerar_graficos(
    history
):
    """
    Gera gráfico de Loss e Accuracy.
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "EcoVision ViT - Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    loss_path = (
        RESULTS_DIR
        / "vit_scratch_loss.png"
    )

    plt.savefig(
        loss_path
    )

    plt.close()


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "EcoVision ViT - Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    accuracy_path = (
        RESULTS_DIR
        / "vit_scratch_accuracy.png"
    )

    plt.savefig(
        accuracy_path
    )

    plt.close()


    print()
    print(
        "Gráficos salvos em:"
    )

    print(
        loss_path
    )

    print(
        accuracy_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "EcoVision AI - Treinamento completo do ViT do zero"
    )

    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        device
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "Treinamento executado em CPU."
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
        f"Treino:     {len(train_dataset)}"
    )

    print(
        f"Validação:  {len(val_dataset)}"
    )

    print(
        f"Teste:      {len(test_dataset)}"
    )


    # ========================================================
    # MODELO
    # ========================================================

    model = EcoVisionViT(

        image_size=IMAGE_SIZE,

        patch_size=PATCH_SIZE,

        num_classes=NUM_CLASSES,

        embed_dim=EMBED_DIM,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        mlp_dim=MLP_DIM,

        dropout=DROPOUT
    )


    model = model.to(
        device
    )


    # --------------------------------------------------------
    # Número de parâmetros
    # --------------------------------------------------------

    total_parametros = sum(
        p.numel()
        for p in model.parameters()
    )


    treinaveis = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print()
    print(
        f"Parâmetros totais: "
        f"{total_parametros:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{treinaveis:,}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OTIMIZADOR
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # SCHEDULER
    # ========================================================
    #
    # Se a validation loss parar de melhorar,
    # reduzimos o learning rate.
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=2
    )


    # ========================================================
    # HISTÓRICO
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_time_seconds": []
    }


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    melhor_val_loss = float(
        "inf"
    )

    epochs_sem_melhora = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        inicio_epoch = time.time()


        print()
        print("=" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = treinar_epoca(

            model=model,

            dataloader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            device=device
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = avaliar(

            model=model,

            dataloader=val_loader,

            criterion=criterion,

            device=device
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )


        # ----------------------------------------------------
        # Tempo
        # ----------------------------------------------------

        tempo_epoch = (
            time.time()
            - inicio_epoch
        )


        # ----------------------------------------------------
        # Histórico
        # ----------------------------------------------------

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
            tempo_epoch
        )


        # ----------------------------------------------------
        # Mostrar resultados
        # ----------------------------------------------------

        learning_rate_atual = (
            optimizer.param_groups[0]["lr"]
        )


        print()
        print(
            f"Train Loss:     {train_loss:.4f}"
        )

        print(
            f"Validation Loss:{val_loss:.4f}"
        )

        print()

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Accuracy:   "
            f"{val_accuracy * 100:.2f}%"
        )

        print()

        print(
            f"Learning Rate: "
            f"{learning_rate_atual:.8f}"
        )

        print(
            f"Tempo da epoch: "
            f"{tempo_epoch / 60:.2f} minutos"
        )


        # ====================================================
        # SALVAR ÚLTIMO CHECKPOINT
        # ====================================================

        salvar_checkpoint(

            LAST_MODEL_PATH,

            epoch,

            model,

            optimizer,

            val_loss,

            val_accuracy
        )


        # ====================================================
        # MELHOR MODELO
        # ====================================================

        if val_loss < melhor_val_loss:

            print()
            print(
                "Nova melhor Validation Loss."
            )

            print(
                "Salvando melhor modelo..."
            )


            melhor_val_loss = val_loss

            epochs_sem_melhora = 0


            salvar_checkpoint(

                BEST_MODEL_PATH,

                epoch,

                model,

                optimizer,

                val_loss,

                val_accuracy
            )


        else:

            epochs_sem_melhora += 1


            print()
            print(
                f"Sem melhoria por "
                f"{epochs_sem_melhora} epoch(s)."
            )


        # ----------------------------------------------------
        # Salvar histórico após TODA epoch.
        #
        # Assim não perdemos resultados se o processo
        # for interrompido posteriormente.
        # ----------------------------------------------------

        salvar_historico(
            history
        )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_sem_melhora
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print("=" * 70)

            print(
                "Early Stopping ativado."
            )

            print("=" * 70)

            break


    # ========================================================
    # FINALIZAÇÃO
    # ========================================================

    gerar_graficos(
        history
    )


    print()
    print("=" * 70)

    print(
        "TREINAMENTO FINALIZADO"
    )

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
        "Último checkpoint:"
    )

    print(
        LAST_MODEL_PATH
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