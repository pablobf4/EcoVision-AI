# ============================================================
# EcoVision AI
# Avaliação final do ViT treinado do zero
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

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

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ecovision_vit_scratch_best.pth"
)

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# CARREGAR MODELO
# ============================================================

def carregar_modelo():
    """
    Carrega o melhor checkpoint salvo durante o treinamento.
    """

    print("Carregando checkpoint:")

    print(
        MODEL_PATH
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    config = checkpoint[
        "config"
    ]


    model = EcoVisionViT(

        image_size=config["image_size"],

        patch_size=config["patch_size"],

        num_classes=config["num_classes"],

        embed_dim=config["embed_dim"],

        num_heads=config["num_heads"],

        num_layers=config["num_layers"],

        mlp_dim=config["mlp_dim"],

        dropout=config["dropout"]
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(
        device
    )


    # Modo de avaliação:
    #
    # desativa comportamento de treinamento de
    # Dropout e outras camadas específicas.
    model.eval()


    print()
    print(
        f"Melhor checkpoint salvo na epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation Loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )


    return model


# ============================================================
# PREDIÇÕES
# ============================================================

def obter_predicoes(
    model,
    test_loader
):
    """
    Executa o modelo sobre todo o conjunto de teste.

    Retorna:

        classes reais
        classes previstas
    """

    y_true = []

    y_pred = []


    with torch.no_grad():

        for imagens, labels in test_loader:

            imagens = imagens.to(
                device
            )


            logits = model(
                imagens
            )


            # Classe com maior logit.
            predicoes = torch.argmax(
                logits,
                dim=1
            )


            y_true.extend(
                labels.cpu().numpy()
            )


            y_pred.extend(
                predicoes.cpu().numpy()
            )


    return (
        np.array(y_true),
        np.array(y_pred)
    )


# ============================================================
# ACCURACY POR CLASSE
# ============================================================

def calcular_accuracy_por_classe(
    y_true,
    y_pred,
    classes
):
    """
    Calcula accuracy individual de cada classe.

    Exemplo:

        Plastic:
        80 imagens corretas / 100 imagens

        accuracy = 80%
    """

    resultados = {}


    print()
    print("=" * 70)

    print(
        "ACCURACY POR CLASSE"
    )

    print("=" * 70)


    for classe_id, nome_classe in enumerate(
        classes
    ):

        # Selecionamos apenas amostras pertencentes
        # à classe atual.
        mascara = (
            y_true == classe_id
        )


        total_classe = mascara.sum()


        corretos = (
            y_pred[mascara]
            == classe_id
        ).sum()


        accuracy = (
            corretos
            / total_classe
            if total_classe > 0
            else 0
        )


        resultados[
            nome_classe
        ] = float(accuracy)


        print(
            f"{nome_classe:25s}: "
            f"{accuracy * 100:6.2f}% "
            f"({corretos}/{total_classe})"
        )


    return resultados


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def gerar_matriz_confusao(
    y_true,
    y_pred,
    classes
):
    """
    Gera e salva a matriz de confusão.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )


    plt.figure(
        figsize=(11, 9)
    )


    # Usamos imshow para desenhar a matriz.
    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        "EcoVision ViT do Zero - Matriz de Confusão"
    )


    plt.colorbar()


    indices = np.arange(
        len(classes)
    )


    plt.xticks(
        indices,
        classes,
        rotation=45,
        ha="right"
    )


    plt.yticks(
        indices,
        classes
    )


    # Escrever os números dentro das células.
    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )


    plt.xlabel(
        "Classe prevista"
    )

    plt.ylabel(
        "Classe real"
    )


    plt.tight_layout()


    caminho = (
        RESULTS_DIR
        / "vit_scratch_confusion_matrix.png"
    )


    plt.savefig(
        caminho,
        dpi=150
    )


    plt.close()


    print()
    print(
        "Matriz de confusão salva em:"
    )

    print(
        caminho
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "EcoVision AI - Avaliação do ViT do zero"
    )

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


    classes = dataset_base.classes


    print()
    print(
        f"Imagens de teste: "
        f"{len(test_dataset)}"
    )


    # ========================================================
    # MODELO
    # ========================================================

    model = carregar_modelo()


    # ========================================================
    # PREDIÇÕES
    # ========================================================

    y_true, y_pred = obter_predicoes(
        model,
        test_loader
    )


    # ========================================================
    # ACCURACY GLOBAL
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    print()
    print("=" * 70)

    print(
        "RESULTADOS GLOBAIS"
    )

    print("=" * 70)


    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    (
        precision,
        recall,
        f1,
        _
    ) = precision_recall_fscore_support(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0
    )


    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )


    print(
        f"Recall: "
        f"{recall * 100:.2f}%"
    )


    print(
        f"F1-score: "
        f"{f1 * 100:.2f}%"
    )


    # ========================================================
    # ACCURACY POR CLASSE
    # ========================================================

    accuracy_classes = calcular_accuracy_por_classe(

        y_true,

        y_pred,

        classes
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)


    report = classification_report(

        y_true,

        y_pred,

        target_names=classes,

        digits=4,

        zero_division=0
    )


    print(
        report
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    gerar_matriz_confusao(

        y_true,

        y_pred,

        classes
    )


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    resultados = {

        "model": "EcoVisionViT Scratch",

        "checkpoint_epoch": 13,

        "accuracy": float(
            accuracy
        ),

        "precision_weighted": float(
            precision
        ),

        "recall_weighted": float(
            recall
        ),

        "f1_weighted": float(
            f1
        ),

        "accuracy_per_class": (
            accuracy_classes
        )
    }


    caminho_json = (
        RESULTS_DIR
        / "vit_scratch_test_results.json"
    )


    with open(
        caminho_json,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resultados,
            arquivo,
            indent=4
        )


    print()
    print(
        "Resultados salvos em:"
    )

    print(
        caminho_json
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()