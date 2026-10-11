# ============================================================
# EcoVision AI
# Avaliação do ViT pré-treinado no conjunto de teste
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from .data_pipeline import (
    localizar_dataset,
    criar_datasets,
    criar_dataloaders
)

from .models.pretrained_vit import EcoVisionPretrainedViT


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ecovision_vit_pretrained_best.pth"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # pretrained=False porque os pesos serão carregados
    # diretamente do nosso checkpoint.
    model = EcoVisionPretrainedViT(
        num_classes=checkpoint["num_classes"],
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print()
    print("Checkpoint carregado:")
    print(MODEL_PATH)

    print()
    print(
        f"Fase: {checkpoint['fase']}"
    )

    print(
        f"Melhor epoch: {checkpoint['epoch']}"
    )

    print(
        f"Validation Loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    return model, checkpoint


# ============================================================
# PREDIÇÕES
# ============================================================

def obter_predicoes(
    model,
    test_loader
):

    labels_reais = []
    labels_previstos = []

    model.eval()

    with torch.no_grad():

        for imagens, labels in test_loader:

            imagens = imagens.to(device)

            logits = model(
                imagens
            )

            predicoes = torch.argmax(
                logits,
                dim=1
            )

            labels_reais.extend(
                labels.numpy()
            )

            labels_previstos.extend(
                predicoes.cpu().numpy()
            )

    return (
        np.array(labels_reais),
        np.array(labels_previstos)
    )


# ============================================================
# ACCURACY POR CLASSE
# ============================================================

def calcular_accuracy_por_classe(
    y_true,
    y_pred,
    classes
):

    resultados = {}

    for indice, classe in enumerate(classes):

        mascara = (
            y_true == indice
        )

        total = mascara.sum()

        corretos = (
            y_pred[mascara]
            == indice
        ).sum()

        accuracy = (
            corretos / total
            if total > 0
            else 0
        )

        resultados[classe] = {
            "accuracy": float(accuracy),
            "corretos": int(corretos),
            "total": int(total)
        }

    return resultados


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def salvar_matriz_confusao(
    y_true,
    y_pred,
    classes
):

    matriz = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        matriz
    )

    plt.title(
        "ViT Pré-treinado - Matriz de Confusão"
    )

    plt.xlabel(
        "Classe prevista"
    )

    plt.ylabel(
        "Classe real"
    )

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(classes)),
        classes
    )

    # Mostrar números dentro das células.
    for i in range(
        matriz.shape[0]
    ):

        for j in range(
            matriz.shape[1]
        ):

            plt.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    caminho = (
        RESULTS_DIR
        / "vit_pretrained_confusion_matrix.png"
    )

    plt.savefig(
        caminho,
        dpi=150
    )

    plt.close()

    return caminho


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Avaliação ViT Pré-treinado")
    print("=" * 70)

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
        f"Quantidade de imagens de teste: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # MODELO
    # ========================================================

    model, checkpoint = carregar_modelo()

    # ========================================================
    # PREDIÇÕES
    # ========================================================

    y_true, y_pred = obter_predicoes(
        model,
        test_loader
    )

    # ========================================================
    # MÉTRICAS GLOBAIS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    (
        precision_macro,
        recall_macro,
        f1_macro,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("RESULTADOS GLOBAIS")
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision weighted: "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Recall weighted: "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"F1 weighted: "
        f"{f1_weighted * 100:.2f}%"
    )

    print()
    print(
        f"Precision macro: "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Recall macro: "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"F1 macro: "
        f"{f1_macro * 100:.2f}%"
    )

    # ========================================================
    # ACCURACY POR CLASSE
    # ========================================================

    accuracy_classes = calcular_accuracy_por_classe(
        y_true,
        y_pred,
        classes
    )

    print()
    print("=" * 70)
    print("ACCURACY POR CLASSE")
    print("=" * 70)

    for classe, dados in accuracy_classes.items():

        print(
            f"{classe:25s}: "
            f"{dados['accuracy'] * 100:6.2f}% "
            f"({dados['corretos']}/{dados['total']})"
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print(
        report_text
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    caminho_matriz = salvar_matriz_confusao(
        y_true,
        y_pred,
        classes
    )

    print()
    print(
        "Matriz de confusão salva em:"
    )

    print(
        caminho_matriz
    )

    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    resultados = {

        "checkpoint": {
            "fase": checkpoint["fase"],
            "epoch": checkpoint["epoch"],
            "validation_loss": checkpoint["val_loss"],
            "validation_accuracy": checkpoint["val_accuracy"]
        },

        "test": {
            "accuracy": float(accuracy),

            "precision_weighted": float(
                precision_weighted
            ),

            "recall_weighted": float(
                recall_weighted
            ),

            "f1_weighted": float(
                f1_weighted
            ),

            "precision_macro": float(
                precision_macro
            ),

            "recall_macro": float(
                recall_macro
            ),

            "f1_macro": float(
                f1_macro
            )
        },

        "accuracy_por_classe": accuracy_classes,

        "classification_report": report_dict
    }


    caminho_resultados = (
        RESULTS_DIR
        / "vit_pretrained_test_results.json"
    )


    with open(
        caminho_resultados,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resultados,
            arquivo,
            indent=4,
            ensure_ascii=False
        )


    print()
    print(
        "Resultados salvos em:"
    )

    print(
        caminho_resultados
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()