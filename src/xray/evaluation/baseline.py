# ============================================================
# EcoVision AI
# Atividade 4.1
# Avaliação do baseline - Raio-X / COVID-19
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.xray.data_pipeline import (
    localizar_dataset,
    criar_datasets,
    criar_dataloaders
)

from src.xray.models.classifier import (
    XRayClassifier
)


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT


CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "xray_baseline_best.pth"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xray"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_PATH = (
    RESULTS_DIR
    / "xray_baseline_test_results.json"
)


CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "xray_baseline_confusion_matrix.png"
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
# AVALIAR
# ============================================================

def avaliar(
    model,
    test_loader
):
    """
    Executa inferência sobre o conjunto de teste.

    Não são calculados gradientes e nenhum peso
    do modelo é alterado.
    """

    model.eval()


    predictions = []

    targets = []


    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )


            logits = model(
                images
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


    return (
        targets,
        predictions
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    targets,
    predictions,
    classes,
    covid_index
):
    """
    Calcula:

        Accuracy global

        Precision weighted
        Recall weighted
        F1 weighted

        Precision macro
        Recall macro
        F1 macro

        Recall específico de COVID

        Accuracy/recall por classe
    """

    # ========================================================
    # MÉTRICAS GLOBAIS
    # ========================================================

    accuracy = accuracy_score(
        targets,
        predictions
    )


    precision_weighted = precision_score(

        targets,
        predictions,

        average="weighted",

        zero_division=0
    )


    recall_weighted = recall_score(

        targets,
        predictions,

        average="weighted",

        zero_division=0
    )


    f1_weighted = f1_score(

        targets,
        predictions,

        average="weighted",

        zero_division=0
    )


    precision_macro = precision_score(

        targets,
        predictions,

        average="macro",

        zero_division=0
    )


    recall_macro = recall_score(

        targets,
        predictions,

        average="macro",

        zero_division=0
    )


    f1_macro = f1_score(

        targets,
        predictions,

        average="macro",

        zero_division=0
    )


    # ========================================================
    # RECALL COVID
    # ========================================================

    covid_recall = recall_score(

        targets,
        predictions,

        labels=[
            covid_index
        ],

        average="macro",

        zero_division=0
    )


    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    matriz = confusion_matrix(

        targets,
        predictions,

        labels=list(
            range(
                len(classes)
            )
        )
    )


    # ========================================================
    # RESULTADO POR CLASSE
    # ========================================================

    por_classe = {}


    for indice, classe in enumerate(
        classes
    ):

        # Linha da matriz:
        #
        # todos os exemplos cuja classe real é "classe".
        total_classe = int(
            matriz[
                indice,
                :
            ].sum()
        )


        # Diagonal:
        #
        # quantidade classificada corretamente.
        acertos = int(
            matriz[
                indice,
                indice
            ]
        )


        if total_classe > 0:

            recall_classe = (
                acertos
                / total_classe
            )

        else:

            recall_classe = 0.0


        por_classe[
            classe
        ] = {

            "acertos": acertos,

            "total": total_classe,

            "recall": recall_classe
        }


    resultados = {

        "accuracy": accuracy,

        "precision_weighted":
            precision_weighted,

        "recall_weighted":
            recall_weighted,

        "f1_weighted":
            f1_weighted,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro,

        "covid_recall":
            covid_recall,

        "por_classe":
            por_classe
    }


    return (
        resultados,
        matriz
    )


# ============================================================
# SALVAR MATRIZ DE CONFUSÃO
# ============================================================

def salvar_matriz_confusao(
    matriz,
    classes
):
    """
    Gera representação visual da matriz
    de confusão.
    """

    figura, eixo = plt.subplots(
        figsize=(
            7,
            6
        )
    )


    display = ConfusionMatrixDisplay(

        confusion_matrix=matriz,

        display_labels=classes
    )


    display.plot(

        ax=eixo,

        values_format="d",

        colorbar=False
    )


    plt.title(
        "Raio-X Baseline - Matriz de Confusão"
    )


    plt.tight_layout()


    plt.savefig(

        CONFUSION_MATRIX_PATH,

        dpi=150,

        bbox_inches="tight"
    )


    plt.close(
        figura
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Avaliação Baseline Raio-X")
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
        _,
        _,
        test_loader
    ) = criar_dataloaders(

        train_dataset,

        val_dataset,

        test_dataset
    )


    classes = (
        dataset_base.classes
    )


    class_to_idx = (
        dataset_base.class_to_idx
    )


    covid_index = (
        class_to_idx[
            "COVID"
        ]
    )


    print()
    print(
        "Classes:",
        class_to_idx
    )


    print(
        "Quantidade de imagens de teste:",
        len(test_dataset)
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint não encontrado:\n"
            f"{CHECKPOINT_PATH}"
        )


    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE
    )


    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    print()
    print(
        "Melhor epoch:",
        checkpoint[
            "epoch"
        ]
    )


    print(
        "Validation Loss:",
        f"{checkpoint['val_loss']:.4f}"
    )


    print(
        "Validation Accuracy:",
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )


    print(
        "Validation COVID Recall:",
        f"{checkpoint['val_covid_recall'] * 100:.2f}%"
    )


    # ========================================================
    # MODELO
    # ========================================================
    #
    # pretrained=False:
    #
    # não precisamos baixar pesos do ImageNet novamente,
    # porque os pesos completos serão carregados do nosso
    # checkpoint.
    # ========================================================

    model = XRayClassifier(

        num_classes=len(
            classes
        ),

        pretrained=False
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        DEVICE
    )


    # ========================================================
    # AVALIAÇÃO
    # ========================================================

    (
        targets,
        predictions
    ) = avaliar(

        model,

        test_loader
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    (
        resultados,
        matriz
    ) = calcular_metricas(

        targets,

        predictions,

        classes,

        covid_index
    )


    # ========================================================
    # RESULTADOS GLOBAIS
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADOS GLOBAIS")
    print("=" * 70)


    print(
        f"Accuracy: "
        f"{resultados['accuracy'] * 100:.2f}%"
    )


    print(
        f"Precision weighted: "
        f"{resultados['precision_weighted'] * 100:.2f}%"
    )


    print(
        f"Recall weighted: "
        f"{resultados['recall_weighted'] * 100:.2f}%"
    )


    print(
        f"F1 weighted: "
        f"{resultados['f1_weighted'] * 100:.2f}%"
    )


    print()


    print(
        f"Precision macro: "
        f"{resultados['precision_macro'] * 100:.2f}%"
    )


    print(
        f"Recall macro: "
        f"{resultados['recall_macro'] * 100:.2f}%"
    )


    print(
        f"F1 macro: "
        f"{resultados['f1_macro'] * 100:.2f}%"
    )


    # ========================================================
    # COVID
    # ========================================================

    print()
    print("=" * 70)
    print("MÉTRICA PRINCIPAL - COVID-19")
    print("=" * 70)


    print(
        f"COVID Recall: "
        f"{resultados['covid_recall'] * 100:.2f}%"
    )


    covid_result = (
        resultados[
            "por_classe"
        ][
            "COVID"
        ]
    )


    print(

        f"Casos detectados: "

        f"{covid_result['acertos']}/"

        f"{covid_result['total']}"
    )


    print(

        f"Casos COVID não detectados: "

        f"{covid_result['total'] - covid_result['acertos']}"
    )


    # ========================================================
    # RESULTADO POR CLASSE
    # ========================================================

    print()
    print("=" * 70)
    print("RECALL POR CLASSE")
    print("=" * 70)


    for classe in classes:

        item = (
            resultados[
                "por_classe"
            ][
                classe
            ]
        )


        print(

            f"{classe:10s}: "

            f"{item['recall'] * 100:6.2f}% "

            f"({item['acertos']}/{item['total']})"
        )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)


    relatorio_texto = classification_report(

        targets,

        predictions,

        target_names=classes,

        digits=4,

        zero_division=0
    )


    print(
        relatorio_texto
    )


    # ========================================================
    # SALVAR MATRIZ
    # ========================================================

    salvar_matriz_confusao(

        matriz,

        classes
    )


    # ========================================================
    # SALVAR JSON
    # ========================================================

    resultados[
        "checkpoint_epoch"
    ] = checkpoint[
        "epoch"
    ]


    resultados[
        "validation_loss"
    ] = checkpoint[
        "val_loss"
    ]


    resultados[
        "validation_accuracy"
    ] = checkpoint[
        "val_accuracy"
    ]


    resultados[
        "validation_covid_recall"
    ] = checkpoint[
        "val_covid_recall"
    ]


    resultados[
        "test_size"
    ] = len(
        test_dataset
    )


    resultados[
        "confusion_matrix"
    ] = matriz.tolist()


    with open(

        RESULTS_PATH,

        "w",

        encoding="utf-8"

    ) as arquivo:

        json.dump(

            resultados,

            arquivo,

            indent=4,

            ensure_ascii=False
        )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "Matriz de confusão salva em:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )


    print()
    print(
        "Resultados salvos em:"
    )

    print(
        RESULTS_PATH
    )


    print()
    print("=" * 70)
    print("AVALIAÇÃO BASELINE FINALIZADA")
    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()