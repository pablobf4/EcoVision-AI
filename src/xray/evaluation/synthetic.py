# ============================================================
# EcoVision AI
# Atividade 4.1
# Avaliação do classificador treinado com COVID sintético
#
# Objetivo:
#
# Comparar diretamente com o baseline:
#
# Baseline:
#     treino apenas com imagens reais
#
# Novo experimento:
#     treino real + 84 imagens COVID sintéticas
#
# IMPORTANTE:
#
# O conjunto de teste é exatamente o mesmo conjunto REAL
# utilizado no baseline.
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json

import matplotlib.pyplot as plt
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
    / "xray_synthetic_best.pth"
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
    / "xray_synthetic_test_results.json"
)


CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "xray_synthetic_confusion_matrix.png"
)


COMPARISON_PATH = (
    RESULTS_DIR
    / "xray_baseline_vs_synthetic.json"
)


# ============================================================
# BASELINE
# ============================================================

BASELINE_RESULTS_PATH = (
    RESULTS_DIR
    / "xray_baseline_test_results.json"
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
# AVALIAÇÃO
# ============================================================

def avaliar(
    model,
    test_loader
):
    """
    Executa inferência sobre as 180 imagens reais
    pertencentes ao conjunto de teste.

    Nenhum gradiente é calculado.
    """

    model.eval()


    predictions = []

    targets = []


    with torch.inference_mode():

        for (
            images,
            labels
        ) in test_loader:

            images = images.to(
                DEVICE
            )


            labels = labels.to(
                DEVICE
            )


            # ------------------------------------------------
            # Forward.
            # ------------------------------------------------

            logits = model(
                images
            )


            # ------------------------------------------------
            # Classe com maior logit.
            # ------------------------------------------------

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
# CALCULAR MÉTRICAS
# ============================================================

def calcular_metricas(
    targets,
    predictions,
    classes,
    covid_index
):
    """
    Calcula as mesmas métricas utilizadas no baseline.

    Isso permite comparação direta entre os experimentos.
    """

    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(

        targets,
        predictions
    )


    # ========================================================
    # WEIGHTED
    # ========================================================

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


    # ========================================================
    # MACRO
    # ========================================================

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
    # COVID RECALL
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
    # RESULTADOS POR CLASSE
    # ========================================================

    por_classe = {}


    for indice, classe in enumerate(
        classes
    ):

        total_classe = int(

            matriz[
                indice,
                :
            ].sum()
        )


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

            "acertos":
                acertos,

            "total":
                total_classe,

            "recall":
                recall_classe
        }


    resultados = {

        "accuracy":
            accuracy,

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
# MATRIZ DE CONFUSÃO
# ============================================================

def salvar_matriz_confusao(
    matriz,
    classes
):
    """
    Gera a matriz de confusão do experimento
    com augmentation sintética.
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
        "Raio-X + COVID Sintético - Matriz de Confusão"
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
# COMPARAR COM BASELINE
# ============================================================

def comparar_com_baseline(
    synthetic_results
):
    """
    Carrega os resultados anteriores do baseline e cria
    uma comparação quantitativa.

    Métrica principal:

        Recall COVID

    Também comparamos:

        Accuracy
        Macro F1
        Macro Recall
    """

    if not BASELINE_RESULTS_PATH.exists():

        print()
        print(
            "Aviso: resultados do baseline não encontrados."
        )

        return None


    with open(

        BASELINE_RESULTS_PATH,

        "r",

        encoding="utf-8"

    ) as arquivo:

        baseline = json.load(
            arquivo
        )


    # ========================================================
    # DIFERENÇAS
    # ========================================================

    covid_difference = (

        synthetic_results[
            "covid_recall"
        ]

        - baseline[
            "covid_recall"
        ]
    )


    accuracy_difference = (

        synthetic_results[
            "accuracy"
        ]

        - baseline[
            "accuracy"
        ]
    )


    macro_f1_difference = (

        synthetic_results[
            "f1_macro"
        ]

        - baseline[
            "f1_macro"
        ]
    )


    macro_recall_difference = (

        synthetic_results[
            "recall_macro"
        ]

        - baseline[
            "recall_macro"
        ]
    )


    comparison = {

        "baseline": {

            "accuracy":
                baseline[
                    "accuracy"
                ],

            "macro_f1":
                baseline[
                    "f1_macro"
                ],

            "macro_recall":
                baseline[
                    "recall_macro"
                ],

            "covid_recall":
                baseline[
                    "covid_recall"
                ]
        },

        "synthetic": {

            "accuracy":
                synthetic_results[
                    "accuracy"
                ],

            "macro_f1":
                synthetic_results[
                    "f1_macro"
                ],

            "macro_recall":
                synthetic_results[
                    "recall_macro"
                ],

            "covid_recall":
                synthetic_results[
                    "covid_recall"
                ]
        },

        "difference": {

            "accuracy":
                accuracy_difference,

            "macro_f1":
                macro_f1_difference,

            "macro_recall":
                macro_recall_difference,

            "covid_recall":
                covid_difference
        }
    }


    # ========================================================
    # SALVAR
    # ========================================================

    with open(

        COMPARISON_PATH,

        "w",

        encoding="utf-8"

    ) as arquivo:

        json.dump(

            comparison,

            arquivo,

            indent=4
        )


    # ========================================================
    # MOSTRAR COMPARAÇÃO
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE VS AUGMENTATION SINTÉTICA")
    print("=" * 70)


    print()

    print(
        f"{'Métrica':20s} "
        f"{'Baseline':>12s} "
        f"{'Sintético':>12s} "
        f"{'Diferença':>12s}"
    )


    print(
        "-" * 60
    )


    print(

        f"{'Accuracy':20s} "

        f"{baseline['accuracy'] * 100:11.2f}% "

        f"{synthetic_results['accuracy'] * 100:11.2f}% "

        f"{accuracy_difference * 100:+11.2f} pp"
    )


    print(

        f"{'Macro F1':20s} "

        f"{baseline['f1_macro'] * 100:11.2f}% "

        f"{synthetic_results['f1_macro'] * 100:11.2f}% "

        f"{macro_f1_difference * 100:+11.2f} pp"
    )


    print(

        f"{'Macro Recall':20s} "

        f"{baseline['recall_macro'] * 100:11.2f}% "

        f"{synthetic_results['recall_macro'] * 100:11.2f}% "

        f"{macro_recall_difference * 100:+11.2f} pp"
    )


    print()

    print(

        f"{'COVID Recall':20s} "

        f"{baseline['covid_recall'] * 100:11.2f}% "

        f"{synthetic_results['covid_recall'] * 100:11.2f}% "

        f"{covid_difference * 100:+11.2f} pp"
    )


    # ========================================================
    # COVID DETECTADOS
    # ========================================================

    baseline_covid = (
        baseline[
            "por_classe"
        ][
            "COVID"
        ]
    )


    synthetic_covid = (
        synthetic_results[
            "por_classe"
        ][
            "COVID"
        ]
    )


    print()

    print(
        "Casos COVID detectados:"
    )


    print(

        f"Baseline:  "

        f"{baseline_covid['acertos']}/"

        f"{baseline_covid['total']}"
    )


    print(

        f"Sintético: "

        f"{synthetic_covid['acertos']}/"

        f"{synthetic_covid['total']}"
    )


    print()

    print(
        "Comparação salva:"
    )

    print(
        COMPARISON_PATH
    )


    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "EcoVision AI - Avaliação com COVID Sintético"
    )

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
        "Teste REAL:",
        len(
            test_dataset
        )
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
        "Experimento:",
        checkpoint[
            "experiment"
        ]
    )


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
    # TESTE
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
    # RESULTADOS
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


    covid_result = (
        resultados[
            "por_classe"
        ][
            "COVID"
        ]
    )


    print(

        f"COVID Recall: "
        f"{resultados['covid_recall'] * 100:.2f}%"
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
    # RECALL POR CLASSE
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


    relatorio = classification_report(

        targets,

        predictions,

        target_names=classes,

        digits=4,

        zero_division=0
    )


    print(
        relatorio
    )


    # ========================================================
    # MATRIZ
    # ========================================================

    salvar_matriz_confusao(

        matriz,

        classes
    )


    # ========================================================
    # JSON
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
        "synthetic_covid_count"
    ] = 84


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
    # COMPARAÇÃO COM BASELINE
    # ========================================================

    comparar_com_baseline(
        resultados
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "Matriz salva:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )


    print()
    print(
        "Resultados salvos:"
    )

    print(
        RESULTS_PATH
    )


    print()
    print("=" * 70)

    print(
        "AVALIAÇÃO COM COVID SINTÉTICO FINALIZADA"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()