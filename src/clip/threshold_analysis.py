# ============================================================
# EcoVision AI
# Atividade 2 - CLIP
# Análise de sensibilidade do threshold
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT

import pandas as pd
import torch

from transformers import (
    CLIPModel,
    CLIPProcessor
)


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT


EMBEDDINGS_PATH = (
    PROJECT_DIR
    / "results"
    / "clip"
    / "clip_image_embeddings.pt"
)


RESULTS_PATH = (
    PROJECT_DIR
    / "results"
    / "clip"
    / "clip_threshold_analysis.csv"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


THRESHOLDS = [
    0.22,
    0.23,
    0.24,
    0.25
]


CONCEPTS = [

    "a person",
    "a human face",
    "a man",
    "a woman",
    "a child",

    "a car",
    "a bicycle",
    "a motorcycle",

    "a food product",
    "a beverage",

    "clothing",
    "shoes",

    "a smartphone",
    "a computer",
    "an electronic device",

    "furniture",

    "an animal",
    "a pet",

    "outdoor scenery",
    "a beach",
    "a city or building",

    "sports",

    "text and logo",
    "a packaged product"
]


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar_embeddings(
    embeddings
):

    normas = embeddings.norm(
        dim=-1,
        keepdim=True
    )

    normas = normas.clamp(
        min=1e-12
    )

    return (
        embeddings
        / normas
    )


# ============================================================
# EMBEDDINGS DOS TEXTOS
# ============================================================

def gerar_embeddings_texto(
    model,
    processor
):

    inputs = processor(
        text=CONCEPTS,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    input_ids = (
        inputs[
            "input_ids"
        ]
        .to(DEVICE)
    )


    attention_mask = (
        inputs[
            "attention_mask"
        ]
        .to(DEVICE)
    )


    with torch.inference_mode():

        outputs = model.text_model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )


        text_embeddings = (
            model.text_projection(
                outputs.pooler_output
            )
        )


    return normalizar_embeddings(
        text_embeddings
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Análise de Threshold CLIP")
    print("=" * 70)


    # ========================================================
    # CARREGAR EMBEDDINGS DAS IMAGENS
    # ========================================================

    image_embeddings = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu"
    )


    print()
    print(
        "Embeddings das imagens:",
        image_embeddings.shape
    )


    # ========================================================
    # CARREGAR CLIP
    # ========================================================

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )


    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    # ========================================================
    # EMBEDDINGS DOS CONCEITOS
    # ========================================================

    text_embeddings = gerar_embeddings_texto(

        model,

        processor
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    similarities = (

        image_embeddings
        @ text_embeddings.cpu().T

    )


    resultados = []


    # ========================================================
    # ANALISAR CADA THRESHOLD
    # ========================================================

    for threshold in THRESHOLDS:

        print()
        print("=" * 70)

        print(
            f"THRESHOLD = {threshold:.2f}"
        )

        print("=" * 70)


        total_acima = 0


        ranking = []


        for indice, conceito in enumerate(
            CONCEPTS
        ):

            scores = (
                similarities[
                    :,
                    indice
                ]
            )


            mascara = (
                scores
                >= threshold
            )


            frequencia = int(
                mascara.sum()
            )


            total_acima += frequencia


            if frequencia > 0:

                score_medio = float(
                    scores[
                        mascara
                    ].mean()
                )

            else:

                score_medio = 0.0


            ranking.append({

                "threshold": threshold,

                "conceito": conceito,

                "frequencia": frequencia,

                "percentual_imagens": (

                    frequencia
                    / image_embeddings.shape[0]
                    * 100

                ),

                "score_medio": score_medio
            })


        # ----------------------------------------------------
        # Ordenação.
        # ----------------------------------------------------

        ranking = sorted(

            ranking,

            key=lambda item: (
                item["frequencia"],
                item["score_medio"]
            ),

            reverse=True
        )


        # ----------------------------------------------------
        # Mostrar Top-10.
        # ----------------------------------------------------

        for posicao, item in enumerate(
            ranking[:10],
            start=1
        ):

            print(

                f"{posicao:02d}. "

                f"{item['conceito']:25s}"

                f" | "

                f"{item['frequencia']:3d}"

                f" | "

                f"{item['percentual_imagens']:6.2f}%"
            )


        # ----------------------------------------------------
        # Percentual de todas as comparações que
        # ficaram acima do threshold.
        # ----------------------------------------------------

        total_comparacoes = (
            similarities.shape[0]
            * similarities.shape[1]
        )


        percentual_global = (

            total_acima
            / total_comparacoes
            * 100
        )


        print()

        print(
            f"Comparações acima do threshold: "
            f"{total_acima}/"
            f"{total_comparacoes}"
        )

        print(
            f"Percentual global: "
            f"{percentual_global:.2f}%"
        )


        resultados.extend(
            ranking
        )


    # ========================================================
    # SALVAR
    # ========================================================

    dataframe = pd.DataFrame(
        resultados
    )


    dataframe.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8"
    )


    print()
    print("=" * 70)
    print("ANÁLISE FINALIZADA")
    print("=" * 70)

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":

    main()