# ============================================================
# EcoVision AI
# Atividade 2 - CLIP
# Busca semântica por consulta textual
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch

from PIL import Image

from transformers import (
    CLIPModel,
    CLIPProcessor
)


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT


MANIFEST_PATH = (
    PROJECT_DIR
    / "data"
    / "ads16_clip_manifest.csv"
)


EMBEDDINGS_PATH = (
    PROJECT_DIR
    / "results"
    / "clip"
    / "clip_image_embeddings.pt"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "clip"
)


SEARCH_RESULTS_DIR = (
    RESULTS_DIR
    / "semantic_search"
)


SEARCH_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


TOP_K = 5


# ============================================================
# CONSULTAS
# ============================================================
#
# A atividade exige pelo menos 8 consultas variando:
#
# - genérico -> específico
# - concreto -> abstrato
#
# Essas consultas foram escolhidas justamente para permitir
# analisar diferentes capacidades do CLIP.
# ============================================================

QUERIES = [

    # --------------------------------------------------------
    # CONSULTAS GENÉRICAS / CONCRETAS
    # --------------------------------------------------------

    "a person",

    "an animal",

    "a food product",

    # --------------------------------------------------------
    # CONSULTAS MAIS ESPECÍFICAS
    # --------------------------------------------------------

    "a person outdoors",

    "a child playing sports",

    "a person using an electronic device",

    # --------------------------------------------------------
    # CONSULTAS MAIS ABSTRATAS
    # --------------------------------------------------------

    "a relaxing outdoor scene",

    "a happy social moment"
]


# ============================================================
# DISPOSITIVO
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NORMALIZAR EMBEDDINGS
# ============================================================

def normalizar_embeddings(
    embeddings
):
    """
    Normaliza embeddings utilizando norma L2.

    Depois disso, o produto escalar equivale
    à cosine similarity.
    """

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
# CARREGAR DADOS
# ============================================================

def carregar_dados():
    """
    Carrega:

    - manifest das 600 imagens
    - embeddings visuais já calculados
    """

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest não encontrado:\n"
            f"{MANIFEST_PATH}"
        )


    if not EMBEDDINGS_PATH.exists():

        raise FileNotFoundError(
            f"Embeddings não encontrados:\n"
            f"{EMBEDDINGS_PATH}"
        )


    dataframe = pd.read_csv(
        MANIFEST_PATH
    )


    image_embeddings = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu"
    )


    print()
    print(
        "Imagens:",
        len(dataframe)
    )


    print(
        "Embeddings:",
        image_embeddings.shape
    )


    # --------------------------------------------------------
    # Validação.
    # --------------------------------------------------------

    assert (
        image_embeddings.shape[0]
        == len(dataframe)
    )


    return (
        dataframe,
        image_embeddings
    )


# ============================================================
# CARREGAR CLIP
# ============================================================

def carregar_clip():
    """
    Carrega somente o CLIP necessário para processar
    as consultas textuais.

    Não precisamos passar novamente as 600 imagens
    pelo encoder visual.
    """

    print()
    print("=" * 70)
    print("CARREGANDO CLIP")
    print("=" * 70)


    print(
        "Modelo:",
        MODEL_NAME
    )


    print(
        "Dispositivo:",
        DEVICE
    )


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


    return (
        model,
        processor
    )


# ============================================================
# GERAR EMBEDDING DA CONSULTA
# ============================================================

def gerar_embedding_consulta(
    consulta,
    model,
    processor
):
    """
    Transforma uma consulta textual em um embedding CLIP.

    Fluxo:

        consulta
           ↓
        tokenizer
           ↓
        CLIP Text Transformer
           ↓
        pooler_output
           ↓
        text_projection
           ↓
        embedding 512 dimensões
           ↓
        normalização L2
    """

    inputs = processor(
        text=[consulta],
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

        text_outputs = model.text_model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )


        text_embedding = (
            model.text_projection(
                text_outputs.pooler_output
            )
        )


    text_embedding = normalizar_embeddings(
        text_embedding
    )


    return (
        text_embedding.cpu()
    )


# ============================================================
# REALIZAR BUSCA
# ============================================================

def buscar(
    consulta,
    model,
    processor,
    dataframe,
    image_embeddings
):
    """
    Executa busca semântica.

    Para uma consulta:

        1. gera embedding textual
        2. compara contra as 600 imagens
        3. calcula cosine similarity
        4. ordena os scores
        5. retorna as Top-5 imagens
    """

    text_embedding = gerar_embedding_consulta(

        consulta,

        model,

        processor
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================
    #
    # image_embeddings:
    #
    #     [600, 512]
    #
    # text_embedding:
    #
    #     [1, 512]
    #
    # Resultado:
    #
    #     [600]
    # ========================================================

    similarities = (

        image_embeddings
        @ text_embedding.T

    ).squeeze(1)


    # ========================================================
    # TOP-K
    # ========================================================

    scores, indices = torch.topk(

        similarities,

        k=TOP_K
    )


    resultados = []


    for posicao, (
        indice,
        score
    ) in enumerate(
        zip(
            indices.tolist(),
            scores.tolist()
        ),
        start=1
    ):

        linha = dataframe.iloc[
            indice
        ]


        resultados.append({

            "posicao": posicao,

            "indice": indice,

            "score": float(
                score
            ),

            "arquivo": linha[
                "arquivo"
            ],

            "usuario": linha[
                "usuario"
            ],

            "sentimento": linha[
                "sentimento"
            ]
        })


    return resultados


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

def imprimir_resultados(
    consulta,
    resultados
):
    """
    Exibe o Top-5 da consulta no terminal.
    """

    print()
    print("=" * 70)

    print(
        f'CONSULTA: "{consulta}"'
    )

    print("=" * 70)


    for item in resultados:

        print(

            f"{item['posicao']}. "

            f"similaridade="
            f"{item['score']:.4f}"

            f" | usuário="
            f"{item['usuario']}"

            f" | grupo="
            f"{item['sentimento']}"
        )


# ============================================================
# CARREGAR IMAGEM
# ============================================================

def carregar_imagem(
    caminho
):
    """
    Carrega uma imagem em RGB.
    """

    with Image.open(
        caminho
    ) as imagem:

        return (
            imagem
            .convert("RGB")
            .copy()
        )


# ============================================================
# CRIAR NOME SEGURO
# ============================================================

def criar_slug(
    texto
):
    """
    Transforma a consulta em um nome de arquivo seguro.

    Exemplo:

        "a person outdoors"

    vira:

        a_person_outdoors
    """

    texto = texto.lower()


    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto
    )


    return texto.strip(
        "_"
    )


# ============================================================
# SALVAR VISUALIZAÇÃO
# ============================================================

def salvar_visualizacao(
    consulta,
    resultados
):
    """
    Cria uma figura contendo as cinco imagens
    recuperadas para uma consulta.
    """

    figura, eixos = plt.subplots(

        1,

        TOP_K,

        figsize=(
            15,
            4
        )
    )


    for indice, item in enumerate(
        resultados
    ):

        imagem = carregar_imagem(
            item[
                "arquivo"
            ]
        )


        eixo = eixos[
            indice
        ]


        eixo.imshow(
            imagem
        )


        eixo.axis(
            "off"
        )


        eixo.set_title(

            f"Top {item['posicao']}\n"

            f"sim="
            f"{item['score']:.3f}"
        )


    figura.suptitle(

        f'Consulta: "{consulta}"',

        fontsize=14
    )


    plt.tight_layout()


    slug = criar_slug(
        consulta
    )


    caminho = (

        SEARCH_RESULTS_DIR

        / f"{slug}.png"
    )


    plt.savefig(

        caminho,

        dpi=150,

        bbox_inches="tight"
    )


    plt.close(
        figura
    )


    return caminho


# ============================================================
# SALVAR RESULTADOS
# ============================================================

def salvar_resultados(
    todos_resultados
):
    """
    Salva os resultados completos em JSON e CSV.
    """

    # ========================================================
    # JSON
    # ========================================================

    json_path = (

        RESULTS_DIR

        / "clip_semantic_search_results.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(

            todos_resultados,

            arquivo,

            indent=4,

            ensure_ascii=False
        )


    # ========================================================
    # CSV
    # ========================================================

    linhas = []


    for consulta in todos_resultados:

        for resultado in consulta[
            "resultados"
        ]:

            linhas.append({

                "consulta": consulta[
                    "consulta"
                ],

                **resultado
            })


    dataframe = pd.DataFrame(
        linhas
    )


    csv_path = (

        RESULTS_DIR

        / "clip_semantic_search_results.csv"
    )


    dataframe.to_csv(

        csv_path,

        index=False,

        encoding="utf-8"
    )


    return (
        json_path,
        csv_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - CLIP Semantic Search")
    print("=" * 70)


    # ========================================================
    # DADOS
    # ========================================================

    (
        dataframe,
        image_embeddings
    ) = carregar_dados()


    # ========================================================
    # CLIP
    # ========================================================

    (
        model,
        processor
    ) = carregar_clip()


    todos_resultados = []


    # ========================================================
    # EXECUTAR AS 8 CONSULTAS
    # ========================================================

    for numero, consulta in enumerate(
        QUERIES,
        start=1
    ):

        resultados = buscar(

            consulta,

            model,

            processor,

            dataframe,

            image_embeddings
        )


        imprimir_resultados(

            consulta,

            resultados
        )


        imagem_resultado = salvar_visualizacao(

            consulta,

            resultados
        )


        todos_resultados.append({

            "numero": numero,

            "consulta": consulta,

            "visualizacao": str(
                imagem_resultado
            ),

            "resultados": resultados
        })


    # ========================================================
    # SALVAR
    # ========================================================

    (
        json_path,
        csv_path
    ) = salvar_resultados(
        todos_resultados
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("BUSCA SEMÂNTICA FINALIZADA")
    print("=" * 70)


    print()
    print(
        "Consultas executadas:",
        len(QUERIES)
    )


    print()
    print(
        "JSON:"
    )

    print(
        json_path
    )


    print()
    print(
        "CSV:"
    )

    print(
        csv_path
    )


    print()
    print(
        "Visualizações:"
    )

    print(
        SEARCH_RESULTS_DIR
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()