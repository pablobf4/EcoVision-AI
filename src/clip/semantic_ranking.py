# ============================================================
# EcoVision AI
# Atividade 2 - CLIP
# Ranking de conceitos por frequência semântica
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json

import matplotlib.pyplot as plt
import numpy as np
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


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "clip"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


BATCH_SIZE = 16


# ------------------------------------------------------------
# Threshold inicial.
#
# Vamos utilizar 0.25 inicialmente.
#
# IMPORTANTE:
# Esse valor ainda não será considerado definitivo.
#
# Depois da execução vamos analisar:
#
# - média
# - mediana
# - percentil 75
# - percentil 90
# - máximo
#
# para verificar se 0.25 representa adequadamente
# uma correspondência semântica relevante.
# ------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.22


# ============================================================
# CONCEITOS
# ============================================================
#
# A atividade exige pelo menos 20 descrições.
#
# Estamos utilizando 24 conceitos.
#
# Eles incluem:
#
# - pessoas
# - veículos
# - alimentos
# - eletrônicos
# - objetos
# - ambientes
# - conceitos visuais
#
# ============================================================

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


# ============================================================
# DISPOSITIVO
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CARREGAR CLIP
# ============================================================

def carregar_clip():
    """
    Carrega o modelo CLIP e seu processor.

    Nesta atividade NÃO realizamos treinamento.

    Utilizamos o modelo pré-treinado para extrair:

        embeddings de texto
        embeddings de imagem

    Esses embeddings serão comparados através
    de cosine similarity.
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


    # --------------------------------------------------------
    # Processor
    #
    # Responsável por:
    #
    # texto:
    # - tokenização
    # - padding
    # - attention mask
    #
    # imagem:
    # - resize
    # - crop
    # - normalização
    #
    # --------------------------------------------------------

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )


    # --------------------------------------------------------
    # Modelo CLIP.
    # --------------------------------------------------------

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )


    # Envia modelo para CPU ou GPU.
    model = model.to(
        DEVICE
    )


    # Modo de avaliação.
    #
    # Desativa comportamentos específicos
    # de treinamento, como dropout.
    model.eval()


    return (
        model,
        processor
    )


# ============================================================
# CARREGAR MANIFEST
# ============================================================

def carregar_manifest():
    """
    Carrega o manifest produzido anteriormente.

    O arquivo contém as 600 imagens selecionadas
    do corpus ADS-16.
    """

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest não encontrado:\n"
            f"{MANIFEST_PATH}"
        )


    dataframe = pd.read_csv(
        MANIFEST_PATH
    )


    # --------------------------------------------------------
    # Validação básica.
    # --------------------------------------------------------

    if "arquivo" not in dataframe.columns:

        raise ValueError(
            "O manifest não possui a coluna 'arquivo'."
        )


    if len(dataframe) < 500:

        raise ValueError(
            "O corpus possui menos de 500 imagens."
        )


    print()
    print(
        f"Imagens no manifest: "
        f"{len(dataframe)}"
    )


    return dataframe


# ============================================================
# NORMALIZAÇÃO DOS EMBEDDINGS
# ============================================================

def normalizar_embeddings(
    embeddings
):
    """
    Aplica normalização L2 aos embeddings.

    Exemplo:

        embedding
            ↓
        divide pela norma L2
            ↓
        vetor de tamanho 1

    Depois disso:

        A @ B.T

    equivale à cosine similarity.

    Isso acontece porque:

        cosine_similarity(A, B)

             A . B
        = -----------
          ||A|| ||B||

    Como os vetores possuem norma 1:

        ||A|| = 1
        ||B|| = 1

    então:

        cosine_similarity(A, B)
        =
        A . B
    """

    normas = embeddings.norm(
        dim=-1,
        keepdim=True
    )


    # Evita eventual divisão por zero.
    normas = normas.clamp(
        min=1e-12
    )


    return (
        embeddings
        / normas
    )


# ============================================================
# EMBEDDINGS DE TEXTO
# ============================================================

def gerar_embeddings_texto(
    model,
    processor
):
    """
    Converte os conceitos textuais em embeddings CLIP.

    Fluxo:

        texto
          ↓
        tokenizer
          ↓
        input_ids
        attention_mask
          ↓
        CLIP Text Transformer
          ↓
        pooler_output
          ↓
        text_projection
          ↓
        embedding multimodal
          ↓
        normalização L2


    IMPORTANTE:

    Não utilizamos diretamente:

        model.get_text_features()

    porque algumas versões recentes do Transformers
    podem retornar um objeto estruturado em vez de
    um Tensor diretamente.

    Acessamos explicitamente:

        model.text_model

    e depois:

        model.text_projection
    """

    # --------------------------------------------------------
    # Tokenização
    # --------------------------------------------------------

    inputs = processor(
        text=CONCEPTS,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    # Enviar tensores para o dispositivo.
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


    # --------------------------------------------------------
    # inference_mode:
    #
    # não calcula gradientes.
    #
    # É ideal porque estamos fazendo apenas inferência.
    # --------------------------------------------------------

    with torch.inference_mode():

        # ====================================================
        # TEXT ENCODER
        # ====================================================

        text_outputs = model.text_model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )


        # ----------------------------------------------------
        # pooler_output
        #
        # Representação global da sequência textual.
        # ----------------------------------------------------

        pooled_output = (
            text_outputs.pooler_output
        )


        # ----------------------------------------------------
        # Projeção
        #
        # O encoder textual e o encoder visual possuem
        # representações internas diferentes.
        #
        # text_projection leva o texto para o espaço
        # multimodal compartilhado pelo CLIP.
        # ----------------------------------------------------

        text_features = (
            model.text_projection(
                pooled_output
            )
        )


    # --------------------------------------------------------
    # Normalização.
    # --------------------------------------------------------

    text_features = normalizar_embeddings(
        text_features
    )


    print()
    print(
        "Embeddings textuais:",
        text_features.shape
    )


    return text_features


# ============================================================
# CARREGAR UMA IMAGEM
# ============================================================

def carregar_imagem(
    caminho
):
    """
    Carrega uma imagem e converte para RGB.

    O .copy() permite fechar o arquivo original
    imediatamente após a leitura.
    """

    with Image.open(
        caminho
    ) as imagem:

        imagem_rgb = (
            imagem
            .convert("RGB")
            .copy()
        )


    return imagem_rgb


# ============================================================
# EMBEDDINGS DAS IMAGENS
# ============================================================

def gerar_embeddings_imagens(
    model,
    processor,
    dataframe
):
    """
    Converte as 600 imagens em embeddings CLIP.

    Processamos em batches para reduzir o uso
    de memória.

    Fluxo:

        imagem
          ↓
        CLIP Processor
          ↓
        pixel_values
          ↓
        Vision Transformer
          ↓
        pooler_output
          ↓
        visual_projection
          ↓
        embedding multimodal
          ↓
        normalização L2
    """

    todos_embeddings = []


    caminhos = dataframe[
        "arquivo"
    ].tolist()


    total_batches = (

        len(caminhos)
        + BATCH_SIZE
        - 1

    ) // BATCH_SIZE


    print()
    print("=" * 70)
    print("GERANDO EMBEDDINGS VISUAIS")
    print("=" * 70)


    # ========================================================
    # PROCESSAMENTO EM BATCHES
    # ========================================================

    for inicio in range(
        0,
        len(caminhos),
        BATCH_SIZE
    ):

        fim = min(

            inicio
            + BATCH_SIZE,

            len(caminhos)
        )


        batch_paths = caminhos[
            inicio:fim
        ]


        imagens = []


        # ====================================================
        # CARREGAR IMAGENS
        # ====================================================

        for caminho in batch_paths:

            imagem = carregar_imagem(
                caminho
            )


            imagens.append(
                imagem
            )


        # ====================================================
        # PREPROCESSAMENTO CLIP
        # ====================================================

        inputs = processor(
            images=imagens,
            return_tensors="pt"
        )


        pixel_values = (
            inputs[
                "pixel_values"
            ]
            .to(DEVICE)
        )


        # ====================================================
        # IMAGE ENCODER
        # ====================================================

        with torch.inference_mode():

            vision_outputs = model.vision_model(
                pixel_values=pixel_values
            )


            # ------------------------------------------------
            # Representação global da imagem.
            # ------------------------------------------------

            pooled_output = (
                vision_outputs.pooler_output
            )


            # ------------------------------------------------
            # Projeta para o mesmo espaço vetorial
            # utilizado pelo texto.
            # ------------------------------------------------

            image_features = (
                model.visual_projection(
                    pooled_output
                )
            )


        # ====================================================
        # NORMALIZAÇÃO
        # ====================================================

        image_features = normalizar_embeddings(
            image_features
        )


        # ----------------------------------------------------
        # Guardamos os embeddings na CPU.
        #
        # Isso reduz consumo de GPU e permite salvar depois.
        # ----------------------------------------------------

        todos_embeddings.append(
            image_features.cpu()
        )


        batch_atual = (
            inicio // BATCH_SIZE
            + 1
        )


        print(
            f"Batch "
            f"{batch_atual:02d}/"
            f"{total_batches:02d}"
        )


    # ========================================================
    # CONCATENAR TODOS OS BATCHES
    # ========================================================

    embeddings = torch.cat(
        todos_embeddings,
        dim=0
    )


    print()
    print(
        "Embeddings visuais:",
        embeddings.shape
    )


    return embeddings


# ============================================================
# SIMILARIDADE
# ============================================================

def calcular_similaridades(
    image_embeddings,
    text_embeddings
):
    """
    Calcula cosine similarity entre:

        todas as imagens
        ×
        todos os conceitos

    Formatos esperados:

        image_embeddings
        [600, 512]

        text_embeddings
        [24, 512]

    Resultado:

        [600, 24]

    Cada posição representa:

        similaridade(
            imagem_i,
            conceito_j
        )
    """

    similarities = (

        image_embeddings
        @ text_embeddings.cpu().T

    )


    print()
    print(
        "Matriz de similaridade:",
        similarities.shape
    )


    return similarities


# ============================================================
# DISTRIBUIÇÃO DOS SCORES
# ============================================================

def analisar_distribuicao(
    similarities
):
    """
    Analisa a distribuição global das similaridades.

    Essas estatísticas serão utilizadas depois
    para justificar tecnicamente o threshold.
    """

    valores = (
        similarities
        .numpy()
        .flatten()
    )


    print()
    print("=" * 70)
    print("DISTRIBUIÇÃO DAS SIMILARIDADES")
    print("=" * 70)


    print(
        f"Mínimo: "
        f"{valores.min():.4f}"
    )


    print(
        f"Média: "
        f"{valores.mean():.4f}"
    )


    print(
        f"Mediana: "
        f"{np.median(valores):.4f}"
    )


    print(
        f"Desvio padrão: "
        f"{valores.std():.4f}"
    )


    print(
        f"Percentil 75: "
        f"{np.percentile(valores, 75):.4f}"
    )


    print(
        f"Percentil 90: "
        f"{np.percentile(valores, 90):.4f}"
    )


    print(
        f"Percentil 95: "
        f"{np.percentile(valores, 95):.4f}"
    )


    print(
        f"Máximo: "
        f"{valores.max():.4f}"
    )


# ============================================================
# RANKING
# ============================================================

def gerar_ranking(
    similarities
):
    """
    Para cada conceito:

    1. pega todos os scores;
    2. verifica quantas imagens possuem score
       maior ou igual ao threshold;
    3. calcula frequência;
    4. calcula percentual;
    5. calcula score médio das ocorrências.

    Depois ordenamos pela frequência.
    """

    resultados = []


    for indice, conceito in enumerate(
        CONCEPTS
    ):

        scores = (
            similarities[
                :,
                indice
            ]
            .numpy()
        )


        # ----------------------------------------------------
        # Imagens consideradas semanticamente relacionadas.
        # ----------------------------------------------------

        mascara = (
            scores
            >= SIMILARITY_THRESHOLD
        )


        frequencia = int(
            mascara.sum()
        )


        percentual = (

            frequencia
            / similarities.shape[0]
            * 100

        )


        # ----------------------------------------------------
        # Score médio apenas das imagens acima do threshold.
        # ----------------------------------------------------

        if frequencia > 0:

            score_medio = float(
                scores[
                    mascara
                ].mean()
            )

        else:

            score_medio = 0.0


        # ----------------------------------------------------
        # Também guardamos o maior score encontrado.
        # ----------------------------------------------------

        score_maximo = float(
            scores.max()
        )


        resultados.append({

            "conceito": conceito,

            "frequencia": frequencia,

            "percentual": percentual,

            "score_medio": score_medio,

            "score_maximo": score_maximo
        })


    # ========================================================
    # ORDENAR
    # ========================================================

    resultados = sorted(

        resultados,

        key=lambda item: (

            item[
                "frequencia"
            ],

            item[
                "score_medio"
            ]

        ),

        reverse=True
    )


    return resultados


# ============================================================
# MOSTRAR RANKING
# ============================================================

def imprimir_ranking(
    ranking
):
    """
    Exibe o ranking completo no terminal.
    """

    print()
    print("=" * 70)
    print("RANKING DE CONCEITOS")
    print("=" * 70)


    print(
        f"Threshold: "
        f"{SIMILARITY_THRESHOLD:.2f}"
    )


    print()


    for posicao, item in enumerate(
        ranking,
        start=1
    ):

        print(

            f"{posicao:02d}. "

            f"{item['conceito']:25s} | "

            f"Freq: "
            f"{item['frequencia']:3d} | "

            f"{item['percentual']:6.2f}% | "

            f"Média: "
            f"{item['score_medio']:.4f} | "

            f"Máx: "
            f"{item['score_maximo']:.4f}"
        )


# ============================================================
# SALVAR RANKING
# ============================================================

def salvar_ranking(
    ranking
):
    """
    Salva o ranking em:

        CSV
        JSON
    """

    dataframe = pd.DataFrame(
        ranking
    )


    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        RESULTS_DIR
        / "clip_concept_ranking.csv"
    )


    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # JSON
    # ========================================================

    json_path = (
        RESULTS_DIR
        / "clip_concept_ranking.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            ranking,
            arquivo,
            indent=4,
            ensure_ascii=False
        )


    return (
        csv_path,
        json_path
    )


# ============================================================
# GRÁFICO TOP-5
# ============================================================

def salvar_grafico_top5(
    ranking
):
    """
    Gera gráfico dos cinco conceitos
    semanticamente mais frequentes.
    """

    top5 = ranking[
        :5
    ]


    nomes = [
        item[
            "conceito"
        ]
        for item in top5
    ]


    frequencias = [
        item[
            "frequencia"
        ]
        for item in top5
    ]


    plt.figure(
        figsize=(10, 6)
    )


    plt.barh(
        nomes[::-1],
        frequencias[::-1]
    )


    plt.xlabel(
        "Número de imagens acima do threshold"
    )


    plt.ylabel(
        "Conceito"
    )


    plt.title(
        "CLIP - Top 5 conceitos por frequência semântica"
    )


    plt.tight_layout()


    caminho = (
        RESULTS_DIR
        / "clip_top5_concepts.png"
    )


    plt.savefig(
        caminho,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    return caminho


# ============================================================
# EXEMPLOS VISUAIS TOP-5
# ============================================================

def salvar_exemplos_top5(
    ranking,
    similarities,
    dataframe
):
    """
    Para cada um dos cinco conceitos mais frequentes,
    seleciona as três imagens com maior similaridade.

    Resultado:

        5 conceitos
        ×
        3 exemplos

        =
        15 imagens

    Isso ajuda a verificar visualmente se o ranking
    produzido pelo CLIP faz sentido.
    """

    top5 = ranking[
        :5
    ]


    figura, eixos = plt.subplots(

        len(top5),

        3,

        figsize=(
            12,
            16
        )
    )


    # ========================================================
    # PERCORRER TOP-5
    # ========================================================

    for linha, item in enumerate(
        top5
    ):

        conceito = item[
            "conceito"
        ]


        indice_conceito = (
            CONCEPTS.index(
                conceito
            )
        )


        scores = (
            similarities[
                :,
                indice_conceito
            ]
            .numpy()
        )


        # ----------------------------------------------------
        # Índices das três imagens com maior similaridade.
        # ----------------------------------------------------

        melhores_indices = (
            np.argsort(
                scores
            )[-3:][::-1]
        )


        # ====================================================
        # MOSTRAR EXEMPLOS
        # ====================================================

        for coluna, indice_imagem in enumerate(
            melhores_indices
        ):

            caminho = dataframe.iloc[
                indice_imagem
            ][
                "arquivo"
            ]


            imagem = carregar_imagem(
                caminho
            )


            eixo = eixos[
                linha,
                coluna
            ]


            eixo.imshow(
                imagem
            )


            eixo.axis(
                "off"
            )


            eixo.set_title(

                f"{conceito}\n"

                f"sim="
                f"{scores[indice_imagem]:.3f}"
            )


    plt.tight_layout()


    caminho = (
        RESULTS_DIR
        / "clip_top5_examples.png"
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
# SALVAR EMBEDDINGS
# ============================================================

def salvar_embeddings(
    image_embeddings
):
    """
    Salva os embeddings das 600 imagens.

    Isso é importante porque a próxima etapa será:

        busca semântica por texto

    Como os embeddings das imagens não mudam,
    não precisamos passar novamente todas as
    600 imagens pelo CLIP.
    """

    caminho = (
        RESULTS_DIR
        / "clip_image_embeddings.pt"
    )


    torch.save(
        image_embeddings,
        caminho
    )


    return caminho


# ============================================================
# SALVAR METADADOS
# ============================================================

def salvar_configuracao():
    """
    Guarda informações importantes do experimento.

    Isso facilita a reprodução posterior.
    """

    configuracao = {

        "modelo": MODEL_NAME,

        "batch_size": BATCH_SIZE,

        "threshold": SIMILARITY_THRESHOLD,

        "quantidade_conceitos": len(
            CONCEPTS
        ),

        "conceitos": CONCEPTS
    }


    caminho = (
        RESULTS_DIR
        / "clip_config.json"
    )


    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            configuracao,
            arquivo,
            indent=4,
            ensure_ascii=False
        )


    return caminho


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - CLIP Semantic Ranking")
    print("=" * 70)


    # ========================================================
    # MANIFEST
    # ========================================================

    dataframe = carregar_manifest()


    # ========================================================
    # CLIP
    # ========================================================

    (
        model,
        processor
    ) = carregar_clip()


    # ========================================================
    # EMBEDDINGS DE TEXTO
    # ========================================================

    text_embeddings = gerar_embeddings_texto(

        model,

        processor
    )


    # ========================================================
    # EMBEDDINGS DAS IMAGENS
    # ========================================================

    image_embeddings = gerar_embeddings_imagens(

        model,

        processor,

        dataframe
    )


    # ========================================================
    # VALIDAÇÕES
    # ========================================================

    assert (
        image_embeddings.ndim
        == 2
    )


    assert (
        text_embeddings.ndim
        == 2
    )


    assert (
        image_embeddings.shape[0]
        == len(dataframe)
    )


    assert (
        text_embeddings.shape[0]
        == len(CONCEPTS)
    )


    assert (
        image_embeddings.shape[1]
        == text_embeddings.shape[1]
    )


    # ========================================================
    # SIMILARIDADE
    # ========================================================

    similarities = calcular_similaridades(

        image_embeddings,

        text_embeddings
    )


    assert similarities.shape == (

        len(dataframe),

        len(CONCEPTS)
    )


    # ========================================================
    # ANALISAR SCORES
    # ========================================================

    analisar_distribuicao(
        similarities
    )


    # ========================================================
    # RANKING
    # ========================================================

    ranking = gerar_ranking(
        similarities
    )


    imprimir_ranking(
        ranking
    )


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    (
        csv_path,
        json_path
    ) = salvar_ranking(
        ranking
    )


    grafico_path = salvar_grafico_top5(
        ranking
    )


    exemplos_path = salvar_exemplos_top5(

        ranking,

        similarities,

        dataframe
    )


    embeddings_path = salvar_embeddings(
        image_embeddings
    )


    configuracao_path = salvar_configuracao()


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADOS SALVOS")
    print("=" * 70)


    print()
    print(
        "Ranking CSV:"
    )

    print(
        csv_path
    )


    print()
    print(
        "Ranking JSON:"
    )

    print(
        json_path
    )


    print()
    print(
        "Gráfico Top-5:"
    )

    print(
        grafico_path
    )


    print()
    print(
        "Exemplos Top-5:"
    )

    print(
        exemplos_path
    )


    print()
    print(
        "Embeddings:"
    )

    print(
        embeddings_path
    )


    print()
    print(
        "Configuração:"
    )

    print(
        configuracao_path
    )


    print()
    print("=" * 70)

    print(
        "CLIP SEMANTIC RANKING FINALIZADO"
    )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()