# ============================================================
# EcoVision AI
# Atividade 2 - ADS-16 para CLIP
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
from collections import Counter

import kagglehub


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_DIR = PROJECT_ROOT

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif"
}


# ============================================================
# DOWNLOAD
# ============================================================

def baixar_dataset():
    """
    Baixa o ADS-16 disponível publicamente no Kaggle.

    Dataset:
        groffo/ads16-dataset
    """

    print()
    print("=" * 70)
    print("EcoVision AI - Download ADS-16")
    print("=" * 70)

    caminho = kagglehub.dataset_download(
        "groffo/ads16-dataset"
    )

    caminho = Path(caminho)

    print()
    print("Dataset localizado em:")
    print(caminho)

    return caminho


# ============================================================
# LOCALIZAR IMAGENS
# ============================================================

def localizar_imagens(dataset_path):
    """
    Procura recursivamente arquivos de imagem
    dentro do dataset.
    """

    imagens = [
        arquivo
        for arquivo in dataset_path.rglob("*")
        if (
            arquivo.is_file()
            and arquivo.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    return imagens


# ============================================================
# ANALISAR PASTAS
# ============================================================

def analisar_pastas(
    dataset_path,
    imagens
):
    """
    Conta quantas imagens aparecem em cada pasta-pai.

    Isso ajuda a identificar:
      - anúncios
      - categorias
      - fotos de usuários
      - possíveis arquivos auxiliares
    """

    contagem = Counter()

    for imagem in imagens:

        pasta_relativa = (
            imagem.parent
            .relative_to(dataset_path)
        )

        contagem[str(pasta_relativa)] += 1


    print()
    print("=" * 70)
    print("PASTAS CONTENDO IMAGENS")
    print("=" * 70)

    for pasta, quantidade in sorted(
        contagem.items()
    ):

        print(
            f"{pasta:60s} : "
            f"{quantidade}"
        )


# ============================================================
# MOSTRAR ESTRUTURA
# ============================================================

def mostrar_diretorios(
    dataset_path
):
    """
    Exibe os primeiros diretórios encontrados.
    """

    diretorios = [
        item
        for item in dataset_path.rglob("*")
        if item.is_dir()
    ]

    print()
    print("=" * 70)
    print("ESTRUTURA DO DATASET")
    print("=" * 70)

    for diretorio in diretorios[:100]:

        try:

            relativo = diretorio.relative_to(
                dataset_path
            )

        except ValueError:

            relativo = diretorio

        print(relativo)


# ============================================================
# MAIN
# ============================================================

def main():

    dataset_path = baixar_dataset()

    mostrar_diretorios(
        dataset_path
    )

    imagens = localizar_imagens(
        dataset_path
    )

    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)

    print(
        f"Total de arquivos de imagem encontrados: "
        f"{len(imagens)}"
    )

    analisar_pastas(
        dataset_path,
        imagens
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()