# ============================================================
# EcoVision AI
# Atividade 3 - Dataset para CNN Pré-treinada
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
from collections import Counter

import kagglehub


# ============================================================
# DIRETÓRIOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT

DATA_DIR = PROJECT_DIR / "data"

CNN_DATA_DIR = DATA_DIR / "cnn_kaggle"

CNN_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Extensões de imagem que vamos considerar.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# DOWNLOAD
# ============================================================

def baixar_dataset():
    """
    Baixa o dataset público:

        pavansanagapati/images-dataset

    utilizando kagglehub.

    O kagglehub utiliza cache local e retorna
    o caminho onde o dataset foi armazenado.
    """

    print()
    print("=" * 70)
    print("DOWNLOAD DO DATASET CNN")
    print("=" * 70)

    caminho = kagglehub.dataset_download(
        "pavansanagapati/images-dataset"
    )

    caminho = Path(caminho)

    print()
    print(
        "Dataset localizado em:"
    )

    print(
        caminho
    )

    return caminho


# ============================================================
# INSPECIONAR ESTRUTURA
# ============================================================

def listar_diretorios(caminho):
    """
    Exibe os principais diretórios encontrados.

    Isso é importante porque ainda precisamos
    confirmar como o dataset do Kaggle está organizado.
    """

    print()
    print("=" * 70)
    print("ESTRUTURA DE DIRETÓRIOS")
    print("=" * 70)

    diretorios = [
        item
        for item in caminho.rglob("*")
        if item.is_dir()
    ]

    # Mostrar somente os primeiros diretórios para
    # não gerar uma saída enorme no terminal.
    for diretorio in diretorios[:40]:

        print(
            diretorio.relative_to(caminho)
        )


# ============================================================
# LOCALIZAR IMAGENS
# ============================================================

def localizar_imagens(caminho):
    """
    Localiza todas as imagens do dataset.
    """

    imagens = [
        arquivo
        for arquivo in caminho.rglob("*")
        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return imagens


# ============================================================
# INFERIR CLASSES PELAS PASTAS
# ============================================================

def analisar_classes(
    imagens,
    caminho_base
):
    """
    Assume inicialmente que o nome da pasta
    imediatamente acima da imagem representa sua classe.

    Exemplo:

        dataset/
            cars/
                imagem01.jpg

    Classe:

        cars
    """

    contagem = Counter()

    for imagem in imagens:

        classe = imagem.parent.name

        contagem[
            classe
        ] += 1


    print()
    print("=" * 70)
    print("CONTAGEM INICIAL POR PASTA")
    print("=" * 70)

    for classe, quantidade in sorted(
        contagem.items()
    ):

        print(
            f"{classe:30s}: "
            f"{quantidade}"
        )


    print()
    print(
        f"Total de possíveis classes: "
        f"{len(contagem)}"
    )


    print(
        f"Total de imagens: "
        f"{len(imagens)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    caminho = baixar_dataset()

    listar_diretorios(
        caminho
    )

    imagens = localizar_imagens(
        caminho
    )

    print()
    print(
        f"Imagens encontradas: "
        f"{len(imagens)}"
    )

    analisar_classes(
        imagens,
        caminho
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()