# ============================================================
# EcoVision AI
# Limpeza do dataset da atividade CNN
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import shutil

from PIL import Image


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT

SOURCE_DIR = Path(
    r"C:\Users\pablo\.cache\kagglehub"
    r"\datasets\pavansanagapati"
    r"\images-dataset\versions\1"
    r"\data\data"
)

DESTINATION_DIR = (
    PROJECT_DIR
    / "data"
    / "cnn_clean"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# VALIDAR IMAGEM
# ============================================================

def imagem_valida(caminho):
    """
    Verifica se a imagem pode ser aberta corretamente.

    Image.verify() identifica vários casos de
    arquivos corrompidos ou inválidos.
    """

    try:

        with Image.open(caminho) as imagem:

            imagem.verify()

        return True

    except Exception as erro:

        print(
            f"Imagem inválida: {caminho}"
        )

        print(
            f"Erro: {erro}"
        )

        return False


# ============================================================
# LIMPAR / COPIAR DATASET
# ============================================================

def limpar_dataset():

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado:\n{SOURCE_DIR}"
        )


    # --------------------------------------------------------
    # Se já existir uma versão limpa anterior,
    # removemos para não acumular arquivos.
    # --------------------------------------------------------

    if DESTINATION_DIR.exists():

        shutil.rmtree(
            DESTINATION_DIR
        )


    DESTINATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    total_copiadas = 0

    total_invalidas = 0


    # ========================================================
    # CLASSES
    # ========================================================

    classes = [
        pasta
        for pasta in SOURCE_DIR.iterdir()
        if pasta.is_dir()
    ]


    print()
    print("=" * 70)

    print(
        "LIMPANDO DATASET CNN"
    )

    print("=" * 70)


    for classe_dir in sorted(classes):

        nome_classe = classe_dir.name


        destino_classe = (
            DESTINATION_DIR
            / nome_classe
        )


        destino_classe.mkdir(
            parents=True,
            exist_ok=True
        )


        imagens_classe = 0


        for arquivo in classe_dir.iterdir():

            if not arquivo.is_file():

                continue


            if (
                arquivo.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):

                continue


            # ------------------------------------------------
            # Verificar se a imagem está íntegra.
            # ------------------------------------------------

            if not imagem_valida(
                arquivo
            ):

                total_invalidas += 1

                continue


            # ------------------------------------------------
            # Copiar imagem válida.
            # ------------------------------------------------

            shutil.copy2(

                arquivo,

                destino_classe
                / arquivo.name
            )


            imagens_classe += 1

            total_copiadas += 1


        print(
            f"{nome_classe:20s}: "
            f"{imagens_classe} imagens"
        )


    print()
    print("=" * 70)

    print(
        f"Imagens válidas copiadas: "
        f"{total_copiadas}"
    )

    print(
        f"Imagens inválidas: "
        f"{total_invalidas}"
    )

    print()

    print(
        "Dataset limpo salvo em:"
    )

    print(
        DESTINATION_DIR
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    limpar_dataset()