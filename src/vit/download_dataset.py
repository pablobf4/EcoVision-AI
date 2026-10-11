from pathlib import Path
from src.common.paths import PROJECT_ROOT
import urllib.request
import zipfile


# ============================================================
# EcoVision AI
# Download e preparação do dataset RealWaste
# ============================================================


# ------------------------------------------------------------
# Caminhos do projeto
# ------------------------------------------------------------

# __file__ representa o arquivo atual: src/dataset.py
#
# .resolve() pega o caminho absoluto.
#
# .parent pega a pasta "src".
#
# .parent novamente sobe para a raiz "EcoVision-AI".
PROJECT_DIR = PROJECT_ROOT


# Pasta onde armazenaremos os dados.
DATA_DIR = PROJECT_DIR / "data"


# Caminho do arquivo ZIP.
ZIP_PATH = DATA_DIR / "realwaste.zip"


# Pasta onde o dataset será extraído.
EXTRACT_DIR = DATA_DIR / "realwaste"


# ------------------------------------------------------------
# URL do dataset
# ------------------------------------------------------------

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
)


def criar_diretorios():
    """
    Cria os diretórios necessários para armazenar o dataset.

    parents=True:
        cria também diretórios superiores caso não existam.

    exist_ok=True:
        não gera erro caso a pasta já exista.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def baixar_dataset():
    """
    Faz o download do dataset RealWaste.

    O download só acontece caso o arquivo ZIP ainda não exista.
    """

    if ZIP_PATH.exists():

        print("Dataset já foi baixado.")

        return

    print("Baixando dataset RealWaste...")

    urllib.request.urlretrieve(
        DATASET_URL,
        ZIP_PATH
    )

    print("Download concluído.")


def extrair_dataset():
    """
    Extrai o conteúdo do arquivo ZIP.
    """

    print("Extraindo dataset...")

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as arquivo_zip:

        arquivo_zip.extractall(
            EXTRACT_DIR
        )

    print("Extração concluída.")


def localizar_dataset():
    """
    Procura automaticamente pela pasta chamada 'RealWaste'.

    Isso evita depender da estrutura interna exata do arquivo ZIP.
    """

    pastas_encontradas = list(
        EXTRACT_DIR.rglob("RealWaste")
    )

    if not pastas_encontradas:

        raise FileNotFoundError(
            "Não foi possível localizar a pasta RealWaste."
        )

    return pastas_encontradas[0]


def listar_classes(dataset_path):
    """
    Lista as classes existentes no dataset.

    Cada pasta dentro de RealWaste representa uma classe.
    """

    classes = [
        pasta.name
        for pasta in dataset_path.iterdir()
        if pasta.is_dir()
    ]

    classes.sort()

    print()
    print("Classes encontradas:")

    for indice, classe in enumerate(classes):

        print(
            f"{indice}: {classe}"
        )

    return classes

def main():

    print("=" * 60)
    print("EcoVision AI - Preparação do dataset")
    print("=" * 60)

    criar_diretorios()

    baixar_dataset()

    # Para evitar extrair novamente em toda execução,
    # verifica primeiro se já existe uma pasta RealWaste.
    pastas_existentes = list(
        EXTRACT_DIR.rglob("RealWaste")
    )

    if not pastas_existentes:

        extrair_dataset()

    dataset_path = localizar_dataset()

    print()
    print("Dataset localizado em:")
    print(dataset_path)
    listar_classes(dataset_path)

# Esse bloco só é executado quando chama:
#
# python src/dataset.py
#
# Se futuramente importarmos dataset.py em outro arquivo,
# o main() não será executado automaticamente.
if __name__ == "__main__":
    main()