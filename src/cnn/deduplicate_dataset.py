# ============================================================
# EcoVision AI
# Remoção de imagens duplicadas do dataset CNN
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
from collections import defaultdict
import hashlib
import shutil


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT

SOURCE_DIR = (
    PROJECT_DIR
    / "data"
    / "cnn_clean"
)

DESTINATION_DIR = (
    PROJECT_DIR
    / "data"
    / "cnn_dedup"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# HASH
# ============================================================

def calcular_sha256(caminho):
    """
    Calcula SHA-256 utilizando o conteúdo da imagem.

    Arquivos com conteúdo exatamente igual
    terão o mesmo hash, mesmo que possuam nomes diferentes.
    """

    sha256 = hashlib.sha256()

    with open(caminho, "rb") as arquivo:

        for bloco in iter(
            lambda: arquivo.read(8192),
            b""
        ):
            sha256.update(bloco)

    return sha256.hexdigest()


# ============================================================
# LOCALIZAR IMAGENS
# ============================================================

def localizar_imagens():

    imagens = []

    for classe_dir in SOURCE_DIR.iterdir():

        if not classe_dir.is_dir():
            continue

        for arquivo in classe_dir.iterdir():

            if (
                arquivo.is_file()
                and arquivo.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                imagens.append(
                    arquivo
                )

    return imagens


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Remoção de Duplicatas")
    print("=" * 70)

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado:\n{SOURCE_DIR}"
        )


    # --------------------------------------------------------
    # Recriar diretório destino
    # --------------------------------------------------------

    if DESTINATION_DIR.exists():

        shutil.rmtree(
            DESTINATION_DIR
        )

    DESTINATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    imagens = localizar_imagens()

    print()
    print(
        f"Arquivos encontrados: "
        f"{len(imagens)}"
    )


    # ========================================================
    # AGRUPAR PELO HASH
    # ========================================================

    grupos = defaultdict(list)

    for imagem in imagens:

        hash_imagem = calcular_sha256(
            imagem
        )

        grupos[hash_imagem].append(
            imagem
        )


    print(
        f"Hashes únicos: "
        f"{len(grupos)}"
    )


    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    copiadas = 0

    duplicatas_removidas = 0

    conflitos_classes = 0


    for hash_imagem, arquivos in grupos.items():

        # Descobrir as classes das cópias.
        classes = {
            arquivo.parent.name
            for arquivo in arquivos
        }


        # ----------------------------------------------------
        # CASO 1:
        # Todas as cópias pertencem à mesma classe.
        #
        # Mantemos somente uma.
        # ----------------------------------------------------

        if len(classes) == 1:

            arquivo_escolhido = arquivos[0]

            classe = arquivo_escolhido.parent.name

            destino_classe = (
                DESTINATION_DIR
                / classe
            )

            destino_classe.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                arquivo_escolhido,
                destino_classe
                / arquivo_escolhido.name
            )

            copiadas += 1

            duplicatas_removidas += (
                len(arquivos) - 1
            )


        # ----------------------------------------------------
        # CASO 2:
        # A MESMA imagem aparece em classes diferentes.
        #
        # Isso representa conflito de rótulo.
        #
        # Para não escolher arbitrariamente uma classe,
        # removemos o grupo inteiro do dataset deduplicado.
        # ----------------------------------------------------

        else:

            conflitos_classes += 1

            print()
            print("-" * 70)

            print(
                "CONFLITO DE CLASSE:"
            )

            for arquivo in arquivos:

                print(
                    f"{arquivo.parent.name:15s} | "
                    f"{arquivo.name}"
                )


    # ========================================================
    # CRIAR PASTAS VAZIAS DAS CLASSES
    # ========================================================
    #
    # Garante que todas as classes continuem presentes,
    # mesmo em um caso extremo de remoção.
    # ========================================================

    for classe_dir in SOURCE_DIR.iterdir():

        if classe_dir.is_dir():

            (
                DESTINATION_DIR
                / classe_dir.name
            ).mkdir(
                parents=True,
                exist_ok=True
            )


    print()
    print("=" * 70)

    print(
        f"Imagens originais: "
        f"{len(imagens)}"
    )

    print(
        f"Hashes únicos: "
        f"{len(grupos)}"
    )

    print(
        f"Imagens copiadas: "
        f"{copiadas}"
    )

    print(
        f"Duplicatas removidas: "
        f"{duplicatas_removidas}"
    )

    print(
        f"Grupos com conflito de classe: "
        f"{conflitos_classes}"
    )

    print()

    print(
        "Dataset deduplicado:"
    )

    print(
        DESTINATION_DIR
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()