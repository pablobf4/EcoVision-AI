# ============================================================
# EcoVision AI
# Atividade 2 - Preparação do Corpus ADS-16 para CLIP
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
from collections import defaultdict
import csv
import hashlib
import random
import shutil
import re

import kagglehub


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_DIR = PROJECT_ROOT

DESTINATION_DIR = (
    PROJECT_DIR
    / "data"
    / "ads16_clip"
)

MANIFEST_PATH = (
    PROJECT_DIR
    / "data"
    / "ads16_clip_manifest.csv"
)

SEED = 42

# Queremos 5 imagens por usuário.
#
# 120 usuários × 5 imagens
# =
# 600 imagens
IMAGES_PER_USER = 5


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# DOWNLOAD / LOCALIZAÇÃO
# ============================================================

def localizar_dataset():
    """
    Localiza o ADS-16 utilizando o cache do KaggleHub.

    Se necessário, o KaggleHub faz o download.
    """

    caminho = kagglehub.dataset_download(
        "groffo/ads16-dataset"
    )

    return Path(caminho)


# ============================================================
# SHA-256
# ============================================================

def calcular_sha256(caminho):
    """
    Calcula um hash do conteúdo da imagem.

    Isso permite detectar imagens exatamente duplicadas,
    mesmo quando possuem nomes diferentes.
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
# LOCALIZAR DIRETÓRIOS DOS USUÁRIOS
# ============================================================

def localizar_usuarios(dataset_path):
    """
    Procura diretórios no formato:

        U0001
        U0002
        ...
        U0120

    Consideramos somente diretórios que contenham
    as pastas IM-POS ou IM-NEG.
    """

    usuarios = {}

    padrao_usuario = re.compile(
        r"^U\d{4}$"
    )

    for diretorio in dataset_path.rglob("*"):

        if not diretorio.is_dir():
            continue

        if not padrao_usuario.match(
            diretorio.name
        ):
            continue


        subpastas = [
            pasta
            for pasta in diretorio.iterdir()
            if pasta.is_dir()
        ]


        possui_imagens_corpus = any(

            (
                pasta.name.endswith(
                    "-IM-POS"
                )
                or
                pasta.name.endswith(
                    "-IM-NEG"
                )
            )

            for pasta in subpastas
        )


        if possui_imagens_corpus:

            usuarios[
                diretorio.name
            ] = diretorio


    return dict(
        sorted(
            usuarios.items()
        )
    )


# ============================================================
# LOCALIZAR IMAGENS DO USUÁRIO
# ============================================================

def localizar_imagens_usuario(
    usuario_dir
):
    """
    Separa as imagens de um usuário em:

        POS
        NEG
    """

    imagens = {
        "POS": [],
        "NEG": []
    }


    for pasta in usuario_dir.iterdir():

        if not pasta.is_dir():
            continue


        if pasta.name.endswith(
            "-IM-POS"
        ):

            sentimento = "POS"


        elif pasta.name.endswith(
            "-IM-NEG"
        ):

            sentimento = "NEG"


        else:

            continue


        for arquivo in pasta.rglob("*"):

            if (
                arquivo.is_file()
                and arquivo.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                imagens[
                    sentimento
                ].append(
                    arquivo
                )


    return imagens


# ============================================================
# REMOVER DUPLICATAS
# ============================================================

def remover_duplicatas(
    imagens_por_usuario
):
    """
    Remove arquivos idênticos utilizando SHA-256.

    A deduplicação é global:
    se a mesma imagem aparecer para dois usuários,
    apenas uma ocorrência será mantida.
    """

    hashes_encontrados = set()

    resultado = {}


    duplicatas = 0


    for usuario, grupos in imagens_por_usuario.items():

        resultado[
            usuario
        ] = {
            "POS": [],
            "NEG": []
        }


        for sentimento in [
            "POS",
            "NEG"
        ]:

            for caminho in grupos[
                sentimento
            ]:

                hash_imagem = calcular_sha256(
                    caminho
                )


                if hash_imagem in hashes_encontrados:

                    duplicatas += 1

                    continue


                hashes_encontrados.add(
                    hash_imagem
                )


                resultado[
                    usuario
                ][
                    sentimento
                ].append(
                    (
                        caminho,
                        hash_imagem
                    )
                )


    return (
        resultado,
        duplicatas
    )


# ============================================================
# SELEÇÃO REPRESENTATIVA
# ============================================================

def selecionar_imagens(
    imagens_por_usuario
):
    """
    Seleciona 5 imagens por usuário.

    Para equilibrar POS e NEG:

    usuários alternados recebem:

        3 POS + 2 NEG

    ou:

        2 POS + 3 NEG

    Com 120 usuários isso tende a produzir:

        aproximadamente 300 POS
        aproximadamente 300 NEG
    """

    selecionadas = []


    for indice_usuario, (
        usuario,
        grupos
    ) in enumerate(
        imagens_por_usuario.items()
    ):

        rng = random.Random(
            SEED
            + indice_usuario
        )


        pos = grupos["POS"].copy()

        neg = grupos["NEG"].copy()


        rng.shuffle(
            pos
        )

        rng.shuffle(
            neg
        )


        # Alternar 3/2 e 2/3
        # ajuda a manter equilíbrio global.
        if indice_usuario % 2 == 0:

            quantidade_pos = 3
            quantidade_neg = 2

        else:

            quantidade_pos = 2
            quantidade_neg = 3


        escolhidas_usuario = []


        # ----------------------------------------------------
        # Seleção inicial
        # ----------------------------------------------------

        for caminho, hash_imagem in pos[
            :quantidade_pos
        ]:

            escolhidas_usuario.append(
                (
                    caminho,
                    hash_imagem,
                    "POS"
                )
            )


        for caminho, hash_imagem in neg[
            :quantidade_neg
        ]:

            escolhidas_usuario.append(
                (
                    caminho,
                    hash_imagem,
                    "NEG"
                )
            )


        # ----------------------------------------------------
        # FALLBACK
        #
        # Se algum usuário não tiver imagens suficientes
        # em POS ou NEG, completamos utilizando as imagens
        # restantes disponíveis.
        # ----------------------------------------------------

        if len(
            escolhidas_usuario
        ) < IMAGES_PER_USER:

            usados = {
                hash_imagem
                for _, hash_imagem, _
                in escolhidas_usuario
            }


            restantes = []


            for sentimento, grupo in [
                ("POS", pos),
                ("NEG", neg)
            ]:

                for (
                    caminho,
                    hash_imagem
                ) in grupo:

                    if (
                        hash_imagem
                        not in usados
                    ):

                        restantes.append(
                            (
                                caminho,
                                hash_imagem,
                                sentimento
                            )
                        )


            rng.shuffle(
                restantes
            )


            faltam = (
                IMAGES_PER_USER
                - len(
                    escolhidas_usuario
                )
            )


            escolhidas_usuario.extend(
                restantes[:faltam]
            )


        # ----------------------------------------------------
        # Guardar seleção
        # ----------------------------------------------------

        for (
            caminho,
            hash_imagem,
            sentimento
        ) in escolhidas_usuario[
            :IMAGES_PER_USER
        ]:

            selecionadas.append({

                "usuario": usuario,

                "sentimento": sentimento,

                "source": caminho,

                "sha256": hash_imagem
            })


    return selecionadas


# ============================================================
# COPIAR CORPUS
# ============================================================

def copiar_corpus(
    selecionadas
):
    """
    Copia as imagens selecionadas para:

        data/
            ads16_clip/
                U0001/
                    POS/
                    NEG/
                U0002/
                    ...
    """

    if DESTINATION_DIR.exists():

        shutil.rmtree(
            DESTINATION_DIR
        )


    DESTINATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    manifest = []


    for indice, item in enumerate(
        selecionadas,
        start=1
    ):

        usuario = item[
            "usuario"
        ]

        sentimento = item[
            "sentimento"
        ]

        origem = item[
            "source"
        ]


        destino_dir = (
            DESTINATION_DIR
            / usuario
            / sentimento
        )


        destino_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # Prefixamos o nome para garantir
        # unicidade dentro do nosso corpus.
        nome_destino = (

            f"{usuario}_"
            f"{sentimento}_"
            f"{indice:04d}_"
            f"{origem.name}"
        )


        destino = (
            destino_dir
            / nome_destino
        )


        shutil.copy2(
            origem,
            destino
        )


        manifest.append({

            "usuario": usuario,

            "sentimento": sentimento,

            "arquivo": str(
                destino
            ),

            "arquivo_original": str(
                origem
            ),

            "sha256": item[
                "sha256"
            ]
        })


    return manifest


# ============================================================
# SALVAR MANIFEST
# ============================================================

def salvar_manifest(
    manifest
):
    """
    Salva um CSV contendo a origem e os metadados
    de todas as imagens selecionadas.
    """

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        writer = csv.DictWriter(

            arquivo,

            fieldnames=[
                "usuario",
                "sentimento",
                "arquivo",
                "arquivo_original",
                "sha256"
            ]
        )


        writer.writeheader()

        writer.writerows(
            manifest
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Preparação do Corpus CLIP")
    print("=" * 70)


    dataset_path = localizar_dataset()


    print()
    print(
        "Dataset ADS-16:"
    )

    print(
        dataset_path
    )


    # ========================================================
    # USUÁRIOS
    # ========================================================

    usuarios = localizar_usuarios(
        dataset_path
    )


    print()
    print(
        f"Usuários encontrados: "
        f"{len(usuarios)}"
    )


    # ========================================================
    # IMAGENS
    # ========================================================

    imagens_por_usuario = {}


    for usuario, caminho in usuarios.items():

        imagens_por_usuario[
            usuario
        ] = localizar_imagens_usuario(
            caminho
        )


    total_original = sum(

        len(grupos["POS"])
        + len(grupos["NEG"])

        for grupos
        in imagens_por_usuario.values()
    )


    print(
        f"Imagens do Corpus encontradas: "
        f"{total_original}"
    )


    # ========================================================
    # DEDUPLICAÇÃO
    # ========================================================

    (
        imagens_unicas,
        duplicatas
    ) = remover_duplicatas(
        imagens_por_usuario
    )


    total_unicas = sum(

        len(grupos["POS"])
        + len(grupos["NEG"])

        for grupos
        in imagens_unicas.values()
    )


    print(
        f"Imagens únicas: "
        f"{total_unicas}"
    )

    print(
        f"Duplicatas removidas: "
        f"{duplicatas}"
    )


    # ========================================================
    # SUBCONJUNTO
    # ========================================================

    selecionadas = selecionar_imagens(
        imagens_unicas
    )


    print()
    print(
        f"Imagens selecionadas: "
        f"{len(selecionadas)}"
    )


    total_pos = sum(

        1

        for item in selecionadas

        if item[
            "sentimento"
        ] == "POS"
    )


    total_neg = sum(

        1

        for item in selecionadas

        if item[
            "sentimento"
        ] == "NEG"
    )


    print(
        f"POS: {total_pos}"
    )

    print(
        f"NEG: {total_neg}"
    )


    # ========================================================
    # COPIAR
    # ========================================================

    manifest = copiar_corpus(
        selecionadas
    )


    salvar_manifest(
        manifest
    )


    # ========================================================
    # VALIDAÇÕES
    # ========================================================

    hashes = [
        item["sha256"]
        for item in manifest
    ]


    assert (
        len(hashes)
        == len(set(hashes))
    )


    assert (
        len(manifest)
        >= 500
    )


    print()
    print("=" * 70)

    print(
        "CORPUS CLIP PREPARADO"
    )

    print("=" * 70)

    print(
        f"Total final: "
        f"{len(manifest)}"
    )

    print(
        f"Hashes únicos: "
        f"{len(set(hashes))}"
    )

    print()

    print(
        "Corpus:"
    )

    print(
        DESTINATION_DIR
    )

    print()

    print(
        "Manifest:"
    )

    print(
        MANIFEST_PATH
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()