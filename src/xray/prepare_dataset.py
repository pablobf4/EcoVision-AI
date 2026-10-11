# ============================================================
# EcoVision AI
# Atividade 4.1
# Preparação do dataset de Raio-X / COVID-19
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import hashlib
import random
import shutil
import csv

import kagglehub

from PIL import Image


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_DIR = PROJECT_ROOT


DESTINATION_DIR = (
    PROJECT_DIR
    / "data"
    / "xray_covid"
)


MANIFEST_PATH = (
    PROJECT_DIR
    / "data"
    / "xray_covid_manifest.csv"
)


SEED = 42


# ============================================================
# QUANTIDADE DE IMAGENS
# ============================================================
#
# Essas quantidades reproduzem o cenário descrito
# no estudo de caso da atividade:
#
# Normal     = 840
# Pneumonia  = 240
# COVID      = 120
#
# Total      = 1200
#
# ============================================================

TARGET_COUNTS = {

    "Normal": 840,

    "Pneumonia": 240,

    "COVID": 120
}


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}


# ============================================================
# DOWNLOAD
# ============================================================

def baixar_dataset():
    """
    Baixa o COVID-19 Radiography Database
    utilizando KaggleHub.

    O KaggleHub reutiliza o cache caso o dataset
    já esteja presente no computador.
    """

    print()
    print("=" * 70)
    print("EcoVision AI - Dataset Raio-X COVID-19")
    print("=" * 70)


    caminho = kagglehub.dataset_download(

        "tawsifurrahman/"
        "covid19-radiography-database"
    )


    caminho = Path(
        caminho
    )


    print()
    print(
        "Dataset localizado em:"
    )

    print(
        caminho
    )


    return caminho


# ============================================================
# SHA-256
# ============================================================

def calcular_sha256(
    caminho
):
    """
    Calcula o hash SHA-256 do conteúdo da imagem.

    Isso permite identificar arquivos exatamente
    duplicados mesmo que possuam nomes diferentes.
    """

    sha256 = hashlib.sha256()


    with open(
        caminho,
        "rb"
    ) as arquivo:

        for bloco in iter(

            lambda: arquivo.read(
                8192
            ),

            b""
        ):

            sha256.update(
                bloco
            )


    return sha256.hexdigest()


# ============================================================
# ENCONTRAR DIRETÓRIO PRINCIPAL
# ============================================================

def localizar_dataset_principal(
    dataset_path
):
    """
    Localiza a pasta:

        COVID-19_Radiography_Dataset

    dentro do dataset baixado.
    """

    candidatos = []


    if (
        dataset_path.name
        == "COVID-19_Radiography_Dataset"
    ):

        candidatos.append(
            dataset_path
        )


    candidatos.extend(

        caminho

        for caminho
        in dataset_path.rglob(
            "COVID-19_Radiography_Dataset"
        )

        if caminho.is_dir()
    )


    if not candidatos:

        raise FileNotFoundError(
            "Não foi possível localizar "
            "'COVID-19_Radiography_Dataset'."
        )


    return candidatos[0]


# ============================================================
# LOCALIZAR PASTA DE IMAGENS
# ============================================================

def localizar_pasta_imagens(
    dataset_root,
    nome_classe
):
    """
    Localiza a pasta 'images' de uma classe.

    Estrutura esperada:

        COVID/
            images/

        Normal/
            images/

        Viral Pneumonia/
            images/
    """

    classe_dir = (
        dataset_root
        / nome_classe
    )


    if not classe_dir.exists():

        raise FileNotFoundError(
            f"Classe não encontrada:\n"
            f"{classe_dir}"
        )


    images_dir = (
        classe_dir
        / "images"
    )


    if images_dir.exists():

        return images_dir


    # --------------------------------------------------------
    # Fallback:
    # caso a estrutura da versão seja ligeiramente diferente.
    # --------------------------------------------------------

    candidatos = [

        caminho

        for caminho
        in classe_dir.rglob("*")

        if (
            caminho.is_dir()
            and caminho.name.lower()
            == "images"
        )
    ]


    if candidatos:

        return candidatos[0]


    raise FileNotFoundError(
        f"Pasta de imagens não encontrada para "
        f"a classe {nome_classe}."
    )


# ============================================================
# LOCALIZAR IMAGENS
# ============================================================

def listar_imagens(
    pasta
):
    """
    Retorna todas as imagens válidas encontradas
    dentro de uma pasta.
    """

    return sorted([

        caminho

        for caminho
        in pasta.rglob("*")

        if (
            caminho.is_file()
            and caminho.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])


# ============================================================
# VALIDAR IMAGEM
# ============================================================

def imagem_valida(
    caminho
):
    """
    Tenta abrir e validar a imagem com Pillow.

    Retorna:
        True  -> imagem válida
        False -> arquivo corrompido/inválido
    """

    try:

        with Image.open(
            caminho
        ) as imagem:

            imagem.verify()


        return True


    except Exception:

        return False


# ============================================================
# REMOVER DUPLICATAS
# ============================================================

def obter_imagens_unicas(
    caminhos
):
    """
    Remove duplicatas exatas através de SHA-256.

    Retorna:

        lista de imagens únicas
        quantidade de duplicatas
        quantidade de inválidas
    """

    hashes = set()

    imagens_unicas = []

    duplicatas = 0

    invalidas = 0


    for caminho in caminhos:

        # ----------------------------------------------------
        # Validar arquivo.
        # ----------------------------------------------------

        if not imagem_valida(
            caminho
        ):

            invalidas += 1

            continue


        # ----------------------------------------------------
        # Calcular hash.
        # ----------------------------------------------------

        hash_imagem = calcular_sha256(
            caminho
        )


        # ----------------------------------------------------
        # Detectar duplicata.
        # ----------------------------------------------------

        if hash_imagem in hashes:

            duplicatas += 1

            continue


        hashes.add(
            hash_imagem
        )


        imagens_unicas.append(
            (
                caminho,
                hash_imagem
            )
        )


    return (
        imagens_unicas,
        duplicatas,
        invalidas
    )


# ============================================================
# SELECIONAR AMOSTRA
# ============================================================

def selecionar_amostra(
    imagens,
    quantidade,
    seed
):
    """
    Seleciona uma quantidade fixa de imagens de forma
    reprodutível utilizando uma seed.
    """

    if len(imagens) < quantidade:

        raise ValueError(

            f"Não existem imagens suficientes.\n"

            f"Disponíveis: {len(imagens)}\n"

            f"Necessárias: {quantidade}"
        )


    rng = random.Random(
        seed
    )


    imagens = imagens.copy()


    rng.shuffle(
        imagens
    )


    return imagens[
        :quantidade
    ]


# ============================================================
# COPIAR IMAGENS
# ============================================================

def copiar_classe(
    classe_destino,
    imagens
):
    """
    Copia as imagens selecionadas para o dataset
    local do EcoVision AI.
    """

    destino_classe = (
        DESTINATION_DIR
        / classe_destino
    )


    destino_classe.mkdir(
        parents=True,
        exist_ok=True
    )


    registros = []


    for indice, (
        origem,
        hash_imagem
    ) in enumerate(
        imagens,
        start=1
    ):

        nome_destino = (

            f"{classe_destino.lower()}_"

            f"{indice:04d}"

            f"{origem.suffix.lower()}"
        )


        destino = (
            destino_classe
            / nome_destino
        )


        shutil.copy2(
            origem,
            destino
        )


        registros.append({

            "classe": classe_destino,

            "arquivo": str(
                destino
            ),

            "arquivo_original": str(
                origem
            ),

            "sha256": hash_imagem
        })


    return registros


# ============================================================
# SALVAR MANIFEST
# ============================================================

def salvar_manifest(
    registros
):
    """
    Salva informações das 1200 imagens utilizadas
    no experimento.
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
                "classe",
                "arquivo",
                "arquivo_original",
                "sha256"
            ]
        )


        writer.writeheader()


        writer.writerows(
            registros
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DOWNLOAD
    # ========================================================

    dataset_path = baixar_dataset()


    dataset_root = localizar_dataset_principal(
        dataset_path
    )


    print()
    print(
        "Dataset principal:"
    )

    print(
        dataset_root
    )


    # ========================================================
    # MAPEAMENTO
    # ========================================================
    #
    # Nome utilizado no nosso projeto:
    #
    #     COVID
    #     Normal
    #     Pneumonia
    #
    # Nome existente no dataset público:
    #
    #     COVID
    #     Normal
    #     Viral Pneumonia
    #
    # ========================================================

    classes_origem = {

        "COVID": "COVID",

        "Normal": "Normal",

        "Pneumonia": "Viral Pneumonia"
    }


    # ========================================================
    # LIMPAR DESTINO ANTERIOR
    # ========================================================

    if DESTINATION_DIR.exists():

        shutil.rmtree(
            DESTINATION_DIR
        )


    DESTINATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    todos_registros = []


    # Hashes globais.
    #
    # Também verificaremos se uma imagem idêntica
    # apareceu em classes diferentes.
    hashes_globais = set()


    print()
    print("=" * 70)
    print("PROCESSAMENTO DAS CLASSES")
    print("=" * 70)


    # ========================================================
    # PROCESSAR CADA CLASSE
    # ========================================================

    for numero_classe, (
        classe_destino,
        classe_origem
    ) in enumerate(
        classes_origem.items()
    ):

        print()
        print(
            f"Classe: {classe_destino}"
        )


        pasta_imagens = localizar_pasta_imagens(

            dataset_root,

            classe_origem
        )


        imagens = listar_imagens(
            pasta_imagens
        )


        print(
            f"Imagens encontradas: "
            f"{len(imagens)}"
        )


        (
            imagens_unicas,
            duplicatas,
            invalidas
        ) = obter_imagens_unicas(
            imagens
        )


        print(
            f"Imagens únicas: "
            f"{len(imagens_unicas)}"
        )


        print(
            f"Duplicatas internas: "
            f"{duplicatas}"
        )


        print(
            f"Inválidas: "
            f"{invalidas}"
        )


        # ----------------------------------------------------
        # Remover eventual duplicação entre classes.
        # ----------------------------------------------------

        sem_conflito = []


        conflitos = 0


        for (
            caminho,
            hash_imagem
        ) in imagens_unicas:

            if hash_imagem in hashes_globais:

                conflitos += 1

                continue


            hashes_globais.add(
                hash_imagem
            )


            sem_conflito.append(
                (
                    caminho,
                    hash_imagem
                )
            )


        print(
            f"Duplicatas entre classes removidas: "
            f"{conflitos}"
        )


        # ----------------------------------------------------
        # Quantidade desejada.
        # ----------------------------------------------------

        quantidade = TARGET_COUNTS[
            classe_destino
        ]


        selecionadas = selecionar_amostra(

            sem_conflito,

            quantidade,

            SEED
            + numero_classe
        )


        print(
            f"Selecionadas: "
            f"{len(selecionadas)}"
        )


        registros = copiar_classe(

            classe_destino,

            selecionadas
        )


        todos_registros.extend(
            registros
        )


    # ========================================================
    # MANIFEST
    # ========================================================

    salvar_manifest(
        todos_registros
    )


    # ========================================================
    # VALIDAÇÕES FINAIS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDAÇÃO FINAL")
    print("=" * 70)


    total = len(
        todos_registros
    )


    hashes = [

        registro[
            "sha256"
        ]

        for registro
        in todos_registros
    ]


    print(
        f"Total final: "
        f"{total}"
    )


    print(
        f"Hashes únicos: "
        f"{len(set(hashes))}"
    )


    for classe, quantidade in TARGET_COUNTS.items():

        quantidade_real = sum(

            1

            for registro
            in todos_registros

            if registro[
                "classe"
            ] == classe
        )


        print(

            f"{classe:10s}: "
            f"{quantidade_real}"
        )


        assert (
            quantidade_real
            == quantidade
        )


    # --------------------------------------------------------
    # Deve haver exatamente 1200 imagens.
    # --------------------------------------------------------

    assert (
        total
        == 1200
    )


    # --------------------------------------------------------
    # Nenhuma duplicata exata.
    # --------------------------------------------------------

    assert (
        len(hashes)
        == len(set(hashes))
    )


    print()
    print(
        "Dataset preparado com sucesso."
    )


    print()
    print(
        "Dataset:"
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