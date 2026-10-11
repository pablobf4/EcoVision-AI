# ============================================================
# EcoVision AI
# Etapa 3 - Análise Exploratória do Dataset
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
from collections import Counter

import matplotlib.pyplot as plt

from torchvision import datasets
from PIL import Image


# ------------------------------------------------------------
# Caminhos do projeto
# ------------------------------------------------------------

PROJECT_DIR = PROJECT_ROOT

DATA_DIR = PROJECT_DIR / "data" / "realwaste"


# ------------------------------------------------------------
# Localiza automaticamente a pasta RealWaste
# ------------------------------------------------------------

def localizar_dataset():
    """
    Procura pela pasta chamada RealWaste dentro do diretório data.
    """

    pastas = list(
        DATA_DIR.rglob("RealWaste")
    )

    if not pastas:
        raise FileNotFoundError(
            "Pasta RealWaste não encontrada."
        )

    return pastas[0]

def carregar_dataset(dataset_path):
    """
    Carrega o dataset utilizando ImageFolder.

    O ImageFolder assume que cada pasta representa uma classe.
    """

    dataset = datasets.ImageFolder(
        root=dataset_path
    )

    return dataset


def mostrar_informacoes(dataset):
    """
    Exibe informações gerais do dataset.
    """

    print("=" * 60)
    print("EcoVision AI - Informações do Dataset")
    print("=" * 60)

    print(
        f"\nQuantidade total de imagens: {len(dataset)}"
    )

    print(
        f"Quantidade de classes: {len(dataset.classes)}"
    )

    print("\nClasses:")

    for indice, classe in enumerate(dataset.classes):

        print(
            f"{indice}: {classe}"
        )

    print("\nMapeamento classe -> índice:")

    print(
        dataset.class_to_idx
    )


def contar_imagens_por_classe(dataset):
    """
    Conta quantas imagens existem em cada classe.

    dataset.targets contém o índice da classe de cada imagem.
    """

    contagem = Counter(
        dataset.targets
    )

    print("\nDistribuição das classes:")
    print("-" * 60)

    for classe_id, quantidade in sorted(contagem.items()):

        nome_classe = dataset.classes[
            classe_id
        ]

        print(
            f"{nome_classe:25s}: {quantidade} imagens"
        )

    return contagem


def plotar_distribuicao(dataset, contagem):
    """
    Cria um gráfico de barras mostrando
    a quantidade de imagens por classe.
    """

    nomes_classes = []
    quantidades = []

    for classe_id, quantidade in sorted(contagem.items()):

        nomes_classes.append(
            dataset.classes[classe_id]
        )

        quantidades.append(
            quantidade
        )


    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        nomes_classes,
        quantidades
    )

    plt.title(
        "Distribuição das Classes - RealWaste"
    )

    plt.xlabel(
        "Classe"
    )

    plt.ylabel(
        "Quantidade de Imagens"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()



def visualizar_exemplos(dataset):
    """
    Mostra uma imagem de exemplo de cada classe.
    """

    quantidade_classes = len(
        dataset.classes
    )

    # Como temos 9 classes,
    # criamos uma grade 3x3.
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12)
    )

    axes = axes.flatten()


    for classe_id, nome_classe in enumerate(dataset.classes):

        # Procura a primeira imagem que pertence à classe atual.
        indice_imagem = dataset.targets.index(
            classe_id
        )

        # dataset.samples contém:
        #
        # (caminho_da_imagem, classe)
        caminho_imagem, _ = dataset.samples[
            indice_imagem
        ]

        # Abre a imagem.
        imagem = Image.open(
            caminho_imagem
        ).convert("RGB")


        # Exibe a imagem.
        axes[classe_id].imshow(
            imagem
        )

        axes[classe_id].set_title(
            nome_classe
        )

        axes[classe_id].axis(
            "off"
        )


    plt.tight_layout()

    plt.show()


def main():

    dataset_path = localizar_dataset()

    print(
        f"Dataset localizado em:\n{dataset_path}"
    )

    dataset = carregar_dataset(
        dataset_path
    )

    mostrar_informacoes(
        dataset
    )

    contagem = contar_imagens_por_classe(
        dataset
    )

    plotar_distribuicao(
        dataset,
        contagem
    )

    visualizar_exemplos(
        dataset
    )


if __name__ == "__main__":
    main()