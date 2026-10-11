# ============================================================
# EcoVision AI
# Atividade 4.1
# Pipeline de dados - Raio-X / COVID-19
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
from collections import Counter

import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import (
    datasets,
    transforms
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

SEED = 42


# ============================================================
# NORMALIZAÇÃO IMAGENET
# ============================================================
#
# Posteriormente utilizaremos uma ResNet-18 pré-treinada
# no ImageNet como classificador.
#
# Por isso usamos a mesma normalização do ImageNet.
# ============================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)


IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)


# ============================================================
# LOCALIZAR DATASET
# ============================================================

def localizar_dataset():
    """
    Retorna o caminho do dataset de raio-X preparado
    anteriormente.
    """

    project_dir = PROJECT_ROOT


    dataset_path = (
        project_dir
        / "data"
        / "xray_covid"
    )


    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado:\n"
            f"{dataset_path}"
        )


    return dataset_path


# ============================================================
# TRANSFORMS
# ============================================================

def criar_transforms():
    """
    Cria os pipelines de transformação.

    TREINO:
        pequenas variações geométricas e de contraste.

    VALIDAÇÃO / TESTE:
        apenas resize e normalização.

    As augmentations são deliberadamente leves,
    pois estamos trabalhando com imagens médicas.
    """

    # ========================================================
    # TREINAMENTO
    # ========================================================

    train_transform = transforms.Compose([

        # ----------------------------------------------------
        # Pequenas rotações simulam pequenas diferenças
        # de posicionamento do paciente.
        # ----------------------------------------------------

        transforms.RandomRotation(
            degrees=7
        ),


        # ----------------------------------------------------
        # Pequena transformação geométrica.
        #
        # Evitamos alterações agressivas que poderiam
        # distorcer estruturas anatômicas.
        # ----------------------------------------------------

        transforms.RandomAffine(

            degrees=0,

            translate=(
                0.03,
                0.03
            ),

            scale=(
                0.95,
                1.05
            )
        ),


        # ----------------------------------------------------
        # Pequenas diferenças de contraste podem ocorrer
        # entre equipamentos e condições de aquisição.
        # ----------------------------------------------------

        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.10
        ),


        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),


        transforms.ToTensor(),


        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])


    # ========================================================
    # VALIDAÇÃO / TESTE
    # ========================================================

    eval_transform = transforms.Compose([

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),


        transforms.ToTensor(),


        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])


    return (
        train_transform,
        eval_transform
    )


# ============================================================
# CRIAR DATASETS
# ============================================================

def criar_datasets(
    dataset_path
):
    """
    Divide o dataset em:

        70% treino
        15% validação
        15% teste

    utilizando divisão estratificada.

    Isso mantém aproximadamente a mesma proporção
    de Normal, Pneumonia e COVID em todos os conjuntos.
    """

    (
        train_transform,
        eval_transform
    ) = criar_transforms()


    # ========================================================
    # DATASET BASE
    # ========================================================

    dataset_base = datasets.ImageFolder(
        dataset_path
    )


    labels = [

        label

        for _, label
        in dataset_base.samples
    ]


    indices = list(
        range(
            len(dataset_base)
        )
    )


    # ========================================================
    # PRIMEIRO SPLIT
    #
    # 70% treino
    # 30% temporário
    # ========================================================

    (
        train_indices,
        temp_indices
    ) = train_test_split(

        indices,

        test_size=0.30,

        random_state=SEED,

        stratify=labels
    )


    temp_labels = [

        labels[i]

        for i in temp_indices
    ]


    # ========================================================
    # SEGUNDO SPLIT
    #
    # 15% validação
    # 15% teste
    # ========================================================

    (
        val_indices,
        test_indices
    ) = train_test_split(

        temp_indices,

        test_size=0.50,

        random_state=SEED,

        stratify=temp_labels
    )


    # ========================================================
    # DATASET DE TREINO
    # ========================================================

    dataset_train = datasets.ImageFolder(

        dataset_path,

        transform=train_transform
    )


    # ========================================================
    # DATASET DE AVALIAÇÃO
    # ========================================================

    dataset_eval = datasets.ImageFolder(

        dataset_path,

        transform=eval_transform
    )


    # ========================================================
    # SUBSETS
    # ========================================================

    train_dataset = Subset(

        dataset_train,

        train_indices
    )


    val_dataset = Subset(

        dataset_eval,

        val_indices
    )


    test_dataset = Subset(

        dataset_eval,

        test_indices
    )


    return (

        dataset_base,

        train_dataset,

        val_dataset,

        test_dataset
    )


# ============================================================
# DISTRIBUIÇÃO POR CLASSE
# ============================================================

def contar_classes(
    subset
):
    """
    Conta quantas imagens de cada classe existem
    em um Subset.
    """

    dataset = subset.dataset


    contador = Counter()


    for indice in subset.indices:

        _, label = dataset.samples[
            indice
        ]


        classe = dataset.classes[
            label
        ]


        contador[
            classe
        ] += 1


    return contador


# ============================================================
# DATALOADERS
# ============================================================

def criar_dataloaders(

    train_dataset,
    val_dataset,
    test_dataset
):
    """
    Cria DataLoaders para treino, validação e teste.
    """

    usar_pin_memory = (
        torch.cuda.is_available()
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    return (

        train_loader,

        val_loader,

        test_loader
    )


# ============================================================
# MOSTRAR DISTRIBUIÇÃO
# ============================================================

def mostrar_distribuicao(
    nome,
    subset
):
    """
    Mostra a quantidade de imagens por classe
    dentro de um subconjunto.
    """

    contador = contar_classes(
        subset
    )


    print()
    print(
        nome
    )


    for classe in sorted(
        contador
    ):

        print(

            f"  {classe:10s}: "

            f"{contador[classe]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Pipeline Raio-X COVID-19")
    print("=" * 70)


    # ========================================================
    # DATASET
    # ========================================================

    dataset_path = localizar_dataset()


    print()
    print(
        "Dataset:"
    )

    print(
        dataset_path
    )


    # ========================================================
    # SPLITS
    # ========================================================

    (
        dataset_base,
        train_dataset,
        val_dataset,
        test_dataset
    ) = criar_datasets(
        dataset_path
    )


    # ========================================================
    # CLASSES
    # ========================================================

    print()
    print(
        "Classes:"
    )


    for indice, classe in enumerate(
        dataset_base.classes
    ):

        print(
            f"{indice}: {classe}"
        )


    # ========================================================
    # TAMANHOS
    # ========================================================

    print()
    print(
        f"Total:     "
        f"{len(dataset_base)}"
    )


    print(
        f"Treino:    "
        f"{len(train_dataset)}"
    )


    print(
        f"Validação: "
        f"{len(val_dataset)}"
    )


    print(
        f"Teste:     "
        f"{len(test_dataset)}"
    )


    # ========================================================
    # DISTRIBUIÇÃO
    # ========================================================

    mostrar_distribuicao(

        "Distribuição TREINO:",

        train_dataset
    )


    mostrar_distribuicao(

        "Distribuição VALIDAÇÃO:",

        val_dataset
    )


    mostrar_distribuicao(

        "Distribuição TESTE:",

        test_dataset
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader
    ) = criar_dataloaders(

        train_dataset,

        val_dataset,

        test_dataset
    )


    # ========================================================
    # TESTAR BATCH
    # ========================================================

    imagens, labels = next(
        iter(train_loader)
    )


    print()
    print(
        "Batch imagens:",
        imagens.shape
    )


    print(
        "Batch labels:",
        labels.shape
    )


    # ========================================================
    # TESTES AUTOMÁTICOS
    # ========================================================

    assert (
        len(dataset_base)
        == 1200
    )


    assert (
        len(dataset_base.classes)
        == 3
    )


    assert (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
        == len(dataset_base)
    )


    assert imagens.shape[1:] == (

        3,

        IMAGE_SIZE,

        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # Garantir que os splits não compartilham índices.
    # --------------------------------------------------------

    train_set = set(
        train_dataset.indices
    )


    val_set = set(
        val_dataset.indices
    )


    test_set = set(
        test_dataset.indices
    )


    assert train_set.isdisjoint(
        val_set
    )


    assert train_set.isdisjoint(
        test_set
    )


    assert val_set.isdisjoint(
        test_set
    )


    print()
    print(
        "Pipeline Raio-X: TESTE OK"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()