# ============================================================
# EcoVision AI
# Etapa 4 - Preparação dos dados
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import torch

# ============================================================
# Configurações gerais
# ============================================================

# Raiz do projeto.
PROJECT_DIR = PROJECT_ROOT

# Pasta onde o dataset foi extraído.
DATA_DIR = PROJECT_DIR / "data" / "realwaste"

# Tamanho que todas as imagens terão antes de entrar no modelo.
IMAGE_SIZE = 224

# Quantidade de imagens processadas simultaneamente.
BATCH_SIZE = 32

# Seed para garantir reprodutibilidade.
SEED = 42


def localizar_dataset():
    """
    Localiza automaticamente a pasta RealWaste.
    """

    pastas = list(
        DATA_DIR.rglob("RealWaste")
    )

    if not pastas:
        raise FileNotFoundError(
            "Dataset RealWaste não encontrado."
        )

    return pastas[0]


# ============================================================
# Normalização
# ============================================================

# Média e desvio padrão utilizados nos modelos
# pré-treinados no ImageNet.

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
# Data Augmentation - Treino
# ============================================================

train_transform = transforms.Compose([

    # Faz um recorte aleatório da imagem e redimensiona
    # para 224 x 224.
    #
    # scale=(0.8, 1.0) significa que o recorte utilizará
    # entre 80% e 100% da imagem original.
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    # Possui 50% de chance de espelhar horizontalmente.
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # Aplica uma pequena rotação.
    #
    # Isso ajuda o modelo a reconhecer o objeto mesmo
    # quando ele aparece levemente inclinado.
    transforms.RandomRotation(
        degrees=10
    ),

    # Pequena variação de brilho e contraste.
    #
    # Simula diferentes condições de iluminação.
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    # Converte a imagem PIL para Tensor.
    transforms.ToTensor(),

    # Normaliza os canais RGB.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])



# ============================================================
# Transformações de validação e teste
# ============================================================

eval_transform = transforms.Compose([

    # Apenas redimensionamos.
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Converte para Tensor.
    transforms.ToTensor(),

    # Aplica a mesma normalização.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])



def criar_indices(dataset):
    """
    Divide o dataset em:

    70% treino
    15% validação
    15% teste

    utilizando divisão estratificada.
    """

    # Cria um array:
    #
    # [0, 1, 2, 3, ..., tamanho_dataset]
    indices = np.arange(
        len(dataset)
    )

    # Classes de cada imagem.
    targets = np.array(
        dataset.targets
    )


    # --------------------------------------------------------
    # Primeira divisão
    #
    # 70% treino
    # 30% temporário
    # --------------------------------------------------------

    train_indices, temp_indices = train_test_split(

        indices,

        test_size=0.30,

        random_state=SEED,

        # Mantém a proporção de cada classe.
        stratify=targets
    )


    # Pegamos somente os labels pertencentes
    # ao conjunto temporário.
    temp_targets = targets[
        temp_indices
    ]


    # --------------------------------------------------------
    # Segunda divisão
    #
    # 30% temporário vira:
    #
    # 15% validação
    # 15% teste
    # --------------------------------------------------------

    val_indices, test_indices = train_test_split(

        temp_indices,

        test_size=0.50,

        random_state=SEED,

        stratify=temp_targets
    )


    return (
        train_indices,
        val_indices,
        test_indices
    )


def criar_datasets(dataset_path):
    """
    Cria datasets separados para:

    treino
    validação
    teste
    """

    # Dataset base usado somente para encontrar
    # classes e labels.
    dataset_base = datasets.ImageFolder(
        root=dataset_path
    )


    # Cria os índices de cada conjunto.
    (
        train_indices,
        val_indices,
        test_indices
    ) = criar_indices(
        dataset_base
    )


    # Dataset de treino com augmentation.
    train_full = datasets.ImageFolder(
        root=dataset_path,
        transform=train_transform
    )


    # Dataset de validação.
    val_full = datasets.ImageFolder(
        root=dataset_path,
        transform=eval_transform
    )


    # Dataset de teste.
    test_full = datasets.ImageFolder(
        root=dataset_path,
        transform=eval_transform
    )


    # Subset permite utilizar apenas determinados índices.
    train_dataset = Subset(
        train_full,
        train_indices
    )

    val_dataset = Subset(
        val_full,
        val_indices
    )

    test_dataset = Subset(
        test_full,
        test_indices
    )


    return (
        dataset_base,
        train_dataset,
        val_dataset,
        test_dataset
    )


def criar_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):
    """
    Cria os DataLoaders utilizados durante
    treinamento e avaliação.
    """

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        # Embaralha as imagens a cada época.
        shuffle=True,

        num_workers=2,

        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=2,

        pin_memory=torch.cuda.is_available()
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=2,

        pin_memory=torch.cuda.is_available()
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )




def main():

    dataset_path = localizar_dataset()

    print("=" * 60)
    print("EcoVision AI - Data Pipeline")
    print("=" * 60)

    print(
        f"\nDataset localizado em:\n{dataset_path}"
    )


    (
        dataset_base,
        train_dataset,
        val_dataset,
        test_dataset
    ) = criar_datasets(
        dataset_path
    )


    print("\nDivisão dos dados:")
    print("-" * 60)

    print(
        f"Treino:     {len(train_dataset)} imagens"
    )

    print(
        f"Validação:  {len(val_dataset)} imagens"
    )

    print(
        f"Teste:      {len(test_dataset)} imagens"
    )


    (
        train_loader,
        val_loader,
        test_loader
    ) = criar_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )


    # --------------------------------------------------------
    # Teste do primeiro batch
    # --------------------------------------------------------

    imagens, labels = next(
        iter(train_loader)
    )


    print("\nPrimeiro batch:")
    print("-" * 60)

    print(
        "Formato das imagens:",
        imagens.shape
    )

    print(
        "Formato dos labels:",
        labels.shape
    )

    print(
        "Número de classes:",
        len(dataset_base.classes)
    )

    print(
        "\nClasses:",
        dataset_base.classes
    )


if __name__ == "__main__":
    main()