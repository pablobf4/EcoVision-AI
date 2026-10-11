# ============================================================
# EcoVision AI
# Atividade 3 - Pipeline de dados da CNN
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT

import torch

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import (
    datasets,
    transforms
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

SEED = 42


# ============================================================
# NORMALIZAÇÃO DO IMAGENET
# ============================================================
#
# Como vamos utilizar uma CNN pré-treinada no ImageNet,
# precisamos utilizar a mesma normalização usada durante
# o pré-treinamento.
#
# Isso mantém os dados de entrada compatíveis com a
# distribuição utilizada pela EfficientNet-B0.
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
    Retorna o caminho do dataset da CNN após a remoção
    das imagens duplicadas.

    Estrutura esperada:

    data/
        cnn_dedup/
            bike/
            cars/
            cats/
            dogs/
            flowers/
            horses/
            human/
    """

    project_dir = PROJECT_ROOT

    dataset_path = (
        project_dir
        / "data"
        / "cnn_dedup"
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado em:\n"
            f"{dataset_path}"
        )

    return dataset_path


# ============================================================
# TRANSFORMS
# ============================================================

def criar_transforms():
    """
    Cria dois pipelines de transformação:

    1. Treinamento:
       aplica Data Augmentation.

    2. Validação/Teste:
       aplica apenas redimensionamento e normalização.
    """

    # ========================================================
    # TRANSFORMS DE TREINAMENTO
    # ========================================================
    #
    # Estratégias de Data Augmentation utilizadas:
    #
    # 1. RandomResizedCrop
    #    Recorta e redimensiona regiões diferentes da imagem.
    #
    # 2. RandomHorizontalFlip
    #    Inverte horizontalmente algumas imagens.
    #
    # 3. RandomRotation
    #    Aplica pequenas rotações.
    #
    # 4. ColorJitter
    #    Modifica brilho, contraste e saturação.
    #
    # A rubrica exige ao menos três estratégias.
    # Aqui utilizamos quatro.
    # ========================================================

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])


    # ========================================================
    # TRANSFORMS DE VALIDAÇÃO / TESTE
    # ========================================================
    #
    # Não utilizamos Data Augmentation durante validação
    # e teste porque queremos medir o desempenho do modelo
    # sobre imagens estáveis.
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

    A divisão é estratificada para preservar aproximadamente
    a proporção das classes em cada conjunto.
    """

    (
        train_transform,
        eval_transform
    ) = criar_transforms()


    # ========================================================
    # DATASET BASE
    # ========================================================
    #
    # ImageFolder identifica automaticamente:
    #
    # - as pastas das classes
    # - os arquivos das imagens
    # - os labels numéricos
    #
    # Exemplo:
    #
    # bike    -> 0
    # cars    -> 1
    # cats    -> 2
    # ...
    # ========================================================

    dataset_base = datasets.ImageFolder(
        dataset_path
    )


    # --------------------------------------------------------
    # Labels de todas as imagens.
    # --------------------------------------------------------

    labels = [
        label
        for _, label
        in dataset_base.samples
    ]


    # --------------------------------------------------------
    # Índices:
    #
    # [0, 1, 2, 3, ..., N]
    # --------------------------------------------------------

    indices = list(
        range(
            len(dataset_base)
        )
    )


    # ========================================================
    # PRIMEIRO SPLIT
    # ========================================================
    #
    # 70% -> treino
    # 30% -> temporário
    #
    # stratify=labels mantém a distribuição das classes.
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


    # --------------------------------------------------------
    # Labels correspondentes aos 30% temporários.
    # --------------------------------------------------------

    temp_labels = [
        labels[i]
        for i in temp_indices
    ]


    # ========================================================
    # SEGUNDO SPLIT
    # ========================================================
    #
    # Divide os 30% restantes em duas partes iguais:
    #
    # 15% -> validação
    # 15% -> teste
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
    # DATASET DE TREINAMENTO
    # ========================================================
    #
    # Esse ImageFolder utiliza Data Augmentation.
    # ========================================================

    dataset_train = datasets.ImageFolder(

        dataset_path,

        transform=train_transform
    )


    # ========================================================
    # DATASET DE VALIDAÇÃO / TESTE
    # ========================================================
    #
    # Validação e teste não utilizam augmentation.
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
# CRIAR DATALOADERS
# ============================================================

def criar_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):
    """
    Cria os DataLoaders utilizados durante:

        treinamento
        validação
        teste
    """

    # --------------------------------------------------------
    # pin_memory ajuda principalmente quando utilizamos GPU.
    #
    # Como o código também precisa funcionar em CPU,
    # ativamos automaticamente apenas quando CUDA estiver
    # disponível.
    # --------------------------------------------------------

    usar_pin_memory = torch.cuda.is_available()


    # ========================================================
    # TRAIN LOADER
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    # ========================================================
    # VALIDATION LOADER
    # ========================================================

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=usar_pin_memory
    )


    # ========================================================
    # TEST LOADER
    # ========================================================

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
# TESTAR PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Pipeline CNN")
    print("=" * 70)


    # ========================================================
    # LOCALIZAR DATASET
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
    # CRIAR DATASETS
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
    # MOSTRAR CLASSES
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
    # QUANTIDADES
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
    # CRIAR DATALOADERS
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
    # TESTAR UM BATCH
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

    # --------------------------------------------------------
    # O dataset precisa possuir exatamente as 7 classes
    # utilizadas nesta atividade.
    # --------------------------------------------------------

    assert len(
        dataset_base.classes
    ) == 7


    # --------------------------------------------------------
    # A soma de treino + validação + teste deve ser
    # exatamente igual ao total de imagens do dataset.
    #
    # Não fixamos mais 1803 ou 1764 porque isso deixaria
    # o pipeline dependente de uma quantidade específica.
    # --------------------------------------------------------

    assert (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    ) == len(dataset_base)


    # --------------------------------------------------------
    # Cada imagem precisa possuir:
    #
    # 3 canais RGB
    # 224 pixels de altura
    # 224 pixels de largura
    #
    # O formato do batch é:
    #
    # [batch, canais, altura, largura]
    # --------------------------------------------------------

    assert imagens.shape[1:] == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # Deve existir um label para cada imagem no batch.
    # --------------------------------------------------------

    assert (
        imagens.shape[0]
        == labels.shape[0]
    )


    # --------------------------------------------------------
    # As classes esperadas são:
    # --------------------------------------------------------

    classes_esperadas = [
        "bike",
        "cars",
        "cats",
        "dogs",
        "flowers",
        "horses",
        "human"
    ]


    assert (
        dataset_base.classes
        == classes_esperadas
    )


    print()
    print(
        "Pipeline CNN: TESTE OK"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()