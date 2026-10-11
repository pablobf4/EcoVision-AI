# ============================================================
# EcoVision AI
# Atividade 4.1
# Pipeline de dados para Conditional GAN
# ============================================================

import torch

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import (
    datasets,
    transforms
)

from src.xray.data_pipeline import (
    localizar_dataset,
    criar_datasets
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

GAN_IMAGE_SIZE = 64

GAN_BATCH_SIZE = 32


# ============================================================
# TRANSFORM
# ============================================================

def criar_transform_gan():
    """
    Transform utilizado pela GAN.

    As imagens são convertidas para:

        grayscale
        64 x 64
        intervalo [-1, 1]

    O intervalo [-1, 1] é utilizado porque
    a saída do gerador utilizará Tanh.
    """

    return transforms.Compose([

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize(
            (
                GAN_IMAGE_SIZE,
                GAN_IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        # ----------------------------------------------------
        # ToTensor produz valores aproximadamente em:
        #
        # [0, 1]
        #
        # Normalize(0.5, 0.5) transforma para:
        #
        # [-1, 1]
        # ----------------------------------------------------

        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])


# ============================================================
# DATASET DE TREINO
# ============================================================

def criar_dataset_gan():
    """
    Utiliza EXATAMENTE os índices pertencentes
    ao conjunto de treino original.

    Validação e teste nunca entram na GAN.
    """

    dataset_path = localizar_dataset()


    (
        dataset_base,
        train_dataset,
        _,
        _
    ) = criar_datasets(
        dataset_path
    )


    # --------------------------------------------------------
    # Dataset com transform específico da GAN.
    # --------------------------------------------------------

    dataset_gan_base = datasets.ImageFolder(

        dataset_path,

        transform=criar_transform_gan()
    )


    # --------------------------------------------------------
    # Reutilizar exatamente os mesmos índices do treino.
    # --------------------------------------------------------

    dataset_gan = Subset(

        dataset_gan_base,

        train_dataset.indices
    )


    return (
        dataset_gan_base,
        dataset_gan
    )


# ============================================================
# DATALOADER
# ============================================================

def criar_gan_loader():
    """
    DataLoader utilizado no treinamento adversarial.
    """

    (
        dataset_base,
        dataset_gan
    ) = criar_dataset_gan()


    loader = DataLoader(

        dataset_gan,

        batch_size=GAN_BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        drop_last=True
    )


    return (
        dataset_base,
        dataset_gan,
        loader
    )


# ============================================================
# TESTE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Pipeline GAN Raio-X")
    print("=" * 70)


    (
        dataset_base,
        dataset_gan,
        loader
    ) = criar_gan_loader()


    print()
    print(
        "Classes:",
        dataset_base.class_to_idx
    )


    print(
        "Imagens usadas pela GAN:",
        len(dataset_gan)
    )


    images, labels = next(
        iter(loader)
    )


    print()
    print(
        "Batch imagens:",
        images.shape
    )


    print(
        "Batch labels:",
        labels.shape
    )


    print(
        "Valor mínimo:",
        f"{images.min().item():.4f}"
    )


    print(
        "Valor máximo:",
        f"{images.max().item():.4f}"
    )


    # ========================================================
    # TESTES
    # ========================================================

    assert (
        len(dataset_gan)
        == 840
    )


    assert images.shape[1:] == (

        1,

        GAN_IMAGE_SIZE,

        GAN_IMAGE_SIZE
    )


    assert (
        images.min()
        >= -1.01
    )


    assert (
        images.max()
        <= 1.01
    )


    print()
    print(
        "Pipeline GAN: TESTE OK"
    )


if __name__ == "__main__":

    main()