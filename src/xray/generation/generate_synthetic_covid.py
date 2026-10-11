# ============================================================
# EcoVision AI
# Atividade 4.1
# Geração de imagens COVID sintéticas
#
# Utiliza o Generator treinado com:
#
# Conditional GAN
# + Label Smoothing
# + Instance Noise
# + Learning Rate assimétrico
# + Mode-Seeking Regularization
#
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import random
import shutil

import numpy as np
import torch

from PIL import Image

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from src.xray.models.cgan import (
    ConditionalGenerator,
    LATENT_DIM
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42

NUM_SYNTHETIC_IMAGES = 84

COVID_CLASS = 0


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT


CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "xray_cgan_diversity_last.pth"
)


OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "xray_synthetic"
    / "COVID"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xray"
)


PREVIEW_PATH = (
    RESULTS_DIR
    / "synthetic_covid_preview.png"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def configurar_seed():
    """
    Define seeds para que a geração possa ser reproduzida.
    """

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


# ============================================================
# CARREGAR GENERATOR
# ============================================================

def carregar_generator():
    """
    Carrega o Generator treinado.

    Não precisamos do Discriminator para gerar imagens.
    """

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint não encontrado:\n"
            f"{CHECKPOINT_PATH}"
        )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    generator = ConditionalGenerator()


    generator.load_state_dict(
        checkpoint[
            "generator_state_dict"
        ]
    )


    generator = generator.to(
        DEVICE
    )


    generator.eval()


    print()
    print(
        "Checkpoint carregado:"
    )

    print(
        CHECKPOINT_PATH
    )


    print(
        "Experimento:",
        checkpoint.get(
            "experiment",
            "desconhecido"
        )
    )


    print(
        "Epoch:",
        checkpoint.get(
            "epoch",
            "?"
        )
    )


    return generator


# ============================================================
# GERAR IMAGENS
# ============================================================

def gerar_imagens(
    generator
):
    """
    Gera NUM_SYNTHETIC_IMAGES imagens da classe COVID.

    Entrada:

        z aleatório
        +
        label COVID = 0

    Saída:

        [84, 1, 64, 64]
    """

    noise = torch.randn(

        NUM_SYNTHETIC_IMAGES,

        LATENT_DIM,

        device=DEVICE
    )


    labels = torch.full(

        (
            NUM_SYNTHETIC_IMAGES,
        ),

        COVID_CLASS,

        dtype=torch.long,

        device=DEVICE
    )


    with torch.inference_mode():

        images = generator(

            noise,

            labels
        )


    print()
    print(
        "Tensor gerado:",
        images.shape
    )


    print(
        "Mínimo:",
        f"{images.min().item():.4f}"
    )


    print(
        "Máximo:",
        f"{images.max().item():.4f}"
    )


    return images


# ============================================================
# CONVERTER PARA [0, 1]
# ============================================================

def desnormalizar(
    images
):
    """
    Generator produz:

        [-1, 1]

    Para salvar as imagens precisamos converter:

        [-1, 1]
            ↓
        [0, 1]
    """

    images = (
        images
        + 1
    ) / 2


    return images.clamp(
        0,
        1
    )


# ============================================================
# SALVAR IMAGENS
# ============================================================

def salvar_imagens(
    images
):
    """
    Salva cada imagem sintética em PNG.

    IMPORTANTE:

    O Generator trabalha em grayscale (1 canal).

    Porém o classificador ResNet utiliza imagens RGB e
    normalização ImageNet com 3 canais.

    Por isso:

        grayscale
            ↓
        RGB com 3 canais iguais

    antes de salvar.
    """

    # --------------------------------------------------------
    # Limpar geração anterior.
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    images = desnormalizar(
        images
    )


    images_cpu = images.cpu()


    for indice, image_tensor in enumerate(
        images_cpu,
        start=1
    ):

        # ====================================================
        # [1, 64, 64]
        # ->
        # [64, 64]
        # ====================================================

        image_array = (
            image_tensor
            .squeeze(0)
            .numpy()
        )


        # ====================================================
        # [0, 1]
        # ->
        # [0, 255]
        # ====================================================

        image_array = (
            image_array
            * 255.0
        )


        image_array = np.clip(
            image_array,
            0,
            255
        )


        image_array = image_array.astype(
            np.uint8
        )


        # ====================================================
        # PIL GRAYSCALE
        # ====================================================

        image = Image.fromarray(
            image_array,
            mode="L"
        )


        # ====================================================
        # GRAYSCALE -> RGB
        # ====================================================

        image = image.convert(
            "RGB"
        )


        nome = (
            f"covid_synthetic_"
            f"{indice:04d}.png"
        )


        caminho = (
            OUTPUT_DIR
            / nome
        )


        image.save(
            caminho
        )


    print()
    print(
        "Imagens salvas:",
        NUM_SYNTHETIC_IMAGES
    )


    print(
        "Pasta:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# PREVIEW
# ============================================================

def salvar_preview(
    images
):
    """
    Salva uma grade com as primeiras 24 imagens
    sintéticas.

    Isso permitirá documentar visualmente as imagens
    utilizadas no experimento de augmentation.
    """

    images = desnormalizar(
        images[:24]
    )


    grid = make_grid(

        images.cpu(),

        nrow=8,

        padding=2
    )


    plt.figure(
        figsize=(
            12,
            5
        )
    )


    plt.imshow(

        grid
        .permute(
            1,
            2,
            0
        )
        .squeeze(),

        cmap="gray"
    )


    plt.axis(
        "off"
    )


    plt.title(
        "COVID-19 - Imagens sintéticas utilizadas na augmentation"
    )


    plt.tight_layout()


    plt.savefig(

        PREVIEW_PATH,

        dpi=150,

        bbox_inches="tight"
    )


    plt.close()


    print()
    print(
        "Preview salvo:"
    )

    print(
        PREVIEW_PATH
    )


# ============================================================
# VALIDAR ARQUIVOS
# ============================================================

def validar_imagens():
    """
    Confirma:

    - exatamente 84 imagens;
    - arquivos válidos;
    - todas em RGB;
    - dimensão 64x64.
    """

    arquivos = sorted(
        OUTPUT_DIR.glob(
            "*.png"
        )
    )


    assert (
        len(arquivos)
        == NUM_SYNTHETIC_IMAGES
    )


    for caminho in arquivos:

        with Image.open(
            caminho
        ) as image:

            assert image.size == (
                64,
                64
            )


            assert image.mode == "RGB"


    print()
    print(
        "Validação das imagens: OK"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    configurar_seed()


    print()
    print("=" * 70)

    print(
        "EcoVision AI - Geração COVID Sintético"
    )

    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        DEVICE
    )


    print(
        "Quantidade a gerar:",
        NUM_SYNTHETIC_IMAGES
    )


    print(
        "Classe:",
        COVID_CLASS,
        "(COVID)"
    )


    # ========================================================
    # GENERATOR
    # ========================================================

    generator = carregar_generator()


    # ========================================================
    # GERAR
    # ========================================================

    images = gerar_imagens(
        generator
    )


    # ========================================================
    # SALVAR
    # ========================================================

    salvar_imagens(
        images
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    salvar_preview(
        images
    )


    # ========================================================
    # VALIDAR
    # ========================================================

    validar_imagens()


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)

    print(
        "GERAÇÃO FINALIZADA"
    )

    print("=" * 70)


    print()
    print(
        "COVID reais no treino:      84"
    )


    print(
        "COVID sintéticos gerados:   84"
    )


    print(
        "COVID disponíveis no novo treino: 168"
    )


if __name__ == "__main__":

    main()