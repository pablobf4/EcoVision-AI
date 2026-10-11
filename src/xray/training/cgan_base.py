# ============================================================
# EcoVision AI
# Atividade 4.1
# Treinamento da Conditional GAN - Raio-X
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import make_grid

from src.xray.gan_data import criar_dataset_gan
from src.xray.models.cgan import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    LATENT_DIM
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42

EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 0.0002

BETA1 = 0.5
BETA2 = 0.999

NUM_CLASSES = 3

COVID_CLASS = 0

SAMPLES_PER_CLASS = 8


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT


MODELS_DIR = (
    PROJECT_DIR
    / "models"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "xray"
    / "cgan"
)


SAMPLES_DIR = (
    RESULTS_DIR
    / "samples"
)


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SAMPLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHECKPOINT_PATH = (
    MODELS_DIR
    / "xray_cgan_base_last.pth"
)


HISTORY_PATH = (
    RESULTS_DIR
    / "cgan_base_history.json"
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
# SEED
# ============================================================

def configurar_seed():

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
# INICIALIZAÇÃO DCGAN
# ============================================================

def inicializar_pesos(
    modulo
):
    """
    Inicialização tradicional utilizada em DCGAN.

    Camadas convolucionais:
        normal média 0 e desvio 0.02

    BatchNorm:
        peso média 1 e desvio 0.02
        bias = 0
    """

    nome = modulo.__class__.__name__


    if (
        "Conv" in nome
    ):

        if hasattr(
            modulo,
            "weight"
        ) and modulo.weight is not None:

            nn.init.normal_(
                modulo.weight.data,
                0.0,
                0.02
            )


    elif (
        "BatchNorm" in nome
    ):

        if hasattr(
            modulo,
            "weight"
        ) and modulo.weight is not None:

            nn.init.normal_(
                modulo.weight.data,
                1.0,
                0.02
            )


        if hasattr(
            modulo,
            "bias"
        ) and modulo.bias is not None:

            nn.init.constant_(
                modulo.bias.data,
                0
            )


# ============================================================
# DATALOADER BALANCEADO
# ============================================================

def criar_loader_balanceado():
    """
    Cria DataLoader com WeightedRandomSampler.

    Distribuição original do treino:

        COVID      = 84
        Normal     = 588
        Pneumonia  = 168

    Sem correção, a GAN veria muito mais exemplos Normal.

    O sampler aumenta a chance das classes minoritárias
    aparecerem durante o treinamento.
    """

    (
        dataset_base,
        dataset_gan
    ) = criar_dataset_gan()


    # --------------------------------------------------------
    # Labels pertencentes somente aos índices de treino.
    # --------------------------------------------------------

    labels = []

    for indice in dataset_gan.indices:

        _, label = dataset_base.samples[
            indice
        ]

        labels.append(
            label
        )


    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long
    )


    # --------------------------------------------------------
    # Contagem das classes.
    # --------------------------------------------------------

    class_counts = torch.bincount(
        labels_tensor,
        minlength=NUM_CLASSES
    )


    print()
    print(
        "Distribuição real do treino:"
    )


    for classe, indice in dataset_base.class_to_idx.items():

        print(
            f"{classe:10s}: "
            f"{class_counts[indice].item()}"
        )


    # --------------------------------------------------------
    # Peso inversamente proporcional à frequência.
    #
    # Classe rara -> maior peso.
    # --------------------------------------------------------

    class_weights = (
        1.0
        / class_counts.float()
    )


    sample_weights = class_weights[
        labels_tensor
    ]


    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(
            dataset_gan
        ),

        replacement=True
    )


    loader = DataLoader(

        dataset_gan,

        batch_size=BATCH_SIZE,

        sampler=sampler,

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
# RUÍDO FIXO
# ============================================================

def criar_amostras_fixas():
    """
    Mantemos o mesmo ruído ao longo das épocas.

    Assim podemos observar se a qualidade das imagens
    realmente evolui durante o treinamento.
    """

    total = (
        NUM_CLASSES
        * SAMPLES_PER_CLASS
    )


    fixed_noise = torch.randn(

        total,

        LATENT_DIM,

        device=DEVICE
    )


    labels = []


    for classe in range(
        NUM_CLASSES
    ):

        labels.extend(
            [classe]
            * SAMPLES_PER_CLASS
        )


    fixed_labels = torch.tensor(

        labels,

        dtype=torch.long,

        device=DEVICE
    )


    return (
        fixed_noise,
        fixed_labels
    )


# ============================================================
# SALVAR AMOSTRAS
# ============================================================

def salvar_amostras(
    generator,
    fixed_noise,
    fixed_labels,
    epoch
):
    """
    Salva imagens geradas sempre a partir do mesmo
    conjunto de ruídos.

    Linhas:

        COVID
        Normal
        Pneumonia
    """

    generator.eval()


    with torch.inference_mode():

        fake_images = generator(

            fixed_noise,

            fixed_labels
        )


    # --------------------------------------------------------
    # [-1, 1] -> [0, 1]
    # --------------------------------------------------------

    fake_images = (
        fake_images
        + 1
    ) / 2


    fake_images = fake_images.clamp(
        0,
        1
    )


    grid = make_grid(

        fake_images.cpu(),

        nrow=SAMPLES_PER_CLASS,

        padding=2
    )


    plt.figure(
        figsize=(12, 5)
    )


    plt.imshow(
        grid.permute(
            1,
            2,
            0
        ).squeeze(),
        cmap="gray"
    )


    plt.axis(
        "off"
    )


    plt.title(

        f"Conditional GAN - Epoch {epoch}\n"
        f"COVID | Normal | Pneumonia"
    )


    caminho = (

        SAMPLES_DIR

        / f"epoch_{epoch:03d}.png"
    )


    plt.tight_layout()


    plt.savefig(

        caminho,

        dpi=150,

        bbox_inches="tight"
    )


    plt.close()


    generator.train()


# ============================================================
# DIVERSIDADE COVID
# ============================================================

def calcular_diversidade_covid(
    generator
):
    """
    Métrica simples para acompanhar possível mode collapse.

    Geramos 32 imagens COVID diferentes e calculamos
    o desvio padrão médio dos pixels.

    Se esse valor cair fortemente em direção a zero,
    pode ser um indício de que o gerador está produzindo
    imagens muito semelhantes.

    Observação:
    esta não é uma métrica clínica de qualidade.
    É apenas um indicador auxiliar de diversidade.
    """

    generator.eval()


    quantidade = 32


    noise = torch.randn(

        quantidade,

        LATENT_DIM,

        device=DEVICE
    )


    labels = torch.full(

        (
            quantidade,
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


    # --------------------------------------------------------
    # Desvio entre diferentes imagens geradas.
    # --------------------------------------------------------

    diversidade = (
        images
        .std(
            dim=0
        )
        .mean()
        .item()
    )


    generator.train()


    return diversidade


# ============================================================
# GRÁFICOS
# ============================================================

def salvar_curvas(
    history
):

    epochs = list(
        range(
            1,
            len(
                history[
                    "generator_loss"
                ]
            )
            + 1
        )
    )


    # ========================================================
    # LOSSES
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        epochs,

        history[
            "generator_loss"
        ],

        label="Generator"
    )


    plt.plot(

        epochs,

        history[
            "discriminator_loss"
        ],

        label="Discriminator"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Conditional GAN - Adversarial Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_base_losses.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # DIVERSIDADE COVID
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        epochs,

        history[
            "covid_diversity"
        ]
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Desvio padrão médio"
    )


    plt.title(
        "Conditional GAN - Diversidade das imagens COVID"
    )


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_base_covid_diversity.png",

        dpi=150
    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    configurar_seed()


    print()
    print("=" * 70)
    print("EcoVision AI - Treinamento Conditional GAN")
    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        DEVICE
    )


    # ========================================================
    # DADOS
    # ========================================================

    (
        dataset_base,
        dataset_gan,
        loader
    ) = criar_loader_balanceado()


    print()
    print(
        "Imagens de treino:",
        len(dataset_gan)
    )


    print(
        "Batches por epoch:",
        len(loader)
    )


    # ========================================================
    # MODELOS
    # ========================================================

    generator = (
        ConditionalGenerator()
        .to(DEVICE)
    )


    discriminator = (
        ConditionalDiscriminator()
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Inicialização DCGAN.
    # --------------------------------------------------------

    generator.apply(
        inicializar_pesos
    )


    discriminator.apply(
        inicializar_pesos
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = (
        nn.BCEWithLogitsLoss()
    )


    # ========================================================
    # OPTIMIZERS
    # ========================================================

    optimizer_g = Adam(

        generator.parameters(),

        lr=LEARNING_RATE,

        betas=(
            BETA1,
            BETA2
        )
    )


    optimizer_d = Adam(

        discriminator.parameters(),

        lr=LEARNING_RATE,

        betas=(
            BETA1,
            BETA2
        )
    )


    # ========================================================
    # FIXED NOISE
    # ========================================================

    (
        fixed_noise,
        fixed_labels
    ) = criar_amostras_fixas()


    # ========================================================
    # HISTORY
    # ========================================================

    history = {

        "generator_loss": [],

        "discriminator_loss": [],

        "discriminator_real_score": [],

        "discriminator_fake_score": [],

        "covid_diversity": []
    }


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        inicio = time.time()


        soma_loss_g = 0.0

        soma_loss_d = 0.0

        soma_real_score = 0.0

        soma_fake_score = 0.0


        generator.train()

        discriminator.train()


        for real_images, labels in loader:

            real_images = real_images.to(
                DEVICE
            )


            labels = labels.to(
                DEVICE
            )


            batch_size = real_images.size(
                0
            )


            # =================================================
            # TARGETS
            # =================================================

            real_targets = torch.ones(

                batch_size,

                device=DEVICE
            )


            fake_targets = torch.zeros(

                batch_size,

                device=DEVICE
            )


            # =================================================
            # 1. TREINAR DISCRIMINADOR
            # =================================================

            optimizer_d.zero_grad()


            # -------------------------------------------------
            # Imagens reais.
            # -------------------------------------------------

            real_logits = discriminator(

                real_images,

                labels
            )


            loss_real = criterion(

                real_logits,

                real_targets
            )


            # -------------------------------------------------
            # Imagens falsas.
            # -------------------------------------------------

            noise = torch.randn(

                batch_size,

                LATENT_DIM,

                device=DEVICE
            )


            fake_images = generator(

                noise,

                labels
            )


            fake_logits = discriminator(

                fake_images.detach(),

                labels
            )


            loss_fake = criterion(

                fake_logits,

                fake_targets
            )


            # -------------------------------------------------
            # Loss total do discriminador.
            # -------------------------------------------------

            loss_d = (
                loss_real
                + loss_fake
            ) / 2


            loss_d.backward()


            optimizer_d.step()


            # =================================================
            # 2. TREINAR GERADOR
            # =================================================

            optimizer_g.zero_grad()


            # -------------------------------------------------
            # O gerador quer enganar o discriminador.
            #
            # Portanto, para o treinamento do gerador,
            # as imagens falsas possuem target = REAL.
            # -------------------------------------------------

            generated_logits = discriminator(

                fake_images,

                labels
            )


            loss_g = criterion(

                generated_logits,

                real_targets
            )


            loss_g.backward()


            optimizer_g.step()


            # =================================================
            # ESTATÍSTICAS
            # =================================================

            soma_loss_d += (
                loss_d.item()
            )


            soma_loss_g += (
                loss_g.item()
            )


            # -------------------------------------------------
            # Converter logits para probabilidades apenas
            # para diagnóstico.
            # -------------------------------------------------

            real_score = torch.sigmoid(
                real_logits
            ).mean().item()


            fake_score = torch.sigmoid(
                fake_logits
            ).mean().item()


            soma_real_score += (
                real_score
            )


            soma_fake_score += (
                fake_score
            )


        # ====================================================
        # MÉDIAS DA EPOCH
        # ====================================================

        quantidade_batches = len(
            loader
        )


        generator_loss = (
            soma_loss_g
            / quantidade_batches
        )


        discriminator_loss = (
            soma_loss_d
            / quantidade_batches
        )


        real_score = (
            soma_real_score
            / quantidade_batches
        )


        fake_score = (
            soma_fake_score
            / quantidade_batches
        )


        diversidade = calcular_diversidade_covid(
            generator
        )


        history[
            "generator_loss"
        ].append(
            generator_loss
        )


        history[
            "discriminator_loss"
        ].append(
            discriminator_loss
        )


        history[
            "discriminator_real_score"
        ].append(
            real_score
        )


        history[
            "discriminator_fake_score"
        ].append(
            fake_score
        )


        history[
            "covid_diversity"
        ].append(
            diversidade
        )


        tempo = (
            time.time()
            - inicio
        )


        # ====================================================
        # LOG
        # ====================================================

        print()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"G Loss: {generator_loss:.4f} | "
            f"D Loss: {discriminator_loss:.4f} | "
            f"D(real): {real_score:.4f} | "
            f"D(fake): {fake_score:.4f} | "
            f"Diversidade COVID: {diversidade:.4f} | "
            f"Tempo: {tempo / 60:.2f} min"
        )


        # ====================================================
        # AMOSTRAS VISUAIS
        # ====================================================

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            salvar_amostras(

                generator,

                fixed_noise,

                fixed_labels,

                epoch
            )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    torch.save({

        "generator_state_dict":
            generator.state_dict(),

        "discriminator_state_dict":
            discriminator.state_dict(),

        "optimizer_g_state_dict":
            optimizer_g.state_dict(),

        "optimizer_d_state_dict":
            optimizer_d.state_dict(),

        "epoch":
            EPOCHS,

        "latent_dim":
            LATENT_DIM,

        "num_classes":
            NUM_CLASSES,

        "class_to_idx":
            dataset_base.class_to_idx,

        "experiment":
            "cgan_base"

    }, CHECKPOINT_PATH)


    # ========================================================
    # HISTORY
    # ========================================================

    with open(

        HISTORY_PATH,

        "w",

        encoding="utf-8"

    ) as arquivo:

        json.dump(

            history,

            arquivo,

            indent=4
        )


    salvar_curvas(
        history
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("TREINAMENTO CGAN FINALIZADO")
    print("=" * 70)


    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    print()
    print(
        "Histórico:"
    )

    print(
        HISTORY_PATH
    )


    print()
    print(
        "Amostras:"
    )

    print(
        SAMPLES_DIR
    )


if __name__ == "__main__":

    main()