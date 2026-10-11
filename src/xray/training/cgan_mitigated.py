# ============================================================
# EcoVision AI
# Atividade 4.1
# Conditional GAN mitigada - Raio-X
#
# Mitigações aplicadas:
#
# 1. Learning rate menor no Discriminator
# 2. One-sided label smoothing
# 3. Instance noise com redução progressiva
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

NUM_CLASSES = 3

COVID_CLASS = 0


# ============================================================
# LEARNING RATES
# ============================================================
#
# Gerador mantém LR original.
#
# Discriminator aprende mais devagar para evitar
# que domine muito rapidamente o treinamento.
# ============================================================

LEARNING_RATE_G = 0.0002

LEARNING_RATE_D = 0.0001


BETA1 = 0.5

BETA2 = 0.999


# ============================================================
# LABEL SMOOTHING
# ============================================================
#
# Na GAN base:
#
# real = 1.0
#
# Agora:
#
# real = 0.9
#
# Isso reduz excesso de confiança do Discriminator.
# ============================================================

REAL_LABEL = 0.9

FAKE_LABEL = 0.0


# ============================================================
# INSTANCE NOISE
# ============================================================
#
# Adicionamos pequeno ruído às imagens recebidas pelo
# Discriminator.
#
# O ruído começa em 0.05 e vai diminuindo até 0.
# ============================================================

INITIAL_INSTANCE_NOISE = 0.05


# ============================================================
# VISUALIZAÇÃO
# ============================================================

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
    / "cgan_mitigated"
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
    / "xray_cgan_mitigated_last.pth"
)


HISTORY_PATH = (
    RESULTS_DIR
    / "cgan_mitigated_history.json"
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

    nome = (
        modulo
        .__class__
        .__name__
    )


    if "Conv" in nome:

        if (
            hasattr(
                modulo,
                "weight"
            )
            and
            modulo.weight is not None
        ):

            nn.init.normal_(

                modulo.weight.data,

                0.0,

                0.02
            )


    elif "BatchNorm" in nome:

        if (
            hasattr(
                modulo,
                "weight"
            )
            and
            modulo.weight is not None
        ):

            nn.init.normal_(

                modulo.weight.data,

                1.0,

                0.02
            )


        if (
            hasattr(
                modulo,
                "bias"
            )
            and
            modulo.bias is not None
        ):

            nn.init.constant_(

                modulo.bias.data,

                0
            )


# ============================================================
# LOADER BALANCEADO
# ============================================================

def criar_loader_balanceado():

    (
        dataset_base,
        dataset_gan
    ) = criar_dataset_gan()


    labels = []


    for indice in dataset_gan.indices:

        _, label = (
            dataset_base.samples[
                indice
            ]
        )


        labels.append(
            label
        )


    labels_tensor = torch.tensor(

        labels,

        dtype=torch.long
    )


    class_counts = torch.bincount(

        labels_tensor,

        minlength=NUM_CLASSES
    )


    print()
    print(
        "Distribuição original:"
    )


    for (
        classe,
        indice
    ) in dataset_base.class_to_idx.items():

        print(

            f"{classe:10s}: "

            f"{class_counts[indice].item()}"
        )


    # ========================================================
    # PESOS
    # ========================================================

    class_weights = (

        1.0

        / class_counts.float()
    )


    sample_weights = (

        class_weights[
            labels_tensor
        ]
    )


    sampler = WeightedRandomSampler(

        sample_weights,

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

    quantidade = (

        NUM_CLASSES

        * SAMPLES_PER_CLASS
    )


    fixed_noise = torch.randn(

        quantidade,

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
# INSTANCE NOISE
# ============================================================

def obter_noise_std(
    epoch
):
    """
    Reduz progressivamente o instance noise.

    Epoch 1:
        aproximadamente 0.05

    Epoch final:
        0
    """

    progresso = (

        (epoch - 1)

        / max(
            1,
            EPOCHS - 1
        )
    )


    noise_std = (

        INITIAL_INSTANCE_NOISE

        * (
            1.0
            - progresso
        )
    )


    return max(
        0.0,
        noise_std
    )


def adicionar_instance_noise(
    images,
    std
):
    """
    Adiciona ruído gaussiano leve.

    Depois mantém os pixels no intervalo [-1, 1].
    """

    if std <= 0:

        return images


    noise = torch.randn_like(
        images
    )


    noisy_images = (

        images

        + noise
        * std
    )


    return noisy_images.clamp(
        -1,
        1
    )


# ============================================================
# DIVERSIDADE COVID
# ============================================================

def calcular_diversidade_covid(
    generator
):

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
# SALVAR AMOSTRAS
# ============================================================

def salvar_amostras(
    generator,
    fixed_noise,
    fixed_labels,
    epoch
):

    generator.eval()


    with torch.inference_mode():

        images = generator(

            fixed_noise,

            fixed_labels
        )


    # [-1, 1] -> [0, 1]

    images = (

        images
        + 1
    ) / 2


    images = images.clamp(
        0,
        1
    )


    grid = make_grid(

        images.cpu(),

        nrow=SAMPLES_PER_CLASS,

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

        f"Conditional GAN Mitigada - Epoch {epoch}\n"

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
# SALVAR CURVAS
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
        figsize=(
            9,
            5
        )
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
        "cGAN Mitigada - Adversarial Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_mitigated_losses.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # DIVERSIDADE
    # ========================================================

    plt.figure(
        figsize=(
            9,
            5
        )
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
        "cGAN Mitigada - Diversidade COVID"
    )


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_mitigated_covid_diversity.png",

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
    print("EcoVision AI - cGAN Mitigada")
    print("=" * 70)


    print()
    print(
        "Dispositivo:",
        DEVICE
    )


    print()
    print(
        "Learning Rate Generator:",
        LEARNING_RATE_G
    )


    print(
        "Learning Rate Discriminator:",
        LEARNING_RATE_D
    )


    print(
        "Real Label:",
        REAL_LABEL
    )


    print(
        "Instance Noise inicial:",
        INITIAL_INSTANCE_NOISE
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
        "Imagens:",
        len(dataset_gan)
    )


    print(
        "Batches:",
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

        lr=LEARNING_RATE_G,

        betas=(
            BETA1,
            BETA2
        )
    )


    optimizer_d = Adam(

        discriminator.parameters(),

        lr=LEARNING_RATE_D,

        betas=(
            BETA1,
            BETA2
        )
    )


    # ========================================================
    # FIXED SAMPLES
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

        "covid_diversity": [],

        "instance_noise_std": []
    }


    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        inicio = time.time()


        generator.train()

        discriminator.train()


        soma_g = 0.0

        soma_d = 0.0

        soma_real_score = 0.0

        soma_fake_score = 0.0


        # ====================================================
        # INSTANCE NOISE DA EPOCH
        # ====================================================

        noise_std = obter_noise_std(
            epoch
        )


        for (
            real_images,
            labels
        ) in loader:

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

            real_targets = torch.full(

                (
                    batch_size,
                ),

                REAL_LABEL,

                device=DEVICE
            )


            fake_targets = torch.full(

                (
                    batch_size,
                ),

                FAKE_LABEL,

                device=DEVICE
            )


            # =================================================
            # TREINAR DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad()


            # -------------------------------------------------
            # Reais + instance noise
            # -------------------------------------------------

            real_images_noisy = (
                adicionar_instance_noise(

                    real_images,

                    noise_std
                )
            )


            real_logits = discriminator(

                real_images_noisy,

                labels
            )


            loss_real = criterion(

                real_logits,

                real_targets
            )


            # -------------------------------------------------
            # Gerar falsas
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


            # -------------------------------------------------
            # Falsas + instance noise
            # -------------------------------------------------

            fake_images_noisy = (
                adicionar_instance_noise(

                    fake_images.detach(),

                    noise_std
                )
            )


            fake_logits = discriminator(

                fake_images_noisy,

                labels
            )


            loss_fake = criterion(

                fake_logits,

                fake_targets
            )


            loss_d = (

                loss_real

                + loss_fake

            ) / 2


            loss_d.backward()


            optimizer_d.step()


            # =================================================
            # TREINAR GENERATOR
            # =================================================

            optimizer_g.zero_grad()


            # -------------------------------------------------
            # Para o Generator queremos que o Discriminator
            # considere as imagens falsas como reais.
            #
            # Aqui usamos target = 1.0.
            # -------------------------------------------------

            generator_targets = torch.ones(

                batch_size,

                device=DEVICE
            )


            fake_for_generator = (
                adicionar_instance_noise(

                    fake_images,

                    noise_std
                )
            )


            generated_logits = discriminator(

                fake_for_generator,

                labels
            )


            loss_g = criterion(

                generated_logits,

                generator_targets
            )


            loss_g.backward()


            optimizer_g.step()


            # =================================================
            # ESTATÍSTICAS
            # =================================================

            soma_g += (
                loss_g.item()
            )


            soma_d += (
                loss_d.item()
            )


            soma_real_score += (

                torch.sigmoid(
                    real_logits
                )
                .mean()
                .item()
            )


            soma_fake_score += (

                torch.sigmoid(
                    fake_logits
                )
                .mean()
                .item()
            )


        # ====================================================
        # MÉDIAS
        # ====================================================

        batches = len(
            loader
        )


        generator_loss = (
            soma_g
            / batches
        )


        discriminator_loss = (
            soma_d
            / batches
        )


        real_score = (
            soma_real_score
            / batches
        )


        fake_score = (
            soma_fake_score
            / batches
        )


        diversidade = (
            calcular_diversidade_covid(
                generator
            )
        )


        # ====================================================
        # HISTORY
        # ====================================================

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


        history[
            "instance_noise_std"
        ].append(
            noise_std
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

            f"Noise: {noise_std:.4f} | "

            f"Tempo: {tempo / 60:.2f} min"
        )


        # ====================================================
        # IMAGENS
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
            "cgan_mitigated",

        "learning_rate_g":
            LEARNING_RATE_G,

        "learning_rate_d":
            LEARNING_RATE_D,

        "real_label":
            REAL_LABEL,

        "initial_instance_noise":
            INITIAL_INSTANCE_NOISE

    }, CHECKPOINT_PATH)


    # ========================================================
    # HISTORY JSON
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


    # ========================================================
    # GRÁFICOS
    # ========================================================

    salvar_curvas(
        history
    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("CGAN MITIGADA FINALIZADA")
    print("=" * 70)


    print()
    print(
        "Diversidade COVID inicial:",
        f"{history['covid_diversity'][0]:.4f}"
    )


    print(
        "Diversidade COVID final:",
        f"{history['covid_diversity'][-1]:.4f}"
    )


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