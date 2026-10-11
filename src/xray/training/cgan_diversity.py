# ============================================================
# EcoVision AI
# Atividade 4.1
# Conditional GAN com regularização de diversidade
#
# Objetivo:
# Combater o mode collapse observado nos experimentos
# anteriores.
#
# Estratégias utilizadas:
#
# 1. Amostragem balanceada das classes
# 2. Learning rate menor no Discriminator
# 3. One-sided label smoothing
# 4. Instance noise com redução progressiva
# 5. Mode-Seeking Regularization
#
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

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from torchvision.utils import make_grid

from src.xray.gan_data import (
    criar_dataset_gan
)

from src.xray.models.cgan import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    LATENT_DIM
)


# ============================================================
# CONFIGURAÇÕES GERAIS
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
# O Generator mantém learning rate maior.
#
# O Discriminator recebe learning rate menor para evitar
# que se torne dominante rapidamente.
# ============================================================

LEARNING_RATE_G = 0.0002

LEARNING_RATE_D = 0.0001


# ============================================================
# PARÂMETROS DO ADAM
# ============================================================

BETA1 = 0.5

BETA2 = 0.999


# ============================================================
# LABEL SMOOTHING
# ============================================================
#
# Imagens reais:
#
# antes:
#     target = 1.0
#
# agora:
#     target = 0.9
#
# Isso reduz a confiança excessiva do Discriminator.
# ============================================================

REAL_LABEL = 0.9

FAKE_LABEL = 0.0


# ============================================================
# INSTANCE NOISE
# ============================================================
#
# Pequeno ruído é adicionado às imagens vistas pelo
# Discriminator.
#
# O valor começa em 0.05 e decresce progressivamente
# até zero na última época.
# ============================================================

INITIAL_INSTANCE_NOISE = 0.05


# ============================================================
# MODE-SEEKING REGULARIZATION
# ============================================================
#
# Essa é a principal nova mitigação.
#
# Queremos que:
#
# z1 diferente de z2
#
# produza:
#
# G(z1) diferente de G(z2)
#
# Se diferentes vetores latentes gerarem sempre a mesma
# imagem, temos um sinal de mode collapse.
# ============================================================

MODE_SEEKING_LAMBDA = 1.0

EPSILON = 1e-8


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
    / "cgan_diversity"
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
    / "xray_cgan_diversity_last.pth"
)


HISTORY_PATH = (
    RESULTS_DIR
    / "cgan_diversity_history.json"
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
    Configura seeds para tornar o experimento
    mais reprodutível.
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
# INICIALIZAÇÃO DOS PESOS
# ============================================================

def inicializar_pesos(
    modulo
):
    """
    Inicialização tradicional de pesos utilizada em DCGAN.

    Conv:
        média = 0
        std   = 0.02

    BatchNorm:
        média = 1
        std   = 0.02
    """

    nome = (
        modulo
        .__class__
        .__name__
    )


    # ========================================================
    # CONVOLUÇÕES
    # ========================================================

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


    # ========================================================
    # BATCH NORMALIZATION
    # ========================================================

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
# ATIVAR / DESATIVAR GRADIENTES
# ============================================================

def configurar_gradientes(
    model,
    enabled
):
    """
    Ativa ou desativa cálculo de gradientes
    de todos os parâmetros de um modelo.

    Durante o treinamento do Generator,
    não precisamos calcular gradientes do Discriminator.
    """

    for parametro in model.parameters():

        parametro.requires_grad_(
            enabled
        )


# ============================================================
# DATALOADER BALANCEADO
# ============================================================

def criar_loader_balanceado():
    """
    Cria DataLoader com WeightedRandomSampler.

    Distribuição original:

        COVID      = 84
        Normal     = 588
        Pneumonia  = 168

    A GAN não deve receber 7 vezes mais Normal que COVID.

    Portanto, cada amostra recebe peso inversamente
    proporcional à frequência de sua classe.
    """

    (
        dataset_base,
        dataset_gan
    ) = criar_dataset_gan()


    labels = []


    # ========================================================
    # RECUPERAR LABELS DO SUBSET DE TREINO
    # ========================================================

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


    # ========================================================
    # CONTAGEM DAS CLASSES
    # ========================================================

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
    # PESO INVERSO
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


    # ========================================================
    # SAMPLER
    # ========================================================

    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(
            dataset_gan
        ),

        replacement=True
    )


    # ========================================================
    # DATALOADER
    # ========================================================

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
# CRIAR AMOSTRAS FIXAS
# ============================================================

def criar_amostras_fixas():
    """
    Cria ruídos fixos.

    Utilizamos exatamente os mesmos ruídos ao longo
    das épocas.

    Assim conseguimos observar visualmente a evolução
    da GAN.
    """

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


    # --------------------------------------------------------
    # 8 COVID
    # 8 Normal
    # 8 Pneumonia
    # --------------------------------------------------------

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
    Calcula o desvio do instance noise.

    Epoch inicial:
        0.05

    Epoch final:
        0.0
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


# ============================================================
# ADICIONAR INSTANCE NOISE
# ============================================================

def adicionar_instance_noise(
    images,
    std
):
    """
    Adiciona ruído gaussiano leve às imagens.

    Depois limita novamente os valores para:

        [-1, 1]
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
# MODE-SEEKING LOSS
# ============================================================

def calcular_mode_seeking_loss(
    generator,
    labels
):
    """
    Combate diretamente o mode collapse.

    Criamos:

        z1
        z2

    diferentes, porém utilizando as mesmas classes.

    Depois geramos:

        image1 = G(z1, classe)
        image2 = G(z2, classe)

    Queremos que uma diferença maior no espaço latente
    gere também diferença nas imagens.

    Score:

        diferença das imagens
        ----------------------
        diferença dos vetores z

    Queremos MAXIMIZAR esse score.

    Como o otimizador minimiza losses:

        loss = -score
    """

    batch_size = labels.size(
        0
    )


    # ========================================================
    # DOIS VETORES LATENTES DIFERENTES
    # ========================================================

    noise_1 = torch.randn(

        batch_size,

        LATENT_DIM,

        device=DEVICE
    )


    noise_2 = torch.randn(

        batch_size,

        LATENT_DIM,

        device=DEVICE
    )


    # ========================================================
    # GERAR DUAS IMAGENS
    # ========================================================

    image_1 = generator(

        noise_1,

        labels
    )


    image_2 = generator(

        noise_2,

        labels
    )


    # ========================================================
    # DIFERENÇA ENTRE IMAGENS
    # ========================================================

    image_difference = (

        torch.abs(

            image_1

            - image_2
        )
        .mean()
    )


    # ========================================================
    # DIFERENÇA ENTRE VETORES LATENTES
    # ========================================================

    latent_difference = (

        torch.abs(

            noise_1

            - noise_2
        )
        .mean()
    )


    # ========================================================
    # SCORE
    # ========================================================

    mode_seeking_score = (

        image_difference

        / (
            latent_difference

            + EPSILON
        )
    )


    # ========================================================
    # LOSS
    # ========================================================
    #
    # Quanto MAIOR o score, melhor.
    #
    # Portanto usamos valor negativo para que a minimização
    # aumente a diversidade.
    # ========================================================

    mode_seeking_loss = (

        -mode_seeking_score
    )


    return (

        mode_seeking_loss,

        mode_seeking_score
    )


# ============================================================
# DIVERSIDADE COVID
# ============================================================

def calcular_diversidade_covid(
    generator
):
    """
    Gera 32 imagens COVID a partir de ruídos diferentes.

    Calculamos o desvio padrão entre elas.

    Quanto maior esse valor, maior tende a ser
    a diversidade das imagens.

    IMPORTANTE:

    Essa não é uma métrica de qualidade clínica.

    É apenas um indicador auxiliar para diagnóstico
    de mode collapse.
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
    """
    Salva 24 imagens:

        linha 1 -> COVID
        linha 2 -> Normal
        linha 3 -> Pneumonia

    São utilizados sempre os mesmos ruídos.
    """

    generator.eval()


    with torch.inference_mode():

        images = generator(

            fixed_noise,

            fixed_labels
        )


    # ========================================================
    # [-1, 1] -> [0, 1]
    # ========================================================

    images = (

        images

        + 1
    ) / 2


    images = images.clamp(
        0,
        1
    )


    # ========================================================
    # GRID
    # ========================================================

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

        f"Conditional GAN + Mode Seeking - Epoch {epoch}\n"

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
    """
    Salva:

        losses
        diversidade COVID
        mode-seeking score
        scores do Discriminator
    """

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
    # GENERATOR / DISCRIMINATOR LOSS
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
        "cGAN + Mode Seeking - Adversarial Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_diversity_losses.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # DIVERSIDADE COVID
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
        "cGAN + Mode Seeking - Diversidade COVID"
    )


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_diversity_covid.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # MODE-SEEKING SCORE
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
            "mode_seeking_score"
        ]
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Mode-Seeking Score"
    )


    plt.title(
        "cGAN - Mode-Seeking Score"
    )


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_mode_seeking_score.png",

        dpi=150
    )


    plt.close()


    # ========================================================
    # D(real) / D(fake)
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
            "discriminator_real_score"
        ],

        label="D(real)"
    )


    plt.plot(

        epochs,

        history[
            "discriminator_fake_score"
        ],

        label="D(fake)"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Score"
    )


    plt.title(
        "cGAN - Discriminator Scores"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "cgan_discriminator_scores.png",

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

    print(
        "EcoVision AI - cGAN com Mode-Seeking Regularization"
    )

    print("=" * 70)


    print()

    print(
        "Dispositivo:",
        DEVICE
    )


    print()

    print(
        "Generator LR:",
        LEARNING_RATE_G
    )


    print(
        "Discriminator LR:",
        LEARNING_RATE_D
    )


    print(
        "Real Label:",
        REAL_LABEL
    )


    print(
        "Instance Noise:",
        INITIAL_INSTANCE_NOISE
    )


    print(
        "Mode-Seeking Lambda:",
        MODE_SEEKING_LAMBDA
    )


    # ========================================================
    # DATASET
    # ========================================================

    (
        dataset_base,
        dataset_gan,
        loader
    ) = criar_loader_balanceado()


    print()

    print(
        "Imagens de treino:",
        len(
            dataset_gan
        )
    )


    print(
        "Batches por epoch:",
        len(
            loader
        )
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


    # ========================================================
    # INICIALIZAÇÃO
    # ========================================================

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
    # AMOSTRAS FIXAS
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

        "generator_adversarial_loss": [],

        "discriminator_loss": [],

        "discriminator_real_score": [],

        "discriminator_fake_score": [],

        "covid_diversity": [],

        "mode_seeking_score": [],

        "instance_noise_std": []
    }


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        inicio = time.time()


        generator.train()

        discriminator.train()


        # ====================================================
        # ACUMULADORES
        # ====================================================

        soma_g = 0.0

        soma_g_adversarial = 0.0

        soma_d = 0.0

        soma_real_score = 0.0

        soma_fake_score = 0.0

        soma_mode_seeking = 0.0


        # ====================================================
        # INSTANCE NOISE DA ÉPOCA
        # ====================================================

        noise_std = obter_noise_std(
            epoch
        )


        # ====================================================
        # BATCHES
        # ====================================================

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
            # 1. TREINAR DISCRIMINATOR
            # =================================================

            configurar_gradientes(

                discriminator,

                True
            )


            optimizer_d.zero_grad()


            # -------------------------------------------------
            # REAIS
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
            # GERAR IMAGENS FALSAS
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
            # FAKE + INSTANCE NOISE
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


            # -------------------------------------------------
            # LOSS DISCRIMINATOR
            # -------------------------------------------------

            loss_d = (

                loss_real

                + loss_fake

            ) / 2


            loss_d.backward()


            optimizer_d.step()


            # =================================================
            # 2. TREINAR GENERATOR
            # =================================================

            # -------------------------------------------------
            # Congelar Discriminator.
            # -------------------------------------------------

            configurar_gradientes(

                discriminator,

                False
            )


            optimizer_g.zero_grad()


            # -------------------------------------------------
            # Target do Generator:
            #
            # queremos que D(fake) -> real.
            # -------------------------------------------------

            generator_targets = torch.ones(

                batch_size,

                device=DEVICE
            )


            # -------------------------------------------------
            # Imagens geradas utilizadas pelo Generator.
            # -------------------------------------------------

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


            # =================================================
            # ADVERSARIAL LOSS
            # =================================================

            adversarial_loss = criterion(

                generated_logits,

                generator_targets
            )


            # =================================================
            # MODE-SEEKING REGULARIZATION
            # =================================================

            (
                mode_seeking_loss,
                mode_seeking_score
            ) = calcular_mode_seeking_loss(

                generator,

                labels
            )


            # =================================================
            # LOSS TOTAL DO GENERATOR
            # =================================================
            #
            # mode_seeking_loss é negativa.
            #
            # Portanto:
            #
            # minimizar loss_g
            #
            # também aumenta o Mode-Seeking Score.
            # =================================================

            loss_g = (

                adversarial_loss

                + MODE_SEEKING_LAMBDA
                * mode_seeking_loss
            )


            loss_g.backward()


            optimizer_g.step()


            # =================================================
            # ESTATÍSTICAS
            # =================================================

            soma_g += (
                loss_g.item()
            )


            soma_g_adversarial += (
                adversarial_loss.item()
            )


            soma_d += (
                loss_d.item()
            )


            soma_mode_seeking += (
                mode_seeking_score
                .detach()
                .item()
            )


            soma_real_score += (

                torch.sigmoid(
                    real_logits.detach()
                )
                .mean()
                .item()
            )


            soma_fake_score += (

                torch.sigmoid(
                    fake_logits.detach()
                )
                .mean()
                .item()
            )


        # ====================================================
        # REATIVAR D
        # ====================================================

        configurar_gradientes(

            discriminator,

            True
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


        generator_adversarial_loss = (

            soma_g_adversarial

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


        mode_seeking_medio = (

            soma_mode_seeking

            / batches
        )


        # ====================================================
        # DIVERSIDADE COVID
        # ====================================================

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
            "generator_adversarial_loss"
        ].append(
            generator_adversarial_loss
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
            "mode_seeking_score"
        ].append(
            mode_seeking_medio
        )


        history[
            "instance_noise_std"
        ].append(
            noise_std
        )


        # ====================================================
        # TEMPO
        # ====================================================

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

            f"G Adv: {generator_adversarial_loss:.4f} | "

            f"D Loss: {discriminator_loss:.4f} | "

            f"D(real): {real_score:.4f} | "

            f"D(fake): {fake_score:.4f} | "

            f"Diversidade COVID: {diversidade:.4f} | "

            f"ModeSeek: {mode_seeking_medio:.4f} | "

            f"Noise: {noise_std:.4f} | "

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
            "cgan_diversity",

        "learning_rate_g":
            LEARNING_RATE_G,

        "learning_rate_d":
            LEARNING_RATE_D,

        "real_label":
            REAL_LABEL,

        "initial_instance_noise":
            INITIAL_INSTANCE_NOISE,

        "mode_seeking_lambda":
            MODE_SEEKING_LAMBDA

    }, CHECKPOINT_PATH)


    # ========================================================
    # SALVAR HISTORY
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
    # FINAL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "CGAN COM MODE-SEEKING FINALIZADA"
    )

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


    print(
        "Maior diversidade COVID:",
        f"{max(history['covid_diversity']):.4f}"
    )


    print()

    print(
        "Mode-Seeking inicial:",
        f"{history['mode_seeking_score'][0]:.4f}"
    )


    print(
        "Mode-Seeking final:",
        f"{history['mode_seeking_score'][-1]:.4f}"
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


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()