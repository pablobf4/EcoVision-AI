# ============================================================
# EcoVision AI
# Atividade 4.1
# Conditional GAN para radiografias de tórax
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# CONFIGURAÇÕES
# ============================================================

LATENT_DIM = 100

NUM_CLASSES = 3

IMAGE_SIZE = 64

IMAGE_CHANNELS = 1


# ============================================================
# GERADOR CONDICIONAL
# ============================================================

class ConditionalGenerator(nn.Module):
    """
    Gerador da Conditional GAN.

    Entrada:

        ruído z
        [batch, 100]

        +

        label da classe
        [batch]

    Classes:

        0 -> COVID
        1 -> Normal
        2 -> Pneumonia

    Saída:

        imagem sintética
        [batch, 1, 64, 64]

    A saída utiliza Tanh, portanto os pixels ficam
    aproximadamente no intervalo [-1, 1].
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ====================================================
        # EMBEDDING DA CLASSE
        # ====================================================
        #
        # O número da classe é transformado em um vetor.
        #
        # Exemplo:
        #
        # COVID = 0
        #
        #       ↓ embedding
        #
        # vetor de 100 dimensões
        #
        # Depois concatenamos esse vetor ao ruído z.
        # ====================================================

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim
        )

        # ====================================================
        # REDE GERADORA
        # ====================================================
        #
        # Entrada:
        #
        # ruído            = 100
        # embedding classe = 100
        #
        # total            = 200 canais
        #
        # Evolução espacial:
        #
        # 1x1
        #  ↓
        # 4x4
        #  ↓
        # 8x8
        #  ↓
        # 16x16
        #  ↓
        # 32x32
        #  ↓
        # 64x64
        # ====================================================

        input_channels = (
            latent_dim
            * 2
        )

        self.network = nn.Sequential(

            # ------------------------------------------------
            # [B, 200, 1, 1]
            # ->
            # [B, 512, 4, 4]
            # ------------------------------------------------

            nn.ConvTranspose2d(
                input_channels,
                512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 4x4 -> 8x8
            # ------------------------------------------------

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 8x8 -> 16x16
            # ------------------------------------------------

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 16x16 -> 32x32
            # ------------------------------------------------

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 32x32 -> 64x64
            # ------------------------------------------------

            nn.ConvTranspose2d(
                64,
                IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            # ------------------------------------------------
            # Mantém saída entre -1 e 1.
            # ------------------------------------------------

            nn.Tanh()
        )


    def forward(
        self,
        noise,
        labels
    ):
        """
        Gera imagens condicionadas pela classe.
        """

        # ----------------------------------------------------
        # Label:
        #
        # [B]
        # ->
        # [B, 100]
        # ----------------------------------------------------

        label_vector = self.label_embedding(
            labels
        )


        # ----------------------------------------------------
        # Concatenar:
        #
        # noise        [B, 100]
        # label_vector [B, 100]
        #
        # resultado    [B, 200]
        # ----------------------------------------------------

        x = torch.cat(

            (
                noise,
                label_vector
            ),

            dim=1
        )


        # ----------------------------------------------------
        # Transformar para formato convolucional:
        #
        # [B, 200]
        #
        # ->
        #
        # [B, 200, 1, 1]
        # ----------------------------------------------------

        x = x.unsqueeze(
            2
        ).unsqueeze(
            3
        )


        return self.network(
            x
        )


# ============================================================
# DISCRIMINADOR CONDICIONAL
# ============================================================

class ConditionalDiscriminator(nn.Module):
    """
    Discriminador da Conditional GAN.

    Ele recebe:

        imagem
        +
        label

    e aprende a responder:

        essa imagem é real/falsa
        PARA ESTA classe?

    Isso é importante porque não queremos apenas saber
    se a radiografia parece real.

    Queremos saber, por exemplo:

        "essa imagem parece uma radiografia COVID real?"
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES
    ):
        super().__init__()

        self.num_classes = num_classes

        # ====================================================
        # EMBEDDING DA CLASSE
        # ====================================================
        #
        # Cada label vira um mapa 64x64.
        #
        # Exemplo:
        #
        # label COVID
        #
        #      ↓
        #
        # vetor com 4096 valores
        #
        #      ↓ reshape
        #
        # [1, 64, 64]
        # ====================================================

        self.label_embedding = nn.Embedding(

            num_classes,

            IMAGE_SIZE
            * IMAGE_SIZE
        )


        # ====================================================
        # REDE DISCRIMINADORA
        # ====================================================
        #
        # Entrada:
        #
        # imagem = 1 canal
        # label  = 1 canal
        #
        # total = 2 canais
        # ====================================================

        self.network = nn.Sequential(

            # ------------------------------------------------
            # 64x64 -> 32x32
            # ------------------------------------------------

            nn.Conv2d(
                2,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # ------------------------------------------------
            # 32x32 -> 16x16
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # ------------------------------------------------
            # 16x16 -> 8x8
            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # ------------------------------------------------
            # 8x8 -> 4x4
            # ------------------------------------------------

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # ------------------------------------------------
            # 4x4 -> 1x1
            # ------------------------------------------------
            #
            # Não colocamos Sigmoid aqui.
            #
            # No treinamento utilizaremos:
            #
            # BCEWithLogitsLoss
            #
            # que já incorpora a operação sigmoid de maneira
            # numericamente mais estável.
            # ------------------------------------------------

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )


    def forward(
        self,
        images,
        labels
    ):
        """
        Avalia se uma imagem é real/falsa considerando
        também a classe informada.
        """

        batch_size = images.size(
            0
        )


        # ----------------------------------------------------
        # [B]
        # ->
        # [B, 4096]
        # ----------------------------------------------------

        label_map = self.label_embedding(
            labels
        )


        # ----------------------------------------------------
        # [B, 4096]
        # ->
        # [B, 1, 64, 64]
        # ----------------------------------------------------

        label_map = label_map.view(

            batch_size,

            1,

            IMAGE_SIZE,

            IMAGE_SIZE
        )


        # ----------------------------------------------------
        # imagem:
        # [B, 1, 64, 64]
        #
        # label:
        # [B, 1, 64, 64]
        #
        # concat:
        # [B, 2, 64, 64]
        # ----------------------------------------------------

        x = torch.cat(

            (
                images,
                label_map
            ),

            dim=1
        )


        output = self.network(
            x
        )


        # ----------------------------------------------------
        # [B, 1, 1, 1]
        #
        # ->
        #
        # [B]
        # ----------------------------------------------------

        return output.view(
            -1
        )


# ============================================================
# TESTE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Teste Conditional GAN")
    print("=" * 70)


    batch_size = 8


    # ========================================================
    # MODELOS
    # ========================================================

    generator = ConditionalGenerator()


    discriminator = ConditionalDiscriminator()


    # ========================================================
    # RUÍDO
    # ========================================================

    noise = torch.randn(

        batch_size,

        LATENT_DIM
    )


    # ========================================================
    # LABELS
    # ========================================================
    #
    # Criamos labels misturadas para verificar se o modelo
    # aceita todas as classes.
    # ========================================================

    labels = torch.tensor(

        [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1
        ],

        dtype=torch.long
    )


    # ========================================================
    # GERADOR
    # ========================================================

    with torch.no_grad():

        fake_images = generator(

            noise,

            labels
        )


    print()
    print(
        "Noise:",
        noise.shape
    )


    print(
        "Labels:",
        labels.shape
    )


    print(
        "Imagens geradas:",
        fake_images.shape
    )


    print(
        "Mínimo:",
        f"{fake_images.min().item():.4f}"
    )


    print(
        "Máximo:",
        f"{fake_images.max().item():.4f}"
    )


    # ========================================================
    # DISCRIMINADOR
    # ========================================================

    with torch.no_grad():

        logits = discriminator(

            fake_images,

            labels
        )


    print()
    print(
        "Saída discriminador:",
        logits.shape
    )


    print(
        "Logits:",
        logits
    )


    # ========================================================
    # TESTES AUTOMÁTICOS
    # ========================================================

    assert fake_images.shape == (

        batch_size,

        1,

        64,

        64
    )


    assert logits.shape == (
        batch_size,
    )


    assert fake_images.min() >= -1.01


    assert fake_images.max() <= 1.01


    # ========================================================
    # PARÂMETROS
    # ========================================================

    generator_params = sum(

        p.numel()

        for p in generator.parameters()
    )


    discriminator_params = sum(

        p.numel()

        for p in discriminator.parameters()
    )


    print()
    print(
        f"Parâmetros Generator: "
        f"{generator_params:,}"
    )


    print(
        f"Parâmetros Discriminator: "
        f"{discriminator_params:,}"
    )


    print()
    print(
        "Conditional GAN: TESTE OK"
    )


if __name__ == "__main__":

    main()