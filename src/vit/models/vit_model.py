# ============================================================
# EcoVision AI
# Vision Transformer completo
# ============================================================

import torch
import torch.nn as nn

from .vit import ViTEmbeddings
from .transformer import TransformerEncoderBlock


class EcoVisionViT(nn.Module):
    """
    Vision Transformer para classificação de resíduos.

    Pipeline:

        Imagem
          ↓
        Patch Embedding
          ↓
        CLS Token + Positional Embedding
          ↓
        Transformer Encoder Blocks
          ↓
        LayerNorm
          ↓
        CLS Token
          ↓
        Classification Head
          ↓
        Logits das 9 classes
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=9,
        embed_dim=192,
        num_heads=3,
        num_layers=4,
        mlp_dim=768,
        dropout=0.1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers

        # ====================================================
        # EMBEDDINGS
        # ====================================================

        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # ====================================================
        # TRANSFORMER ENCODERS
        # ====================================================
        #
        # Criamos vários blocos Encoder.
        #
        # Exemplo:
        #
        # Encoder 1
        # Encoder 2
        # Encoder 3
        # Encoder 4
        # ====================================================

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Normalização final.
        self.norm = nn.LayerNorm(
            embed_dim
        )

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================
        #
        # Recebe:
        #
        # CLS token = 192 dimensões
        #
        # Retorna:
        #
        # 9 logits
        # ====================================================

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )


    def forward(self, x, return_attention=False):
        """
        Entrada:

            x:
            [batch, 3, 224, 224]

        Saída padrão:

            logits:
            [batch, 9]

        Se return_attention=True:

            retorna também os attention weights
            de todos os encoder blocks.
        """

        # ----------------------------------------------------
        # 1. Patch Embedding + CLS + Positional Encoding
        #
        # [batch, 3, 224, 224]
        #
        # →
        #
        # [batch, 197, 192]
        # ----------------------------------------------------

        x = self.embeddings(
            x
        )


        # Lista onde guardaremos os pesos de attention
        # de cada Transformer Encoder.
        all_attention_weights = []


        # ----------------------------------------------------
        # 2. Transformer Encoder Blocks
        # ----------------------------------------------------

        for encoder in self.encoder_blocks:

            x, attention_weights = encoder(
                x
            )

            all_attention_weights.append(
                attention_weights
            )


        # ----------------------------------------------------
        # 3. Normalização final
        # ----------------------------------------------------

        x = self.norm(
            x
        )


        # ----------------------------------------------------
        # 4. Selecionar o CLS Token
        #
        # x:
        #
        # [batch, 197, 192]
        #
        # x[:, 0]:
        #
        # [batch, 192]
        #
        # A posição 0 corresponde ao CLS Token.
        # ----------------------------------------------------

        cls_token = x[
            :,
            0
        ]


        # ----------------------------------------------------
        # 5. Classification Head
        #
        # [batch, 192]
        #
        # →
        #
        # [batch, 9]
        # ----------------------------------------------------

        logits = self.classifier(
            cls_token
        )


        # Attention weights serão úteis futuramente
        # para os Attention Maps.
        if return_attention:

            return (
                logits,
                all_attention_weights
            )


        return logits


# ============================================================
# TESTE DO MODELO COMPLETO
# ============================================================

def testar_ecovision_vit():

    print()
    print("=" * 60)
    print("Teste - EcoVision ViT completo")
    print("=" * 60)


    # Configuração do nosso modelo.
    image_size = 224

    patch_size = 16

    num_classes = 9

    embed_dim = 192

    num_heads = 3

    num_layers = 4

    mlp_dim = 768


    # --------------------------------------------------------
    # Criar modelo
    # --------------------------------------------------------

    model = EcoVisionViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout=0.1
    )


    # Duas imagens artificiais.
    imagens = torch.randn(
        2,
        3,
        224,
        224
    )


    # --------------------------------------------------------
    # Forward normal
    # --------------------------------------------------------

    logits = model(
        imagens
    )


    print(
        "Entrada:",
        imagens.shape
    )


    print(
        "Logits:",
        logits.shape
    )


    # --------------------------------------------------------
    # Teste com Attention
    # --------------------------------------------------------

    logits_attention, attention_weights = model(
        imagens,
        return_attention=True
    )


    print()

    print(
        "Número de Encoder Blocks:",
        len(attention_weights)
    )


    print(
        "Attention do primeiro Encoder:",
        attention_weights[0].shape
    )


    print(
        "Attention do último Encoder:",
        attention_weights[-1].shape
    )


    # ========================================================
    # TESTES AUTOMÁTICOS
    # ========================================================

    assert logits.shape == (
        2,
        num_classes
    )


    assert len(attention_weights) == (
        num_layers
    )


    assert attention_weights[0].shape == (
        2,
        num_heads,
        197,
        197
    )


    assert not torch.isnan(
        logits
    ).any()


    print()

    print(
        "EcoVision ViT completo: TESTE OK"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    testar_ecovision_vit()