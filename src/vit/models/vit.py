# ============================================================
# EcoVision AI
# Vision Transformer - Componentes iniciais
#
# Nesta etapa implementamos:
#
# 1. Patch Embedding
# 2. CLS Token
# 3. Positional Encoding
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Converte uma imagem em uma sequência de patches.

    Exemplo:

        Imagem:
        224 x 224

        Patch:
        16 x 16

        224 / 16 = 14

        14 x 14 = 196 patches

    Cada patch será transformado em um vetor de tamanho:

        embed_dim = 192

    Entrada:

        [batch, 3, 224, 224]

    Saída:

        [batch, 196, 192]
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192
    ):
        super().__init__()

        # ----------------------------------------------------
        # Verificação
        # ----------------------------------------------------

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size deve ser divisível por patch_size."
            )


        self.image_size = image_size

        self.patch_size = patch_size

        self.embed_dim = embed_dim


        # ----------------------------------------------------
        # Quantidade de patches
        # ----------------------------------------------------
        #
        # 224 / 16 = 14
        #
        # 14 x 14 = 196
        # ----------------------------------------------------

        self.patches_per_side = (
            image_size // patch_size
        )

        self.num_patches = (
            self.patches_per_side ** 2
        )


        # ----------------------------------------------------
        # Projeção dos patches
        # ----------------------------------------------------
        #
        # Podemos utilizar Conv2d para:
        #
        # 1. dividir a imagem em patches
        # 2. transformar cada patch em embedding
        #
        # kernel_size = 16
        # stride      = 16
        #
        # Como o stride possui o mesmo tamanho do kernel,
        # os patches não se sobrepõem.
        #
        # Entrada:
        #
        # [batch, 3, 224, 224]
        #
        # Saída:
        #
        # [batch, 192, 14, 14]
        # ----------------------------------------------------

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Verificação do tamanho da imagem.
        # ----------------------------------------------------

        batch_size, channels, height, width = x.shape


        if height != self.image_size or width != self.image_size:

            raise ValueError(
                f"A imagem deve possuir tamanho "
                f"{self.image_size}x{self.image_size}, "
                f"mas recebeu {height}x{width}."
            )


        # ----------------------------------------------------
        # Projeção
        #
        # Antes:
        #
        # [batch, 3, 224, 224]
        #
        # Depois:
        #
        # [batch, 192, 14, 14]
        # ----------------------------------------------------

        x = self.projection(
            x
        )


        # ----------------------------------------------------
        # Flatten das dimensões espaciais.
        #
        # [batch, 192, 14, 14]
        #
        # vira:
        #
        # [batch, 192, 196]
        # ----------------------------------------------------

        x = x.flatten(
            start_dim=2
        )


        # ----------------------------------------------------
        # Reorganizar dimensões.
        #
        # [batch, 192, 196]
        #
        # vira:
        #
        # [batch, 196, 192]
        #
        # Agora cada patch é tratado como um token.
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2
        )


        return x


# ============================================================
# ViT EMBEDDINGS
# ============================================================

class ViTEmbeddings(nn.Module):
    """
    Recebe os patch embeddings e adiciona:

    1. CLS Token
    2. Positional Embedding

    Resultado:

        196 patches
            +
        1 CLS Token

            =

        197 tokens

    Formato final:

        [batch, 197, embed_dim]
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        dropout=0.1
    ):
        super().__init__()


        # ----------------------------------------------------
        # Patch Embedding
        # ----------------------------------------------------

        self.patch_embedding = PatchEmbedding(

            image_size=image_size,

            patch_size=patch_size,

            in_channels=in_channels,

            embed_dim=embed_dim
        )


        self.embed_dim = embed_dim


        self.num_patches = (
            self.patch_embedding.num_patches
        )


        # ====================================================
        # CLS TOKEN
        # ====================================================
        #
        # Criamos um parâmetro treinável.
        #
        # Shape:
        #
        # [1, 1, embed_dim]
        #
        # O primeiro 1 representa um template de batch.
        #
        # O segundo 1 representa um único token.
        #
        # embed_dim = representação do token.
        #
        # O CLS Token será aprendido durante o treinamento.
        # ====================================================

        self.cls_token = nn.Parameter(

            torch.zeros(
                1,
                1,
                embed_dim
            )

        )


        # ====================================================
        # POSITIONAL EMBEDDING
        # ====================================================
        #
        # Attention, sozinha, não sabe onde cada patch estava
        # localizado na imagem.
        #
        # Por isso adicionamos informação posicional.
        #
        # Precisamos de:
        #
        # 196 patches + 1 CLS Token
        #
        # = 197 posições
        #
        # Shape:
        #
        # [1, 197, embed_dim]
        # ====================================================

        self.position_embedding = nn.Parameter(

            torch.zeros(
                1,
                self.num_patches + 1,
                embed_dim
            )

        )


        self.dropout = nn.Dropout(
            dropout
        )


        # Inicialização dos parâmetros aprendíveis.
        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )


    def forward(self, x):

        # ----------------------------------------------------
        # 1. Transformar imagem em patches.
        #
        # [batch, 3, 224, 224]
        #
        # →
        #
        # [batch, 196, 192]
        # ----------------------------------------------------

        x = self.patch_embedding(
            x
        )


        batch_size = x.size(0)


        # ----------------------------------------------------
        # 2. Preparar CLS Token para todo o batch.
        #
        # Original:
        #
        # [1, 1, 192]
        #
        # Exemplo batch = 32:
        #
        # [32, 1, 192]
        # ----------------------------------------------------

        cls_tokens = self.cls_token.expand(

            batch_size,
            -1,
            -1
        )


        # ----------------------------------------------------
        # 3. Adicionar CLS no início da sequência.
        #
        # Antes:
        #
        # [batch, 196, 192]
        #
        # Depois:
        #
        # [batch, 197, 192]
        # ----------------------------------------------------

        x = torch.cat(

            (
                cls_tokens,
                x
            ),

            dim=1
        )


        # ----------------------------------------------------
        # 4. Adicionar informação posicional.
        #
        # Cada posição recebe um vetor aprendível diferente.
        # ----------------------------------------------------

        x = x + self.position_embedding


        # Regularização.
        x = self.dropout(
            x
        )


        return x


# ============================================================
# TESTE DO PATCH EMBEDDING
# ============================================================

def testar_patch_embedding():

    print()
    print("=" * 60)
    print("Teste - Patch Embedding")
    print("=" * 60)


    batch_size = 2

    image_size = 224

    patch_size = 16

    embed_dim = 192


    # --------------------------------------------------------
    # Duas imagens RGB artificiais.
    #
    # Shape:
    #
    # [2, 3, 224, 224]
    # --------------------------------------------------------

    imagens = torch.randn(

        batch_size,
        3,
        image_size,
        image_size
    )


    patch_embedding = PatchEmbedding(

        image_size=image_size,

        patch_size=patch_size,

        embed_dim=embed_dim
    )


    output = patch_embedding(
        imagens
    )


    print(
        "Entrada:",
        imagens.shape
    )


    print(
        "Número de patches:",
        patch_embedding.num_patches
    )


    print(
        "Saída:",
        output.shape
    )


    # --------------------------------------------------------
    # Esperamos:
    #
    # 224 / 16 = 14
    #
    # 14 x 14 = 196 patches
    #
    # output:
    #
    # [2, 196, 192]
    # --------------------------------------------------------

    assert patch_embedding.num_patches == 196


    assert output.shape == (
        batch_size,
        196,
        embed_dim
    )


    print()
    print(
        "Patch Embedding: TESTE OK"
    )


# ============================================================
# TESTE DOS EMBEDDINGS COMPLETOS
# ============================================================

def testar_vit_embeddings():

    print()
    print("=" * 60)
    print("Teste - CLS Token + Positional Embedding")
    print("=" * 60)


    batch_size = 2

    image_size = 224

    patch_size = 16

    embed_dim = 192


    imagens = torch.randn(

        batch_size,
        3,
        image_size,
        image_size
    )


    embeddings = ViTEmbeddings(

        image_size=image_size,

        patch_size=patch_size,

        embed_dim=embed_dim,

        dropout=0.0
    )


    output = embeddings(
        imagens
    )


    print(
        "Entrada:",
        imagens.shape
    )


    print(
        "Patches:",
        embeddings.num_patches
    )


    print(
        "CLS Tokens:",
        1
    )


    print(
        "Total de tokens:",
        embeddings.num_patches + 1
    )


    print(
        "Saída:",
        output.shape
    )


    print()

    print(
        "Shape do CLS Token:",
        embeddings.cls_token.shape
    )


    print(
        "Shape do Positional Embedding:",
        embeddings.position_embedding.shape
    )


    # --------------------------------------------------------
    # Verificações automáticas
    # --------------------------------------------------------

    assert output.shape == (
        batch_size,
        197,
        embed_dim
    )


    assert embeddings.cls_token.shape == (
        1,
        1,
        embed_dim
    )


    assert embeddings.position_embedding.shape == (
        1,
        197,
        embed_dim
    )


    # Confirma que CLS Token é parâmetro treinável.
    assert embeddings.cls_token.requires_grad


    # Positional Embedding também será aprendido.
    assert embeddings.position_embedding.requires_grad


    print()

    print(
        "CLS + Positional Embedding: TESTE OK"
    )


# ============================================================
# EXECUTAR TESTES
# ============================================================

if __name__ == "__main__":

    testar_patch_embedding()

    testar_vit_embeddings()