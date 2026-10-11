# ============================================================
# EcoVision AI
# Transformer Encoder Block
# ============================================================

import torch
import torch.nn as nn

# Importamos o Multi-Head Attention que implementamos do zero.
from .attention import MultiHeadAttention


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    """
    Rede Feed Forward utilizada dentro do Transformer.

    Estrutura:

        embedding
            ↓
        Linear
            ↓
        GELU
            ↓
        Dropout
            ↓
        Linear
            ↓
        Dropout

    Cada token é processado individualmente pela mesma rede.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()


        # Primeira camada:
        #
        # aumenta a dimensionalidade.
        #
        # Exemplo:
        #
        # 192 → 768

        self.linear1 = nn.Linear(
            embed_dim,
            hidden_dim
        )


        # GELU é uma função de ativação muito utilizada
        # em arquiteturas Transformer.
        self.activation = nn.GELU()


        self.dropout1 = nn.Dropout(
            dropout
        )


        # Segunda camada:
        #
        # reduz novamente para a dimensão original.
        #
        # 768 → 192

        self.linear2 = nn.Linear(
            hidden_dim,
            embed_dim
        )


        self.dropout2 = nn.Dropout(
            dropout
        )


    def forward(self, x):

        # Primeira transformação linear.
        x = self.linear1(x)

        # Não linearidade.
        x = self.activation(x)

        # Regularização.
        x = self.dropout1(x)

        # Volta para embed_dim.
        x = self.linear2(x)

        x = self.dropout2(x)

        return x


# ============================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """
    Bloco Encoder utilizado pelo Vision Transformer.

    Estrutura utilizada:

        x
        │
        LayerNorm
        │
        Multi-Head Attention
        │
        +
        │ residual
        ▼

        x
        │
        LayerNorm
        │
        Feed Forward
        │
        +
        │ residual
        ▼

        output

    Também retornamos os attention weights para que
    posteriormente possamos gerar Attention Maps.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.1
    ):
        super().__init__()


        # ----------------------------------------------------
        # Primeira normalização
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(
            embed_dim
        )


        # ----------------------------------------------------
        # Multi-Head Attention
        #
        # Esta é a implementação feita por nós.
        # ----------------------------------------------------

        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )


        self.dropout1 = nn.Dropout(
            dropout
        )


        # ----------------------------------------------------
        # Segunda normalização
        # ----------------------------------------------------

        self.norm2 = nn.LayerNorm(
            embed_dim
        )


        # ----------------------------------------------------
        # Feed Forward
        #
        # Exemplo:
        #
        # 192
        # ↓
        # 768
        # ↓
        # 192
        # ----------------------------------------------------

        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=mlp_dim,
            dropout=dropout
        )


        self.dropout2 = nn.Dropout(
            dropout
        )


    def forward(self, x):
        """
        Entrada:

            x:
            [batch, tokens, embed_dim]

        Exemplo futuro:

            [32, 197, 192]

        Retorna:

            output:
            [batch, tokens, embed_dim]

            attention_weights:
            [batch, heads, tokens, tokens]
        """


        # ====================================================
        # BLOCO 1
        # MULTI-HEAD ATTENTION + RESIDUAL
        # ====================================================

        # Guardamos a entrada.
        residual = x


        # Pre-Normalization.
        normalized = self.norm1(
            x
        )


        # Multi-Head Attention.
        attention_output, attention_weights = self.attention(
            normalized
        )


        attention_output = self.dropout1(
            attention_output
        )


        # ----------------------------------------------------
        # Residual Connection
        #
        # saída = entrada + transformação(entrada)
        # ----------------------------------------------------

        x = residual + attention_output


        # ====================================================
        # BLOCO 2
        # FEED FORWARD + RESIDUAL
        # ====================================================

        residual = x


        normalized = self.norm2(
            x
        )


        ff_output = self.feed_forward(
            normalized
        )


        ff_output = self.dropout2(
            ff_output
        )


        # Segunda residual connection.
        x = residual + ff_output


        return x, attention_weights


# ============================================================
# TESTE DO FEED FORWARD
# ============================================================

def testar_feed_forward():

    print()
    print("=" * 60)
    print("Teste - Feed Forward")
    print("=" * 60)


    batch_size = 2

    num_tokens = 5

    embed_dim = 192

    mlp_dim = 768


    x = torch.randn(
        batch_size,
        num_tokens,
        embed_dim
    )


    feed_forward = FeedForward(
        embed_dim=embed_dim,
        hidden_dim=mlp_dim
    )


    output = feed_forward(
        x
    )


    print(
        "Entrada:",
        x.shape
    )


    print(
        "Saída:",
        output.shape
    )


    # A dimensão deve permanecer igual.
    assert output.shape == (
        batch_size,
        num_tokens,
        embed_dim
    )


    print()
    print("Feed Forward: TESTE OK")


# ============================================================
# TESTE DO TRANSFORMER ENCODER
# ============================================================

def testar_transformer_encoder():

    print()
    print("=" * 60)
    print("Teste - Transformer Encoder Block")
    print("=" * 60)


    # --------------------------------------------------------
    # Configuração semelhante ao nosso futuro EcoVision ViT.
    # --------------------------------------------------------

    batch_size = 2

    num_tokens = 197

    embed_dim = 192

    num_heads = 3

    mlp_dim = 768


    # Entrada artificial:
    #
    # [batch, tokens, embedding]

    x = torch.randn(
        batch_size,
        num_tokens,
        embed_dim
    )


    encoder = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=0.1
    )


    output, attention_weights = encoder(
        x
    )


    print(
        "Entrada:",
        x.shape
    )


    print(
        "Saída:",
        output.shape
    )


    print(
        "Attention weights:",
        attention_weights.shape
    )


    # --------------------------------------------------------
    # Verificações automáticas
    # --------------------------------------------------------

    assert output.shape == (
        batch_size,
        num_tokens,
        embed_dim
    )


    assert attention_weights.shape == (
        batch_size,
        num_heads,
        num_tokens,
        num_tokens
    )


    # Verifica se nenhuma operação gerou NaN.
    assert not torch.isnan(
        output
    ).any()


    print()
    print(
        "Transformer Encoder Block: TESTE OK"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    testar_feed_forward()

    testar_transformer_encoder()