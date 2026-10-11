# ============================================================
# EcoVision AI
# Attention Mechanisms
#
# Implementações:
# 1. Scaled Dot-Product Attention
# 2. Multi-Head Attention
# ============================================================

import math

import torch
import torch.nn as nn


# ============================================================
# 1. SCALED DOT-PRODUCT ATTENTION
# ============================================================

class ScaledDotProductAttention(nn.Module):
    """
    Implementação do mecanismo de atenção:

        Attention(Q, K, V) =
        softmax(QK^T / sqrt(d_k)) V

    Onde:

        Q = Query
        K = Key
        V = Value

    A classe recebe Q, K e V já projetados pelos
    attention heads.
    """

    def __init__(self, dropout=0.0):
        super().__init__()

        # Dropout aplicado sobre os pesos de attention.
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value):
        """
        Formato esperado:

        query: [batch, heads, tokens, head_dim]
        key:   [batch, heads, tokens, head_dim]
        value: [batch, heads, tokens, head_dim]

        Retorna:

        output:
            [batch, heads, tokens, head_dim]

        attention_weights:
            [batch, heads, tokens, tokens]
        """

        # ----------------------------------------------------
        # 1. Descobrir a dimensão utilizada pelo head.
        #
        # Exemplo:
        #
        # query.shape = [2, 3, 5, 64]
        #
        # head_dim = 64
        # ----------------------------------------------------

        head_dim = query.size(-1)


        # ----------------------------------------------------
        # 2. Calcular similaridade entre Query e Key.
        #
        # Fórmula:
        #
        # QK^T
        #
        # key.transpose(-2, -1):
        #
        # [tokens, head_dim]
        #
        # vira:
        #
        # [head_dim, tokens]
        # ----------------------------------------------------

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )


        # ----------------------------------------------------
        # 3. Scaling.
        #
        # Dividimos os scores por sqrt(d_k).
        #
        # Isso evita valores excessivamente grandes antes
        # do softmax e ajuda a manter gradientes estáveis.
        # ----------------------------------------------------

        scores = scores / math.sqrt(head_dim)


        # ----------------------------------------------------
        # 4. Softmax.
        #
        # Transforma os scores em pesos de atenção.
        #
        # Cada token passa a possuir uma distribuição
        # de importância sobre os demais tokens.
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        # Dropout opcional sobre os pesos de atenção.
        attention_weights = self.dropout(
            attention_weights
        )


        # ----------------------------------------------------
        # 5. Aplicar os pesos sobre os Values.
        #
        # Tokens com maior attention weight contribuem mais
        # para a nova representação.
        # ----------------------------------------------------

        output = torch.matmul(
            attention_weights,
            value
        )


        # Retornamos os pesos porque eles serão utilizados
        # posteriormente nos Attention Maps do EcoVision.
        return output, attention_weights


# ============================================================
# 2. MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Implementação de Multi-Head Attention do zero.

    Cada head possui projeções próprias para:

        Query
        Key
        Value

    Depois:

        Head 1 \
        Head 2  -> concatenação -> projeção final
        Head 3 /

    Exemplo:

        embed_dim = 192
        num_heads = 3

        head_dim = 192 / 3
                 = 64
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0
    ):
        super().__init__()


        # ----------------------------------------------------
        # Validação
        # ----------------------------------------------------

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim deve ser divisível por num_heads."
            )


        self.embed_dim = embed_dim

        self.num_heads = num_heads

        # Dimensão individual de cada attention head.
        self.head_dim = embed_dim // num_heads


        # ====================================================
        # PROJEÇÕES DE QUERY
        # ====================================================
        #
        # Cada attention head possui sua própria camada linear.
        #
        # Por exemplo:
        #
        # Head 1 → Wq1
        # Head 2 → Wq2
        # Head 3 → Wq3
        # ====================================================

        self.query_projections = nn.ModuleList([

            nn.Linear(
                embed_dim,
                self.head_dim
            )

            for _ in range(num_heads)
        ])


        # ====================================================
        # PROJEÇÕES DE KEY
        # ====================================================

        self.key_projections = nn.ModuleList([

            nn.Linear(
                embed_dim,
                self.head_dim
            )

            for _ in range(num_heads)
        ])


        # ====================================================
        # PROJEÇÕES DE VALUE
        # ====================================================

        self.value_projections = nn.ModuleList([

            nn.Linear(
                embed_dim,
                self.head_dim
            )

            for _ in range(num_heads)
        ])


        # Nosso mecanismo de atenção implementado acima.
        self.attention = ScaledDotProductAttention(
            dropout=dropout
        )


        # ----------------------------------------------------
        # Projeção final
        #
        # Depois de concatenar:
        #
        # Head 1 = 64
        # Head 2 = 64
        # Head 3 = 64
        #
        # temos novamente:
        #
        # 64 + 64 + 64 = 192
        # ----------------------------------------------------

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):
        """
        Formato esperado:

            x = [batch, tokens, embed_dim]

        Futuramente no EcoVision:

            [32, 197, 192]

        onde:

            32  = imagens no batch
            197 = CLS token + 196 patches
            192 = dimensão do embedding
        """

        # Lista que armazenará a saída de cada head.
        head_outputs = []

        # Lista para armazenar os attention weights.
        attention_weights_heads = []


        # ----------------------------------------------------
        # Processamos cada head individualmente.
        # ----------------------------------------------------

        for head in range(self.num_heads):


            # =================================================
            # QUERY
            # =================================================

            query = self.query_projections[
                head
            ](
                x
            )


            # =================================================
            # KEY
            # =================================================

            key = self.key_projections[
                head
            ](
                x
            )


            # =================================================
            # VALUE
            # =================================================

            value = self.value_projections[
                head
            ](
                x
            )


            # Neste ponto:
            #
            # query = [batch, tokens, head_dim]
            # key   = [batch, tokens, head_dim]
            # value = [batch, tokens, head_dim]


            # -------------------------------------------------
            # Adicionar dimensão referente ao head.
            #
            # Antes:
            #
            # [batch, tokens, head_dim]
            #
            # Depois:
            #
            # [batch, 1, tokens, head_dim]
            # -------------------------------------------------

            query = query.unsqueeze(1)

            key = key.unsqueeze(1)

            value = value.unsqueeze(1)


            # -------------------------------------------------
            # Scaled Dot-Product Attention
            # -------------------------------------------------

            head_output, attention_weights = self.attention(
                query,
                key,
                value
            )


            # -------------------------------------------------
            # Remover dimensão temporária do head.
            #
            # [batch, 1, tokens, head_dim]
            #
            # →
            #
            # [batch, tokens, head_dim]
            # -------------------------------------------------

            head_output = head_output.squeeze(1)


            # Guardamos saída do head.
            head_outputs.append(
                head_output
            )


            # Guardamos os attention weights.
            attention_weights_heads.append(
                attention_weights
            )


        # ====================================================
        # CONCATENAÇÃO DOS HEADS
        # ====================================================
        #
        # Exemplo:
        #
        # Head 1:
        # [batch, tokens, 64]
        #
        # Head 2:
        # [batch, tokens, 64]
        #
        # Head 3:
        # [batch, tokens, 64]
        #
        # concat:
        #
        # [batch, tokens, 192]
        # ====================================================

        concatenated = torch.cat(
            head_outputs,
            dim=-1
        )


        # ----------------------------------------------------
        # Projeção linear final.
        # ----------------------------------------------------

        output = self.output_projection(
            concatenated
        )


        output = self.dropout(
            output
        )


        # ====================================================
        # ATTENTION WEIGHTS
        # ====================================================
        #
        # Cada head possui:
        #
        # [batch, 1, tokens, tokens]
        #
        # Concatenando:
        #
        # [batch, num_heads, tokens, tokens]
        #
        # Futuramente isso permitirá observar individualmente
        # os attention maps de cada head.
        # ====================================================

        attention_weights = torch.cat(
            attention_weights_heads,
            dim=1
        )


        return output, attention_weights


# ============================================================
# TESTE 1
# Scaled Dot-Product Attention
# ============================================================

def testar_scaled_dot_product_attention():

    print()
    print("=" * 60)
    print("Teste - Scaled Dot-Product Attention")
    print("=" * 60)


    # Dados artificiais para teste.

    batch_size = 2

    num_heads = 3

    num_tokens = 5

    head_dim = 64


    query = torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        head_dim
    )


    key = torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        head_dim
    )


    value = torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        head_dim
    )


    attention = ScaledDotProductAttention()


    output, attention_weights = attention(
        query,
        key,
        value
    )


    print(
        "Query:",
        query.shape
    )

    print(
        "Key:",
        key.shape
    )

    print(
        "Value:",
        value.shape
    )


    print()

    print(
        "Output:",
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
        num_heads,
        num_tokens,
        head_dim
    )


    assert attention_weights.shape == (
        batch_size,
        num_heads,
        num_tokens,
        num_tokens
    )


    # Softmax deve produzir pesos cuja soma seja 1.
    soma_attention = attention_weights.sum(
        dim=-1
    )


    print()

    print(
        "Soma dos pesos de attention:"
    )

    print(
        soma_attention[0, 0]
    )


    assert torch.allclose(
        soma_attention,
        torch.ones_like(soma_attention),
        atol=1e-6
    )


    print()

    print(
        "Scaled Dot-Product Attention: TESTE OK"
    )


# ============================================================
# TESTE 2
# Multi-Head Attention
# ============================================================

def testar_multi_head_attention():

    print()
    print("=" * 60)
    print("Teste - Multi-Head Attention")
    print("=" * 60)


    batch_size = 2

    num_tokens = 5

    embed_dim = 192

    num_heads = 3


    # Entrada artificial:
    #
    # [batch, tokens, embedding]

    x = torch.randn(
        batch_size,
        num_tokens,
        embed_dim
    )


    multi_head_attention = MultiHeadAttention(

        embed_dim=embed_dim,

        num_heads=num_heads
    )


    output, attention_weights = multi_head_attention(
        x
    )


    print(
        "Entrada:",
        x.shape
    )


    print(
        "Output:",
        output.shape
    )


    print(
        "Attention weights:",
        attention_weights.shape
    )


    print()

    print(
        "Dimensão por head:",
        multi_head_attention.head_dim
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


    assert multi_head_attention.head_dim == 64


    # --------------------------------------------------------
    # Verificar softmax do primeiro head.
    # --------------------------------------------------------

    soma_attention = attention_weights[
        0,
        0
    ].sum(
        dim=-1
    )


    print()

    print(
        "Soma dos pesos do Head 1:"
    )


    print(
        soma_attention
    )


    assert torch.allclose(
        soma_attention,
        torch.ones_like(soma_attention),
        atol=1e-6
    )


    print()

    print(
        "Multi-Head Attention: TESTE OK"
    )


# ============================================================
# EXECUÇÃO DOS TESTES
# ============================================================
#
# É importante que este bloco fique NO FINAL DO ARQUIVO.
#
# Assim todas as classes e funções já terão sido definidas
# antes de os testes serem executados.
# ============================================================

if __name__ == "__main__":

    testar_scaled_dot_product_attention()

    testar_multi_head_attention()