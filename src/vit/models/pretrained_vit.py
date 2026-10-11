# ============================================================
# EcoVision AI
# Vision Transformer Pré-treinado
# ============================================================

import torch
import torch.nn as nn
import timm


class EcoVisionPretrainedViT(nn.Module):
    """
    Vision Transformer pré-treinado adaptado
    para o dataset RealWaste.

    Modelo utilizado:

        vit_tiny_patch16_224

    O classification head original é substituído
    por uma nova camada com 9 classes.
    """

    def __init__(
        self,
        num_classes=9,
        pretrained=True
    ):
        super().__init__()

        # ====================================================
        # CARREGAR ViT PRÉ-TREINADO
        # ====================================================

        self.model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained
        )

        # Quantidade de features que chegam ao
        # classification head original.
        in_features = self.model.head.in_features

        print(
            "Features de entrada do classification head:",
            in_features
        )

        # ====================================================
        # NOVO CLASSIFICATION HEAD
        # ====================================================
        #
        # O modelo original possui classes diferentes
        # das 9 classes do RealWaste.
        #
        # Por isso substituímos o head.
        # ====================================================

        self.model.head = nn.Linear(
            in_features,
            num_classes
        )


    def forward(self, x):
        """
        Entrada:

            [batch, 3, 224, 224]

        Saída:

            [batch, 9]
        """

        return self.model(x)


    # ========================================================
    # CONGELAR BACKBONE
    # ========================================================

    def congelar_backbone(self):
        """
        Congela todos os parâmetros do ViT.

        Depois deixa treinável apenas o novo
        classification head.

        Essa etapa corresponde à feature extraction.
        """

        # Congela todo o modelo.
        for parametro in self.model.parameters():
            parametro.requires_grad = False

        # Descongela somente o classification head.
        for parametro in self.model.head.parameters():
            parametro.requires_grad = True


    # ========================================================
    # DESCONGELAR MODELO
    # ========================================================

    def descongelar_modelo(self):
        """
        Descongela todos os parâmetros.

        Essa etapa permite realizar fine-tuning
        de todo o Vision Transformer.
        """

        for parametro in self.model.parameters():
            parametro.requires_grad = True


# ============================================================
# FUNÇÃO AUXILIAR
# ============================================================

def contar_parametros(model):
    """
    Retorna:

        total de parâmetros
        quantidade de parâmetros treináveis
    """

    total = sum(
        parametro.numel()
        for parametro in model.parameters()
    )

    treinaveis = sum(
        parametro.numel()
        for parametro in model.parameters()
        if parametro.requires_grad
    )

    return total, treinaveis


# ============================================================
# TESTE DO MODELO
# ============================================================

def testar_modelo():

    print()
    print("=" * 70)
    print("EcoVision AI - Teste do ViT Pré-treinado")
    print("=" * 70)

    # --------------------------------------------------------
    # Criar modelo
    # --------------------------------------------------------

    model = EcoVisionPretrainedViT(
        num_classes=9,
        pretrained=True
    )

    # --------------------------------------------------------
    # Imagens artificiais para teste
    # --------------------------------------------------------

    imagens = torch.randn(
        2,
        3,
        224,
        224
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits = model(
        imagens
    )

    # --------------------------------------------------------
    # Parâmetros antes do congelamento
    # --------------------------------------------------------

    total, treinaveis = contar_parametros(
        model
    )

    print()
    print(
        "Entrada:",
        imagens.shape
    )

    print(
        "Saída:",
        logits.shape
    )

    print()
    print(
        f"Parâmetros totais: "
        f"{total:,}"
    )

    print(
        f"Parâmetros treináveis inicialmente: "
        f"{treinaveis:,}"
    )

    # ========================================================
    # TESTAR CONGELAMENTO
    # ========================================================

    model.congelar_backbone()

    total_congelado, treinaveis_congelado = contar_parametros(
        model
    )

    print()
    print("=" * 70)
    print("Teste - Backbone Congelado")
    print("=" * 70)

    print(
        f"Parâmetros totais: "
        f"{total_congelado:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{treinaveis_congelado:,}"
    )

    # Classification head:
    #
    # 192 entradas x 9 classes
    # +
    # 9 bias
    #
    # = 1737 parâmetros

    assert treinaveis_congelado == 1737

    # ========================================================
    # TESTAR DESCONGELAMENTO
    # ========================================================

    model.descongelar_modelo()

    total_descongelado, treinaveis_descongelado = contar_parametros(
        model
    )

    print()
    print("=" * 70)
    print("Teste - Modelo Descongelado")
    print("=" * 70)

    print(
        f"Parâmetros treináveis: "
        f"{treinaveis_descongelado:,}"
    )

    assert (
        treinaveis_descongelado
        == total_descongelado
    )

    # ========================================================
    # TESTES DO FORWARD
    # ========================================================

    assert logits.shape == (
        2,
        9
    )

    assert not torch.isnan(
        logits
    ).any()

    print()
    print(
        "ViT pré-treinado: TESTE OK"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    testar_modelo()