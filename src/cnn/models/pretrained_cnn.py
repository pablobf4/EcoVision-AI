# ============================================================
# EcoVision AI
# Atividade 3 - CNN Pré-treinada
# EfficientNet-B0
# ============================================================

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class EcoVisionPretrainedCNN(nn.Module):
    """
    EfficientNet-B0 pré-treinada no ImageNet,
    adaptada para as 7 classes do nosso dataset.

    Classes:
        bike
        cars
        cats
        dogs
        flowers
        horses
        human
    """

    def __init__(
        self,
        num_classes=7,
        pretrained=True
    ):
        super().__init__()


        # ====================================================
        # CARREGAR PESOS PRÉ-TREINADOS
        # ====================================================

        if pretrained:

            weights = (
                EfficientNet_B0_Weights.DEFAULT
            )

        else:

            weights = None


        self.model = efficientnet_b0(
            weights=weights
        )


        # ====================================================
        # CLASSIFICATION HEAD ORIGINAL
        # ====================================================
        #
        # EfficientNet-B0:
        #
        # classifier =
        # Sequential(
        #     Dropout(...)
        #     Linear(1280, 1000)
        # )
        #
        # Vamos trocar as 1000 classes do ImageNet
        # pelas nossas 7 classes.
        # ====================================================

        in_features = (
            self.model
            .classifier[1]
            .in_features
        )


        print(
            "Features de entrada do classification head:",
            in_features
        )


        # ====================================================
        # NOVO CLASSIFICATION HEAD
        # ====================================================

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):
        """
        Entrada:

            [batch, 3, 224, 224]

        Saída:

            [batch, 7]
        """

        return self.model(
            x
        )


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def congelar_backbone(self):
        """
        Congela todos os parâmetros da CNN.

        Depois deixa treinável somente
        o classification head.
        """

        # Congelar tudo.
        for parametro in self.model.parameters():

            parametro.requires_grad = False


        # Liberar classifier.
        for parametro in self.model.classifier.parameters():

            parametro.requires_grad = True


    # ========================================================
    # FINE-TUNING
    # ========================================================

    def descongelar_modelo(self):
        """
        Libera todos os parâmetros da EfficientNet
        para fine-tuning completo.
        """

        for parametro in self.model.parameters():

            parametro.requires_grad = True


# ============================================================
# CONTAR PARÂMETROS
# ============================================================

def contar_parametros(
    model
):

    total = sum(

        parametro.numel()

        for parametro in model.parameters()
    )


    treinaveis = sum(

        parametro.numel()

        for parametro in model.parameters()

        if parametro.requires_grad
    )


    return (
        total,
        treinaveis
    )


# ============================================================
# TESTE
# ============================================================

def testar_modelo():

    print()
    print("=" * 70)
    print("EcoVision AI - EfficientNet-B0 Pré-treinada")
    print("=" * 70)


    # ========================================================
    # MODELO
    # ========================================================

    model = EcoVisionPretrainedCNN(

        num_classes=7,

        pretrained=True
    )


    # ========================================================
    # IMAGENS FICTÍCIAS
    # ========================================================

    imagens = torch.randn(

        2,
        3,
        224,
        224
    )


    # ========================================================
    # FORWARD
    # ========================================================

    logits = model(
        imagens
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


    # ========================================================
    # PARÂMETROS INICIAIS
    # ========================================================

    total, treinaveis = contar_parametros(
        model
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
    # BACKBONE CONGELADO
    # ========================================================

    model.congelar_backbone()


    total, treinaveis = contar_parametros(
        model
    )


    print()
    print("=" * 70)
    print("Teste - Backbone Congelado")
    print("=" * 70)


    print(
        f"Parâmetros totais: "
        f"{total:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{treinaveis:,}"
    )


    # O novo Linear possui:
    #
    # 1280 * 7 + 7
    #
    # = 8.967 parâmetros

    assert treinaveis == 8967


    # ========================================================
    # DESCONGELAR
    # ========================================================

    model.descongelar_modelo()


    total, treinaveis = contar_parametros(
        model
    )


    print()
    print("=" * 70)
    print("Teste - Modelo Descongelado")
    print("=" * 70)


    print(
        f"Parâmetros treináveis: "
        f"{treinaveis:,}"
    )


    assert treinaveis == total


    # ========================================================
    # TESTES DO FORWARD
    # ========================================================

    assert logits.shape == (
        2,
        7
    )


    assert not torch.isnan(
        logits
    ).any()


    print()

    print(
        "EfficientNet-B0: TESTE OK"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    testar_modelo()