# ============================================================
# EcoVision AI
# Atividade 4.1
# Classificador de Raio-X
# ============================================================

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# MODELO
# ============================================================

class XRayClassifier(
    nn.Module
):
    """
    Classificador baseado em ResNet-18 pré-treinada
    no ImageNet.

    Classes:

        COVID
        Normal
        Pneumonia

    A camada final original da ResNet é substituída
    por uma nova camada com 3 saídas.
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True
    ):

        super().__init__()


        # ====================================================
        # PESOS PRÉ-TREINADOS
        # ====================================================

        if pretrained:

            weights = (
                ResNet18_Weights.DEFAULT
            )

        else:

            weights = None


        # ====================================================
        # RESNET-18
        # ====================================================

        self.model = resnet18(
            weights=weights
        )


        # Número de features recebidas pela camada final.
        in_features = (
            self.model.fc.in_features
        )


        print(
            "Features da camada final:",
            in_features
        )


        # ====================================================
        # SUBSTITUIR CLASSIFICATION HEAD
        # ====================================================

        self.model.fc = nn.Linear(

            in_features,

            num_classes
        )


    def forward(
        self,
        x
    ):
        """
        Forward:

            imagens
               ↓
            ResNet-18
               ↓
            logits de 3 classes
        """

        return self.model(
            x
        )


# ============================================================
# TESTE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Teste XRayClassifier")
    print("=" * 70)


    model = XRayClassifier(
        num_classes=3,
        pretrained=True
    )


    # Batch fictício.
    x = torch.randn(
        2,
        3,
        224,
        224
    )


    with torch.no_grad():

        output = model(
            x
        )


    print()
    print(
        "Entrada:",
        x.shape
    )


    print(
        "Saída:",
        output.shape
    )


    assert output.shape == (
        2,
        3
    )


    total_params = sum(

        p.numel()

        for p in model.parameters()
    )


    print()
    print(
        f"Parâmetros totais: "
        f"{total_params:,}"
    )


    print()
    print(
        "XRayClassifier: TESTE OK"
    )


if __name__ == "__main__":

    main()