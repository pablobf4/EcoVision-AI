# ============================================================
# EcoVision AI
# Visualização de Attention Maps
# ============================================================

from pathlib import Path
from src.common.paths import PROJECT_ROOT

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from .data_pipeline import (
    localizar_dataset,
    criar_datasets,
    criar_dataloaders
)

from .models.vit_model import EcoVisionViT


# ============================================================
# CAMINHOS
# ============================================================

PROJECT_DIR = PROJECT_ROOT

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ecovision_vit_scratch_best.pth"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "attention_maps"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NORMALIZAÇÃO UTILIZADA NO DATASET
# ============================================================

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(
    3,
    1,
    1
)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(
    3,
    1,
    1
)


# ============================================================
# CARREGAR MODELO
# ============================================================

def carregar_modelo():
    """
    Carrega o melhor checkpoint do ViT treinado do zero.
    """

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    config = checkpoint[
        "config"
    ]

    model = EcoVisionViT(

        image_size=config["image_size"],

        patch_size=config["patch_size"],

        num_classes=config["num_classes"],

        embed_dim=config["embed_dim"],

        num_heads=config["num_heads"],

        num_layers=config["num_layers"],

        mlp_dim=config["mlp_dim"],

        dropout=config["dropout"]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"Checkpoint carregado: "
        f"epoch {checkpoint['epoch']}"
    )

    print(
        f"Validation Accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    return model


# ============================================================
# DESNORMALIZAR IMAGEM
# ============================================================

def desnormalizar_imagem(
    imagem
):
    """
    Reverte a normalização ImageNet.

    Durante o treinamento utilizamos:

        imagem_normalizada =
        (imagem - média) / desvio

    Aqui fazemos o inverso:

        imagem =
        imagem_normalizada * desvio + média
    """

    imagem = imagem.cpu()

    imagem = (
        imagem
        * IMAGENET_STD
        + IMAGENET_MEAN
    )

    # Garantir intervalo válido para imagem.
    imagem = torch.clamp(
        imagem,
        0,
        1
    )

    # PyTorch:
    #
    # [C, H, W]
    #
    # matplotlib:
    #
    # [H, W, C]

    imagem = imagem.permute(
        1,
        2,
        0
    )

    return imagem.numpy()


# ============================================================
# ENCONTRAR EXEMPLO CORRETAMENTE CLASSIFICADO
# ============================================================

def encontrar_exemplo_correto(
    model,
    test_dataset,
    classe_desejada=None
):
    """
    Procura uma imagem corretamente classificada.

    Se classe_desejada for informada, procura
    especificamente por essa classe.

    Exemplo:

        classe_desejada = 6

    para procurar Plastic.
    """

    model.eval()

    with torch.no_grad():

        for indice in range(
            len(test_dataset)
        ):

            imagem, label = (
                test_dataset[
                    indice
                ]
            )

            # Caso tenhamos escolhido uma classe,
            # ignoramos as demais.
            if (
                classe_desejada is not None
                and label != classe_desejada
            ):

                continue


            entrada = imagem.unsqueeze(
                0
            ).to(
                device
            )


            logits = model(
                entrada
            )


            predicao = torch.argmax(
                logits,
                dim=1
            ).item()


            if predicao == label:

                return (
                    imagem,
                    label,
                    indice
                )


    raise RuntimeError(
        "Nenhum exemplo corretamente classificado foi encontrado."
    )


# ============================================================
# EXTRAIR ATTENTION MAP
# ============================================================

def extrair_attention_map(
    model,
    imagem,
    head_index=0
):
    """
    Extrai o attention map do CLS Token.

    Utilizamos o último Transformer Encoder.

    Attention shape:

        [batch, heads, 197, 197]

    Queremos:

        CLS → patches

    Portanto:

        attention[0, head, 0, 1:]

    Isso produz:

        196 valores

    que reorganizamos como:

        14 x 14
    """

    entrada = imagem.unsqueeze(
        0
    ).to(
        device
    )


    with torch.no_grad():

        logits, all_attention = model(

            entrada,

            return_attention=True
        )


    # --------------------------------------------------------
    # Último Transformer Encoder
    # --------------------------------------------------------

    attention_ultimo_encoder = (
        all_attention[-1]
    )


    print(
        "Attention shape:",
        attention_ultimo_encoder.shape
    )


    # --------------------------------------------------------
    # Attention do CLS Token
    #
    # posição 0:
    #
    # CLS Token
    #
    # posições 1:197:
    #
    # patches
    # --------------------------------------------------------

    cls_attention = (
        attention_ultimo_encoder[
            0,
            head_index,
            0,
            1:
        ]
    )


    # --------------------------------------------------------
    # Temos 196 patches.
    #
    # sqrt(196) = 14
    # --------------------------------------------------------

    num_patches = (
        cls_attention.shape[0]
    )

    grid_size = int(
        np.sqrt(
            num_patches
        )
    )


    attention_map = (
        cls_attention.reshape(
            1,
            1,
            grid_size,
            grid_size
        )
    )


    # --------------------------------------------------------
    # Redimensionar:
    #
    # 14 x 14
    #
    # →
    #
    # 224 x 224
    # --------------------------------------------------------

    attention_map = F.interpolate(

        attention_map,

        size=(
            224,
            224
        ),

        mode="bilinear",

        align_corners=False
    )


    attention_map = (
        attention_map
        .squeeze()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Normalizar para intervalo:
    #
    # 0 → 1
    # --------------------------------------------------------

    attention_map = (
        attention_map
        - attention_map.min()
    )

    attention_map = (
        attention_map
        / (
            attention_map.max()
            + 1e-8
        )
    )


    # --------------------------------------------------------
    # Predição
    # --------------------------------------------------------

    probabilidades = torch.softmax(
        logits,
        dim=1
    )


    predicao = torch.argmax(
        probabilidades,
        dim=1
    ).item()


    confianca = (
        probabilidades[
            0,
            predicao
        ].item()
    )


    return (
        attention_map,
        predicao,
        confianca
    )


# ============================================================
# VISUALIZAR ATTENTION MAP
# ============================================================

def visualizar_attention(
    imagem,
    attention_map,
    classe_real,
    classe_prevista,
    confianca,
    classes,
    head_index
):
    """
    Cria três visualizações:

    1. imagem original
    2. attention map
    3. attention sobreposta na imagem
    """

    imagem_original = desnormalizar_imagem(
        imagem
    )


    # --------------------------------------------------------
    # IMAGEM ORIGINAL
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        imagem_original
    )

    plt.title(
        f"Imagem Original\n"
        f"Real: {classes[classe_real]} | "
        f"Predição: {classes[classe_prevista]} "
        f"({confianca * 100:.2f}%)"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()


    caminho_original = (
        RESULTS_DIR
        / "attention_original.png"
    )


    plt.savefig(
        caminho_original,
        dpi=150
    )

    plt.close()


    # --------------------------------------------------------
    # ATTENTION MAP
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        attention_map
    )

    plt.title(
        f"Attention Map - Head {head_index}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()


    caminho_attention = (
        RESULTS_DIR
        / f"attention_head_{head_index}.png"
    )


    plt.savefig(
        caminho_attention,
        dpi=150
    )

    plt.close()


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )


    # Primeiro desenhamos a imagem.
    plt.imshow(
        imagem_original
    )


    # Depois desenhamos o attention map
    # parcialmente transparente.
    plt.imshow(
        attention_map,
        alpha=0.45
    )


    plt.title(

        f"EcoVision AI - Attention Head {head_index}\n"

        f"Real: {classes[classe_real]} | "

        f"Predição: {classes[classe_prevista]} | "

        f"Confiança: {confianca * 100:.2f}%"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    caminho_overlay = (
        RESULTS_DIR
        / f"attention_overlay_head_{head_index}.png"
    )


    plt.savefig(
        caminho_overlay,
        dpi=150
    )


    # Também mostramos na tela.
    plt.show()


    print()
    print(
        "Attention Map salvo em:"
    )

    print(
        caminho_overlay
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "EcoVision AI - Attention Maps"
    )

    print("=" * 70)


    # ========================================================
    # DATASET
    # ========================================================

    dataset_path = localizar_dataset()


    (
        dataset_base,
        train_dataset,
        val_dataset,
        test_dataset
    ) = criar_datasets(
        dataset_path
    )


    classes = dataset_base.classes


    print()
    print(
        "Classes:"
    )


    for indice, classe in enumerate(
        classes
    ):

        print(
            f"{indice}: {classe}"
        )


    # ========================================================
    # MODELO
    # ========================================================

    model = carregar_modelo()


    # ========================================================
    # ESCOLHER CLASSE
    # ========================================================
    #
    # Nosso dataset:
    #
    # 0 Cardboard
    # 1 Food Organics
    # 2 Glass
    # 3 Metal
    # 4 Miscellaneous Trash
    # 5 Paper
    # 6 Plastic
    # 7 Textile Trash
    # 8 Vegetation
    #
    # Vamos começar com Plastic porque foi uma
    # das classes com maior recall.
    # ========================================================

    CLASSE_DESEJADA = 6


    imagem, label, indice = encontrar_exemplo_correto(

        model,

        test_dataset,

        classe_desejada=CLASSE_DESEJADA
    )


    print()
    print(
        "Imagem selecionada:"
    )

    print(
        f"Índice no conjunto de teste: "
        f"{indice}"
    )

    print(
        f"Classe real: "
        f"{classes[label]}"
    )


    # ========================================================
    # ESCOLHER HEAD
    # ========================================================

    HEAD_INDEX = 2


    (
        attention_map,
        predicao,
        confianca
    ) = extrair_attention_map(

        model,

        imagem,

        head_index=HEAD_INDEX
    )


    print()
    print(
        f"Predição: "
        f"{classes[predicao]}"
    )

    print(
        f"Confiança: "
        f"{confianca * 100:.2f}%"
    )


    # ========================================================
    # VISUALIZAÇÃO
    # ========================================================

    visualizar_attention(

        imagem=imagem,

        attention_map=attention_map,

        classe_real=label,

        classe_prevista=predicao,

        confianca=confianca,

        classes=classes,

        head_index=HEAD_INDEX
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()