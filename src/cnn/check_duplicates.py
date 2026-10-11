# ============================================================
# EcoVision AI
# Verificação de duplicatas no dataset da CNN
# ============================================================

from collections import defaultdict
from pathlib import Path
import hashlib

from .data_pipeline import (
    localizar_dataset,
    criar_datasets
)


# ============================================================
# HASH DO ARQUIVO
# ============================================================

def calcular_sha256(caminho):
    """
    Calcula o SHA-256 do conteúdo do arquivo.

    Se dois arquivos tiverem exatamente o mesmo conteúdo,
    terão o mesmo hash.
    """

    sha256 = hashlib.sha256()

    with open(caminho, "rb") as arquivo:

        for bloco in iter(
            lambda: arquivo.read(8192),
            b""
        ):
            sha256.update(bloco)

    return sha256.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EcoVision AI - Verificação de Duplicatas")
    print("=" * 70)

    dataset_path = localizar_dataset()

    (
        dataset_base,
        train_dataset,
        val_dataset,
        test_dataset
    ) = criar_datasets(
        dataset_path
    )

    # ImageFolder mantém a lista original:
    #
    # [(caminho_imagem, label), ...]
    samples = dataset_base.samples

    # Os Subsets possuem os índices correspondentes
    # ao dataset original.
    splits = {
        "train": train_dataset.indices,
        "validation": val_dataset.indices,
        "test": test_dataset.indices
    }

    # hash -> lista de ocorrências
    hashes = defaultdict(list)

    for nome_split, indices in splits.items():

        for indice in indices:

            caminho, label = samples[indice]

            hash_imagem = calcular_sha256(
                caminho
            )

            hashes[hash_imagem].append({
                "split": nome_split,
                "path": caminho,
                "label": label
            })

    # ========================================================
    # LOCALIZAR DUPLICATAS
    # ========================================================

    duplicatas = {
        hash_imagem: ocorrencias
        for hash_imagem, ocorrencias in hashes.items()
        if len(ocorrencias) > 1
    }

    print()
    print(
        f"Total de imagens analisadas: "
        f"{len(dataset_base)}"
    )

    print(
        f"Hashes únicos: "
        f"{len(hashes)}"
    )

    print(
        f"Grupos de arquivos duplicados: "
        f"{len(duplicatas)}"
    )

    # ========================================================
    # VERIFICAR DATA LEAKAGE
    # ========================================================

    vazamentos = []

    for hash_imagem, ocorrencias in duplicatas.items():

        splits_encontrados = {
            item["split"]
            for item in ocorrencias
        }

        # Se a mesma imagem aparecer em mais de um split,
        # existe vazamento de dados.
        if len(splits_encontrados) > 1:

            vazamentos.append(
                ocorrencias
            )

    print()
    print("=" * 70)
    print("VERIFICAÇÃO DE DATA LEAKAGE")
    print("=" * 70)

    print(
        f"Duplicatas entre splits: "
        f"{len(vazamentos)}"
    )

    if len(vazamentos) == 0:

        print()
        print(
            "OK: nenhuma duplicata exata foi encontrada "
            "entre treino, validação e teste."
        )

    else:

        print()
        print(
            "ATENÇÃO: existem imagens duplicadas "
            "entre os conjuntos."
        )

        for grupo in vazamentos:

            print()
            print("-" * 70)

            for item in grupo:

                print(
                    f"{item['split']:12s} | "
                    f"{Path(item['path']).name}"
                )


if __name__ == "__main__":
    main()