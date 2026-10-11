# EcoVision AI

Projeto acadêmico de Visão Computacional organizado por atividade/domínio.

## Estrutura

```text
EcoVision-AI/
├── src/
│   ├── common/              # caminhos/utilidades compartilhadas
│   ├── vit/                 # Attention, Transformer e Vision Transformer
│   │   └── models/
│   ├── cnn/                 # CNN + transfer learning
│   │   └── models/
│   ├── clip/                # CLIP, ranking e busca semântica
│   └── xray/                # estudo de caso médico + cGAN
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── generation/
├── notebooks/               # notebooks finais A1-A4
├── data/                    # datasets locais (não versionados)
├── models/                  # checkpoints (não versionados)
├── results/                 # métricas, gráficos e evidências
└── reports/
    ├── academic/
    └── study/
```

## Organização das atividades

### A1 — Vision Transformers
- `src/vit/models/attention.py`
- `src/vit/models/transformer.py`
- `src/vit/models/vit.py`
- `src/vit/models/vit_model.py`
- `src/vit/train_scratch.py`
- `src/vit/evaluate_scratch.py`
- `src/vit/attention_maps.py`
- `src/vit/train_pretrained.py`
- `src/vit/evaluate_pretrained.py`

### A2 — CLIP / ADS-16
- `src/clip/download_dataset.py`
- `src/clip/prepare_corpus.py`
- `src/clip/semantic_ranking.py`
- `src/clip/threshold_analysis.py`
- `src/clip/semantic_search.py`

### A3 — CNN / Transfer Learning
- `src/cnn/download_dataset.py`
- `src/cnn/clean_dataset.py`
- `src/cnn/deduplicate_dataset.py`
- `src/cnn/check_duplicates.py`
- `src/cnn/data_pipeline.py`
- `src/cnn/models/pretrained_cnn.py`
- `src/cnn/train.py`
- `src/cnn/evaluate.py`

### A4 — Raio-X / cGAN
- `src/xray/prepare_dataset.py`
- `src/xray/data_pipeline.py`
- `src/xray/models/classifier.py`
- `src/xray/models/cgan.py`
- `src/xray/training/baseline.py`
- `src/xray/training/cgan_base.py`
- `src/xray/training/cgan_mitigated.py`
- `src/xray/training/cgan_diversity.py`
- `src/xray/generation/generate_synthetic_covid.py`
- `src/xray/training/synthetic.py`
- `src/xray/evaluation/baseline.py`
- `src/xray/evaluation/synthetic.py`

## Execução

Execute os módulos a partir da raiz do projeto:

```powershell
python -m src.vit.train_scratch
python -m src.vit.evaluate_scratch

python -m src.cnn.train
python -m src.cnn.evaluate

python -m src.clip.semantic_ranking
python -m src.clip.semantic_search

python -m src.xray.training.baseline
python -m src.xray.training.cgan_diversity
python -m src.xray.generation.generate_synthetic_covid
python -m src.xray.training.synthetic
python -m src.xray.evaluation.synthetic
```

## Observação

A reorganização é estrutural. O objetivo é preservar os algoritmos e resultados dos experimentos já realizados, alterando apenas a localização dos módulos, imports e resolução da raiz do projeto.
