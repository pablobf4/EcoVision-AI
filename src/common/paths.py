"""Caminhos compartilhados do projeto EcoVision AI."""

from pathlib import Path

# .../EcoVision-AI/src/common/paths.py -> raiz EcoVision-AI
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
