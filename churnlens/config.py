"""Project-wide constants.

Kept in one place so every split, model and sample uses the same seed and paths,
which makes every number in `reports/` reproducible.
"""

from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "datasets"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
