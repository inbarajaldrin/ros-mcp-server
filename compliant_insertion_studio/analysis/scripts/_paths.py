# Reference: shared path config so scripts are runnable from anywhere and overridable per iteration.
# Ralph loop can override DATA_DIR via INSERT_ANALYSIS_DATA env var to keep iterations isolated.
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "compliant_insertion_studio" / "logs"
ANALYSIS_DIR = REPO_ROOT / "compliant_insertion_studio" / "analysis"
DATA_DIR = Path(os.environ.get("INSERT_ANALYSIS_DATA", ANALYSIS_DIR / "data"))
PER_SAMPLE_DIR = DATA_DIR / "per_sample"
ITERATIONS_DIR = ANALYSIS_DIR / "iterations"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
