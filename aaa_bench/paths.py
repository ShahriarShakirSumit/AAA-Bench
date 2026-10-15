"""Default locations inside the repository. Every script also accepts explicit paths."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("AAA_BENCH_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
HUMAN_STUDY = ROOT / "human_study"
MODEL_OUTPUTS = ROOT / "model_outputs"
REFERENCE_OUTPUTS = MODEL_OUTPUTS / "reference"
CONSTRUCTION = ROOT / "construction"
RESULTS = ROOT / "results"
