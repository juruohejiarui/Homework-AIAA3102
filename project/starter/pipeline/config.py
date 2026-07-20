from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PREDICTIONS = ROOT / "predictions"
EXPERIMENTS = ROOT / "experiments"
TICKETS = ROOT / "tickets"
SEED = 3102

EXPECTED_COUNTS = {"train": 4567, "dev": 1523, "heldout": 1523}
EXPECTED_POSITIVES = {"train": 1962, "dev": 655, "heldout": 654}

