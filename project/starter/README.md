# Text Classification Pipeline Forensics

This repository is a reproducible, CPU-only implementation of Topic A using Python 3.10+ (verified on Python 3.13.0), pandas, SciPy, NumPy, scikit-learn, and pytest. The random seed is `3102`.

## Setup and data

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ucbrise/kaggle-nlp-disasters/master/data/train.csv" -OutFile "data/train.csv"
python -m pipeline.cli validate-data
```

On Linux/macOS, activate with `source .venv/bin/activate` and download with `curl -L .../train.csv -o data/train.csv`. Only the full labeled Kaggle file at `data/train.csv` is used. It is intentionally ignored by Git; see [data/README_DATA.md](data/README_DATA.md). The fixed IDs in `data/split_indices.json` are never regenerated.

## Reproduction

These commands were executed successfully from the repository root:

```powershell
python -m pytest -q
python -m pipeline.cli validate-data
python -m pipeline.cli run-all
python -m pipeline.cli validate-artifacts
python -m pipeline.cli run-ticket --ticket 1 --split dev
python -m pipeline.cli freeze-ticket --ticket 1
python -m pipeline.cli run-ticket --ticket 1 --split heldout
```

Replace `1` with `2` through `5` for an individual ticket interface. Ticket commands rebuild the deterministic shared registry so cross-ticket baselines remain consistent. `run-ticket --split heldout` refuses to proceed unless `experiments/decisions.json` contains the ticket's frozen decision. For a clean reproduction, `run-all` performs every dev comparison, writes each frozen decision, and only then evaluates held-out data.

## Evaluation policy

All vectorizers, encoders, scalers, and classifiers fit only the 4,567-row training split. The 1,523-row dev split selects normalization, feature sets, models, hyperparameters, and thresholds. The 1,523-row held-out split is evaluated only after a decision is frozen. Scores are target-1 probabilities for Logistic Regression and decision-function values for LinearSVC; `model_name` and the Ticket 4 decision record disambiguate them.

The main files are `results/summary.csv`, `predictions/heldout_predictions.csv`, `results/threshold_sweep.csv`, `results/error_transitions.csv`, `results/perturbation_stress.csv`, `results/top_features.csv`, `results/data_quality_audit.csv`, `results/environment.json`, and `experiments/decisions.json`. See [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md) for the technical data flow and exact findings.

Per the completion instructions, `report.pdf` was not generated and `logs/chat.md` was not generated or modified. `PROJECT_WALKTHROUGH.md` is a technical implementation walkthrough, not a substitute formal report or chat log.

