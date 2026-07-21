# Text Classification Pipeline Forensics

This repository is a reproducible, CPU-only implementation of Topic A using Python 3.12.9, pandas, SciPy, NumPy, scikit-learn, and pytest. The random seed is `3102`.

## Setup and data

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ucbrise/kaggle-nlp-disasters/master/data/train.csv" -OutFile "data/train.csv"
python -m pipeline.cli validate-data
```

On Linux/macOS, create and activate the environment as follows:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
curl -L https://raw.githubusercontent.com/ucbrise/kaggle-nlp-disasters/master/data/train.csv -o data/train.csv
python -m pipeline.cli validate-data
```

Only the full labeled Kaggle file at `data/train.csv` is used. It is intentionally ignored by Git; see [data/README_DATA.md](data/README_DATA.md). The fixed IDs in `data/split_indices.json` are never regenerated.

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

On Linux/macOS, first run `source .venv/bin/activate`, then run the same `python` commands.

Replace `1` with `2` through `5` for an individual ticket interface. `run-ticket --split dev` fits and scores only train/dev data, writes a pending record to `experiments/pending_decisions.json`, and does not create or update held-out predictions. `freeze-ticket` moves that record into `experiments/decisions.json`. Only then can `run-ticket --split heldout` score the frozen configuration and write `predictions/heldout/ticket-N.csv`. `run-all` is the explicit final regeneration command: it performs every dev comparison, freezes each selected decision, and then writes the aggregate final artifacts.

## Evaluation policy

All vectorizers, encoders, scalers, and classifiers fit only the 4,567-row training split. The 1,523-row dev split selects normalization, feature sets, models, hyperparameters, and thresholds. The 1,523-row held-out split is evaluated only after a decision is frozen. Scores are target-1 probabilities for Logistic Regression and decision-function values for LinearSVC; `model_name` and the Ticket 4 decision record disambiguate them.

The main files are `results/summary.csv`, `predictions/heldout_predictions.csv`, `results/threshold_sweep.csv`, `results/decision_ablation.csv`, `results/ticket4_dev_decision_curve.csv`, `results/ticket4_dev_precision_recall_curve.csv`, `results/figures/ticket4_dev_precision_recall.png`, `results/figures/ticket4_dev_f1_threshold.png`, `results/error_transitions.csv`, `results/perturbation_stress.csv`, `results/top_features.csv`, `results/data_quality_audit.csv`, `results/environment.json`, and `experiments/decisions.json`. The two Ticket 4 plots are dev-only explanations of the already frozen operating point; they are not held-out selection artifacts. See [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md) for the technical data flow and exact findings.

The formal submission is [report.pdf](report.pdf); its source is [REPORT.md](REPORT.md). [logs/chat.md](logs/chat.md) is a concise record of the actual AI-assisted audit and verification work. `PROJECT_WALKTHROUGH.md` remains a technical implementation walkthrough rather than the formal report.

