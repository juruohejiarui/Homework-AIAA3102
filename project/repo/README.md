# Topic A: Text Classification Pipeline Forensics — V3

This is the **V3 merged implementation** of the AIAA 3102 Topic A project. It combines the best elements of two parallel implementations:

- **V1 (starter)**: Extended C value search (C=5, C=10), richer Concrete Examples in ticket markdown files
- **V2 (topic-a-text-classification-pipeline-forensics)**: Modular pipeline architecture, complete SHA-256 provenance tracking, systematic threshold sweep, correction probe in Ticket 5, comprehensive pytest test suite

---

## Project Structure

```
v3/
├── pipeline/               # All Python pipeline modules
│   ├── __init__.py
│   ├── artifacts.py        # CSV/JSON artifact writers and validators
│   ├── baselines.py        # Floor model and reference baseline
│   ├── data.py             # Data loading and split selection
│   ├── data_quality.py     # Duplicate detection and audit logic
│   ├── decision_rule.py    # MODEL_SPECS (11 variants), threshold sweep
│   ├── metrics.py          # metric_bundle (P/R/F1/accuracy/confusion)
│   ├── modeling.py         # Shared model fitting utilities
│   ├── normalization.py    # 7 normalization variants
│   ├── reproducibility.py  # Seed and n_jobs configuration
│   ├── shortcut_features.py # 10 feature variants + perturbation tests
│   ├── splits.py           # Fixed split loading
│   ├── ticket2.py          # Ticket 2 shared utilities
│   ├── versions.py         # Package version capture
│   ├── run_baselines.py    # Step 4: floor + reference baseline
│   ├── run_ticket1_probes.py
│   ├── run_ticket1_heldout.py
│   ├── run_ticket2_dev.py
│   ├── freeze_ticket2.py
│   ├── run_ticket2_heldout.py
│   ├── run_ticket3_dev.py
│   ├── freeze_ticket3.py
│   ├── run_ticket4_dev.py
│   ├── freeze_ticket4.py
│   ├── run_ticket4_heldout.py
│   ├── run_ticket5_dev.py
│   ├── run_ticket5_corrections.py
│   ├── finalize_ticket5_audit.py
│   ├── run_ticket5_heldout.py
│   ├── reproduce_frozen_ticket.py
│   ├── verify_final_reproducibility.py
│   ├── build_frozen_decisions_manifest.py
│   └── validate_submission.py
├── tickets/                # Ticket markdown documentation
│   ├── ticket-1-baseline.md
│   ├── ticket-2-normalization.md
│   ├── ticket-3-shortcuts.md
│   ├── ticket-4-decision-rule.md
│   └── ticket-5-data-quality.md
├── experiments/            # Experiment plans and outputs (generated)
│   ├── step-4-baselines/
│   ├── ticket-1/
│   ├── ticket-2/
│   ├── ticket-3/
│   ├── ticket-4/
│   ├── ticket-5/
│   └── final-reproducibility-audit/
├── results/                # Summary CSVs (generated)
│   ├── summary.csv
│   ├── threshold_sweep.csv
│   ├── data_quality_audit.csv
│   └── figures/
├── predictions/            # Held-out prediction CSVs (generated)
│   ├── ticket-1-heldout-predictions.csv
│   ├── ticket-2-heldout-predictions.csv
│   ├── ticket-3-heldout-predictions.csv (reuses ticket-1)
│   ├── ticket-4-heldout-predictions.csv
│   └── heldout_predictions.csv (final, from ticket-5)
├── configs/
│   └── reproducibility.json
├── starter/
│   └── data/
│       └── split_indices.json
├── data/
│   └── train.csv           # ← YOU MUST PLACE THIS FILE HERE
├── tests/                  # pytest test suite
├── report/
│   ├── main.tex
│   └── references.bib
├── report_assets/
│   ├── tables/             # Generated LaTeX table files
│   └── figures/            # Generated figure PNG files
├── logs/
│   └── chat.md             # AI usage log
├── topic-a-handout.md
├── teacher_clarifications.md
├── report_outline.md
└── requirements.txt
```

---

## Environment Setup

This project requires Python 3.10 or later. All dependencies are pinned in `requirements.txt`.

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

Key packages: `scikit-learn`, `pandas`, `numpy`, `scipy`. See `requirements-lock.txt` for the exact pinned versions used during development.

---

## Data Acquisition

The dataset is the [Kaggle "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started/data) competition dataset.

1. Download `train.csv` from the competition page (requires a free Kaggle account).
2. Place the file at `data/train.csv` relative to the project root:

```
v3_merged/
└── data/
    └── train.csv   ← place here (7,613 rows)
```

The split indices (`starter/data/split_indices.json`) are already included in the repository. **Do not regenerate them** — the split is fixed and all frozen decisions depend on it.

---

## Expected Directory Structure

After cloning the repository and placing `data/train.csv`, the project root should look like:

```
v3_merged/
├── data/
│   └── train.csv                  ← you provide this
├── pipeline/                      ← all Python modules
├── experiments/                   ← generated outputs (empty before first run)
│   ├── step-4-baselines/
│   ├── ticket-1/
│   ├── ticket-2/
│   ├── ticket-3/
│   ├── ticket-4/
│   ├── ticket-5/
│   └── final-reproducibility-audit/
├── results/                       ← generated summary CSVs
├── predictions/                   ← generated held-out prediction files
├── configs/
│   ├── reproducibility.json       ← seed and determinism settings
│   └── frozen_decisions.json      ← generated by Step 7
├── starter/
│   └── data/
│       └── split_indices.json     ← fixed split (do not modify)
├── tickets/                       ← ticket markdown documentation
├── tests/                         ← pytest test suite
├── report/
│   ├── main.tex
│   └── references.bib
├── logs/
│   └── chat.md
├── run_all.sh                     ← one-click full pipeline
└── requirements.txt
```

---

## Reproduce the Baseline

The reference baseline is a raw-text TF-IDF + Logistic Regression pipeline (no preprocessing). To reproduce it in isolation:

```bash
python -m pipeline.run_baselines \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/step-4-baselines
```

Expected output files:
- `experiments/step-4-baselines/predictions/raw_text_tfidf_logistic_regression_dev_predictions.csv`
- `experiments/step-4-baselines/predictions/raw_text_tfidf_logistic_regression_heldout_predictions.csv`
- `experiments/step-4-baselines/dev_metrics.json`

The frozen baseline dev F1 is approximately **0.7574** (class 1). All subsequent tickets are evaluated relative to this value.

---

## Reproduce Each Ticket

Each ticket follows the same three-phase pattern: **dev evaluation → freeze decision → held-out report**. Run the phases strictly in order.

### Ticket 1 — Baseline Reproducibility Probes

```bash
# Phase 1: Dev probes (32 one-lever diagnostics)
python -m pipeline.run_ticket1_probes \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-1/probes/probe_plan.json \
    --output-dir experiments/ticket-1/probes

# Phase 2: Held-out evaluation (baseline + all 32 probes)
python -m pipeline.run_ticket1_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/ticket-1/heldout \
    --overwrite
```

### Ticket 2 — Text Normalisation

```bash
python -m pipeline.run_ticket2_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-2/dev/experiment_plan.json \
    --output-dir experiments/ticket-2/dev

python -m pipeline.freeze_ticket2 \
    --dev-dir experiments/ticket-2/dev \
    --output experiments/ticket-2/frozen_decision.json

python -m pipeline.run_ticket2_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-2/frozen_decision.json \
    --output-dir experiments/ticket-2/heldout \
    --confirm-single-heldout-evaluation
```

### Ticket 3 — Shortcut Feature Audit

```bash
python -m pipeline.run_ticket3_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-3/dev/experiment_plan.json \
    --output-dir experiments/ticket-3/dev

python -m pipeline.freeze_ticket3 \
    --dev-dir experiments/ticket-3/dev \
    --output experiments/ticket-3/frozen_decision.json

python -m pipeline.run_ticket3_heldout \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-3/frozen_decision.json \
    --output-dir experiments/ticket-3/heldout \
    --confirm-single-ticket3-report
```

### Ticket 4 — Decision Rule Optimisation

```bash
python -m pipeline.run_ticket4_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-4/dev/experiment_plan.json \
    --output-dir experiments/ticket-4/dev

python -m pipeline.freeze_ticket4 \
    --dev-dir experiments/ticket-4/dev \
    --output experiments/ticket-4/frozen_decision.json

python -m pipeline.run_ticket4_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-4/frozen_decision.json \
    --output-dir experiments/ticket-4/heldout \
    --confirm-single-ticket4-evaluation
```

### Ticket 5 — Data Quality Audit

```bash
python -m pipeline.run_ticket5_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-5/dev/audit_plan.json \
    --output-dir experiments/ticket-5/dev

python -m pipeline.run_ticket5_corrections \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-5/dev/label_correction_plan.json \
    --output-dir experiments/ticket-5/dev/correction_experiment

python -m pipeline.freeze_ticket5 \
    --dev-dir experiments/ticket-5/dev \
    --output experiments/ticket-5/frozen_decision.json

python -m pipeline.run_ticket5_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-5/frozen_decision.json \
    --output-dir experiments/ticket-5/heldout \
    --confirm-single-ticket5-report
```

---

## Regenerate Final Artifacts

After all five tickets have been run, regenerate the consolidated summary artifacts and reproducibility audit:

```bash
# Step 6e: Finalize Ticket 5 audit records
python -m pipeline.finalize_ticket5_audit \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --records experiments/ticket-5/final_audit_records.json \
    --output results/data_quality_audit.csv

# Step 7: Build frozen decisions manifest
python -m pipeline.build_frozen_decisions_manifest

# Step 8: Run 5 clean-process reproducibility replays (one per ticket)
for ticket in 1 2 3 4 5; do
    python -m pipeline.reproduce_frozen_ticket \
        --ticket "${ticket}" \
        --manifest configs/frozen_decisions.json \
        --output-dir "experiments/final-reproducibility-audit/replays/ticket-${ticket}"
done

# Step 9: Verify final reproducibility
python -m pipeline.verify_final_reproducibility
```

Alternatively, run the entire pipeline end-to-end with a single command:

```bash
bash run_all.sh
```

---

## Validate the Submission

After all steps complete, run the submission validator to confirm all required artifacts are present and well-formed:

```bash
python -m pipeline.validate_submission
```

This checks:
- All five `frozen_decision.json` files exist and are internally consistent
- `results/summary.csv` has exactly 5 rows (one per ticket)
- `predictions/heldout_predictions.csv` has exactly 1,523 rows
- `results/data_quality_audit.csv` is present and non-empty
- `configs/frozen_decisions.json` manifest is complete
- `README.md` contains all required documentation sections
- `logs/chat.md` contains the AI usage log
- `report.pdf` is a valid PDF of the expected size

If any check fails, the validator prints a descriptive error message identifying the missing or malformed artifact.