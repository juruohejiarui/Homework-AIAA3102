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

## Setup

```bash
python venv .venv
source .venv/bin/activate
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
cp /path/to/train.csv data/train.csv
```

---

## Full Execution Guide (Run in Order)

> **Important**: Each step must complete successfully before the next step begins. Do NOT skip steps or run them out of order.

### Step 0: Verify Setup

```bash
cd /path/to/v3
python -m pipeline.data --data data/train.csv --split starter/data/split_indices.json
```

Expected output: dataset statistics (7613 rows, 4567 train, 1523 dev, 1523 heldout).

---

### Step 1: Run Baselines (Floor + Reference)

```bash
python -m pipeline.run_baselines \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/step-4-baselines
```

Outputs:
- `experiments/step-4-baselines/predictions/raw_text_tfidf_logistic_regression_dev_predictions.csv`
- `experiments/step-4-baselines/predictions/raw_text_tfidf_logistic_regression_heldout_predictions.csv`

---

### Step 2: Ticket 1 — Baseline Probes (Dev Only)

```bash
python -m pipeline.run_ticket1_probes \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/ticket-1/dev
```

Then run held-out (after reviewing dev results):

```bash
python -m pipeline.run_ticket1_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/ticket-1/heldout
```

---

### Step 3: Ticket 2 — Normalization (Dev → Freeze → Held-Out)

```bash
# Dev
python -m pipeline.run_ticket2_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-2/dev/experiment_plan.json \
    --output-dir experiments/ticket-2/dev

# Freeze decision
python -m pipeline.freeze_ticket2 \
    --dev-dir experiments/ticket-2/dev \
    --output experiments/ticket-2/frozen_decision.json

# Held-out (only after freeze)
python -m pipeline.run_ticket2_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-2/frozen_decision.json \
    --output-dir experiments/ticket-2/heldout
```

---

### Step 4: Ticket 3 — Feature Audit (Dev Only, No New Held-Out)

```bash
# Dev
python -m pipeline.run_ticket3_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-3/dev/experiment_plan.json \
    --output-dir experiments/ticket-3/dev

# Freeze decision (ticket-3 reuses ticket-1 held-out predictions)
python -m pipeline.freeze_ticket3 \
    --dev-dir experiments/ticket-3/dev \
    --output experiments/ticket-3/frozen_decision.json
```

---

### Step 5: Ticket 4 — Decision Rule (Dev → Freeze → Held-Out)

```bash
# Dev (evaluates 11 model variants + 61-point threshold sweep)
python -m pipeline.run_ticket4_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-4/dev/experiment_plan.json \
    --output-dir experiments/ticket-4/dev

# Freeze decision
python -m pipeline.freeze_ticket4 \
    --dev-dir experiments/ticket-4/dev \
    --output experiments/ticket-4/frozen_decision.json

# Held-out (only after freeze)
python -m pipeline.run_ticket4_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-4/frozen_decision.json \
    --output-dir experiments/ticket-4/heldout
```

---

### Step 6: Ticket 5 — Data Quality Audit (Dev → Correction Probe → Freeze → Held-Out)

```bash
# Dev audit
python -m pipeline.run_ticket5_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --audit-plan experiments/ticket-5/dev/audit_plan.json \
    --correction-plan experiments/ticket-5/dev/label_correction_plan.json \
    --output-dir experiments/ticket-5/dev

# Run correction probe (controlled experiment)
python -m pipeline.run_ticket5_corrections \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --dev-dir experiments/ticket-5/dev \
    --output-dir experiments/ticket-5/dev/correction_experiment

# Finalize audit and freeze decision
python -m pipeline.finalize_ticket5_audit \
    --dev-dir experiments/ticket-5/dev \
    --output experiments/ticket-5/frozen_decision.json

# Held-out (only after freeze)
python -m pipeline.run_ticket5_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-5/frozen_decision.json \
    --output-dir experiments/ticket-5/heldout
```

---

### Step 7: Final Reproducibility Audit

```bash
python -m pipeline.verify_final_reproducibility \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/final-reproducibility-audit
```

---

### Step 8: Build Submission Manifest

```bash
python -m pipeline.validate_submission \
    --data data/train.csv \
    --split starter/data/split_indices.json
```

---

## V3 Key Differences from V1 and V2

### Extended MODEL_SPECS (Ticket 4)

V3 adds C=5 and C=10 variants (both unweighted and balanced) to the Ticket 4 search space, motivated by V1's finding that `lr_c10_balanced` achieved the highest held-out F1 (0.7547) in a parallel study:

```
lr_c5_unweighted_default   (C=5.0, None)
lr_c10_unweighted_default  (C=10.0, None)
lr_c5_balanced_default     (C=5.0, balanced)
lr_c10_balanced_default    (C=10.0, balanced)
```

### Correction Probe (Ticket 5)

V3 includes a controlled label-correction probe in Ticket 5 (from V2). The probe tests whether applying proposed label corrections improves dev F1 by ≥ 0.002. If not, all original labels are retained.

### Modular Architecture

V3 uses V2's modular architecture: each ticket has independent `run_*`, `freeze_*`, and `*_heldout` scripts. This enables clean provenance tracking and reproducibility verification.

---

## Running Tests

```bash
pytest tests/ -v
```

Note: Tests that check for specific frozen decision values (e.g., `test_ticket4.py`) will only pass after the pipeline has been run and the frozen decisions have been written.

---

## Expected Final Outputs

After running all steps, the following files should exist:

| File | Description |
|------|-------------|
| `results/summary.csv` | 5-row summary (one per ticket) |
| `results/threshold_sweep.csv` | 61-row threshold sweep (Ticket 4) |
| `results/data_quality_audit.csv` | Audit manifest (Ticket 5) |
| `predictions/heldout_predictions.csv` | Final held-out predictions (1,523 rows) |
| `predictions/ticket-1-heldout-predictions.csv` | Ticket 1 held-out predictions |
| `predictions/ticket-2-heldout-predictions.csv` | Ticket 2 held-out predictions |
| `predictions/ticket-4-heldout-predictions.csv` | Ticket 4 held-out predictions |
| `experiments/*/frozen_decision.json` | Frozen decisions for each ticket |
| `configs/frozen_decisions_manifest.json` | Complete provenance manifest |
