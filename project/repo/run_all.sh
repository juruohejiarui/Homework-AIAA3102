#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full pipeline execution script (Steps 0–9)
#
# Usage:
#   bash run_all.sh
#
# Prerequisites:
#   - Python environment with requirements.txt installed
#   - data/train.csv placed in the project root
#
# This script:
#   1. Cleans ALL experiment outputs thoroughly (keeps only plan JSON files)
#   2. Runs all pipeline steps in the correct order
#   3. Stops immediately on any error (set -e)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

log()   { echo -e "${BOLD}[run_all]${RESET} $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 0: Verify prerequisites
# ---------------------------------------------------------------------------
log "Step 0: Verifying prerequisites..."

if [ ! -f "data/train.csv" ]; then
    fail "data/train.csv not found. Please place the dataset file at data/train.csv before running."
fi

if [ ! -f "starter/data/split_indices.json" ]; then
    fail "starter/data/split_indices.json not found."
fi

python3 -c "import sklearn, pandas, numpy, scipy" 2>/dev/null \
    || fail "Required Python packages not installed. Run: pip install -r requirements.txt"

ok "Prerequisites verified."

# ---------------------------------------------------------------------------
# THOROUGH cleanup: remove ALL generated outputs, keep only plan JSONs
# ---------------------------------------------------------------------------
log "Cleaning ALL leftover experiment outputs..."

# step-4-baselines: clean entirely (no plan files to keep)
rm -rf experiments/step-4-baselines/

# ticket-1: remove everything except probe_plan.json
# Must handle subdirectories (probes/, heldout/, heldout/predictions/)
if [ -d "experiments/ticket-1" ]; then
    find experiments/ticket-1/ -type f \
        ! -name "probe_plan.json" \
        -delete 2>/dev/null || true
    find experiments/ticket-1/ -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

# ticket-2: remove everything except experiment_plan.json
if [ -d "experiments/ticket-2" ]; then
    find experiments/ticket-2/ -type f \
        ! -name "experiment_plan.json" \
        -delete 2>/dev/null || true
    find experiments/ticket-2/ -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

# ticket-3: remove everything except experiment_plan.json
if [ -d "experiments/ticket-3" ]; then
    find experiments/ticket-3/ -type f \
        ! -name "experiment_plan.json" \
        -delete 2>/dev/null || true
    find experiments/ticket-3/ -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

# ticket-4: remove everything except experiment_plan.json
if [ -d "experiments/ticket-4" ]; then
    find experiments/ticket-4/ -type f \
        ! -name "experiment_plan.json" \
        -delete 2>/dev/null || true
    find experiments/ticket-4/ -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

# ticket-5: remove everything except pre-populated human review files
if [ -d "experiments/ticket-5" ]; then
    find experiments/ticket-5/ -type f \
        ! -name "audit_plan.json" \
        ! -name "label_correction_plan.json" \
        ! -name "curated_dev_review.csv" \
        ! -name "final_audit_records.json" \
        -delete 2>/dev/null || true
    find experiments/ticket-5/ -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

# final-reproducibility-audit: clean entirely
rm -rf experiments/final-reproducibility-audit/

# results: clean generated CSVs and figures (keep directory)
rm -f results/summary.csv results/threshold_sweep.csv results/data_quality_audit.csv
rm -rf results/figures/

# predictions: clean all generated prediction files
rm -f predictions/*.csv

# configs: remove frozen decisions manifest (will be regenerated)
rm -f configs/frozen_decisions.json

ok "Cleanup complete."

# ---------------------------------------------------------------------------
# Step 1: Run Baselines (Floor + Reference)
# ---------------------------------------------------------------------------
log "Step 1: Running baselines..."
python3 -m pipeline.run_baselines \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/step-4-baselines
ok "Step 1 complete."

# ---------------------------------------------------------------------------
# Step 2: Ticket 1 — Baseline Probes (Dev) + Full Held-Out (all probes)
# ---------------------------------------------------------------------------
log "Step 2a: Ticket 1 — Baseline probes (dev only)..."
python3 -m pipeline.run_ticket1_probes \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-1/probes/probe_plan.json \
    --output-dir experiments/ticket-1/probes
ok "Step 2a complete (dev probes)."

log "Step 2b: Ticket 1 — Full held-out evaluation (baseline + all probes)..."
python3 -m pipeline.run_ticket1_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --output-dir experiments/ticket-1/heldout \
    --overwrite
ok "Step 2b complete (held-out: baseline + 32 probes, correlation analysis, frozen_baseline_config.json)."

# ---------------------------------------------------------------------------
# Step 3: Ticket 2 — Normalization (Dev → Freeze → Held-Out)
# ---------------------------------------------------------------------------
log "Step 3a: Ticket 2 — Normalization dev..."
python3 -m pipeline.run_ticket2_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-2/dev/experiment_plan.json \
    --output-dir experiments/ticket-2/dev
ok "Step 3a complete (dev)."

log "Step 3b: Ticket 2 — Freeze decision..."
python3 -m pipeline.freeze_ticket2 \
    --dev-dir experiments/ticket-2/dev \
    --output experiments/ticket-2/frozen_decision.json
ok "Step 3b complete (freeze)."

log "Step 3c: Ticket 2 — Held-out evaluation..."
python3 -m pipeline.run_ticket2_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-2/frozen_decision.json \
    --output-dir experiments/ticket-2/heldout \
    --confirm-single-heldout-evaluation
ok "Step 3c complete (held-out)."

# ---------------------------------------------------------------------------
# Step 4: Ticket 3 — Feature Audit (Dev → Freeze → Held-Out)
# ---------------------------------------------------------------------------
log "Step 4a: Ticket 3 — Feature audit dev..."
python3 -m pipeline.run_ticket3_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-3/dev/experiment_plan.json \
    --output-dir experiments/ticket-3/dev
ok "Step 4a complete (dev)."

log "Step 4b: Ticket 3 — Freeze decision..."
python3 -m pipeline.freeze_ticket3 \
    --dev-dir experiments/ticket-3/dev \
    --output experiments/ticket-3/frozen_decision.json
ok "Step 4b complete (freeze)."

log "Step 4c: Ticket 3 — Held-out report..."
python3 -m pipeline.run_ticket3_heldout \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-3/frozen_decision.json \
    --output-dir experiments/ticket-3/heldout \
    --confirm-single-ticket3-report
ok "Step 4c complete (held-out)."

# ---------------------------------------------------------------------------
# Step 5: Ticket 4 — Decision Rule (Dev → Freeze → Held-Out)
# ---------------------------------------------------------------------------
log "Step 5a: Ticket 4 — Decision rule dev (11 variants + threshold sweep)..."
python3 -m pipeline.run_ticket4_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-4/dev/experiment_plan.json \
    --output-dir experiments/ticket-4/dev
ok "Step 5a complete (dev)."

log "Step 5b: Ticket 4 — Freeze decision..."
python3 -m pipeline.freeze_ticket4 \
    --dev-dir experiments/ticket-4/dev \
    --output experiments/ticket-4/frozen_decision.json
ok "Step 5b complete (freeze)."

log "Step 5c: Ticket 4 — Held-out evaluation..."
python3 -m pipeline.run_ticket4_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-4/frozen_decision.json \
    --output-dir experiments/ticket-4/heldout \
    --confirm-single-ticket4-evaluation
ok "Step 5c complete (held-out)."

# ---------------------------------------------------------------------------
# Step 6: Ticket 5 — Data Quality Audit (Dev → Corrections → Freeze → Held-Out)
# ---------------------------------------------------------------------------
log "Step 6a: Ticket 5 — Data quality audit dev..."
python3 -m pipeline.run_ticket5_dev \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-5/dev/audit_plan.json \
    --output-dir experiments/ticket-5/dev
ok "Step 6a complete (dev audit)."

log "Step 6b: Ticket 5 — Label correction probe..."
python3 -m pipeline.run_ticket5_corrections \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --plan experiments/ticket-5/dev/label_correction_plan.json \
    --output-dir experiments/ticket-5/dev/correction_experiment
ok "Step 6b complete (correction probe)."

log "Step 6c: Ticket 5 — Freeze decision..."
python3 -m pipeline.freeze_ticket5 \
    --dev-dir experiments/ticket-5/dev \
    --output experiments/ticket-5/frozen_decision.json
ok "Step 6c complete (freeze)."

log "Step 6d: Ticket 5 — Held-out report..."
python3 -m pipeline.run_ticket5_heldout \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --freeze experiments/ticket-5/frozen_decision.json \
    --output-dir experiments/ticket-5/heldout \
    --confirm-single-ticket5-report
ok "Step 6d complete (held-out)."

log "Step 6e: Ticket 5 — Finalize audit records (write results/data_quality_audit.csv)..."
python3 -m pipeline.finalize_ticket5_audit \
    --data data/train.csv \
    --split starter/data/split_indices.json \
    --records experiments/ticket-5/final_audit_records.json \
    --output results/data_quality_audit.csv
ok "Step 6e complete (audit finalized)."

# ---------------------------------------------------------------------------
# Step 7: Build Frozen Decisions Manifest
# ---------------------------------------------------------------------------
log "Step 7: Building frozen decisions manifest..."
python3 -m pipeline.build_frozen_decisions_manifest
ok "Step 7 complete."

# ---------------------------------------------------------------------------
# Step 8: Clean-Process Reproducibility Replays (one per ticket)
# ---------------------------------------------------------------------------
log "Step 8: Running 5 clean-process reproducibility replays..."
mkdir -p experiments/final-reproducibility-audit/replays
for ticket in 1 2 3 4 5; do
    log "  Step 8.${ticket}: Reproducing Ticket ${ticket} in clean process..."
    python3 -m pipeline.reproduce_frozen_ticket \
        --ticket "${ticket}" \
        --manifest configs/frozen_decisions.json \
        --output-dir "experiments/final-reproducibility-audit/replays/ticket-${ticket}"
    ok "  Ticket ${ticket} replay complete."
done
ok "Step 8 complete (all 5 replays)."

# ---------------------------------------------------------------------------
# Step 9: Verify Final Reproducibility
# ---------------------------------------------------------------------------
log "Step 9: Verifying final reproducibility..."
python3 -m pipeline.verify_final_reproducibility
ok "Step 9 complete."

# ---------------------------------------------------------------------------
# Step 10: Validate Submission
# ---------------------------------------------------------------------------
log "Step 10: Validating submission artifacts..."
python3 -m pipeline.validate_submission
ok "Step 10 complete."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}============================================${RESET}"
echo -e "${BOLD}${GREEN}  All steps completed successfully!         ${RESET}"
echo -e "${BOLD}${GREEN}============================================${RESET}"
echo ""
echo "Key output files:"
echo "  results/summary.csv              — 5-ticket summary"
echo "  results/threshold_sweep.csv      — Ticket 4 threshold sweep"
echo "  results/data_quality_audit.csv   — Ticket 5 audit"
echo "  predictions/heldout_predictions.csv — Final held-out predictions"
echo "  experiments/ticket-1/heldout/discrepancy_comparison.csv — Probe dev+heldout correlation"
echo "  experiments/ticket-1/heldout/discrepancy_association.json — Pearson/Spearman stats"
echo ""
echo "Upload the entire project directory for analysis."
