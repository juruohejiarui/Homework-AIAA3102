"""Freeze the dev-only Ticket 5 decision before held-out audit/reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .artifacts import DATA_QUALITY_DISPOSITIONS, write_json_artifact, write_text_artifact
from .run_ticket5_dev import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DIR = PROJECT_ROOT / "experiments" / "ticket-5" / "dev"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "ticket-5" / "frozen_decision.json"
DECISION_REASON = (
    "Retain the frozen Ticket 4 balanced Logistic Regression model and do not apply training-label corrections. "
    "The eight-row correction probe preserved all original/proposed labels and source data, but dev F1 fell from "
    "0.7520849128127369 to 0.7488653555219364 (delta -0.003219557290800479), failing the predeclared -0.002 "
    "noninferiority margin. It fixed 3 Ticket 4 dev errors while creating 8. Duplicate conflicts remain important "
    "audit findings, but benchmark inconsistency and annotation ambiguity make silent relabeling unjustified."
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise RuntimeError("Ticket 5 freeze exists; refusing overwrite")
    heldout_dir = PROJECT_ROOT / "experiments" / "ticket-5" / "heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()):
        raise RuntimeError("Ticket 5 held-out artifacts exist before freeze")
    if (PROJECT_ROOT / "predictions" / "ticket-5-heldout-predictions.csv").exists():
        raise RuntimeError("Ticket 5 stable predictions exist before freeze")
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if (summary["ticket"] == "ticket_5").any():
        raise RuntimeError("Ticket 5 summary exists before freeze")

    correction_selection_path = args.dev_dir / "correction_experiment" / "selection_result.json"
    correction = json.loads(correction_selection_path.read_text(encoding="utf-8"))
    if correction["adopt_corrected_training_model"] is not False or correction["selected_variant"] != "lr_c1_balanced_default":
        raise RuntimeError("correction result does not retain the frozen Ticket 4 model")
    if correction["heldout_rows_loaded"] != 0 or correction["dev_labels_modified"] is not False:
        raise RuntimeError("correction selection violated split or label constraints")
    review = pd.read_csv(args.dev_dir / "curated_dev_review.csv")
    if not set(review["disposition"]).issubset(DATA_QUALITY_DISPOSITIONS):
        raise RuntimeError("curated dev review contains invalid dispositions")
    if not pd.to_numeric(review["confidence"]).between(0, 1, inclusive="both").all():
        raise RuntimeError("curated dev confidence is outside [0,1]")

    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.freeze_ticket5", *sys.argv[1:]])
    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "data_quality_source_sha256": PROJECT_ROOT / "pipeline" / "data_quality.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket5_dev.py",
        "correction_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket5_corrections.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket5_heldout.py",
        "audit_plan_sha256": args.dev_dir / "audit_plan.json",
        "duplicate_summary_sha256": args.dev_dir / "results" / "duplicate_summary.csv",
        "curated_dev_review_sha256": args.dev_dir / "curated_dev_review.csv",
        "correction_plan_sha256": args.dev_dir / "label_correction_plan.json",
        "correction_metrics_sha256": args.dev_dir / "correction_experiment" / "dev_metrics.csv",
        "correction_selection_sha256": correction_selection_path,
        "ticket4_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json",
        "ticket4_heldout_predictions_sha256": PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv",
        "baseline_heldout_predictions_sha256": PROJECT_ROOT / "predictions" / "heldout_predictions.csv",
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    freeze = {
        "ticket": 5,
        "freeze_status": "frozen_before_ticket5_heldout_reporting_and_audit",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision_split": "train_ids and dev_ids only",
        "selected_variant": "lr_c1_balanced_default",
        "selected_model_source": "Frozen Ticket 4 model configuration",
        "training_label_corrections_adopted": False,
        "source_dataset_modified": False,
        "dev_labels_modified": False,
        "selected_dev_evidence": correction["original_dev_metrics"],
        "rejected_correction_probe": {
            "proposal_count": correction["proposal_count"],
            "minimum_proposal_confidence": correction["minimum_proposal_confidence"],
            "candidate_dev_metrics": correction["candidate_dev_metrics"],
            "candidate_f1_delta_vs_ticket4": correction["candidate_f1_delta_vs_ticket4"],
            "candidate_transitions_vs_ticket4": correction["candidate_transitions_vs_ticket4"],
            "noninferiority_pass": correction["noninferiority_pass"],
            "error_balance_pass": correction["error_balance_pass"],
        },
        "decision_reason": DECISION_REASON,
        "prior_ticket_heldout_artifacts_exist": True,
        "ticket5_heldout_artifact_used_in_decision": False,
        "ticket5_heldout_reporting_count_at_freeze": 0,
        "heldout_reporting_mode": "Reuse validated Ticket 4 stable held-out predictions; do not refit or repredict. Perform post-freeze duplicate/error audit without changing labels or rows.",
        "heldout_label_modification_permitted": False,
        "heldout_row_removal_permitted": False,
        "selection_reopening_permitted": False,
        "exact_freeze_command": command,
        "integrity": {key: sha256(path) for key, path in paths.items()},
    }
    write_json_artifact(freeze, args.output)
    write_text_artifact(
        "\n".join(
            [
                "# Ticket 5 Freeze Decision",
                "",
                f"Frozen at: {freeze['frozen_at']}",
                "",
                "Selected: retain the frozen Ticket 4 balanced Logistic Regression model; apply no label corrections.",
                "",
                DECISION_REASON,
                "",
                "No Ticket 5 held-out artifact or label informed this decision. Held-out labels and rows are immutable, and held-out reporting cannot reopen selection.",
            ]
        ),
        args.output.with_name("freeze_decision.md"),
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
