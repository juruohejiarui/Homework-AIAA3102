"""Freeze the dev-selected Ticket 4 model, hyperparameters, and threshold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .artifacts import write_json_artifact, write_text_artifact
from .run_ticket4_dev import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DIR = PROJECT_ROOT / "experiments" / "ticket-4" / "dev"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json"
SELECTED_VARIANT = "lr_c1_balanced_default"
DECISION_REASON = (
    "Select balanced Logistic Regression with the frozen raw-text TF-IDF preprocessing, C=1.0, and probability threshold 0.50. "
    "It maximized target-1 dev F1 under the predeclared bounded criterion (0.7520849128127369 versus baseline 0.7388120423108218). "
    "The gain came from fixing 42 baseline false negatives while creating 48 new false positives: recall rose from 0.6931297709923664 to "
    "0.7572519083969466 while precision fell from 0.7909407665505227 to 0.7469879518072289. The best baseline threshold-only candidate "
    "was 0.47 at F1 0.7494071146245059, and the best regularization-only candidate was C=2.0 at F1 0.7504025764895334."
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise RuntimeError("Ticket 4 freeze exists; refusing overwrite")
    heldout_dir = PROJECT_ROOT / "experiments" / "ticket-4" / "heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()):
        raise RuntimeError("Ticket 4 held-out artifacts exist before freeze")
    stable_prediction = PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv"
    if stable_prediction.exists():
        raise RuntimeError("Ticket 4 stable held-out predictions exist before freeze")
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if (summary["ticket"] == "ticket_4").any():
        raise RuntimeError("Ticket 4 summary exists before freeze")

    selection_path = args.dev_dir / "selection_result.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    configurations_path = args.dev_dir / "model_configurations.json"
    configurations = json.loads(configurations_path.read_text(encoding="utf-8"))
    metrics_path = args.dev_dir / "results" / "dev_model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    selected = metrics.loc[metrics["variant"] == SELECTED_VARIANT]
    if len(selected) != 1 or selection["selected_variant"] != SELECTED_VARIANT:
        raise RuntimeError("executed dev selection does not match the expected winning candidate")
    row = selected.iloc[0]
    if float(row["C"]) != 1.0 or row["class_weight"] != "balanced" or float(row["decision_threshold"]) != 0.5:
        raise RuntimeError("selected Ticket 4 hyperparameters differ from the declared decision")
    if selection["heldout_rows_loaded"] != 0 or selection["heldout_evaluations_run"] != 0:
        raise RuntimeError("held-out evidence was present during Ticket 4 selection")

    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.freeze_ticket4", *sys.argv[1:]])
    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "decision_rule_source_sha256": PROJECT_ROOT / "pipeline" / "decision_rule.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket4_dev.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket4_heldout.py",
        "experiment_plan_sha256": args.dev_dir / "experiment_plan.json",
        "dev_metrics_sha256": metrics_path,
        "selection_result_sha256": selection_path,
        "model_configurations_sha256": configurations_path,
        "threshold_sweep_sha256": PROJECT_ROOT / "results" / "threshold_sweep.csv",
        "selected_dev_predictions_sha256": args.dev_dir / "predictions" / f"{SELECTED_VARIANT}_dev_predictions.csv",
        "baseline_heldout_predictions_sha256": PROJECT_ROOT / "predictions" / "heldout_predictions.csv",
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    freeze = {
        "ticket": 4,
        "freeze_status": "frozen_before_ticket4_heldout_evaluation",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision_split": "dev_ids only",
        "selection_criterion": selection["criterion"],
        "selected_variant": SELECTED_VARIANT,
        "selected_model_family": "logistic_regression",
        "selected_preprocessing": "raw text through default TfidfVectorizer fitted only on train_ids",
        "selected_C": 1.0,
        "selected_class_weight": "balanced",
        "selected_threshold": 0.5,
        "selected_prediction_rule": "predict target 1 when class-1 probability >= 0.50",
        "selected_effective_configuration": configurations[SELECTED_VARIANT],
        "selected_dev_evidence": {
            key: (row[key].item() if hasattr(row[key], "item") else row[key])
            for key in (
                "precision_target_1",
                "recall_target_1",
                "f1_target_1",
                "accuracy",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "prediction_changes",
                "fixed_fp",
                "fixed_fn",
                "new_fp",
                "new_fn",
            )
        },
        "best_threshold_only_evidence": selection["threshold_sweep_best"],
        "decision_reason": DECISION_REASON,
        "prior_ticket_heldout_artifacts_exist": True,
        "ticket4_heldout_artifact_used_in_decision": False,
        "ticket4_heldout_evaluation_count_at_freeze": 0,
        "selection_reopening_permitted": False,
        "exact_freeze_command": command,
        "integrity": {key: sha256(path) for key, path in paths.items()},
    }
    write_json_artifact(freeze, args.output)
    write_text_artifact(
        "\n".join(
            [
                "# Ticket 4 Freeze Decision",
                "",
                f"Frozen at: {freeze['frozen_at']}",
                "",
                "Selected: balanced Logistic Regression, C=1.0, raw-text default TF-IDF, probability threshold 0.50.",
                "",
                DECISION_REASON,
                "",
                "No Ticket 4 held-out artifact or metric was used. Model, preprocessing, hyperparameters, and threshold are locked; held-out cannot reopen the decision.",
            ]
        ),
        args.output.with_name("freeze_decision.md"),
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
