"""Report the frozen Ticket 3 retain-text-only decision on held-out without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import SUMMARY_COLUMNS, validate_prediction_frame, write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .metrics import metric_bundle
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import transition_counts
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_PATH = PROJECT_ROOT / "experiments" / "ticket-3" / "frozen_decision.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-3" / "heldout"
BASELINE_PREDICTIONS = PROJECT_ROOT / "predictions" / "heldout_predictions.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--confirm-single-ticket3-report", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_ticket3_configuration(path: str | Path = DEFAULT_FREEZE_PATH) -> dict[str, Any]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze["freeze_status"] != "frozen_before_ticket3_heldout_reporting":
        raise ValueError("Ticket 3 decision is not frozen")
    if freeze["selected_variant"] != "raw_text_tfidf_logistic_regression":
        raise ValueError("Ticket 3 freeze does not retain the frozen baseline")
    if freeze["ticket3_heldout_reporting_count_at_freeze"] != 0:
        raise ValueError("Ticket 3 held-out report count was not zero at freeze")
    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "shortcut_source_sha256": PROJECT_ROOT / "pipeline" / "shortcut_features.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket3_dev.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket3_heldout.py",
        "experiment_plan_sha256": PROJECT_ROOT / "experiments" / "ticket-3" / "dev" / "experiment_plan.json",
        "dev_metrics_sha256": PROJECT_ROOT / "experiments" / "ticket-3" / "dev" / "results" / "dev_metrics.csv",
        "robustness_metrics_sha256": PROJECT_ROOT / "experiments" / "ticket-3" / "dev" / "robustness" / "robustness_metrics.csv",
        "text_control_dev_predictions_sha256": PROJECT_ROOT / "experiments" / "ticket-3" / "dev" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv",
        "baseline_heldout_predictions_sha256": BASELINE_PREDICTIONS,
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    mismatches = {key:{"expected":freeze["integrity"][key],"actual":sha256(file)} for key,file in paths.items() if freeze["integrity"][key] != sha256(file)}
    if mismatches:
        raise ValueError(f"Ticket 3 frozen integrity validation failed: {mismatches}")
    return freeze


def main() -> int:
    args = _arguments()
    if not args.confirm_single_ticket3_report:
        raise RuntimeError("explicit single Ticket 3 held-out reporting confirmation is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Ticket 3 held-out artifacts exist; refusing repeated reporting")
    freeze = validate_frozen_ticket3_configuration(args.freeze)
    command = subprocess.list2cmdline([sys.executable,"-m","pipeline.run_ticket3_heldout",*sys.argv[1:]])
    freeze_hash = sha256(args.freeze)
    write_json_artifact({"status":"ticket3_heldout_reporting_started","ticket3_heldout_reporting_number":1,"new_heldout_model_fit":False,"source_prediction_sha256":sha256(BASELINE_PREDICTIONS),"freeze_sha256":freeze_hash,"exact_command":command,"selection_reopened":False}, output_dir / "heldout_evaluation_started.json")

    split = load_fixed_split(args.split)
    source = pd.read_csv(BASELINE_PREDICTIONS)
    validate_prediction_frame(source, expected_ids=list(split.heldout_ids))
    predictions = source.copy()
    predictions["ticket"] = "ticket_3_shortcuts"
    destination = output_dir / "heldout_predictions.csv"
    write_prediction_artifact(predictions, destination, expected_ids=list(split.heldout_ids))
    write_prediction_artifact(predictions, PROJECT_ROOT / "predictions" / "ticket-3-heldout-predictions.csv", expected_ids=list(split.heldout_ids))
    metrics = metric_bundle(predictions["y_true"], predictions["y_pred"])
    transitions = transition_counts(source, predictions)
    write_csv_artifact(pd.DataFrame([{"split":"heldout","variant":freeze["selected_variant"],**metrics,**transitions,"new_heldout_model_fit":False}]), output_dir / "heldout_metrics.csv")
    write_csv_artifact(pd.DataFrame([{"model_name":freeze["selected_variant"],"actual_label":0,"predicted_0":metrics["true_negative"],"predicted_1":metrics["false_positive"]},{"model_name":freeze["selected_variant"],"actual_label":1,"predicted_0":metrics["false_negative"],"predicted_1":metrics["true_positive"]}]), output_dir / "heldout_confusion_matrix.csv")
    empty_changes = pd.DataFrame(columns=["id","text","keyword","location","y_true","baseline_y_pred","candidate_y_pred","baseline_score","candidate_score","transition","baseline_correct","candidate_correct","outcome"])
    write_csv_artifact(empty_changes, output_dir / "heldout_changes_vs_frozen_baseline.csv")
    for source_name, destination_name in (("heldout_false_positives.csv","heldout_false_positives.csv"),("heldout_false_negatives.csv","heldout_false_negatives.csv")):
        errors = pd.read_csv(PROJECT_ROOT / "experiments" / "ticket-1" / "heldout" / source_name)
        errors["ticket"] = "ticket_3_shortcuts"
        write_csv_artifact(errors, output_dir / destination_name)
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact({"scope":"Ticket 3 held-out report for the frozen decision to retain the existing text-only baseline","exact_command":command,"freeze_sha256":freeze_hash,"reporting_mode":"reuse validated Ticket 1 stable held-out predictions; no refit or new prediction pass","ticket3_heldout_reporting_number":1,"new_heldout_model_fit":False,"selection_or_tuning_on_heldout":False}, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    summary_path = PROJECT_ROOT / "results" / "summary.csv"
    summary = pd.read_csv(summary_path)
    if list(summary.columns) != SUMMARY_COLUMNS or (summary["ticket"] == "ticket_3").any():
        raise RuntimeError("summary schema invalid or Ticket 3 row already exists")
    row = pd.DataFrame([{"ticket":"ticket_3","model_name":freeze["selected_variant"],"dev_f1_target_1":freeze["selected_dev_evidence"]["f1_target_1"],"heldout_f1_target_1":metrics["f1_target_1"],"heldout_accuracy":metrics["accuracy"],"fixed_fp":0,"fixed_fn":0,"new_fp":0,"new_fn":0,"decision":"retain_text_only_reject_shortcuts","decision_reason":freeze["decision_reason"]}],columns=SUMMARY_COLUMNS)
    write_csv_artifact(pd.concat([summary,row],ignore_index=True), summary_path)
    write_json_artifact({"status":"ticket3_heldout_reporting_completed","ticket3_heldout_reporting_count":1,"new_heldout_model_fit":False,"freeze_sha256":freeze_hash,"prediction_sha256":sha256(destination),"source_prediction_sha256":sha256(BASELINE_PREDICTIONS),"selection_reopened":False,"metrics":metrics,"transitions_vs_frozen_baseline":transitions}, output_dir / "heldout_evaluation_completed.json")
    print(json.dumps({"metrics":metrics,"transitions":transitions,"new_heldout_model_fit":False},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
