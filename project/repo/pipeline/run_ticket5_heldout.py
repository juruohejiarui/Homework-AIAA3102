"""Report the frozen Ticket 5 decision and audit full cross-split relationships once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import SUMMARY_COLUMNS, validate_prediction_frame, write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .data import load_labeled_tweets
from .data_quality import attach_split, duplicate_members, duplicate_summary, near_duplicate_pairs
from .metrics import metric_bundle
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .run_ticket5_dev import sha256
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import prediction_change_rows, transition_counts
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_FREEZE_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "frozen_decision.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "heldout"
TICKET4_PREDICTIONS = PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv"
BASELINE_PREDICTIONS = PROJECT_ROOT / "predictions" / "heldout_predictions.csv"
STABLE_PREDICTIONS = PROJECT_ROOT / "predictions" / "ticket-5-heldout-predictions.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--confirm-single-ticket5-report", action="store_true")
    return parser.parse_args()


def validate_frozen_ticket5_configuration(path: str | Path = DEFAULT_FREEZE_PATH) -> dict[str, Any]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze["freeze_status"] != "frozen_before_ticket5_heldout_reporting_and_audit":
        raise ValueError("Ticket 5 decision is not frozen")
    if freeze["selected_variant"] != "lr_c1_balanced_default" or freeze["training_label_corrections_adopted"] is not False:
        raise ValueError("Ticket 5 freeze does not retain the frozen Ticket 4 model")
    if freeze["ticket5_heldout_reporting_count_at_freeze"] != 0:
        raise ValueError("Ticket 5 held-out reporting count was not zero at freeze")
    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "data_quality_source_sha256": PROJECT_ROOT / "pipeline" / "data_quality.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket5_dev.py",
        "correction_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket5_corrections.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket5_heldout.py",
        "audit_plan_sha256": PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "audit_plan.json",
        "duplicate_summary_sha256": PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "results" / "duplicate_summary.csv",
        "curated_dev_review_sha256": PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "curated_dev_review.csv",
        "correction_plan_sha256": PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "label_correction_plan.json",
        "correction_metrics_sha256": PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "correction_experiment" / "dev_metrics.csv",
        "correction_selection_sha256": PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "correction_experiment" / "selection_result.json",
        "ticket4_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json",
        "ticket4_heldout_predictions_sha256": TICKET4_PREDICTIONS,
        "baseline_heldout_predictions_sha256": BASELINE_PREDICTIONS,
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    mismatches = {
        key: {"expected": freeze["integrity"][key], "actual": sha256(file)}
        for key, file in paths.items()
        if freeze["integrity"][key] != sha256(file)
    }
    if mismatches:
        raise ValueError(f"Ticket 5 frozen integrity validation failed: {mismatches}")
    return freeze


def _error_candidates(predictions: pd.DataFrame, data: pd.DataFrame, exact: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    errors = predictions.merge(
        data.loc[:, ["id", "text", "keyword", "location"]],
        on="id",
        validate="one_to_one",
        sort=False,
    )
    errors = errors.loc[errors["y_true"] != errors["y_pred"]].copy()
    errors["error_type"] = errors.apply(lambda row: "false_positive" if int(row["y_true"]) == 0 else "false_negative", axis=1)
    errors["model_confidence_in_wrong_prediction"] = errors.apply(
        lambda row: float(row["score"]) if int(row["y_pred"]) == 1 else 1.0 - float(row["score"]),
        axis=1,
    )
    for prefix, members in (("exact", exact), ("canonical", canonical)):
        flags = members.loc[:, ["id", "group_id", "group_size", "label_conflict", "cross_split"]].rename(
            columns={
                "group_id": f"{prefix}_group_id",
                "group_size": f"{prefix}_group_size",
                "label_conflict": f"{prefix}_label_conflict",
                "cross_split": f"{prefix}_cross_split",
            }
        )
        errors = errors.merge(flags, on="id", how="left", validate="one_to_one", sort=False)
    return errors.sort_values(
        ["error_type", "model_confidence_in_wrong_prediction", "id"],
        ascending=[True, False, True],
        kind="stable",
        ignore_index=True,
    )


def main() -> int:
    args = _arguments()
    if not args.confirm_single_ticket5_report:
        raise RuntimeError("explicit single Ticket 5 reporting confirmation is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Ticket 5 held-out artifacts exist; refusing repeated reporting")
    if STABLE_PREDICTIONS.exists():
        raise RuntimeError("Ticket 5 stable predictions exist; refusing overwrite")
    summary_path = PROJECT_ROOT / "results" / "summary.csv"
    summary = pd.read_csv(summary_path)
    if list(summary.columns) != SUMMARY_COLUMNS or (summary["ticket"] == "ticket_5").any():
        raise RuntimeError("summary schema invalid or Ticket 5 row already exists")
    freeze = validate_frozen_ticket5_configuration(args.freeze)
    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    split_by_id = {int(value): "train" for value in split.train_ids}
    split_by_id.update({int(value): "dev" for value in split.dev_ids})
    split_by_id.update({int(value): "heldout" for value in split.heldout_ids})
    full = attach_split(data, split_by_id)

    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket5_heldout", *sys.argv[1:]])
    freeze_hash = sha256(args.freeze)
    write_json_artifact(
        {
            "status": "ticket5_heldout_reporting_started",
            "ticket5_heldout_reporting_number": 1,
            "new_model_fit": False,
            "source_prediction_sha256": sha256(TICKET4_PREDICTIONS),
            "freeze_sha256": freeze_hash,
            "exact_command": command,
            "selection_reopened": False,
            "heldout_labels_modified": False,
            "heldout_rows_removed": 0,
        },
        output_dir / "heldout_evaluation_started.json",
    )

    exact = duplicate_members(full, kind="exact")
    canonical = duplicate_members(full, kind="canonical")
    near = near_duplicate_pairs(full, threshold=0.88, neighbors=8, n_jobs=settings.n_jobs)
    duplicate_stats = duplicate_summary(exact, canonical, near)
    write_csv_artifact(duplicate_stats, output_dir / "full_duplicate_summary.csv")
    write_csv_artifact(exact, output_dir / "duplicates" / "full_exact_duplicate_members.csv")
    write_csv_artifact(canonical, output_dir / "duplicates" / "full_canonical_duplicate_members.csv")
    write_csv_artifact(near, output_dir / "duplicates" / "full_near_duplicate_pairs.csv")
    write_csv_artifact(exact.loc[exact["splits_present"].str.contains("heldout")].copy(), output_dir / "duplicates" / "heldout_exact_relationships.csv")
    write_csv_artifact(canonical.loc[canonical["splits_present"].str.contains("heldout")].copy(), output_dir / "duplicates" / "heldout_canonical_relationships.csv")
    write_csv_artifact(near.loc[(near["split_a"] == "heldout") | (near["split_b"] == "heldout")].copy(), output_dir / "duplicates" / "heldout_near_relationships.csv")

    ticket4 = pd.read_csv(TICKET4_PREDICTIONS)
    baseline = pd.read_csv(BASELINE_PREDICTIONS)
    validate_prediction_frame(ticket4, expected_ids=list(split.heldout_ids))
    validate_prediction_frame(baseline, expected_ids=list(split.heldout_ids))
    predictions = ticket4.copy()
    predictions["ticket"] = "ticket_5_data_quality"
    write_prediction_artifact(predictions, output_dir / "heldout_predictions.csv", expected_ids=list(split.heldout_ids))
    write_prediction_artifact(predictions, STABLE_PREDICTIONS, expected_ids=list(split.heldout_ids))
    metrics = metric_bundle(predictions["y_true"], predictions["y_pred"])
    transitions = transition_counts(baseline, predictions)
    heldout_context = full.loc[full["split"] == "heldout"].copy()
    changes = prediction_change_rows(baseline, predictions, heldout_context)
    errors = _error_candidates(predictions, heldout_context, exact, canonical)
    write_csv_artifact(pd.DataFrame([{"split": "heldout", "variant": freeze["selected_variant"], **metrics, **transitions, "new_model_fit": False}]), output_dir / "heldout_metrics.csv")
    write_csv_artifact(changes, output_dir / "heldout_changes_vs_frozen_baseline.csv")
    write_csv_artifact(errors, output_dir / "review" / "heldout_model_errors.csv")
    write_csv_artifact(errors.loc[errors["error_type"] == "false_positive"].copy(), output_dir / "review" / "heldout_false_positive_candidates.csv")
    write_csv_artifact(errors.loc[errors["error_type"] == "false_negative"].copy(), output_dir / "review" / "heldout_false_negative_candidates.csv")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact(
        {
            "scope": "Post-freeze Ticket 5 held-out report and full cross-split data-quality candidate audit",
            "exact_command": command,
            "freeze_sha256": freeze_hash,
            "reporting_mode": "Reuse validated Ticket 4 stable held-out predictions; no refit or new prediction pass",
            "ticket5_heldout_reporting_number": 1,
            "new_model_fit": False,
            "selection_or_tuning_on_heldout": False,
            "selection_reopened": False,
            "source_dataset_modified": False,
            "heldout_labels_modified": False,
            "heldout_rows_removed": 0,
            "full_rows_audited": len(full),
        },
        output_dir / "run_config.json",
    )
    write_text_artifact(command, output_dir / "run_command.txt")

    row = pd.DataFrame(
        [
            {
                "ticket": "ticket_5",
                "model_name": freeze["selected_variant"],
                "dev_f1_target_1": freeze["selected_dev_evidence"]["f1_target_1"],
                "heldout_f1_target_1": metrics["f1_target_1"],
                "heldout_accuracy": metrics["accuracy"],
                "fixed_fp": transitions["fixed_fp"],
                "fixed_fn": transitions["fixed_fn"],
                "new_fp": transitions["new_fp"],
                "new_fn": transitions["new_fn"],
                "decision": "retain_ticket4_model_no_label_corrections",
                "decision_reason": freeze["decision_reason"],
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    write_csv_artifact(pd.concat([summary, row], ignore_index=True), summary_path)
    write_json_artifact(
        {
            "status": "ticket5_heldout_reporting_completed",
            "ticket5_heldout_reporting_count": 1,
            "new_model_fit": False,
            "freeze_sha256": freeze_hash,
            "prediction_sha256": sha256(output_dir / "heldout_predictions.csv"),
            "stable_prediction_sha256": sha256(STABLE_PREDICTIONS),
            "selection_reopened": False,
            "heldout_labels_modified": False,
            "heldout_rows_removed": 0,
            "metrics": metrics,
            "transitions_vs_frozen_baseline": transitions,
            "duplicate_summary": duplicate_stats.to_dict(orient="records"),
        },
        output_dir / "heldout_evaluation_completed.json",
    )
    print(duplicate_stats.to_string(index=False))
    print(json.dumps({"metrics": metrics, "transitions": transitions, "new_model_fit": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
