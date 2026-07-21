"""Run one predeclared Ticket 5 training-label correction probe on dev."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import validate_prediction_frame, write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .data import load_labeled_tweets, select_split_by_id
from .decision_rule import MODEL_SPECS, ModelSpec, fit_and_evaluate_spec
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .run_ticket5_dev import sha256
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import error_rows, prediction_change_rows, transition_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "label_correction_plan.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "correction_experiment"
def _get_ticket4_selected_variant() -> str:
    """Read the selected variant from the Ticket 4 frozen decision (supports expanded MODEL_SPECS)."""
    freeze_path = PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json"
    if freeze_path.exists():
        import json as _json
        return _json.loads(freeze_path.read_text(encoding="utf-8"))["selected_variant"]
    # Fallback: use the original default if freeze has not been run yet
    return "lr_c1_balanced_default"


def _ticket4_dev_predictions_path() -> Path:
    variant = _get_ticket4_selected_variant()
    return PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "predictions" / f"{variant}_dev_predictions.csv"
BASELINE_DEV_PREDICTIONS = (
    PROJECT_ROOT / "experiments" / "step-4-baselines" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Ticket 5 correction artifacts exist; refusing repeated execution")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("created_before_correction_experiment") is not True or plan.get("source_dataset_mutation_permitted") is not False:
        raise ValueError("correction experiment plan is invalid")
    proposals = pd.DataFrame(plan["proposals"])
    if proposals.empty or proposals["id"].duplicated().any():
        raise ValueError("correction proposals must contain unique stable IDs")
    if float(proposals["confidence"].min()) < 0.90:
        raise ValueError("correction proposal violates the confidence gate")

    source_hash_before = sha256(args.data)
    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")
    if not set(proposals["id"].astype(int)).issubset(set(train["id"].astype(int))):
        raise ValueError("every correction proposal must be a train ID")

    proposal_labels = proposals.set_index("id")
    observed = train.set_index("id").loc[proposals["id"].astype(int), "target"].astype(int)
    expected = proposal_labels.loc[proposals["id"].astype(int), "original_label"].astype(int)
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("stored training labels differ from correction plan originals")

    ticket4_variant = _get_ticket4_selected_variant()
    ticket4_spec = next((spec for spec in MODEL_SPECS if spec.name == ticket4_variant), None)
    if ticket4_spec is None:
        raise RuntimeError(
            f"Ticket 4 selected variant '{ticket4_variant}' not found in MODEL_SPECS. "
            "Ensure decision_rule.py MODEL_SPECS includes this variant."
        )
    original_model, original = fit_and_evaluate_spec(train, dev, ticket4_spec, settings)
    frozen_predictions = pd.read_csv(_ticket4_dev_predictions_path())
    validate_prediction_frame(frozen_predictions, expected_ids=list(split.dev_ids))
    if not np.array_equal(original.predictions["y_pred"], frozen_predictions["y_pred"]):
        raise AssertionError("correction control does not reproduce frozen Ticket 4 dev predictions")

    corrected_train = train.copy()
    corrected_train["target_original"] = corrected_train["target"].astype(int)
    proposed_by_id = proposal_labels["proposed_label"].astype(int).to_dict()
    mask = corrected_train["id"].isin(proposed_by_id)
    corrected_train.loc[mask, "target"] = corrected_train.loc[mask, "id"].map(proposed_by_id).astype(int)
    candidate_spec = ModelSpec(
        name="lr_c1_balanced_eight_train_label_corrections",
        classifier="logistic_regression",
        c=1.0,
        class_weight="balanced",
        native_threshold=0.5,
        intended_lever="training_label_correction",
    )
    candidate_model, candidate = fit_and_evaluate_spec(corrected_train, dev, candidate_spec, settings)

    transitions_vs_ticket4 = transition_counts(frozen_predictions, candidate.predictions)
    baseline_predictions = pd.read_csv(BASELINE_DEV_PREDICTIONS)
    validate_prediction_frame(baseline_predictions, expected_ids=list(split.dev_ids))
    transitions_vs_baseline = transition_counts(baseline_predictions, candidate.predictions)
    dev_delta = float(candidate.metrics["f1_target_1"] - original.metrics["f1_target_1"])
    noninferior = dev_delta >= -0.002 - 1e-15
    fixes = int(transitions_vs_ticket4["fixed_fp"] + transitions_vs_ticket4["fixed_fn"])
    new_errors = int(transitions_vs_ticket4["new_fp"] + transitions_vs_ticket4["new_fn"])
    adopt = bool(noninferior and fixes >= new_errors and float(proposals["confidence"].min()) >= 0.90)

    proposal_output = proposals.copy()
    proposal_output["split"] = "train"
    proposal_output["applied_only_to_in_memory_experiment"] = True
    proposal_output["source_dataset_modified"] = False
    write_csv_artifact(proposal_output, output_dir / "label_correction_proposals.csv")
    metric_rows = []
    for name, evaluation, transitions in (
        ("ticket4_original_training_labels", original, {"prediction_changes": 0, "fixed_fp": 0, "fixed_fn": 0, "new_fp": 0, "new_fn": 0}),
        (candidate_spec.name, candidate, transitions_vs_ticket4),
    ):
        metric_rows.append({"variant": name, **evaluation.metrics, **transitions})
    write_csv_artifact(pd.DataFrame(metric_rows), output_dir / "dev_metrics.csv")
    write_prediction_artifact(candidate.predictions, output_dir / "candidate_dev_predictions.csv", expected_ids=list(split.dev_ids))
    write_csv_artifact(prediction_change_rows(frozen_predictions, candidate.predictions, dev), output_dir / "changes_vs_ticket4.csv")
    write_csv_artifact(error_rows(dev, candidate, "false_positives"), output_dir / "candidate_false_positives.csv")
    write_csv_artifact(error_rows(dev, candidate, "false_negatives"), output_dir / "candidate_false_negatives.csv")
    selection = {
        "decision_split": "dev_ids only",
        "acceptance_rule": plan["acceptance_rule"],
        "proposal_count": len(proposals),
        "minimum_proposal_confidence": float(proposals["confidence"].min()),
        "source_dataset_modified": False,
        "dev_labels_modified": False,
        "heldout_rows_loaded": 0,
        "heldout_labels_modified": False,
        "original_dev_metrics": original.metrics,
        "candidate_dev_metrics": candidate.metrics,
        "candidate_f1_delta_vs_ticket4": dev_delta,
        "candidate_transitions_vs_ticket4": transitions_vs_ticket4,
        "candidate_transitions_vs_frozen_baseline": transitions_vs_baseline,
        "noninferiority_margin": -0.002,
        "noninferiority_pass": noninferior,
        "fixed_errors": fixes,
        "new_errors": new_errors,
        "error_balance_pass": fixes >= new_errors,
        "adopt_corrected_training_model": adopt,
        "selected_variant": candidate_spec.name if adopt else ticket4_variant,
    }
    write_json_artifact(selection, output_dir / "selection_result.json")
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket5_corrections", *sys.argv[1:]])
    write_json_artifact(
        {
            "scope": "Single predeclared train-label correction comparison on dev only",
            "exact_command": command,
            "data_sha256_before": source_hash_before,
            "data_sha256_after": sha256(args.data),
            "source_dataset_unchanged": source_hash_before == sha256(args.data),
            "split_sha256": sha256(args.split),
            "plan_sha256": sha256(args.plan),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "heldout_rows_loaded": 0,
            "labels_changed_in_experiment_copy": int(mask.sum()),
            "source_labels_changed": 0,
            "dev_labels_changed": 0,
            "heldout_labels_changed": 0,
        },
        output_dir / "run_config.json",
    )
    write_text_artifact(command, output_dir / "run_command.txt")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
