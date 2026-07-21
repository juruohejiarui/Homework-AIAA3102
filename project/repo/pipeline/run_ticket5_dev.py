"""Run the train/dev-only Ticket 5 duplicate and error candidate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .artifacts import validate_prediction_frame, write_csv_artifact, write_json_artifact, write_text_artifact
from .data import load_labeled_tweets, select_split_by_id
from .data_quality import attach_split, duplicate_members, duplicate_summary, near_duplicate_pairs
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "dev" / "audit_plan.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "dev"
def _get_ticket4_dev_predictions_path() -> Path:
    """Dynamically resolve Ticket 4 dev predictions path from frozen_decision.json."""
    freeze_path = PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json"
    if freeze_path.exists():
        import json as _json
        variant = _json.loads(freeze_path.read_text(encoding="utf-8"))["selected_variant"]
    else:
        variant = "lr_c1_balanced_default"
    return PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "predictions" / f"{variant}_dev_predictions.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _join_group_flags(errors: pd.DataFrame, exact: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    result = errors.copy()
    for prefix, members in (("exact", exact), ("canonical", canonical)):
        flags = members.loc[:, ["id", "group_id", "group_size", "label_conflict", "cross_split"]].rename(
            columns={
                "group_id": f"{prefix}_group_id",
                "group_size": f"{prefix}_group_size",
                "label_conflict": f"{prefix}_label_conflict",
                "cross_split": f"{prefix}_cross_split",
            }
        )
        result = result.merge(flags, on="id", how="left", validate="one_to_one", sort=False)
    return result


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    existing = [path for path in output_dir.iterdir()] if output_dir.exists() else []
    allowed_existing = {
        args.plan.resolve(),
        (output_dir / "curated_dev_review.csv").resolve(),
        (output_dir / "label_correction_plan.json").resolve(),
    }
    unexpected = [path for path in existing if path.resolve() not in allowed_existing]
    if unexpected:
        raise RuntimeError("Ticket 5 dev artifacts already exist; refusing repeated execution")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("ticket") != 5 or plan.get("created_before_dev_execution") is not True:
        raise ValueError("Ticket 5 pre-execution audit plan is invalid")
    if plan.get("heldout_access_by_dev_command") is not False:
        raise ValueError("Ticket 5 dev plan must prohibit held-out access")

    ticket4_dev_predictions = _get_ticket4_dev_predictions_path()

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    split_by_id = {int(value): "train" for value in split.train_ids}
    split_by_id.update({int(value): "dev" for value in split.dev_ids})
    train = attach_split(select_split_by_id(data, split, "train"), split_by_id)
    dev = attach_split(select_split_by_id(data, split, "dev"), split_by_id)
    audit_frame = pd.concat([train, dev], ignore_index=True)

    exact = duplicate_members(audit_frame, kind="exact")
    canonical = duplicate_members(audit_frame, kind="canonical")
    near = near_duplicate_pairs(
        audit_frame,
        threshold=float(plan["near_duplicate_configuration"]["similarity_threshold"]),
        neighbors=int(plan["near_duplicate_configuration"]["neighbors_including_self"]),
        n_jobs=settings.n_jobs,
    )
    summary = duplicate_summary(exact, canonical, near)
    conflicts = pd.concat(
        [
            exact.loc[exact["label_conflict"]].assign(review_source="exact_conflict"),
            canonical.loc[canonical["label_conflict"]].assign(review_source="canonical_conflict"),
        ],
        ignore_index=True,
    ).drop_duplicates(["review_source", "group_id", "id"], keep="first")

    predictions = pd.read_csv(ticket4_dev_predictions)
    validate_prediction_frame(predictions, expected_ids=list(split.dev_ids))
    errors = predictions.merge(
        dev.loc[:, ["id", "text", "keyword", "location"]],
        on="id",
        validate="one_to_one",
        sort=False,
    )
    errors = errors.loc[errors["y_true"] != errors["y_pred"]].copy()
    errors["error_type"] = errors.apply(
        lambda row: "false_positive" if int(row["y_true"]) == 0 else "false_negative",
        axis=1,
    )
    errors["margin_from_threshold"] = (errors["score"].astype(float) - 0.5).abs()
    errors["model_confidence_in_wrong_prediction"] = errors.apply(
        lambda row: float(row["score"]) if int(row["y_pred"]) == 1 else 1.0 - float(row["score"]),
        axis=1,
    )
    errors = _join_group_flags(errors, exact, canonical).sort_values(
        ["error_type", "model_confidence_in_wrong_prediction", "id"],
        ascending=[True, False, True],
        kind="stable",
        ignore_index=True,
    )

    write_csv_artifact(summary, output_dir / "results" / "duplicate_summary.csv")
    write_csv_artifact(exact, output_dir / "duplicates" / "exact_duplicate_members.csv")
    write_csv_artifact(canonical, output_dir / "duplicates" / "canonical_duplicate_members.csv")
    write_csv_artifact(near, output_dir / "duplicates" / "near_duplicate_pairs.csv")
    write_csv_artifact(conflicts, output_dir / "review" / "conflicting_duplicate_members.csv")
    write_csv_artifact(errors, output_dir / "review" / "dev_model_errors.csv")
    write_csv_artifact(
        errors.loc[errors["error_type"] == "false_positive"].copy(),
        output_dir / "review" / "dev_false_positive_candidates.csv",
    )
    write_csv_artifact(
        errors.loc[errors["error_type"] == "false_negative"].copy(),
        output_dir / "review" / "dev_false_negative_candidates.csv",
    )
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket5_dev", *sys.argv[1:]])
    run_config = {
        "scope": "Ticket 5 duplicate and frozen-model error candidate discovery on train/dev only",
        "exact_command": command,
        "data_sha256": sha256(args.data),
        "split_sha256": sha256(args.split),
        "plan_sha256": sha256(args.plan),
        "ticket4_dev_predictions_sha256": sha256(ticket4_dev_predictions),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "heldout_rows_loaded": 0,
        "heldout_labels_inspected": False,
        "heldout_evaluations_run": 0,
        "seed": settings.seed,
        "n_jobs": settings.n_jobs,
        "source_rows_modified": 0,
        "labels_modified": 0,
        "rows_removed": 0,
    }
    write_json_artifact(run_config, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    print(summary.to_string(index=False))
    print(
        json.dumps(
            {
                "dev_false_positives": int((errors["error_type"] == "false_positive").sum()),
                "dev_false_negatives": int((errors["error_type"] == "false_negative").sum()),
                "conflicting_duplicate_review_rows": len(conflicts),
                "heldout_rows_loaded": 0,
                "source_rows_modified": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
