"""Re-fit one frozen ticket in an isolated process and compare archived outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .artifacts import build_prediction_frame, validate_prediction_frame, write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .baselines import make_reference_pipeline
from .data import load_labeled_tweets, select_split_by_id
from .decision_rule import MODEL_SPECS, make_ticket4_pipeline, predictions_at_threshold
from .metrics import metric_bundle
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .run_ticket5_dev import sha256
from .splits import load_fixed_split
from .ticket2 import make_ticket2_pipeline
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "frozen_decisions.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _make_model(ticket: int, settings: object) -> object:
    if ticket in {1, 3}:
        return make_reference_pipeline(settings)
    if ticket == 2:
        return make_ticket2_pipeline("normalize_urls_placeholder", settings)
    spec = next(item for item in MODEL_SPECS if item.name == "lr_c1_balanced_default")
    return make_ticket4_pipeline(spec, settings)


def _evaluate(model: object, frame: pd.DataFrame, *, ticket: int, model_name: str, threshold: float) -> tuple[pd.DataFrame, dict[str, float | int]]:
    classifier = model.named_steps["classifier"]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    scores = model.predict_proba(frame["text"])[:, positive_index]
    predictions = predictions_at_threshold(scores, threshold)
    native = model.predict(frame["text"]).astype(int)
    if threshold == 0.5 and not np.array_equal(predictions, native):
        raise AssertionError("explicit threshold differs from native Logistic Regression prediction")
    artifact = build_prediction_frame(
        ids=frame["id"].tolist(),
        y_true=frame["target"].to_numpy(dtype=int),
        y_pred=predictions,
        scores=scores,
        model_name=model_name,
        ticket=f"ticket_{ticket}_clean_replay",
    )
    return artifact, metric_bundle(artifact["y_true"], artifact["y_pred"])


def _compare_prediction_core(replay: pd.DataFrame, archived_path: Path, expected_ids: list[int]) -> dict[str, object]:
    archived = pd.read_csv(archived_path)
    validate_prediction_frame(archived, expected_ids=expected_ids)
    validate_prediction_frame(replay, expected_ids=expected_ids)
    same_ids = np.array_equal(replay["id"], archived["id"])
    same_labels = np.array_equal(replay["y_true"], archived["y_true"])
    same_predictions = np.array_equal(replay["y_pred"], archived["y_pred"])
    score_difference = np.abs(replay["score"].to_numpy(dtype=float) - archived["score"].to_numpy(dtype=float))
    return {
        "same_ids_and_order": bool(same_ids),
        "same_y_true": bool(same_labels),
        "same_y_pred": bool(same_predictions),
        "prediction_changes": int((replay["y_pred"].to_numpy() != archived["y_pred"].to_numpy()).sum()),
        "maximum_absolute_score_difference": float(score_difference.max()),
        "scores_match_within_1e_12": bool(np.all(score_difference <= 1e-12)),
        "archived_sha256": sha256(archived_path),
    }


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("clean replay output exists; refusing overwrite")
    process_started = datetime.now().astimezone().isoformat(timespec="microseconds")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    decision = next(item for item in manifest["decisions"] if item["ticket"] == args.ticket)
    freeze_path = PROJECT_ROOT / decision["freeze_path"]
    if sha256(freeze_path) != decision["freeze_sha256"]:
        raise RuntimeError("ticket freeze hash differs from consolidated manifest")
    if manifest["data_sha256"] != sha256(PROJECT_ROOT / manifest["data_path"]):
        raise RuntimeError("data hash differs from consolidated manifest")
    if manifest["split_sha256"] != sha256(PROJECT_ROOT / manifest["split_path"]):
        raise RuntimeError("split hash differs from consolidated manifest")

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(PROJECT_ROOT / manifest["split_path"])
    data = load_labeled_tweets(PROJECT_ROOT / manifest["data_path"], split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")
    heldout = select_split_by_id(data, split, "heldout")
    model = _make_model(args.ticket, settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train["text"], train["target"])
    threshold = float(decision["recipe"]["threshold"])
    dev_predictions, dev_metrics = _evaluate(model, dev, ticket=args.ticket, model_name=decision["model_name"], threshold=threshold)
    heldout_predictions, heldout_metrics = _evaluate(model, heldout, ticket=args.ticket, model_name=decision["model_name"], threshold=threshold)
    dev_comparison = _compare_prediction_core(dev_predictions, PROJECT_ROOT / decision["archived_dev_predictions"], list(split.dev_ids))
    heldout_comparison = _compare_prediction_core(heldout_predictions, PROJECT_ROOT / decision["archived_heldout_predictions"], list(split.heldout_ids))
    for comparison in (dev_comparison, heldout_comparison):
        if not all(comparison[key] for key in ("same_ids_and_order", "same_y_true", "same_y_pred", "scores_match_within_1e_12")):
            raise AssertionError(f"clean replay differs from archived prediction core: {comparison}")
    for observed, expected, name in ((dev_metrics, decision["expected_dev_metrics"], "dev"), (heldout_metrics, decision["expected_heldout_metrics"], "heldout")):
        for key, expected_value in expected.items():
            if key not in observed:
                continue
            if isinstance(expected_value, float):
                if abs(float(observed[key]) - expected_value) > 1e-15:
                    raise AssertionError(f"{name} metric {key} differs")
            elif int(observed[key]) != int(expected_value):
                raise AssertionError(f"{name} metric {key} differs")

    write_prediction_artifact(dev_predictions, output_dir / "dev_predictions.csv", expected_ids=list(split.dev_ids))
    write_prediction_artifact(heldout_predictions, output_dir / "heldout_predictions.csv", expected_ids=list(split.heldout_ids))
    write_csv_artifact(pd.DataFrame([{"split": "dev", **dev_metrics}, {"split": "heldout", **heldout_metrics}]), output_dir / "metrics.csv")
    warning_payload = {
        "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
        "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        "n_iter": [int(value) for value in np.atleast_1d(model.named_steps["classifier"].n_iter_).tolist()],
    }
    write_json_artifact(warning_payload, output_dir / "warnings.json")
    comparison_payload = {
        "ticket": args.ticket,
        "clean_process_pid": os.getpid(),
        "process_started": process_started,
        "manifest_sha256": sha256(args.manifest),
        "freeze_sha256": sha256(freeze_path),
        "dev": dev_comparison,
        "heldout": heldout_comparison,
        "metrics_exact_within_1e_15": True,
        "result": "PASS",
    }
    write_json_artifact(comparison_payload, output_dir / "comparison.json")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.reproduce_frozen_ticket", *sys.argv[1:]])
    write_text_artifact(command, output_dir / "run_command.txt")
    write_json_artifact(
        {
            "exact_command": command,
            "ticket": args.ticket,
            "scope": "Step 10 clean-process audit replay only; no model selection or artifact overwrite",
            "train_rows": len(train),
            "dev_rows": len(dev),
            "heldout_rows": len(heldout),
            "selection_reopened": False,
            "historical_artifacts_modified": False,
        },
        output_dir / "run_config.json",
    )
    print(json.dumps({"ticket": args.ticket, "pid": os.getpid(), "dev_f1": dev_metrics["f1_target_1"], "heldout_f1": heldout_metrics["f1_target_1"], "result": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
