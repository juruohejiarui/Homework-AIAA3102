"""Verify Step 10 clean replays and publish audit-only reproducibility artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from .artifacts import (
    DATA_QUALITY_AUDIT_COLUMNS,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    THRESHOLD_SWEEP_COLUMNS,
    build_prediction_frame,
    validate_prediction_frame,
    write_csv_artifact,
    write_json_artifact,
    write_prediction_artifact,
    write_text_artifact,
)
from .data_quality import validate_data_quality_audit
from .decision_rule import THRESHOLDS, predictions_at_threshold
from .metrics import metric_bundle
from .run_ticket5_dev import sha256
from .splits import load_fixed_split
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "frozen_decisions.json"
AUDIT_DIR = PROJECT_ROOT / "experiments" / "final-reproducibility-audit"
FINAL_PREDICTION_PATH = PROJECT_ROOT / "predictions" / "final-heldout-predictions.csv"
TOLERANCE = 1e-15


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["ticket"]): item for item in manifest["decisions"]}


def _transition_counts(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, int]:
    if not baseline["id"].equals(candidate["id"]) or not baseline["y_true"].equals(candidate["y_true"]):
        raise AssertionError("transition comparison requires identical stable IDs, order, and labels")
    y_true = baseline["y_true"].to_numpy(dtype=int)
    base = baseline["y_pred"].to_numpy(dtype=int)
    new = candidate["y_pred"].to_numpy(dtype=int)
    return {
        "fixed_fp": int(((y_true == 0) & (base == 1) & (new == 0)).sum()),
        "fixed_fn": int(((y_true == 1) & (base == 0) & (new == 1)).sum()),
        "new_fp": int(((y_true == 0) & (base == 0) & (new == 1)).sum()),
        "new_fn": int(((y_true == 1) & (base == 1) & (new == 0)).sum()),
    }


def _compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> dict[str, Any]:
    try:
        assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            atol=TOLERANCE,
            rtol=0.0,
        )
    except AssertionError as error:
        raise AssertionError(f"{name} did not reproduce: {error}") from error
    return {"semantic_match": True, "rows": len(actual), "columns": list(actual.columns)}


def _reproduce_summary(
    decisions: dict[int, dict[str, Any]],
    dev_predictions: dict[int, pd.DataFrame],
    heldout_predictions: dict[int, pd.DataFrame],
    transitions: dict[int, dict[str, int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticket in range(1, 6):
        dev_metrics = metric_bundle(dev_predictions[ticket]["y_true"], dev_predictions[ticket]["y_pred"])
        heldout_metrics = metric_bundle(heldout_predictions[ticket]["y_true"], heldout_predictions[ticket]["y_pred"])
        rows.append(
            {
                "ticket": f"ticket_{ticket}",
                "model_name": decisions[ticket]["model_name"],
                "dev_f1_target_1": dev_metrics["f1_target_1"],
                "heldout_f1_target_1": heldout_metrics["f1_target_1"],
                "heldout_accuracy": heldout_metrics["accuracy"],
                **transitions[ticket],
                "decision": decisions[ticket]["decision"],
                "decision_reason": decisions[ticket]["summary_decision_reason"],
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _reproduce_threshold_sweep(baseline_dev: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true = baseline_dev["y_true"].to_numpy(dtype=int)
    scores = baseline_dev["score"].to_numpy(dtype=float)
    for threshold in THRESHOLDS:
        metrics = metric_bundle(y_true, predictions_at_threshold(scores, threshold))
        rows.append(
            {
                "ticket": "ticket_4",
                "threshold": threshold,
                "precision_target_1": metrics["precision_target_1"],
                "recall_target_1": metrics["recall_target_1"],
                "f1_target_1": metrics["f1_target_1"],
            }
        )
    return pd.DataFrame(rows, columns=THRESHOLD_SWEEP_COLUMNS)


def _inventory_row(path: Path, *, category: str, note: str, rows: int | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "category": category,
        "sha256": sha256(path),
        "rows": "" if rows is None else rows,
        "status": "verified",
        "note": note,
    }


def _audit_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["ticket_metrics"]
    metric_rows = "\n".join(
        f"| Ticket {ticket} | {row['dev_f1_target_1']:.16f} | {row['heldout_f1_target_1']:.16f} | "
        f"{row['heldout_accuracy']:.16f} | {row['fixed_fp']} / {row['fixed_fn']} / {row['new_fp']} / {row['new_fn']} |"
        for ticket, row in metrics.items()
    )
    return f"""# Step 10 reproducibility audit

This is the reproducibility audit requested before final report assembly. It is not the final project report and it does not reopen any ticket decision.

## Audit conclusion

**PASS.** All five frozen decisions were independently re-fit from the unchanged source data and fixed ID split in five distinct Python processes. Dev and held-out labels, prediction order, predictions, and scores reproduce the archived artifacts. Every scalar metric reproduces within `{TOLERANCE}`. The three active result tables reproduce from source/replay evidence, and the final Ticket 5 held-out artifact has the required six-column schema.

## Decision provenance

- The consolidated manifest contains five decisions and five immutable freeze hashes.
- Every freeze records a dev-only decision basis, zero held-out evaluations at freeze time, `heldout_used_for_selection=false`, and `selection_reopening_permitted=false`.
- Step 10 held-out runs are audit replays only. They are not additional primary evaluations and were not used to alter any model, feature, normalization, label, or threshold choice.

## Clean-process reproduction

- Distinct process IDs: `{', '.join(str(value) for value in payload['clean_process_pids'])}`.
- Archived prediction changes across all ten dev/held-out comparisons: `0`.
- Maximum score discrepancy from an archived artifact: `{payload['maximum_score_difference']:.17g}` (acceptance limit `1e-12`).
- Convergence: all five replays completed without convergence warnings.

| Decision | Dev F1 | Held-out F1 | Held-out accuracy | fixed FP / fixed FN / new FP / new FN vs Ticket 1 |
|---|---:|---:|---:|---:|
{metric_rows}

## Result-table and stable-ID checks

- `results/summary.csv`: exact required schema, tickets 1–5 exactly once, and semantic reproduction from the five clean replays.
- `results/threshold_sweep.csv`: exact required schema and all 61 thresholds reproduced from the Ticket 1 raw-text dev scores.
- `results/data_quality_audit.csv`: exact required schema and all curated records reproduced from the preserved Ticket 5 audit-record source; IDs are valid and dispositions/confidences pass validation.
- All historical held-out prediction artifacts and the final artifact contain `{payload['heldout_rows']}` unique stable held-out IDs in the fixed instructor order, with unchanged true labels.

## Consistent error-transition comparison

Every ticket was recalculated against the same comparator: the clean Ticket 1 held-out replay. These counts reproduce `results/summary.csv`. Ticket 1 and Ticket 3 are intentionally the same prediction core; Ticket 4 and Ticket 5 are intentionally the same prediction core because Ticket 5 retained Ticket 4 without label corrections.

## Stale, duplicate, contradictory, and manual-edit audit

- No unexplained stale active result file was found: each active result table has a source-to-reproduction comparison in this audit.
- No contradictory active metric or transition count was found.
- No unexplained manual edit was found. The summary and threshold sweep are regenerated from replay predictions; the data-quality table is regenerated from its preserved structured record source.
- Two semantic duplicate groups are intentional and documented: Ticket 1 = Ticket 3, and Ticket 4 = Ticket 5 = the final submission core. Their CSV byte hashes differ because the `ticket` provenance field differs.
- `predictions/heldout_predictions.csv` is the historical Ticket 1 baseline despite its generic filename. It remains immutable; the unambiguous final path is `predictions/final-heldout-predictions.csv`.
- Historical pre-freeze and completion ledgers are retained as time-stamped provenance. They are not interpreted as current model state, and none was overwritten during this audit.

## Limitations

This audit demonstrates deterministic reproduction in the locked local environment and verifies internal provenance. It does not claim bit-identical floating-point scores across arbitrary operating systems or future dependency versions; the stronger cross-artifact requirement used here is identical class predictions and score differences no greater than `1e-12`.
"""


def main() -> int:
    manifest = _load_json(MANIFEST_PATH)
    decisions = _decision_map(manifest)
    if set(decisions) != set(range(1, 6)):
        raise AssertionError("manifest must contain exactly Tickets 1 through 5")
    if not all(
        "dev" in item["decision_basis_split"].lower()
        and "heldout" not in item["decision_basis_split"].lower()
        and item["heldout_count_at_freeze"] == 0
        and item["heldout_used_for_selection"] is False
        and item["selection_reopening_permitted"] is False
        for item in decisions.values()
    ):
        raise AssertionError("one or more final decisions lacks dev-only pre-held-out provenance")

    for item in decisions.values():
        if sha256(PROJECT_ROOT / item["freeze_path"]) != item["freeze_sha256"]:
            raise AssertionError(f"freeze changed after manifest creation: Ticket {item['ticket']}")
    if sha256(PROJECT_ROOT / manifest["data_path"]) != manifest["data_sha256"]:
        raise AssertionError("source data hash changed")
    if sha256(PROJECT_ROOT / manifest["split_path"]) != manifest["split_sha256"]:
        raise AssertionError("fixed split hash changed")
    if sha256(PROJECT_ROOT / "requirements-lock.txt") != manifest["requirements_lock_sha256"]:
        raise AssertionError("locked software environment changed")

    split = load_fixed_split(PROJECT_ROOT / manifest["split_path"])
    expected_dev_ids = list(split.dev_ids)
    expected_heldout_ids = list(split.heldout_ids)
    dev_predictions: dict[int, pd.DataFrame] = {}
    heldout_predictions: dict[int, pd.DataFrame] = {}
    comparisons: dict[int, dict[str, Any]] = {}
    pids: list[int] = []
    maximum_score_difference = 0.0
    for ticket in range(1, 6):
        replay_dir = AUDIT_DIR / "replays" / f"ticket-{ticket}"
        comparison = _load_json(replay_dir / "comparison.json")
        diagnostics = _load_json(replay_dir / "warnings.json")
        if comparison["result"] != "PASS" or not comparison["metrics_exact_within_1e_15"]:
            raise AssertionError(f"Ticket {ticket} clean replay did not pass")
        if diagnostics["converged"] is not True or diagnostics["warnings"]:
            raise AssertionError(f"Ticket {ticket} replay had convergence warnings")
        for split_name in ("dev", "heldout"):
            detail = comparison[split_name]
            if not all(detail[key] for key in ("same_ids_and_order", "same_y_true", "same_y_pred", "scores_match_within_1e_12")):
                raise AssertionError(f"Ticket {ticket} {split_name} archived comparison failed")
            maximum_score_difference = max(maximum_score_difference, float(detail["maximum_absolute_score_difference"]))
        pids.append(int(comparison["clean_process_pid"]))
        comparisons[ticket] = comparison
        dev_predictions[ticket] = pd.read_csv(replay_dir / "dev_predictions.csv")
        heldout_predictions[ticket] = pd.read_csv(replay_dir / "heldout_predictions.csv")
        validate_prediction_frame(dev_predictions[ticket], expected_ids=expected_dev_ids)
        validate_prediction_frame(heldout_predictions[ticket], expected_ids=expected_heldout_ids)
    if len(set(pids)) != 5:
        raise AssertionError("clean ticket replays did not use five distinct processes")

    baseline = heldout_predictions[1]
    transitions = {ticket: _transition_counts(baseline, heldout_predictions[ticket]) for ticket in range(1, 6)}
    transition_frame = pd.DataFrame(
        [{"ticket": f"ticket_{ticket}", "baseline": "ticket_1", **transitions[ticket]} for ticket in range(1, 6)],
        columns=["ticket", "baseline", "fixed_fp", "fixed_fn", "new_fp", "new_fn"],
    )
    write_csv_artifact(transition_frame, AUDIT_DIR / "transition_recalculation.csv")

    reproduced_summary = _reproduce_summary(decisions, dev_predictions, heldout_predictions, transitions)
    active_summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if list(active_summary.columns) != SUMMARY_COLUMNS or active_summary["ticket"].tolist() != [f"ticket_{i}" for i in range(1, 6)]:
        raise AssertionError("results/summary.csv violates required schema or ticket order")
    summary_comparison = _compare_frames(reproduced_summary, active_summary, name="summary.csv")
    reproduced_summary_path = write_csv_artifact(reproduced_summary, AUDIT_DIR / "reproduced_summary.csv")
    summary_comparison["byte_exact"] = sha256(reproduced_summary_path) == sha256(PROJECT_ROOT / "results" / "summary.csv")

    reproduced_sweep = _reproduce_threshold_sweep(dev_predictions[1])
    active_sweep = pd.read_csv(PROJECT_ROOT / "results" / "threshold_sweep.csv")
    if list(active_sweep.columns) != THRESHOLD_SWEEP_COLUMNS or len(active_sweep) != len(THRESHOLDS):
        raise AssertionError("results/threshold_sweep.csv violates required schema or row count")
    sweep_comparison = _compare_frames(reproduced_sweep, active_sweep, name="threshold_sweep.csv")
    reproduced_sweep_path = write_csv_artifact(reproduced_sweep, AUDIT_DIR / "reproduced_threshold_sweep.csv")
    sweep_comparison["byte_exact"] = sha256(reproduced_sweep_path) == sha256(PROJECT_ROOT / "results" / "threshold_sweep.csv")

    audit_source = _load_json(PROJECT_ROOT / "experiments" / "ticket-5" / "final_audit_records.json")
    reproduced_quality = pd.DataFrame(audit_source["records"], columns=DATA_QUALITY_AUDIT_COLUMNS)
    validate_data_quality_audit(reproduced_quality, valid_ids=set(split.all_ids))
    active_quality = pd.read_csv(PROJECT_ROOT / "results" / "data_quality_audit.csv")
    quality_comparison = _compare_frames(reproduced_quality, active_quality, name="data_quality_audit.csv")
    reproduced_quality_path = write_csv_artifact(reproduced_quality, AUDIT_DIR / "reproduced_data_quality_audit.csv")
    quality_comparison["byte_exact"] = sha256(reproduced_quality_path) == sha256(PROJECT_ROOT / "results" / "data_quality_audit.csv")

    for name, expected in manifest["result_table_inputs"].items():
        if sha256(PROJECT_ROOT / expected["path"]) != expected["sha256"]:
            raise AssertionError(f"active {name} changed after manifest creation")

    source_final = heldout_predictions[5]
    final_prediction = build_prediction_frame(
        ids=source_final["id"].tolist(),
        y_true=source_final["y_true"].tolist(),
        y_pred=source_final["y_pred"].tolist(),
        scores=source_final["score"].tolist(),
        model_name=decisions[5]["model_name"],
        ticket="ticket_5_final_frozen_decision",
    )
    write_prediction_artifact(final_prediction, FINAL_PREDICTION_PATH, expected_ids=expected_heldout_ids)

    historical_prediction_paths = [PROJECT_ROOT / decisions[ticket]["archived_heldout_predictions"] for ticket in range(1, 6)]
    prediction_paths = historical_prediction_paths + [FINAL_PREDICTION_PATH]
    inventory: list[dict[str, Any]] = []
    for path in prediction_paths:
        frame = pd.read_csv(path)
        validate_prediction_frame(frame, expected_ids=expected_heldout_ids)
        if not frame["y_true"].equals(baseline["y_true"]):
            raise AssertionError(f"held-out labels differ in {path}")
        inventory.append(_inventory_row(path, category="prediction", note="exact schema, stable IDs, labels, and order verified", rows=len(frame)))
    for path, note in (
        (PROJECT_ROOT / "results" / "summary.csv", "reproduced from five clean replay prediction sets"),
        (PROJECT_ROOT / "results" / "threshold_sweep.csv", "reproduced from raw-text baseline dev scores"),
        (PROJECT_ROOT / "results" / "data_quality_audit.csv", "reproduced from preserved structured Ticket 5 records"),
    ):
        inventory.append(_inventory_row(path, category="result_table", note=note, rows=len(pd.read_csv(path))))
    exact_hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in inventory:
        exact_hash_groups[row["sha256"]].append(row["path"])
    unexplained_byte_duplicates = [paths for paths in exact_hash_groups.values() if len(paths) > 1]
    if unexplained_byte_duplicates:
        raise AssertionError(f"unexpected byte-identical active artifacts: {unexplained_byte_duplicates}")

    core = lambda frame: frame[["id", "y_true", "y_pred", "score"]].reset_index(drop=True)
    intentional_groups = []
    for label, tickets in (("raw baseline retained", [1, 3]), ("Ticket 4 retained by Ticket 5 and final", [4, 5])):
        reference = core(heldout_predictions[tickets[0]])
        if not all(_compare_frames(reference, core(heldout_predictions[ticket]), name=label)["semantic_match"] for ticket in tickets[1:]):
            raise AssertionError(f"documented semantic duplicate group failed: {label}")
        intentional_groups.append({"reason": label, "tickets": tickets})
    _compare_frames(core(heldout_predictions[5]), core(final_prediction), name="final Ticket 5 prediction core")

    inventory_frame = pd.DataFrame(inventory, columns=["path", "category", "sha256", "rows", "status", "note"])
    write_csv_artifact(inventory_frame, AUDIT_DIR / "artifact_inventory.csv")
    ticket_metrics: dict[str, dict[str, Any]] = {}
    for ticket in range(1, 6):
        dev_metrics = metric_bundle(dev_predictions[ticket]["y_true"], dev_predictions[ticket]["y_pred"])
        heldout_metrics = metric_bundle(heldout_predictions[ticket]["y_true"], heldout_predictions[ticket]["y_pred"])
        ticket_metrics[str(ticket)] = {
            "dev_f1_target_1": dev_metrics["f1_target_1"],
            "heldout_f1_target_1": heldout_metrics["f1_target_1"],
            "heldout_accuracy": heldout_metrics["accuracy"],
            **transitions[ticket],
        }

    verification = {
        "audit": "Step 10 frozen-decision reproducibility; not the final report",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "result": "PASS",
        "manifest_path": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "all_decisions_dev_based": True,
        "selection_reopened": False,
        "clean_process_pids": pids,
        "distinct_clean_process_count": len(set(pids)),
        "maximum_score_difference": maximum_score_difference,
        "score_acceptance_tolerance": 1e-12,
        "metric_acceptance_tolerance": TOLERANCE,
        "ticket_metrics": ticket_metrics,
        "summary_comparison": summary_comparison,
        "threshold_sweep_comparison": sweep_comparison,
        "data_quality_audit_comparison": quality_comparison,
        "baseline_comparator": "ticket_1 clean held-out replay",
        "intentional_semantic_duplicate_groups": intentional_groups,
        "unexplained_exact_duplicate_groups": [],
        "stale_active_result_files": [],
        "contradictory_active_result_files": [],
        "unexplained_manual_edits": [],
        "historical_naming_hazard": "predictions/heldout_predictions.csv is Ticket 1, not the final submission",
        "final_prediction_path": FINAL_PREDICTION_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "final_prediction_sha256": sha256(FINAL_PREDICTION_PATH),
        "final_prediction_columns": PREDICTION_COLUMNS,
        "heldout_rows": len(expected_heldout_ids),
    }
    write_json_artifact(verification, AUDIT_DIR / "reproducibility_verification.json")
    write_text_artifact(_audit_markdown(verification), AUDIT_DIR / "reproducibility_audit.md")
    write_json_artifact(capture_package_versions(), AUDIT_DIR / "software_versions.json")
    exact_command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.verify_final_reproducibility", *sys.argv[1:]])
    write_text_artifact(exact_command, AUDIT_DIR / "run_command.txt")
    write_json_artifact(
        {
            "exact_command": exact_command,
            "pid": os.getpid(),
            "scope": "Step 10 verification and final prediction publication only",
            "selection_reopened": False,
            "final_report_written": False,
        },
        AUDIT_DIR / "run_config.json",
    )
    print(json.dumps({"result": "PASS", "distinct_processes": len(set(pids)), "final_prediction": str(FINAL_PREDICTION_PATH), "ticket_metrics": ticket_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
