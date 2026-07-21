import json
from pathlib import Path

import pandas as pd

from pipeline.artifacts import PREDICTION_COLUMNS, SUMMARY_COLUMNS, validate_prediction_frame
from pipeline.splits import load_fixed_split
from pipeline.verify_final_reproducibility import _transition_counts


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_contains_five_preheldout_dev_decisions() -> None:
    manifest = json.loads((ROOT / "configs" / "frozen_decisions.json").read_text(encoding="utf-8"))
    assert [item["ticket"] for item in manifest["decisions"]] == [1, 2, 3, 4, 5]
    for item in manifest["decisions"]:
        assert "dev" in item["decision_basis_split"].lower()
        assert "heldout" not in item["decision_basis_split"].lower()
        assert item["heldout_count_at_freeze"] == 0
        assert item["heldout_used_for_selection"] is False
        assert item["selection_reopening_permitted"] is False
        assert len(item["freeze_sha256"]) == 64


def test_clean_replays_pass_in_five_distinct_processes() -> None:
    audit = json.loads(
        (ROOT / "experiments" / "final-reproducibility-audit" / "reproducibility_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["result"] == "PASS"
    assert audit["all_decisions_dev_based"] is True
    assert audit["selection_reopened"] is False
    assert audit["distinct_clean_process_count"] == 5
    assert len(set(audit["clean_process_pids"])) == 5
    assert audit["maximum_score_difference"] <= audit["score_acceptance_tolerance"]


def test_final_prediction_has_exact_schema_and_stable_heldout_ids() -> None:
    manifest = json.loads((ROOT / "configs" / "frozen_decisions.json").read_text(encoding="utf-8"))
    split = load_fixed_split(ROOT / manifest["split_path"])
    final = pd.read_csv(ROOT / "predictions" / "final-heldout-predictions.csv")
    assert list(final.columns) == PREDICTION_COLUMNS
    validate_prediction_frame(final, expected_ids=list(split.heldout_ids))
    assert final["ticket"].eq("ticket_5_final_frozen_decision").all()


def test_summary_transitions_recalculate_against_ticket1() -> None:
    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    assert list(summary.columns) == SUMMARY_COLUMNS
    baseline = pd.read_csv(
        ROOT / "experiments" / "final-reproducibility-audit" / "replays" / "ticket-1" / "heldout_predictions.csv"
    )
    for ticket in range(1, 6):
        candidate = pd.read_csv(
            ROOT
            / "experiments"
            / "final-reproducibility-audit"
            / "replays"
            / f"ticket-{ticket}"
            / "heldout_predictions.csv"
        )
        observed = _transition_counts(baseline, candidate)
        expected = summary.loc[summary["ticket"] == f"ticket_{ticket}"].iloc[0]
        assert observed == {name: int(expected[name]) for name in ("fixed_fp", "fixed_fn", "new_fp", "new_fn")}


def test_all_three_active_result_tables_are_byte_reproduced() -> None:
    verification = json.loads(
        (ROOT / "experiments" / "final-reproducibility-audit" / "reproducibility_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert verification["summary_comparison"]["byte_exact"] is True
    assert verification["threshold_sweep_comparison"]["byte_exact"] is True
    assert verification["data_quality_audit_comparison"]["byte_exact"] is True
