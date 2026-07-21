import json
from pathlib import Path

import pandas as pd

from pipeline.artifacts import THRESHOLD_SWEEP_COLUMNS, validate_prediction_frame
from pipeline.run_ticket4_heldout import validate_frozen_ticket4_configuration
from pipeline.splits import load_fixed_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_ticket4_configuration_matches_current_sources() -> None:
    freeze = validate_frozen_ticket4_configuration()

    assert freeze["freeze_status"] == "frozen_before_ticket4_heldout_evaluation"
    # V3 extended MODEL_SPECS with C=5 and C=10 variants; selected variant may differ
    assert freeze["selected_variant"] in {
        "lr_c1_balanced_default",
        "lr_c5_balanced_default",
        "lr_c10_balanced_default",
        "lr_c1_unweighted_tuned_threshold",
        "lr_c5_unweighted_default",
        "lr_c10_unweighted_default",
    }
    assert freeze["selected_class_weight"] in {"balanced", None}
    assert 0.0 <= freeze["selected_threshold"] <= 1.0
    assert freeze["ticket4_heldout_artifact_used_in_decision"] is False
    assert freeze["ticket4_heldout_evaluation_count_at_freeze"] == 0
    assert freeze["selection_reopening_permitted"] is False


def test_ticket4_selection_records_precision_recall_tradeoff() -> None:
    freeze = validate_frozen_ticket4_configuration()
    evidence = freeze["selected_dev_evidence"]

    assert evidence["recall_target_1"] > 0.6931297709923664
    assert evidence["precision_target_1"] < 0.7909407665505227
    assert evidence["fixed_fn"] > 0
    assert evidence["new_fp"] > 0


def test_ticket4_threshold_sweep_matches_machine_contract() -> None:
    sweep = pd.read_csv(PROJECT_ROOT / "results" / "threshold_sweep.csv")

    assert list(sweep.columns) == THRESHOLD_SWEEP_COLUMNS
    assert len(sweep) == 61
    assert sweep["ticket"].eq("ticket_4").all()
    assert sweep["threshold"].is_unique
    assert sweep["threshold"].iloc[0] == 0.2
    assert sweep["threshold"].iloc[-1] == 0.8
    assert (sweep["threshold"] == 0.5).sum() == 1


def test_ticket4_final_artifacts_are_complete_and_single_run() -> None:
    split = load_fixed_split()
    predictions = pd.read_csv(
        PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv"
    )
    validate_prediction_frame(predictions, expected_ids=list(split.heldout_ids))
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    completion = json.loads(
        (
            PROJECT_ROOT
            / "experiments"
            / "ticket-4"
            / "heldout"
            / "heldout_evaluation_completed.json"
        ).read_text(encoding="utf-8")
    )

    assert (summary["ticket"] == "ticket_4").sum() == 1
    assert completion["ticket4_heldout_evaluation_count"] == 1
    assert completion["selection_reopened"] is False
