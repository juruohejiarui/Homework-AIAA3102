import json
from pathlib import Path

import pandas as pd

from pipeline.artifacts import DATA_QUALITY_AUDIT_COLUMNS, DATA_QUALITY_DISPOSITIONS, validate_prediction_frame
from pipeline.data_quality import validate_data_quality_audit
from pipeline.run_ticket5_heldout import validate_frozen_ticket5_configuration
from pipeline.splits import load_fixed_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_ticket5_configuration_matches_current_sources() -> None:
    freeze = validate_frozen_ticket5_configuration()

    assert freeze["freeze_status"] == "frozen_before_ticket5_heldout_reporting_and_audit"
    assert freeze["selected_variant"] == "lr_c1_balanced_default"
    assert freeze["training_label_corrections_adopted"] is False
    assert freeze["source_dataset_modified"] is False
    assert freeze["ticket5_heldout_artifact_used_in_decision"] is False
    assert freeze["ticket5_heldout_reporting_count_at_freeze"] == 0
    assert freeze["heldout_label_modification_permitted"] is False
    assert freeze["heldout_row_removal_permitted"] is False


def test_ticket5_rejects_controlled_correction_probe() -> None:
    selection = json.loads(
        (
            PROJECT_ROOT
            / "experiments"
            / "ticket-5"
            / "dev"
            / "correction_experiment"
            / "selection_result.json"
        ).read_text(encoding="utf-8")
    )

    assert selection["source_dataset_modified"] is False
    assert selection["dev_labels_modified"] is False
    assert selection["heldout_rows_loaded"] == 0
    assert selection["adopt_corrected_training_model"] is False
    assert selection["noninferiority_pass"] is False
    assert selection["error_balance_pass"] is False


def test_ticket5_required_audit_is_valid_and_complete() -> None:
    split = load_fixed_split()
    audit = pd.read_csv(PROJECT_ROOT / "results" / "data_quality_audit.csv")

    assert list(audit.columns) == DATA_QUALITY_AUDIT_COLUMNS
    assert len(audit) == 64
    assert audit["id"].is_unique
    assert set(audit["disposition"]) == DATA_QUALITY_DISPOSITIONS
    validate_data_quality_audit(audit, valid_ids=set(split.all_ids))


def test_ticket5_final_artifacts_preserve_heldout() -> None:
    split = load_fixed_split()
    predictions = pd.read_csv(
        PROJECT_ROOT / "predictions" / "ticket-5-heldout-predictions.csv"
    )
    validate_prediction_frame(predictions, expected_ids=list(split.heldout_ids))
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    completion = json.loads(
        (
            PROJECT_ROOT
            / "experiments"
            / "ticket-5"
            / "heldout"
            / "heldout_evaluation_completed.json"
        ).read_text(encoding="utf-8")
    )

    assert (summary["ticket"] == "ticket_5").sum() == 1
    assert completion["ticket5_heldout_reporting_count"] == 1
    assert completion["new_model_fit"] is False
    assert completion["heldout_labels_modified"] is False
    assert completion["heldout_rows_removed"] == 0
    assert completion["selection_reopened"] is False
