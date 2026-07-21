import pandas as pd

from pipeline.normalization import NORMALIZATION_PARAMETERS
from pipeline.run_ticket2_heldout import validate_frozen_ticket2_configuration
from pipeline.ticket2 import transition_counts


def test_frozen_ticket2_configuration_matches_current_sources() -> None:
    freeze = validate_frozen_ticket2_configuration()

    assert freeze["freeze_status"] == "frozen_before_ticket2_heldout"
    assert freeze["selected_variant"] == "normalize_urls_placeholder"
    assert freeze["normalization"]["parameters"] == {"replace_urls": True}
    assert freeze["heldout_observed_at_freeze"] is False
    assert freeze["ticket2_heldout_evaluation_count_at_freeze"] == 0
    assert NORMALIZATION_PARAMETERS[freeze["selected_variant"]] == {
        "replace_urls": True
    }


def test_ticket2_transitions_are_always_relative_to_frozen_baseline() -> None:
    baseline = pd.DataFrame(
        {"id": [1, 2, 3, 4], "y_true": [0, 1, 0, 1], "y_pred": [1, 0, 0, 1]}
    )
    candidate = pd.DataFrame(
        {"id": [1, 2, 3, 4], "y_pred": [0, 1, 1, 0]}
    )

    assert transition_counts(baseline, candidate) == {
        "prediction_changes": 4,
        "fixed_fp": 1,
        "fixed_fn": 1,
        "new_fp": 1,
        "new_fn": 1,
    }
