from pipeline.run_ticket1_heldout import (
    DEFAULT_CONTRACT_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_FREEZE_PATH,
    compare_contract,
    validate_frozen_configuration,
)
from pipeline.splits import DEFAULT_SPLIT_PATH


def test_frozen_ticket1_configuration_matches_current_sources() -> None:
    freeze = validate_frozen_configuration(
        DEFAULT_FREEZE_PATH,
        data_path=DEFAULT_DATA_PATH,
        split_path=DEFAULT_SPLIT_PATH,
        contract_path=DEFAULT_CONTRACT_PATH,
    )
    assert freeze["freeze_status"] == "frozen_before_heldout"
    assert freeze["heldout_observed_at_freeze"] is False


def test_contract_comparison_respects_tolerance() -> None:
    reference = 0.7574221578566256
    assert compare_contract(reference + 0.0009, reference, 0.001)["matches_reference"] is True
    assert compare_contract(reference + 0.0011, reference, 0.001)["matches_reference"] is False
