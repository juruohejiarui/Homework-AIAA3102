from pipeline.run_ticket3_heldout import validate_frozen_ticket3_configuration


def test_frozen_ticket3_configuration_matches_current_sources() -> None:
    freeze = validate_frozen_ticket3_configuration()

    assert freeze["freeze_status"] == "frozen_before_ticket3_heldout_reporting"
    assert freeze["selected_variant"] == "raw_text_tfidf_logistic_regression"
    assert freeze["ticket3_heldout_artifact_used_in_decision"] is False
    assert freeze["ticket3_heldout_reporting_count_at_freeze"] == 0
    assert freeze["selection_reopening_permitted"] is False


def test_ticket3_rejects_best_visible_shortcut_candidate() -> None:
    freeze = validate_frozen_ticket3_configuration()
    rejected = freeze["rejected_best_visible_candidate"]

    assert rejected["f1_delta_vs_baseline"] > 0
    assert rejected["keyword_masked_f1"] < freeze["selected_dev_evidence"]["f1_target_1"]
    assert freeze["heldout_reporting_mode"].startswith("Reuse")
