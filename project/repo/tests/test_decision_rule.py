import numpy as np

from pipeline.decision_rule import (
    MODEL_SPECS,
    THRESHOLDS,
    predictions_at_threshold,
    select_best_threshold,
)


def test_threshold_grid_is_bounded_complete_and_contains_default() -> None:
    assert len(THRESHOLDS) == 61
    assert THRESHOLDS[0] == 0.2
    assert THRESHOLDS[-1] == 0.8
    assert 0.5 in THRESHOLDS
    assert len(set(THRESHOLDS)) == len(THRESHOLDS)


def test_threshold_rule_is_inclusive() -> None:
    actual = predictions_at_threshold(np.array([0.49, 0.5, 0.51]), 0.5)
    np.testing.assert_array_equal(actual, np.array([0, 1, 1]))


def test_model_grid_is_bounded_and_has_one_second_classifier() -> None:
    # V3 extends the grid with C=5 and C=10 variants (both unweighted and balanced),
    # motivated by V1 parallel study findings; total is 11 variants
    assert len(MODEL_SPECS) == 11
    assert sum(spec.classifier == "linear_svc" for spec in MODEL_SPECS) == 1
    assert {spec.c for spec in MODEL_SPECS if spec.intended_lever == "regularization"} == {
        0.25,
        0.5,
        2.0,
        4.0,
        5.0,
        10.0,
    }
    assert sum(spec.class_weight == "balanced" for spec in MODEL_SPECS) >= 1


def test_threshold_selection_uses_predeclared_tie_break() -> None:
    rows = [
        {"threshold": 0.4, "f1_target_1": 0.75},
        {"threshold": 0.6, "f1_target_1": 0.75},
        {"threshold": 0.5, "f1_target_1": 0.74},
    ]
    assert select_best_threshold(rows)["threshold"] == 0.4
