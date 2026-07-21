from pathlib import Path

import pytest

from pipeline.baselines import (
    assert_identical_evaluations,
    evaluate_floor_model,
    fit_and_evaluate_reference_baseline,
    make_reference_pipeline,
)
from pipeline.data import load_labeled_tweets, select_split_by_id
from pipeline.reproducibility import (
    configure_reproducibility,
    load_reproducibility_settings,
)
from pipeline.splits import load_fixed_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"
pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="public train.csv is absent")


def _train_and_dev():
    split = load_fixed_split()
    data = load_labeled_tweets(DATA_PATH, split)
    return (
        select_split_by_id(data, split, "train"),
        select_split_by_id(data, split, "dev"),
    )


def test_floor_model_matches_instructor_definition() -> None:
    train, dev = _train_and_dev()

    evaluation = evaluate_floor_model(train, dev)

    assert evaluation.predictions["y_pred"].eq(0).all()
    assert evaluation.metrics["f1_target_1"] == 0.0
    assert evaluation.metrics["false_positive"] == 0
    assert evaluation.metrics["false_negative"] == 655


def test_reference_baseline_effective_core_defaults() -> None:
    settings = load_reproducibility_settings()
    model = make_reference_pipeline(settings)
    tfidf = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    assert tfidf.ngram_range == (1, 1)
    assert tfidf.lowercase is True
    assert tfidf.stop_words is None
    assert tfidf.max_features is None
    assert classifier.C == 1.0
    assert classifier.class_weight is None
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 100
    assert classifier.random_state == settings.seed


def test_real_reference_baseline_is_reproducible_on_dev() -> None:
    train, dev = _train_and_dev()
    settings = load_reproducibility_settings()

    configure_reproducibility(settings)
    _, first = fit_and_evaluate_reference_baseline(train, dev, settings)
    configure_reproducibility(settings)
    _, second = fit_and_evaluate_reference_baseline(train, dev, settings)

    assert_identical_evaluations(first, second)
    assert first.warnings == ()
    assert first.converged is True
