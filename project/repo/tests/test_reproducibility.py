import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from pipeline.modeling import make_leakage_safe_pipeline
from pipeline.reproducibility import (
    configure_estimator,
    configure_reproducibility,
    load_reproducibility_settings,
)


def test_pipeline_predictions_are_deterministic() -> None:
    settings = load_reproducibility_settings()
    train_text = [
        "quiet sunny afternoon",
        "friends having lunch",
        "river flooding homes",
        "wildfire evacuation order",
        "ordinary commute today",
        "earthquake damaged buildings",
    ]
    train_target = np.array([0, 0, 1, 1, 0, 1])
    evaluation_text = ["sunny commute", "flood evacuation", "damaged homes"]
    template = make_leakage_safe_pipeline(
        CountVectorizer(),
        SGDClassifier(loss="log_loss", max_iter=40, tol=None, shuffle=True),
    )

    configure_reproducibility(settings)
    first = configure_estimator(template, settings)
    first.fit(train_text, train_target)

    configure_reproducibility(settings)
    second = configure_estimator(template, settings)
    second.fit(train_text, train_target)

    assert isinstance(first, Pipeline)
    np.testing.assert_array_equal(
        first.predict(evaluation_text), second.predict(evaluation_text)
    )
    np.testing.assert_allclose(
        first.predict_proba(evaluation_text), second.predict_proba(evaluation_text)
    )
