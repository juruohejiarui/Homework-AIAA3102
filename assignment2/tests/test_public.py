import pandas as pd
import pytest

from plant_pathology.data import load_labeled_csv, load_test_csv
from plant_pathology.metrics import classification_metrics
from plant_pathology.validate_submission import validate_submission


def test_metrics_perfect_predictions() -> None:
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert all(metrics[f"recall_class_{index}"] == 1.0 for index in range(4))


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_course_manifests_and_submission(tmp_path) -> None:
    classes = ("healthy", "multiple_diseases", "rust", "scab")
    labeled = pd.DataFrame(
        [
            {"image_id": "Train_0", "healthy": 1, "multiple_diseases": 0, "rust": 0, "scab": 0},
            {"image_id": "Train_1", "healthy": 0, "multiple_diseases": 0, "rust": 1, "scab": 0},
        ]
    )
    labeled_path = tmp_path / "train.csv"
    labeled.to_csv(labeled_path, index=False)
    assert len(load_labeled_csv(labeled_path)) == 2

    test_path = tmp_path / "test.csv"
    pd.DataFrame({"image_id": ["Train_2", "Train_3"]}).to_csv(test_path, index=False)
    assert len(load_test_csv(test_path)) == 2

    submission = pd.DataFrame(
        [
            ["Train_2", 0.1, 0.2, 0.6, 0.1],
            ["Train_3", 0.7, 0.1, 0.1, 0.1],
        ],
        columns=["image_id", *classes],
    )
    submission_path = tmp_path / "submission.csv"
    submission.to_csv(submission_path, index=False)
    validate_submission(test_path, submission_path)
