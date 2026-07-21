import pandas as pd
import pytest

from pipeline.artifacts import (
    PREDICTION_COLUMNS,
    ArtifactValidationError,
    build_prediction_frame,
    validate_prediction_frame,
    write_prediction_artifact,
)


def _prediction_frame() -> pd.DataFrame:
    return build_prediction_frame(
        ids=[101, 7, 44],
        y_true=[0, 1, 1],
        y_pred=[0, 1, 0],
        scores=[0.1, 0.8, 0.4],
        model_name="synthetic_test_model",
        ticket="infrastructure_test",
    )


def test_prediction_artifact_has_expected_columns() -> None:
    frame = _prediction_frame()

    assert list(frame.columns) == PREDICTION_COLUMNS


def test_prediction_validation_rejects_missing_ids() -> None:
    frame = _prediction_frame().iloc[:-1].copy()

    with pytest.raises(ArtifactValidationError, match=r"missing=\[44\]"):
        validate_prediction_frame(frame, expected_ids=[101, 7, 44])


def test_prediction_writer_preserves_stable_ids(tmp_path) -> None:
    frame = _prediction_frame()
    destination = tmp_path / "predictions.csv"

    write_prediction_artifact(frame, destination, expected_ids=[101, 7, 44])
    restored = pd.read_csv(destination)

    assert restored["id"].tolist() == [101, 7, 44]
    assert list(restored.columns) == PREDICTION_COLUMNS
