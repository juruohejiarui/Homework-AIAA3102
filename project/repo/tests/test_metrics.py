from pipeline.metrics import confusion_matrix_counts, metric_bundle


def test_binary_metric_bundle_uses_target_1_as_positive() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    metrics = metric_bundle(y_true, y_pred)
    counts = confusion_matrix_counts(y_true, y_pred)

    assert metrics["precision_target_1"] == 0.5
    assert metrics["recall_target_1"] == 0.5
    assert metrics["f1_target_1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert counts.as_dict() == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
