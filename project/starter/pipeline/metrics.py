"""Binary target-1 metrics and exhaustive error transitions."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def labels_from_scores(scores, threshold: float = .5) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def binary_metrics(y_true, y_pred) -> dict[str, float | int]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p), "recall": float(r),
            "f1": float(f), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "predicted_positive": int(np.sum(y_pred))}


def error_transitions(ids, y_true, old_pred, new_pred, old_name="baseline", new_name="candidate") -> pd.DataFrame:
    rows = []
    for i, y, old, new in zip(ids, y_true, old_pred, new_pred):
        if old == y and new == y: category = "unchanged_correct"
        elif old != y and new != y: category = "unchanged_error"
        elif old == 1 and y == 0 and new == 0: category = "fixed_fp"
        elif old == 0 and y == 1 and new == 1: category = "fixed_fn"
        elif old == 0 and y == 0 and new == 1: category = "new_fp"
        else: category = "new_fn"
        rows.append((int(i), int(y), int(old), int(new), category, old_name, new_name))
    return pd.DataFrame(rows, columns=["id","y_true","old_pred","new_pred","category","baseline_model","candidate_model"])

