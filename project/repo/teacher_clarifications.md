# Instructor Clarifications

## Floor Model (Topic A)

For Topic A, the floor model is only a sanity-check baseline, not a competitive model.

It works as follows:

- Use only the training split (`train_ids`) to find the majority label.
- In our fixed split, the majority label is `target=0`.
- Predict `target=0` for every dev and held-out example.
- Since it never predicts `target=1`, its held-out F1 for `target=1` is expected to be `0.0`.

The purpose of the floor model is to verify that your data loading, split handling, prediction export, and evaluation code are wired correctly before you implement the stronger TF-IDF + Logistic Regression baseline.

---

## `new_fp` and `new_fn` Metrics in `summary.csv`

Question:

For project topic A, do the `new_fp` and `new_fn` metrics in `summary.csv` represent the difference compared to the original baseline model, or compared to the immediate previous model that this version was built upon?

Answer:

The `fixed_fp`, `fixed_fn`, `new_fp`, and `new_fn` values in `summary.csv` should be computed relative to the same frozen baseline model, not the immediate previous model.

This keeps the error transitions comparable across tickets.