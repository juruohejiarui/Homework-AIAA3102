# Project Implementation Walkthrough

This is a technical implementation walkthrough, not the formal project report.

## Repository and data flow

`pipeline/` contains configuration, loading/validation, pure normalization functions, train-fitted feature builders, metrics/transitions, data-quality analysis, stable artifact writers, experiments, and the CLI. `tests/` contains 50 tests. `tickets/` records evidence for the five investigations. `results/`, `predictions/`, and `experiments/decisions.json` are machine-checkable outputs.

The public labeled CSV is downloaded from the starter-specified ucbrise GitHub mirror into `data/train.csv`. `pipeline.data` validates its five columns and uses the Kaggle `id` values in `data/split_indices.json`. Validation confirmed 4,567/1,523/1,523 rows and 1,962/655/654 positives for train/dev/held-out; IDs are unique, splits are disjoint, and their union is all 7,613 rows. No split was regenerated.

The complete flow is CSV -> fixed ID assignment -> integrity checks -> train-only feature fitting -> train-only model fitting -> dev scoring/selection -> frozen JSON decision -> held-out scoring -> transitions and stable CSV artifacts. Every feature builder records its fitted IDs, and tests prove that dev and held-out IDs are absent.

## Tickets and frozen variables

Ticket 1 fixed a text-only word `(1,2)`-gram TF-IDF (`sublinear_tf=True`, maximum 40,000 features) plus `liblinear` Logistic Regression (`C=1`, threshold 0.5). It produced dev F1 0.734034 and held-out F1 0.731984, below the 0.757422 reference. One-factor diagnostics identify n-gram choice as the largest tested source: unigrams raised held-out F1 to 0.751220, whereas changing only the solver was effectively neutral. The discrepancy is reported, not tuned away.

Ticket 2 changed one normalization variable at a time. URL replacement was frozen from dev F1 0.737105 and produced held-out F1 0.739817. Ticket 3 compared text, keyword, location, shallow statistics, and specified combinations. Text-only was retained on dev; keyword-only was surprisingly strong (0.659859 dev F1), so keyword signal is classified as mixed benchmark/task information rather than causal evidence. Ticket 4 compared a limited Logistic Regression grid, thresholds, and LinearSVC. Balanced Logistic Regression with `C=10` and threshold 0.56 was frozen at dev F1 0.753577 and yielded held-out F1 0.755311. Ticket 5 retained original labels: duplicate and high-confidence-error evidence was insufficient for automatic correction.

Each controlled experiment changes the variable named by `experiment_type` in `results/experiment_registry.csv`; normalization candidates hold model/features constant, feature candidates hold training and classifier settings constant, and Ticket 4 holds text features constant while changing `C`, class weighting, family, or threshold. Decisions are in `experiments/decisions.json` and were not revised after held-out inspection.

## Artifacts and reproduction

Run `python -m pipeline.cli run-all`, then `python -m pipeline.cli validate-artifacts`. Important paths are:

- `predictions/heldout_predictions.csv`: 7,615 stable ticket/ID rows with continuous scores.
- `predictions/dev/ticket-*.csv`: dev prediction evidence.
- `results/summary.csv`: five reconstructed summaries.
- `results/confusion_matrices.csv`, `results/error_transitions.csv`: aggregate and example-level errors.
- `results/threshold_sweep.csv`: 81 thresholds with precision, recall, F1, and confusion counts.
- `results/perturbation_stress.csv`: superficial-text and metadata robustness.
- `results/top_features.csv`: linear associations, not causal effects.
- `results/data_quality_audit.csv`: 1,687 evidence rows using the allowed dispositions.
- `results/discrepancy_comparison.csv`, `results/experiment_registry.csv`, `results/environment.json`.

Commands actually executed successfully were `python -m pipeline.cli validate-data`, `python -m pipeline.cli run-all`, `python -m pipeline.cli validate-artifacts`, and `python -m pytest -q`. The artifact validator reconstructed every summary F1 and accuracy directly from predictions.

## Tests, limitations, and unresolved issues

The 50 tests cover split counts/balance/integrity, malformed inputs, every normalization lever, metrics and thresholds, exhaustive transitions, duplicate/near-duplicate validity, dispositions and schemas, train-only fitting, deterministic transformations, CLI behavior, freeze enforcement, a small end-to-end fixture, score reconstruction, stable ordering, and artifact reconstruction. The full run is CPU-only and deterministic.

Limitations: dev selection uncertainty is substantial; metadata associations may not transfer beyond Kaggle; near-duplicate cosine thresholds are heuristic; and model disagreement is not proof of a mislabeled row. The baseline-reference gap remains partially unresolved because the contract does not specify the reference vectorizer/model parameters. The controlled probes make n-gram configuration the most plausible tested source, but do not prove it. No raw labels were changed. `report.pdf` and `logs/chat.md` were intentionally not created.
