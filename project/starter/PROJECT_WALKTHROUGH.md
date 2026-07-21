# Project Implementation Walkthrough

This is a technical implementation walkthrough, not the formal project report.

## Repository and data flow

`pipeline/` contains configuration, loading/validation, pure normalization functions, train-fitted feature builders, metrics/transitions, data-quality analysis, stable artifact writers, experiments, and the CLI. `tests/` contains 60 tests. `tickets/` records evidence for the five investigations. `results/`, `predictions/`, and `experiments/decisions.json` are machine-checkable outputs.

The public labeled CSV is downloaded from the starter-specified ucbrise GitHub mirror into `data/train.csv`. `pipeline.data` validates its five columns and uses the Kaggle `id` values in `data/split_indices.json`. Validation confirmed 4,567/1,523/1,523 rows and 1,962/655/654 positives for train/dev/held-out; IDs are unique, splits are disjoint, and their union is all 7,613 rows. No split was regenerated.

The complete flow is CSV -> fixed ID assignment -> integrity checks -> train-only feature fitting -> train-only model fitting -> dev scoring/selection -> pending JSON decision -> explicit frozen JSON decision -> held-out scoring -> transitions and stable CSV artifacts. `run-ticket --split dev` constructs only train/dev partitions and cannot dispatch to held-out scoring; `freeze-ticket` promotes its pending record; and held-out commands require the frozen record. Every feature builder records its fitted IDs, and tests prove that dev and held-out IDs are absent.

## Tickets and frozen variables

Ticket 1 fixed a text-only word `(1,2)`-gram TF-IDF (sublinear TF, IDF, smooth IDF, 40,000 features, L2 document normalization) plus `liblinear` Logistic Regression (`C=1`, threshold 0.5). It produced dev F1 0.734034 and held-out F1 0.731984, below the 0.757422 reference. A fixed one-factor matrix was evaluated once after the baseline freeze; its 26 dev/held-out F1 deltas have Pearson $r=0.996813$ and Spearman $\rho=0.874915$. This verifies directional agreement for diagnostic factors without allowing a probe to replace the baseline or select a later ticket. The probes establish sensitivity to regularization and vector geometry, but cannot identify the undocumented reference configuration.

Ticket 2 changed one normalization variable at a time. URL replacement was frozen from dev F1 0.738155 and produced held-out F1 0.739817; its selected-model stress probe is invariant to URL replacement but not punctuation removal. Ticket 3 compared text, keyword, location, shallow statistics, and specified combinations. Text-only was retained on dev; keyword-only is strong (0.659859 dev F1) but collapses to 0 when keyword is blanked, so the keyword signal is classified as mixed benchmark/task information rather than causal evidence. Ticket 4 compared a limited Logistic Regression grid, thresholds, and LinearSVC. Balanced Logistic Regression with `C=10` and threshold 0.56 was frozen at dev F1 0.753577 and yielded held-out F1 0.754717. Ticket 5 retained original labels: duplicate and high-confidence-error evidence was insufficient for automatic correction.

Each controlled experiment changes the variable named by `experiment_type` in `results/experiment_registry.csv`; normalization candidates hold model/features constant, feature candidates hold training and classifier settings constant, and Ticket 4 holds text features constant while changing `C`, class weighting, family, or threshold. Decisions are in `experiments/decisions.json` and were not revised after held-out inspection.

## Artifacts and reproduction

Run `python -m pipeline.cli run-all`, then `python -m pipeline.cli validate-artifacts`. Important paths are:

- `predictions/heldout_predictions.csv`: 7,615 stable ticket/ID rows with continuous scores.
- `predictions/dev/ticket-*.csv`: dev prediction evidence.
- `results/summary.csv`: five reconstructed summaries.
- `results/confusion_matrices.csv`, `results/error_transitions.csv`: aggregate and example-level errors.
- `results/threshold_sweep.csv`: 81 thresholds with precision, recall, F1, and confusion counts.
- `results/decision_ablation.csv`: dev-only threshold, class-weight, and regularization comparisons for Ticket 4.
- `results/perturbation_stress.csv`: superficial-text and metadata robustness.
- `results/top_features.csv`: linear associations, not causal effects.
- `results/data_quality_audit.csv`: 1,688 evidence rows using the allowed dispositions.
- `results/discrepancy_comparison.csv`, `results/experiment_registry.csv`, `results/environment.json`.

Commands actually executed successfully with Python 3.12.9 were `python -m pipeline.cli validate-data`, `python -m pipeline.cli run-all`, `python -m pipeline.cli validate-artifacts`, and `python -m pytest -q`. The artifact validator reconstructs every summary F1, accuracy, and all four error-transition counts directly from predictions using the frozen Ticket 1 baseline.

## Tests, limitations, and unresolved issues

The 60 tests cover split counts/balance/integrity, train/dev-only loading, malformed inputs, every normalization lever, configurable TF-IDF diagnostics, metrics and thresholds, exhaustive transitions, duplicate/near-duplicate validity, dispositions and schemas, train-only fitting, deterministic transformations, phase-separated CLI behavior, freeze enforcement, a small end-to-end fixture, score reconstruction, stable ordering, artifact reconstruction, frozen-baseline transition reconstruction, the rejected audit finding, frozen diagnostic-probe held-out evidence, representative probe transition artifacts, dev-only Ticket 4 decision-curve artifacts, and ticket-specific evidence exports. The full run is CPU-only and deterministic.

Limitations: dev selection uncertainty is substantial; metadata associations may not transfer beyond Kaggle; near-duplicate cosine thresholds are heuristic; and model disagreement is not proof of a mislabeled row. The baseline-reference gap remains partially unresolved because the contract does not specify the reference vectorizer/model parameters. The controlled probes show n-grams matter but do not prove the reference configuration. No raw labels were changed. The formal report and a truthful AI-work record are provided separately.
