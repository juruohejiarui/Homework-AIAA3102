# Ticket 1 - Baseline Discrepancy Diagnosis

## Hypothesis
An explicitly frozen TF-IDF + Logistic Regression baseline will be reproducible; controlled probes can bound, but may not uniquely identify, an undocumented reference gap.

## Intended Lever
Baseline specification; diagnostic probes change one factor only.

## Controlled Setup
The submitted baseline is train-only word `(1,2)`-gram TF-IDF with lowercasing, `min_df=1`, `max_df=1.0`, `max_features=40000`, sublinear TF, IDF and smooth IDF enabled, per-document L2 normalization, no accent stripping, and the default two-or-more-character token pattern. It uses `liblinear` Logistic Regression with L2 penalty, `C=1`, no class weighting, seed 3102, 1,000 maximum iterations, and threshold 0.5. The vectorizer vocabulary has 40,000 terms. Full serialized parameters are in `experiments/decisions.json`.

The diagnostic matrix keeps the split, labels, training-only fitting, threshold, and every non-named parameter fixed. It varies exactly one factor in each row: n-gram range; TF scaling; vocabulary cap; `min_df`; `max_df`; IDF, smoothing, and vector norm; accent and token handling; casing; `C`; class weighting; solver; and random state. The full 26-probe matrix is fixed in `pipeline.experiments` before held-out evaluation. After the submitted baseline was frozen in `experiments/decisions.json`, each fixed probe was scored once on held-out data. The recorded policy is explicit: held-out probe scores are forensic evidence only; they cannot replace the baseline, select a later-ticket setting, or create a held-out ranking.

## Dev Results
Floor target-1 F1: 0.000000 (it predicts no positives). Baseline F1: 0.734034; confusion matrix TN=740, FP=128, FN=201, TP=454.

## Frozen Decision
The assignment-driven baseline was frozen before held-out scoring; probes were diagnostic and did not redefine it.

## Held-Out Results
F1=0.731984, accuracy=0.782666, TN=740, FP=129, FN=202, TP=452. This misses the 0.757422 reference by 0.025438. The frozen diagnostic matrix is then evaluated once to check whether dev changes correspond to held-out changes; it does not revise the baseline. All numbers below are target-1 F1.

| Factor | Setting | Dev F1 | Held-out F1 | Dev / held-out delta |
| --- | --- | ---: | ---: | --- |
| Submitted baseline | L2-normalized `(1,2)` TF-IDF, `C=1` | 0.734034 | 0.731984 | 0.000000 / 0.000000 |
| Vector norm | `norm=None` | 0.738333 | 0.758621 | +0.004299 / +0.026637 |
| Regularization | `C=10` | 0.750982 | 0.752137 | +0.016948 / +0.020153 |
| Regularization | `C=4` | 0.745067 | 0.756078 | +0.011033 / +0.024095 |
| N-grams | unigrams only | 0.738812 | 0.751220 | +0.004778 / +0.019236 |
| Frequency pruning | `min_df=3` | 0.741176 | 0.750419 | +0.007142 / +0.018435 |
| Solver | `lbfgs` instead of `liblinear` | 0.734034 | 0.731826 | +0.000000 / -0.000158 |
| Random state | 1 or 42 instead of 3102 | 0.734034 | 0.731984 | 0.000000 / 0.000000 |

The remaining rows are negative controls rather than candidates: `(1,3)` n-grams gives dev/held-out F1 0.728455/0.732787; removing sublinear TF gives 0.732738/0.729201; disabling IDF gives 0.720949/0.723370; removing the feature cap gives 0.733709/0.721154; preserving case gives 0.717619/0.709106; and L1 document normalization gives 0.316176/0.288582. `max_df=0.9` has no observable effect on either split. The full matrix retains these facts rather than reporting only favorable configurations.

## Precision-Recall Interpretation
The baseline has precision 0.777969 and recall 0.691131 on held-out data, so its 0.5 operating point misses more positives than it admits negative tweets. Across all 26 frozen probes, dev and held-out F1 deltas have Pearson $r=0.996813$ and Spearman $\rho=0.874915$ ([results/discrepancy_association.json](results/discrepancy_association.json)). Thus dev improvements generally move in the same direction after freezing: for example, `C=10`, `C=4`, unigrams, `min_df=3`, and `norm=None` all improve on both splits, while the negative controls decline on both. These correlations are descriptive evidence only: the probes share data, baseline, and model family, so they are not independent observations and their p-values are not treated as confirmatory tests.

Two representative transition probes were fixed after dev comparison and before held-out scoring: `C=10` represents the regularization sweep, and `norm=None` represents document-vector geometry. `C=10` fixes 38 FNs and 12 FPs while adding 32 FPs and 6 FNs; it is therefore recall-oriented, not a free F1 gain. It recovers ID 244, a positive tweet about a shooting or airplane accident, but newly flags ID 117's conversational travel-delay wording. `norm=None` fixes 53 FPs and 36 FNs while adding 26 FPs and 26 FNs. It correctly rejects ID 303's non-event “annihilated” promotional wording, but newly misses ID 302's terse positive “Annihilated Abs” text. The exhaustive, stable-ID rows and counts are in `results/discrepancy_error_transitions.csv` and `results/discrepancy_transition_summary.csv`; these examples make the aggregate deltas inspectable rather than treating them as semantic proof.

## Fixed False Positives
Not applicable: this ticket establishes the comparison anchor.

## Fixed False Negatives
Not applicable: this ticket establishes the comparison anchor.

## New False Positives
Baseline FP examples include IDs 198 and 303.

## New False Negatives
Baseline FN examples include IDs 17 and 244.

## Concrete Examples with Stable IDs
ID 198 discusses the lifetime odds of an airplane accident but is not an event; ID 17 begins jokingly before reporting local flooding and is missed. These demonstrate lexical disaster terms without event semantics and informal real-event language.

## Interpretation
The result is deterministic but the submitted baseline is not reference-matching. The random-state result is a useful exclusion: seeds 1, 42, and 3102 produce identical dev and held-out metrics under `liblinear`, so seed choice is not evidence for the 0.025438 gap. The one-factor `lbfgs` probe preserves all baseline preprocessing and classifier parameters except the solver; its held-out F1 is 0.731826 versus 0.731984 for `liblinear`. Thus solver choice alone is not a plausible explanation in this environment.

The probes show that both per-document normalization and regularization materially affect the model, and their dev/held-out deltas are directionally associated after the freeze boundary. Standard L2 normalization rescales every tweet vector to unit length before Logistic Regression; removing it preserves TF-IDF magnitude and changes the feature geometry. Larger `C` weakens L2 regularization. Unigrams and `min_df=3` also change the result, showing that feature sparsity and rare phrase features matter. The evidence supports a bounded conclusion: this submission's split, label balance, train-only fitting, seed, and recorded environment are fixed; the course reference configuration and library versions are undisclosed; and no diagnostic result proves which hidden setting explains the 0.025438 gap. No probe replaced the submitted baseline or altered later ticket decisions after held-out inspection.

## Limitation
The contract omits the reference vectorizer and classifier configuration and provides no reference predictions or dev score. A single reference F1 cannot identify a unique parameter setting: different changes can produce similar aggregate F1 through different precision-recall tradeoffs. The frozen matrix's dev/held-out association supports the relevance of some implementation factors, but cannot prove causality or justify selecting a new baseline by held-out score. Confirmation would require the reference configuration, reference dev predictions, or an independently published reference implementation.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 1 --split dev`, freeze, then run with `--split heldout`.

## Artifact Paths
`results/discrepancy_comparison.csv`, `results/discrepancy_association.json`, `results/discrepancy_error_transitions.csv`, `results/discrepancy_transition_summary.csv`, `results/experiment_registry.csv`, `results/confusion_matrices.csv`, `results/environment.json`, `predictions/dev/ticket-1.csv`, `predictions/heldout_predictions.csv`.

