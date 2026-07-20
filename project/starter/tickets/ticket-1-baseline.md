# Ticket 1 - Baseline Discrepancy Diagnosis

## Hypothesis
An explicitly frozen TF-IDF + Logistic Regression baseline will be reproducible, and a parameter mismatch can explain most of any reference gap.

## Intended Lever
Baseline specification; diagnostic probes change one factor only.

## Controlled Setup
Train-only word `(1,2)`-gram TF-IDF, sublinear TF, 40,000-feature cap; `liblinear` Logistic Regression, `C=1`, seed 3102, 1,000 iterations, threshold 0.5. Full parameters are in `experiments/decisions.json`.

## Dev Results
Floor target-1 F1: 0.000000 (it predicts no positives). Baseline F1: 0.734034; confusion matrix TN=740, FP=128, FN=201, TP=454.

## Frozen Decision
The assignment-driven baseline was frozen before held-out scoring; probes were diagnostic and did not redefine it.

## Held-Out Results
F1=0.731984, accuracy=0.782666, TN=740, FP=129, FN=202, TP=452. This misses the 0.757422 reference by 0.025438. Unigrams-only reached 0.751220; changing only to `lbfgs` reached 0.731826. N-gram configuration is the most plausible tested source, with an unresolved residual gap.

## Precision-Recall Interpretation
Precision is 0.777969 and recall is 0.691131. The conservative 0.5 operating point misses more positives than it admits negative tweets.

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
The result is deterministic but not reference-matching. The controlled probes support parameter specification, especially n-grams, as a meaningful discrepancy source; they do not justify choosing the closest held-out score.

## Limitation
The contract omits the exact reference vectorizer/classifier configuration, so the remaining 0.0062 gap after the unigram probe cannot be conclusively attributed.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 1 --split dev`, freeze, then run with `--split heldout`.

## Artifact Paths
`results/discrepancy_comparison.csv`, `results/confusion_matrices.csv`, `results/environment.json`, `predictions/dev/ticket-1.csv`, `predictions/heldout_predictions.csv`.

