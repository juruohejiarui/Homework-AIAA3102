# Ticket 3 - Feature and Shortcut Audit

## Hypothesis
Keyword and shallow statistics contain usable but potentially brittle benchmark signal; location alone should be weak.

## Intended Lever
Feature source only.

## Controlled Setup
Text-only, keyword-only, location-only, numeric-only, keyword+location, text+keyword, and text+numeric models use the same split, seed, classifier, and threshold. Numeric features include length, counts for words/URLs/mentions/hashtags/punctuation/digits, uppercase ratio, and repeated punctuation.

## Dev Results
F1: text 0.734034; keyword 0.659859; location 0.162730; numeric 0.527511; keyword+location 0.665635; text+keyword 0.720949; text+numeric 0.729951.

## Frozen Decision
Retain text-only because it was the highest dev model; do not add shortcut features.

## Held-Out Results
F1=0.731984 and accuracy=0.782666, identical to the frozen baseline.

## Precision-Recall Interpretation
The retained model keeps baseline precision 0.777969 and recall 0.691131. Keyword-only predicts meaningful signal but loses contextual precision/recall balance.

## Fixed False Positives
0 for the frozen model relative to baseline.

## Fixed False Negatives
0 for the frozen model relative to baseline.

## New False Positives
0 for the frozen model relative to baseline.

## New False Negatives
0 for the frozen model relative to baseline.

## Concrete Examples with Stable IDs
Alternative shortcut models change examples such as IDs 86 and 132 (fixed FP) but also IDs 25 and 36 (new FP); the exhaustive candidate-level rows are in `results/error_transitions.csv`.

## Interpretation
Keyword is mixed evidence: it is supplied task information and strongly associated with labels, but reliance on its benchmark vocabulary may not transfer. Location is weak and noisy; coefficients are associations, not causal effects.

## Limitation
Blank/shuffle perturbations cannot fully simulate real distribution shift, and one-hot categories discard semantics.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 3 --split dev` then freeze and held-out commands.

## Artifact Paths
`results/experiment_registry.csv`, `results/top_features.csv`, `results/perturbation_stress.csv`, `results/error_transitions.csv`, `predictions/dev/ticket-3.csv`.

