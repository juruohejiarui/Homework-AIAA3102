# Ticket 3 - Feature and Shortcut Audit

## Hypothesis
Keyword and shallow statistics contain usable but potentially brittle benchmark signal; location alone should be weak.

## Intended Lever
Feature source only.

## Controlled Setup
Text-only, keyword-only, location-only, numeric-only, keyword+location, text+keyword, and text+numeric models use the same split, seed, classifier, and threshold. Numeric features include length, counts for words/URLs/mentions/hashtags/punctuation/digits, uppercase ratio, and repeated punctuation.

## Dev Results
F1: text 0.734034; keyword 0.659859; location 0.162730; numeric 0.527511; keyword+location 0.665635; text+keyword 0.719937; text+numeric 0.729951.

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
Keyword is mixed evidence: it is supplied task information and strongly associated with labels, but reliance on its benchmark vocabulary is demonstrably brittle. The corrected stress audit now perturbs models that actually consume metadata: blanking `keyword` takes keyword-only F1 from 0.659859 to 0.000000 (618 flips), and shuffling it reduces F1 to 0.434231 (699 flips). For `text_plus_keyword`, blanking and shuffling keyword reduce F1 from 0.719937 to 0.690009 and 0.616242. Location-only is weak (F1 0.162730) and falls to zero when blanked. These results support retaining text-only: it has the best dev F1 and does not depend on these metadata fields. Coefficients remain associations, not causal effects.

## Limitation
Blank/shuffle perturbations are synthetic interventions, not a real deployment shift; they demonstrate field reliance but cannot prove that every use of keyword is illegitimate. One-hot categories also discard semantics.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 3 --split dev` then freeze and held-out commands.

## Artifact Paths
`results/experiment_registry.csv`, `results/top_features.csv`, `results/perturbation_stress.csv`, `results/error_transitions.csv`, `predictions/dev/ticket-3.csv`.

