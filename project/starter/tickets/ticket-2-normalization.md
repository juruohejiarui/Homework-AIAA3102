# Ticket 2 - Text Normalization Lever

## Hypothesis
Replacing URLs with a stable token reduces memorization of opaque URL fragments while retaining the fact that a link exists.

## Intended Lever
One normalization choice at a time: URL, mention, hashtag, casing, punctuation, or emoji handling.

## Controlled Setup
The Ticket 1 feature/model settings and threshold stay fixed. Twelve single-lever candidates are listed in `results/experiment_registry.csv`; combinations were not searched.

## Dev Results
URL replacement was highest at F1 0.737105 versus raw 0.734034. URL removal was 0.734868; hashtag removal was 0.736167; all other candidates are recorded.

## Frozen Decision
Freeze URL replacement on dev before held-out evaluation.

## Held-Out Results
F1=0.739817, accuracy=0.794485, TN=765, FP=104, FN=209, TP=445.

## Precision-Recall Interpretation
Compared with raw text, precision improved because false positives fell substantially, while recall declined because some positive URL contexts also lost discriminating fragments.

## Fixed False Positives
32; examples: IDs 353 and 386.

## Fixed False Negatives
8; examples: IDs 244 and 1596.

## New False Positives
7; examples: IDs 110 and 1272.

## New False Negatives
15; examples: IDs 237 and 390.

## Concrete Examples with Stable IDs
ID 353 is promotional “World Annihilation” content whose opaque link no longer drives a positive. ID 237 is a real airplane-accident post that becomes a new miss, showing the cost of collapsing URL fragments.

## Interpretation
The mechanism is consistent but mixed: URL replacement improves precision and net F1, yet creates more new FN than fixed FN. Perturbation results in `results/perturbation_stress.csv` show sensitivity for URLs, mentions, casing, hashtags, punctuation, and emoji.

## Limitation
The gain is small and selected on one dev split; `<URL>` behavior is represented by the tokenizer-safe `URLTOKEN` string.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 2 --split dev` then freeze and held-out commands.

## Artifact Paths
`results/experiment_registry.csv`, `results/error_transitions.csv`, `results/perturbation_stress.csv`, `predictions/dev/ticket-2.csv`, `predictions/heldout_predictions.csv`.

