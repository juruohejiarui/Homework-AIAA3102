# Ticket 5 - Data Quality and Error Audit

## Hypothesis
Duplicates, conflicting labels, near duplicates, ambiguity, and confident model errors expose benchmark limits, but model disagreement alone is insufficient to alter labels.

## Intended Lever
Audit disposition only; no original data mutation.

## Controlled Setup
Exact text, explicitly normalized duplicate keys (lowercase, URL/user tokens, punctuation removal, whitespace collapse), character 3-5 gram TF-IDF cosine near duplicates, and high-confidence held-out errors from the frozen Ticket 4 model.

## Dev Results
The inherited audit model has dev F1 0.753577. Audit evidence did not justify a training-label-correction experiment.

## Frozen Decision
Keep all original labels; record findings as `keep_but_flag`, `ambiguous`, or rejected false-positive findings as appropriate. No held-out label or row was modified.

## Held-Out Results
The unchanged frozen model has F1 0.755311 and accuracy 0.795798. The audit contains 1,687 evidence rows.

## Precision-Recall Interpretation
High-confidence FP and FN rows show that confidence does not resolve semantic ambiguity. Precision/recall values remain Ticket 4's because the audit is analytical.

## Fixed False Positives
17 relative to Ticket 1, inherited from Ticket 4; IDs 472 and 939 are examples.

## Fixed False Negatives
37; examples: IDs 244 and 509.

## New False Positives
25; examples: IDs 117 and 996.

## New False Negatives
9; examples: IDs 2528 and 5863.

## Concrete Examples with Stable IDs
Duplicate and related IDs, similarity, original labels, model scores, evidence, and rationale are recorded per row in `results/data_quality_audit.csv`. Conflicting-label groups receive `ambiguous`, never automatic `fix`.

## Interpretation
Exact/normalized duplicates are dataset artifacts worth flagging; near duplicates with consistent labels are useful redundancy, while conflicts are ambiguous. Confident errors are candidates for human review, not proof of mislabeling.

## Limitation
The near-duplicate threshold (0.86) and character representation are heuristic, and this work did not include independent human annotation.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 5 --split dev` then freeze and held-out commands.

## Artifact Paths
`results/data_quality_audit.csv`, `results/error_transitions.csv`, `predictions/heldout_predictions.csv`, `experiments/decisions.json`.

