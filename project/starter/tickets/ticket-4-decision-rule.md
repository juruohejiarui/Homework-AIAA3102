# Ticket 4 - Decision Rule and Model

## Hypothesis
A limited regularization/class-weight search and dev-selected threshold can improve recall without unacceptable precision loss.

## Intended Lever
Decision rule and model capacity: Logistic Regression `C`, class weighting, threshold, and one LinearSVC alternative. The final selection is a combination; `results/decision_ablation.csv` separates its components.

## Controlled Setup
Text features and splits are fixed. `C` is in `[0.1,0.25,0.5,1,2,4,10]`, weights are `None`/`balanced`, LR thresholds are 0.20-0.80, and LinearSVC uses its decision function at zero. The general threshold table spans 0.10-0.90 by 0.01.

## Dev Results
Balanced LR with `C=10`, threshold 0.56 won at F1 0.753577. LinearSVC reached 0.752351. The selected dev confusion matrix is TN=739, FP=129, FN=181, TP=474.

## Frozen Decision
Freeze balanced LR, `C=10`, probability threshold 0.56.

## Held-Out Results
F1=0.754717, accuracy=0.795141, TN=731, FP=138, FN=174, TP=480.

## Precision-Recall Interpretation
Relative to baseline, the selected combination uses weaker L2 regularization (`C=10` rather than `C=1`), class balancing, and a higher threshold of 0.56. The threshold partially restores precision. Held-out precision is 0.776699, close to baseline precision 0.777969, while recall rises from 0.691131 to 0.733945. The dev ablation table separates the effects: threshold-only reaches F1 0.745971, `C=10` without class weighting reaches 0.752791, `C=1` with balancing reaches 0.743336, and the selected combination reaches 0.753577.

The dev-only precision-recall curve marks the frozen 0.56 operating point and makes its precision-recall trade-off visible. The companion dev-only F1-versus-threshold curve shows that this threshold is selected from the final LR configuration's dev scores, not from held-out data. These plots explain a threshold decision; they do not calibrate a deployment cost or claim that the held-out curve would have the same optimum.

## Fixed False Positives
17; examples: IDs 472 and 939.

## Fixed False Negatives
37; examples: IDs 244 and 509.

## New False Positives
26; examples: IDs 117 and 996.

## New False Negatives
9; examples: IDs 2528 and 5863.

## Concrete Examples with Stable IDs
ID 244 (“shooting or airplane accident”) is recovered; ID 117 is a conversational mention of an accident and becomes a false positive. ID 2528 is a traffic-collision alert that becomes a new miss.

## Interpretation
The operating point is preferable for target-1 F1 because 37 recovered positives outweigh nine lost positives, while the FP trade is less favorable. The ablation results show that threshold adjustment alone improves F1 but does not explain the full selected result; the improvement belongs to the selected combination rather than to one factor alone. The small dev-to-held-out difference suggests moderate rather than conclusive robustness.

## Limitation
Thresholds were optimized for F1, not calibrated costs, and the same dev set selected model and threshold.

## Reproduction Command
`python -m pipeline.cli run-ticket --ticket 4 --split dev` then freeze and held-out commands.

## Artifact Paths
`results/threshold_sweep.csv`, `results/decision_ablation.csv`, `results/ticket4_dev_decision_curve.csv`, `results/ticket4_dev_precision_recall_curve.csv`, `results/figures/ticket4_dev_precision_recall.png`, `results/figures/ticket4_dev_f1_threshold.png`, `results/experiment_registry.csv`, `results/error_transitions.csv`, `predictions/dev/ticket-4.csv`, `experiments/decisions.json`.

