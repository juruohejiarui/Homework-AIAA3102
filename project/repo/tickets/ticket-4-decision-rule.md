# Ticket 4 — Decision Rule and Model

## Required evidence map

The ticket's **hypothesis** is decomposed under "Predeclared hypotheses." Its **intended lever** is one bounded decision-rule or classifier change: threshold, class weighting, regularization strength, or LinearSVC. The **controlled setup** is defined under "Scope and controls" and "Exact controlled configuration." **Dev evidence** appears under "Bounded dev experiment" and "Threshold sweep and precision-recall tradeoff." The **frozen decision** is timestamped under "Freeze chronology." **Held-out evidence** appears only under "Final held-out evidence." **Concrete prediction changes** are listed by stable ID under "Concrete dev prediction changes" and in the held-out transition discussion. The **interpretation** is given in the precision-recall analysis and "Conclusion," and the **limitation** is explicit under "Limitation." All transition counts use Ticket 1 as the common comparator.

## Required question and answer

Can the operating point of the frozen raw-text TF-IDF classifier be improved through threshold tuning, class weighting, Logistic Regression regularization, or one second CPU-compatible classifier?

Yes, but only narrowly and with an important precision-recall tradeoff. On dev, balanced Logistic Regression with `C=1.0` and the default probability threshold `0.50` achieved the highest target-1 F1 among the bounded, predeclared candidates: `0.7520849128127369`, compared with `0.7388120423108218` for the frozen baseline. It raised recall from `0.6931297709923664` to `0.7572519083969466`, while precision fell from `0.7909407665505227` to `0.7469879518072289`. It fixed 42 baseline false negatives but created 48 new false positives on dev.

That dev-only decision was frozen before Ticket 4 held-out access. On held-out, the frozen candidate achieved F1 `0.7505686125852918`, versus `0.749185667752443` for the original baseline: a small increase of `0.0013829448328488425`. Recall increased by `0.05351681957186549`, precision decreased by `0.05703282596735737`, and accuracy decreased by `0.013788575180564644`. It fixed 35 baseline false negatives and created 56 new false positives. Therefore the operating point did move in the intended recall-oriented direction, but its aggregate F1 benefit is small and should not be described as a general model-quality improvement.

## Scope and controls

Ticket 4 changed only the classifier decision rule or classifier configuration. It did not adopt Ticket 2 normalization, Ticket 3 metadata, keyword, location, length, or shallow features. Every model used only the raw `text` column and the same default `TfidfVectorizer` representation as the frozen Ticket 1 baseline.

The fixed split in `starter/data/split_indices.json` was loaded by stable Kaggle `id`; row positions were never used for membership. The vectorizer and classifier were fitted on the 4,567 `train_ids` rows. All threshold, class-weight, `C`, and classifier choices used the 1,523 `dev_ids` rows only. The dev command recorded `heldout_rows_loaded=0` and `heldout_evaluations_run=0`. The selected model, full effective hyperparameters, threshold, source hashes, dev evidence, and selection criterion were then saved in `experiments/ticket-4/frozen_decision.json`. Only after that freeze was the 1,523-row held-out partition loaded for one Ticket 4 evaluation.

All error transitions in this ticket compare directly against the same frozen Ticket 1 raw-text baseline, as required by `teacher_clarifications.md`. They do not compare against Ticket 2 or Ticket 3.

## Predeclared hypotheses

### Threshold tuning

Hypothesis: the baseline threshold of `0.50` may be conservative for target 1. A modestly lower threshold should convert borderline negative predictions into positives, recover some false negatives, and increase recall. Because the same monotonic scores are thresholded, lowering the threshold cannot fix an existing false positive or create a new false negative; its cost must be new false positives. Raising the threshold should produce the reverse pattern: improved precision and fewer false positives, at the cost of more false negatives and lower recall.

The sweep was fixed before execution at 61 inclusive thresholds from `0.20` through `0.80` in steps of `0.01`. The prediction rule was `target=1` when the baseline Logistic Regression class-1 probability was greater than or equal to the tested threshold. No adaptive refinement around the observed maximum was performed.

### Class weighting

Hypothesis: `class_weight='balanced'` should increase the influence of the less frequent positive class during fitting. It should increase recall, but could lower precision by shifting borderline negatives toward target 1. This was tested independently at `C=1.0` and threshold `0.50`; the class-weighted model was not given its own threshold sweep.

### Logistic Regression regularization

Hypothesis: stronger regularization may suppress rare but useful disaster terms and lose recall, while weaker regularization may separate harder cases but amplify noisy lexical associations. The finite logarithmic-style range was `C ∈ {0.25, 0.5, 1.0, 2.0, 4.0}`, always with `class_weight=None`, the unchanged TF-IDF representation, and threshold `0.50`. The `C=1.0` member was the control. This was not an open-ended hyperparameter search.

### Second CPU classifier

Hypothesis: a linear maximum-margin model may find a different boundary over the same sparse TF-IDF vectors. `LinearSVC(C=1.0, class_weight=None, dual='auto', random_state=3102)` was selected because it is deterministic, CPU-compatible, appropriate for high-dimensional sparse text, and changes the learning objective without changing the representation. It was evaluated at its native decision threshold `0.0`. No SVM hyperparameter or threshold search was conducted.

## Exact controlled configuration

The shared TF-IDF configuration was:

- raw `text` only;
- `analyzer='word'`;
- `lowercase=True`;
- `ngram_range=(1, 1)`;
- `min_df=1`, `max_df=1.0`, and `max_features=None`;
- `token_pattern='(?u)\\b\\w\\w+\\b'`;
- `norm='l2'`, `use_idf=True`, `smooth_idf=True`, and `sublinear_tf=False`;
- no stop words, custom tokenizer, preprocessor, accent stripping, or vocabulary;
- `dtype=float64`.

The selected classifier was `LogisticRegression` with:

- `C=1.0`;
- `class_weight='balanced'`;
- effective L2 behavior (`l1_ratio=0.0` under the scikit-learn 1.9.0 deprecated-penalty representation);
- `solver='lbfgs'`;
- `max_iter=100`;
- `tol=0.0001`;
- `fit_intercept=True`;
- `random_state=3102`;
- `n_jobs=None` in the estimator, with project-level CPU thread limits set to one;
- probability threshold `0.50`, applied inclusively.

It converged in 10 iterations on dev and 10 iterations in the final held-out fit, with no convergence warnings.

## Bounded dev experiment

The pre-execution plan is `experiments/ticket-4/dev/experiment_plan.json`. It specifies exactly 61 thresholds, one non-default class-weight setting, five total `C` values including the control, and one second classifier. The raw control was independently refitted and reproduced all frozen baseline dev labels and predictions exactly. Its saved probabilities differed from the earlier CSV only at floating-point roundoff: maximum absolute difference `1.1102230246251565e-16`, with no differences exceeding `1e-12` and no prediction changes.

The selection criterion was declared before execution: maximize target-1 dev F1 across the bounded candidates. The threshold sweep applied only to the unweighted `C=1.0` baseline scores. Class weighting, regularization, and the second classifier were evaluated independently at their native default rules. If F1 tied within `1e-12`, choose the candidate with fewer departures from the frozen baseline, then the threshold closest to `0.50`, then the lexicographically earlier name. No tie occurred at the maximum.

### Model and operating-point results

| Candidate | Lever | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP | Changes | Fixed FP | Fixed FN | New FP | New FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lr_c1_unweighted_default` | Control | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 | 0 | 0 | 0 | 0 | 0 |
| `lr_c025_unweighted_default` | Stronger regularization | 0.811133 | 0.622901 | 0.704663 | 0.775443 | 773 | 95 | 247 | 408 | 93 | 31 | 5 | 6 | 51 |
| `lr_c05_unweighted_default` | Stronger regularization | 0.791367 | 0.671756 | 0.726672 | 0.782666 | 752 | 116 | 215 | 440 | 32 | 8 | 3 | 4 | 17 |
| `lr_c2_unweighted_default` | Weaker regularization | 0.793867 | 0.711450 | 0.750403 | 0.796454 | 747 | 121 | 189 | 466 | 37 | 8 | 16 | 9 | 4 |
| `lr_c4_unweighted_default` | Weaker regularization | 0.786555 | 0.714504 | 0.748800 | 0.793828 | 741 | 127 | 187 | 468 | 67 | 15 | 22 | 22 | 8 |
| `lr_c1_balanced_default` | Class weighting | 0.746988 | 0.757252 | **0.752085** | 0.785292 | 700 | 168 | 159 | 496 | 90 | 0 | 42 | 48 | 0 |
| `linear_svc_c1_default` | Second classifier | 0.772803 | 0.711450 | 0.740859 | 0.785949 | 731 | 137 | 189 | 466 | 111 | 26 | 27 | 43 | 15 |
| `lr_c1_unweighted_tuned_threshold` | Threshold `0.47` | 0.777049 | 0.723664 | 0.749407 | 0.791858 | 732 | 136 | 181 | 474 | 36 | 0 | 20 | 16 | 0 |

These data distinguish three mechanisms. Lowering the threshold to `0.47` changed only 36 borderline scores and made the expected monotonic exchange: 20 false negatives were fixed and 16 new false positives appeared. Balanced fitting changed 90 decisions and made a stronger recall-oriented exchange: 42 fixed false negatives versus 48 new false positives. Because class weighting changes the fitted objective and coefficients, it is not merely the same baseline score ranking at a different cutoff. `C=2.0` gave the best accuracy (`0.796454`) and a more precision-preserving gain, but the predeclared criterion was F1, under which balanced fitting was higher by `0.0016823363232039`.

The second classifier did not win. Linear SVC improved recall over the baseline but had lower precision, F1, and accuracy than `C=2.0`, balanced Logistic Regression, and the best threshold alternative. Its 111 changes included both directions—26 fixed false positives, 27 fixed false negatives, 43 new false positives, and 15 new false negatives—consistent with learning a different boundary rather than simply shifting one operating point.

## Threshold sweep and precision-recall tradeoff

The machine-checkable sweep is `results/threshold_sweep.csv`. It has the exact contract columns `ticket,threshold,precision_target_1,recall_target_1,f1_target_1` and 61 data rows. A detailed companion under `experiments/ticket-4/dev/results/threshold_sweep_detailed.csv` adds accuracy, confusion counts, and transitions.

| Threshold | Precision 1 | Recall 1 | F1 1 | FP | FN | Fixed FN | New FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.493721 | 0.960305 | 0.652151 | 645 | 26 | 175 | 525 |
| 0.30 | 0.577689 | 0.885496 | 0.699216 | 424 | 75 | 126 | 304 |
| 0.40 | 0.685452 | 0.798473 | 0.737659 | 240 | 132 | 69 | 120 |
| 0.45 | 0.749614 | 0.740458 | 0.745008 | 162 | 170 | 31 | 42 |
| **0.47** | **0.777049** | **0.723664** | **0.749407** | **136** | **181** | **20** | **16** |
| 0.50 | 0.790941 | 0.693130 | 0.738812 | 120 | 201 | 0 | 0 |
| 0.55 | 0.824458 | 0.638168 | 0.719449 | 89 | 237 | 0 | 36 new FN |
| 0.60 | 0.870968 | 0.577099 | 0.694215 | 56 | 277 | 0 | 76 new FN |
| 0.70 | 0.933798 | 0.409160 | 0.569002 | 19 | 387 | 0 | 186 new FN |
| 0.80 | 0.987179 | 0.235115 | 0.379778 | 2 | 501 | 0 | 300 new FN |

The sweep makes the operating choice explicit rather than treating F1 as the only meaningful outcome. At `0.20`, the model detects 96.0% of positive rows but produces 645 false positives and precision below 0.50. At `0.80`, precision reaches 98.7%, but recall falls to 23.5% and 501 disasters are missed. The best threshold-only F1 occurs at `0.47`, not at the default `0.50`; it modestly favors recall without the much larger false-positive increase caused by balanced fitting. A deployment in which false alarms are costly could defensibly prefer `0.47` or even `0.50`, despite the class-weighted model's higher dev F1. Conversely, a disaster triage system that strongly penalizes missed positives may prefer balanced fitting or an even lower threshold. The submitted choice follows the stated F1 criterion, not an unstated operational cost assumption.

## Interpretation of model coefficients

Coefficient tables for every linear candidate are saved in `experiments/ticket-4/dev/interpretability/top_coefficients.csv`. They are technically valid because all candidates are linear models over the same TF-IDF vocabulary. In the baseline, prominent positive terms included `in` (`3.823635`), `california` (`2.681294`), `hiroshima` (`2.653886`), `fires` (`2.584588`), `http` (`2.494384`), and `train` (`2.382401`). Strong negative terms included `you` (`-3.373872`), `my` (`-2.918315`), `new` (`-2.106237`), and `full` (`-1.763207`). The balanced model retained the same leading lexical structure but refitted magnitudes: for example, `in` became `3.704944`, `fires` `2.601811`, `you` `-3.284955`, and `my` `-2.857211`. This supports the interpretation that class weighting primarily repositions the boundary around an otherwise similar representation, while still refitting the model rather than applying a post-hoc threshold only.

Linear SVC's leading positive terms included `hiroshima`, `fires`, `earthquake`, `train`, and `drought`; its leading negative terms included `you`, `new`, `full`, and `my`. The broad semantic agreement shows that its failure to win was not caused by a completely unrelated feature mechanism. It simply did not provide the best precision-recall balance on this fixed dev split.

## Concrete dev prediction changes

All 90 selected-candidate changes are saved with stable IDs, text, metadata context, both predictions, both scores, and outcome in `experiments/ticket-4/dev/changes/lr_c1_balanced_default_changes.csv`.

- ID `105`, target 1, “BigRigRadio Live Accident Awareness”: the baseline probability `0.467047` produced a false negative; balanced fitting raised it to `0.521118` and fixed the error. The short report-like text was borderline under the unweighted model, so the recall-oriented movement matches the hypothesis.

- ID `110`, target 0, “'By accident' they knew what was gon happen ...”: the score rose from `0.486671` to `0.547238`, creating a false positive. The disaster-related word is used casually, illustrating the direct precision cost of weighting positive errors more heavily.

- IDs `353` and `390` have near-duplicate “World Annihilation vs Self Transformation ... Aliens Attack” text but opposite labels. Both received the same baseline score `0.471587` and balanced score `0.526361`. The change fixed ID `390` but created an error for ID `353`. No decision rule can separate identical lexical evidence with contradictory labels; this is a data ambiguity, not evidence that class weighting understood one row better.

- ID `394`, target 1, describes a campaign to stop the “Annihilation” of wild horses. Its score moved from `0.445722` to `0.503492`, fixing a false negative, but the language is metaphorical rather than a conventional acute disaster. This exposes uncertainty in interpreting score improvements row by row.

- ID `402`, target 0, “Apocalypse please”: balanced fitting moved `0.449365` to `0.513235`, creating a false positive. The model responds to a strong disaster token without sufficient context, as predicted by the reduced-precision hypothesis.

- ID `840`, target 1, a report about FedEx stopping transport of bioterror germs, moved from `0.494623` to `0.550903` and became correct. This is a plausible fixed miss near the original boundary.

These examples show why the 42-versus-48 count must accompany the F1 score. Some new positives are useful recoveries, some are figurative or underspecified false alarms, and contradictory/ambiguous labels can make the same movement look simultaneously good and bad.

## Freeze chronology

The dev experiment completed before the freeze and explicitly reported zero held-out rows loaded and zero held-out evaluations. The freeze was written at `2026-07-21T13:26:17+08:00` to `experiments/ticket-4/frozen_decision.json`. It locks:

- raw-text default TF-IDF preprocessing;
- Logistic Regression;
- `C=1.0`;
- `class_weight='balanced'`;
- threshold `0.50`;
- target-1 dev F1 as the selection criterion;
- the complete effective parameter dictionaries;
- hashes of data, split, source files, plan, dev metrics, selection result, threshold sweep, selected dev predictions, baseline held-out predictions, and package lock;
- Ticket 4 held-out evaluation count zero at freeze;
- no permission to reopen selection.

Freeze SHA-256 is `14e3c474f15a04a9997bada41d477447ea2e3c7e49533b8980edfac7896b371e`. The exact freeze command is stored in the freeze. The held-out runner validates all frozen hashes and refuses to execute when held-out artifacts or a Ticket 4 summary row already exist.

## Final held-out evidence

The one post-freeze held-out command was:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket4_heldout --data data\train.csv --split starter\data\split_indices.json --freeze experiments\ticket-4\frozen_decision.json --output-dir experiments\ticket-4\heldout --confirm-single-ticket4-evaluation
```

The frozen candidate produced:

| Split/model | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Ticket 1 baseline | 0.801394 | 0.703364 | 0.749186 | 0.797768 | 755 | 114 | 194 | 460 |
| Ticket 4 balanced LR | 0.744361 | 0.756881 | 0.750569 | 0.783979 | 699 | 170 | 159 | 495 |

Relative to the frozen baseline, there were 91 prediction changes: zero fixed false positives, 35 fixed false negatives, 56 new false positives, and zero new false negatives. The directional tradeoff closely matched dev: more target-1 recall, less target-1 precision, and lower accuracy. However, the F1 benefit shrank from a dev delta of `+0.0132728705019151` to a held-out delta of only `+0.0013829448328488425`. Held-out was not used to reverse or modify the selected decision.

Representative held-out movements also match the dev mechanism:

- ID `556`, target 1, reports that people burned their own house. Its score moved from `0.464323` to `0.529717`, fixing a false negative.

- ID `907`, target 1, “To fight bioterrorism sir.” moved from `0.460814` to `0.530381`, fixing a miss, though the row is extremely short and context-poor.

- ID `59`, target 0, contains only several URLs and `#nsfw`; its score moved from `0.450541` to `0.504457`, creating a false positive. This is a weak and undesirable recall tradeoff.

- ID `767`, target 0, uses “avalanche” metaphorically in a quotation. The machine-generated transition row records a move from `0.4876230020837692` to `0.5448153095708489`, creating a false positive.

- ID `996`, target 0, uses “blazing hot” figuratively. It moved from `0.474603` to `0.536082`, creating a false positive.

- ID `1645`, target 1, references bombing Pearl Harbor and moved from `0.473645` to `0.531109`, fixing a false negative.

The full 91-row ledger is `experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv`; complete final false-positive and false-negative tables are saved alongside it.

## Stable artifacts and reproducibility

Primary artifacts are:

- `results/threshold_sweep.csv`: required 61-row machine-checkable threshold sweep;
- `experiments/ticket-4/dev/experiment_plan.json`: pre-execution hypotheses, bounded ranges, and selection rule;
- `experiments/ticket-4/dev/results/dev_model_metrics.csv`: all native model candidates plus the selected threshold candidate;
- `experiments/ticket-4/dev/results/threshold_sweep_detailed.csv`: confusion and transition detail for all thresholds;
- `experiments/ticket-4/dev/predictions/`, `changes/`, and `errors/`: stable-ID evidence for every reported dev candidate;
- `experiments/ticket-4/dev/interpretability/top_coefficients.csv`: 350 ranked linear coefficients;
- `experiments/ticket-4/frozen_decision.json` and `freeze_decision.md`: pre-held-out lock;
- `experiments/ticket-4/heldout/`: one held-out run with prediction, metric, confusion, transition, error, warning, version, command, configuration, and chronology artifacts;
- `predictions/ticket-4-heldout-predictions.csv`: 1,523 unique IDs in exact held-out order, SHA-256 `d45d3f0b53c29d304bb571a3e9c01dc48d4c742060b410e2fcb8a6095190feeb`;
- `results/summary.csv`: final Ticket 4 row, with transitions relative to the frozen Ticket 1 baseline.

The dev command was:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket4_dev --data data\train.csv --split starter\data\split_indices.json --plan experiments\ticket-4\dev\experiment_plan.json --output-dir experiments\ticket-4\dev
```

The first attempted invocation stopped before artifact writing because the reproduction guard required bitwise-equal saved probabilities. Diagnostic comparison showed identical predictions and a maximum score difference of only `1.1102230246251565e-16`; the guard was corrected to the explicit absolute tolerance `1e-12`, then the documented dev experiment ran successfully. This chronology did not touch held-out or alter the candidate grid.

## Conclusion

Balanced Logistic Regression at `C=1.0` and threshold `0.50` was the correct frozen selection under the predeclared target-1 dev F1 criterion. Threshold tuning alone showed that `0.47` is a more favorable operating point than `0.50` for the unweighted baseline, and `C=2.0` offered a comparatively balanced regularization-only improvement. Nevertheless, balanced fitting had the highest dev F1 and delivered the expected recall increase.

The held-out result supports only a cautious conclusion. The recall-oriented movement reproduced, but F1 improved by just `0.001383`, accuracy fell, and more new false positives were introduced than false negatives were fixed. The candidate is a defensible F1-oriented operating choice, not an unequivocally superior classifier. In a real deployment, the threshold or class weight should be selected from an explicit cost model for missed disasters versus false alarms, rather than F1 alone.

## Limitation

The experiment uses one fixed dev split and a finite grid. Selecting the maximum among 61 thresholds and several model configurations can exploit dev-specific noise even without held-out leakage; the small held-out F1 delta is consistent with that risk. Linear SVC is only one reasonable second classifier, so this ticket does not establish that every CPU-compatible alternative is inferior. Class weighting and threshold tuning were intentionally tested independently to preserve causal interpretability; their interaction was not searched. Finally, labels include duplicates, near-duplicates, figurative uses, and ambiguous examples, so changes in measured F1 do not always correspond to unambiguous semantic improvements. These constraints are reasons to preserve the detailed error ledgers and avoid presenting the selected operating point as universally optimal.
