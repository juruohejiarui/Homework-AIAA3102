# Ticket 1 — Baseline Discrepancy Diagnosis

## Required evidence map

This ticket's **hypothesis** is stated under "Hypothesis." Its **intended lever** is discrepancy diagnosis across split construction, train-only feature fitting, seed, effective TF-IDF/classifier defaults, convergence, and software version; each behavioral probe changes one lever. The **controlled setup** is fully serialized under "Frozen baseline configuration" and bounded by "Evidence boundary and chronology." **Dev evidence** appears under "Dev evidence before the freeze" and "Controlled dev probes." The **frozen decision** and its pre-held-out chronology are recorded under "Evidence boundary and chronology." **Held-out evidence** appears only under "Final held-out evidence." **Concrete prediction changes** are listed by stable ID under "Concrete prediction-change examples." The **interpretation** is in "Systematic discrepancy diagnosis" and "Conclusion," and the **limitation** is stated under "Limitations." This map is descriptive; it does not add or revise any experiment result.

## Required question and answer

**Question:** Does the independently implemented raw-text TF-IDF + Logistic Regression baseline reproduce the reference contract?

**Answer:** No. The frozen implementation produced held-out target-1 F1 `0.749185667752443`, while the contract specifies `0.7574221578566256` with an absolute tolerance of `0.001`. The signed difference is `-0.008236490104182592`; its absolute value, `0.008236490104182592`, is more than eight times the allowed tolerance. Therefore `matches_reference=false`.

This conclusion is about the exact version-locked implementation documented below. The structural checks rule out an incorrect fixed split, row-position splitting, train/dev leakage in the frozen pipeline, a seed mismatch, non-convergence, and several other local implementation errors. Controlled dev probes demonstrate that modest TF-IDF and classifier choices can change F1 by an amount comparable to or larger than the contract gap. However, the contract does not publish the reference implementation's full parameter dictionary or package versions, so the surviving evidence cannot uniquely identify one external reference-setting difference as the cause.

## Hypothesis

The pre-evaluation hypothesis was that the most literal implementation of the assignment description—raw `text`, a default `TfidfVectorizer`, and a default `LogisticRegression` with only the project seed supplied—would reproduce the reference held-out F1 within tolerance. A train-majority floor was first used only as a wiring check. The baseline decision was based on the specification and dev-only evidence; it was frozen before the original held-out comparison.

The diagnostic hypothesis after the mismatch was that one of four broad mechanisms could explain it:

1. a data-flow defect, such as the wrong split or row-position indexing;
2. an apparently minor feature or classifier-default difference;
3. environment-dependent scikit-learn behavior not specified by the contract; or
4. leakage or held-out-driven selection.

The investigation used structural audits for categorical data-flow claims and one-lever dev probes for behavioral claims. Probe results were not used to change the frozen model or rerun the primary comparison.

## Evidence boundary and chronology

Ticket 1 has an unusual but fully recorded recovery history:

1. The original configuration was frozen at `2026-07-21T12:08:53+08:00`. The freeze recorded that held-out had not been observed and that the held-out evaluation count was zero.
2. The frozen model was evaluated once for the historical primary comparison.
3. A user-requested rollback deleted the Ticket 1 files, including the stable prediction rows, while the scalar outcome remained recorded in `PROJECT_STATUS.md`.
4. Because row-level predictions cannot be recovered from aggregate metrics, an explicitly authorized deterministic replay of the same frozen model was performed once solely to reconstruct deleted artifacts. It reproduced the historical metrics exactly and did not create a new decision or tune a model.
5. The historical primary-comparison count is therefore `1`; the artifact-recovery replay count is `1`. No further held-out run was performed while producing this document, and none is permitted.

The current freeze JSON was reconstructed byte-for-byte with SHA-256 `3b1f589fb5b445cf146e63ad176dd5255e99aa342cc0502c8b8df657945ee3e8`. The recovery start and completion ledgers preserve the purpose and counts. This history is a limitation of the artifact chain and is not hidden or described as an ordinary single physical execution.

Primary evidence:

- Freeze: `experiments/ticket-1/frozen_baseline_config.json`
- Human-readable freeze decision: `experiments/ticket-1/freeze_decision.md`
- Recovery chronology: `experiments/ticket-1/heldout/heldout_evaluation_started.json` and `heldout_evaluation_completed.json`
- Contract comparison: `experiments/ticket-1/heldout/primary_contract_comparison.json`
- Held-out metrics and confusion matrix: `experiments/ticket-1/heldout/heldout_metrics.csv` and `heldout_confusion_matrix.csv`
- Stable prediction interface: `predictions/heldout_predictions.csv`
- Dev probe plan and outputs: `experiments/ticket-1/probes/`

## Frozen baseline configuration

### Data and split discipline

- Source: the labeled Kaggle Disaster Tweets `train.csv` named by the starter package.
- Data SHA-256: `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`.
- Fixed-split SHA-256: `db2fd1fdcc24043dd40ed202efe2c6cc19183d75de2becb2bb8645e99d8988f1`.
- Contract SHA-256: `84de6c87a21df0bc352110edafa73891fa1e2eabb380dc648e1af2e99bf4f646`.
- Fit rows: exactly the 4,567 `train_ids`.
- Selection/evaluation before freeze: exactly the 1,523 `dev_ids`.
- Final report split: exactly the 1,523 `heldout_ids`, used only after the freeze.
- Rows are selected by Kaggle `id` membership and emitted in the order stored in `split_indices.json`; dataframe position is never used as identity.

### Features and prediction rule

- Input: raw `text` only.
- Excluded: `keyword`, `location`, text length, and all other metadata or engineered features.
- Manual normalization: none.
- Threshold selection: none.
- Prediction: scikit-learn `Pipeline.predict`.
- Score: `predict_proba` probability for classifier class `1`.
- Leakage barrier: vectorization and classification are composed in one scikit-learn `Pipeline` and fitted on train rows only.

### Effective `TfidfVectorizer` parameters

| Parameter | Frozen value |
|---|---|
| `analyzer` | `word` |
| `binary` | `False` |
| `decode_error` | `strict` |
| `dtype` | `numpy.float64` |
| `encoding` | `utf-8` |
| `input` | `content` |
| `lowercase` | `True` |
| `max_df` | `1.0` |
| `max_features` | `None` |
| `min_df` | `1` |
| `ngram_range` | `(1, 1)` |
| `norm` | `l2` |
| `preprocessor` | `None` |
| `smooth_idf` | `True` |
| `stop_words` | `None` |
| `strip_accents` | `None` |
| `sublinear_tf` | `False` |
| `token_pattern` | `(?u)\b\w\w+\b` |
| `tokenizer` | `None` |
| `use_idf` | `True` |
| `vocabulary` | `None` |

### Effective `LogisticRegression` parameters

| Parameter | Frozen value |
|---|---|
| `C` | `1.0` |
| `class_weight` | `None` |
| `dual` | `False` |
| `fit_intercept` | `True` |
| `intercept_scaling` | `1` |
| `l1_ratio` | `0.0` |
| `max_iter` | `100` |
| `n_jobs` | `None` |
| `penalty` | `deprecated` in scikit-learn 1.9.0's parameter representation |
| effective regularization | pure L2 because `l1_ratio=0.0` |
| `random_state` | `3102` |
| `solver` | `lbfgs` |
| `tol` | `0.0001` |
| `verbose` | `0` |
| `warm_start` | `False` |

Central reproducibility settings are seed `3102` and `n_jobs=1`. The executed environment was CPython 3.13.0, NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, and Matplotlib 3.11.1. The complete environment is pinned in `requirements-lock.txt`.

## Dev evidence before the freeze

The floor and reference pipeline were each executed twice. Both reference fits produced identical prediction labels, class-1 probabilities, metrics, warning state, and iteration count across all 1,523 dev rows.

| Model | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train-majority floor | 0.000000 | 0.000000 | 0.000000 | 0.569928 | 868 | 0 | 655 | 0 |
| Frozen raw-text baseline | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 |

The logistic model converged in 15 of 100 permitted iterations, with no fit warnings. This rules out a local failure to converge as the mechanism behind the frozen dev behavior. The dev prediction file has 1,523 unique IDs in exact fixed dev order and matches the independently repeated fit.

## Final held-out evidence

The held-out result was computed only for the already frozen baseline. It was not used to revise preprocessing, parameters, threshold, or model choice.

| Quantity | Value |
|---|---:|
| Contract metric | `heldout_f1_target_1` |
| Actual target-1 F1 | 0.749185667752443 |
| Reference target-1 F1 | 0.7574221578566256 |
| Signed difference | -0.008236490104182592 |
| Absolute difference | 0.008236490104182592 |
| Allowed tolerance | 0.001 |
| Matches reference | **False** |
| Precision target 1 | 0.8013937282229965 |
| Recall target 1 | 0.7033639143730887 |
| Accuracy | 0.7977675640183848 |
| TN / FP / FN / TP | 755 / 114 / 194 / 460 |
| Convergence iterations | 15 / 100 |
| Fit warnings | none |

The stable held-out artifact contains 1,523 rows, 1,523 unique IDs, exact fixed held-out order, and columns `id,y_true,y_pred,score,model_name,ticket`. Its reconstruction SHA-256 is `0e5b6caab24e1371ff17ae7a95882831b744f4d147c3c56f2ef2e51369c4bba1` and is recorded in the completion ledger.

## Systematic discrepancy diagnosis

All behavioral probes changed one intended lever and used train for fitting and dev for comparison, except the explicitly invalid leakage probe, which deliberately allowed dev text into TF-IDF fitting to demonstrate the effect. No probe accessed held-out or changed the frozen decision.

### Structural causes

| Plausible cause | Evidence | Finding |
|---|---|---|
| Incorrect split assignment | Fixed split has 4,567 train, 1,523 dev, 1,523 held-out, 7,613 unique total IDs, and zero overlap. Data and split hashes match the freeze. | Ruled out for the frozen run. |
| Row-position instead of ID splitting | Selection joins on Kaggle `id` and restores stored JSON order. IDs are sparse: maximum ID is 10,873, and 2,279 IDs are not valid zero-based positions in a 7,613-row frame. Prediction IDs exactly match the fixed lists. | Ruled out for the frozen run. |
| Changed preprocessing | Frozen input is raw `text`; there is no custom preprocessor. Disabling default lowercase changed 59 dev labels and reduced F1 by 0.007665. | No hidden preprocessing exists locally; lowercasing is demonstrably consequential. |
| Changed TF-IDF parameters | Frozen effective parameters are serialized. Adding word bigrams changed 64 labels and reduced dev F1 by 0.010067. | Frozen settings are verified; a plausible reference parameter difference could exceed tolerance. |
| Accidental leakage | Frozen vectorizer is inside a pipeline fitted only on train. The intentionally leaky TF-IDF probe changed 68 dev labels and reduced F1 by 0.013575. | Ruled out for the frozen run; the probe shows leakage can materially alter results and is not necessarily beneficial. |

### Classifier and environment causes

| Plausible cause | Controlled dev evidence | Finding |
|---|---|---|
| Solver | `lbfgs -> liblinear` changed one prediction and F1 by -0.001027. | Frozen solver is verified; solver choice alone is enough to cross the contract tolerance on dev. |
| Regularization | `C=1.0 -> 0.5` changed 32 labels and F1 by -0.012140. Explicit historical `penalty='l2'` was label- and metric-equivalent to the 1.9.0 default representation. | Regularization strength is highly plausible as a cross-implementation cause; current explicit/default L2 representation is not. |
| Maximum iterations or convergence | `max_iter=100 -> 1000` changed no labels, scores, or metrics. Baseline converged in 15 iterations without warnings. | Ruled out in this environment. |
| Class weighting | `None -> 'balanced'` changed 90 labels and increased F1 by 0.013273 while decreasing precision and increasing recall. | A material operating-point lever, but not part of the frozen baseline and not evidence that the reference used it. |
| Random seed | `3102 -> 9999` changed no labels, scores, metrics, or iterations under `lbfgs`. Two seed-3102 fits were also identical. | Ruled out for this deterministic path in the current environment. |
| Package-version behavior | Current environment is scikit-learn 1.9.0. Explicit L2 matched current default behavior but raised the documented deprecation warning. The contract gives no reference scikit-learn version or complete effective parameter dictionary. | Cannot be confirmed or excluded across environments. This is the principal unresolved reproducibility variable. |

## Controlled dev probes

The frozen baseline is the comparator for every transition count, as required by the instructor clarification.

| Probe: single lever | Dev F1 | Delta | Changed labels | Fixed FP | Fixed FN | New FP | New FN | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Frozen baseline | 0.738812 | 0.000000 | 0 | 0 | 0 | 0 | 0 | Comparator |
| `lowercase=True -> False` | 0.731148 | -0.007665 | 59 | 14 | 12 | 13 | 20 | Case handling changes both directions and lowers F1. |
| word `(1,1) -> (1,2)` n-grams | 0.728745 | -0.010067 | 64 | 12 | 13 | 22 | 17 | Added phrase features do not improve this default-regularized fit. |
| `lbfgs -> liblinear` | 0.737785 | -0.001027 | 1 | 0 | 0 | 0 | 1 | A near-boundary positive crosses below 0.5. |
| `C=1.0 -> 0.5` | 0.726672 | -0.012140 | 32 | 8 | 3 | 4 | 17 | Stronger regularization suppresses more true positives than it repairs. |
| `max_iter=100 -> 1000` | 0.738812 | 0.000000 | 0 | 0 | 0 | 0 | 0 | Already-converged fit is unchanged. |
| `class_weight=None -> balanced` | 0.752085 | +0.013273 | 90 | 0 | 42 | 48 | 0 | Recall rises from 0.6931 to 0.7573; precision falls from 0.7909 to 0.7470. |
| `random_state=3102 -> 9999` | 0.738812 | 0.000000 | 0 | 0 | 0 | 0 | 0 | Current deterministic solver path is seed-insensitive. |
| default L2 representation -> explicit `penalty='l2'` | 0.738812 | 0.000000 | 0 | 0 | 0 | 0 | 0 | Equivalent in scikit-learn 1.9.0; explicit spelling emits a future warning. |
| train-only TF-IDF -> train+dev-text TF-IDF | 0.725237 | -0.013575 | 68 | 35 | 0 | 0 | 33 | Invalid leakage shifts toward negative predictions and lowers F1. |

The largest positive dev delta came from class weighting, but it is not adopted: it was a post-mismatch diagnostic, changes the operating point substantially, and using it to revise Ticket 1 would make the held-out comparison no longer correspond to the frozen baseline. It is also more appropriately considered in Ticket 4, which was not started here.

## Concrete prediction-change examples

These examples come from the saved dev change tables. They are evidence of mechanism, not hand-selected training cases.

1. **Lowercasing disabled, new false negative — ID 237.** The real-disaster tweet contains many case-marked hashtags: “#AirPlane #Accident #JetEngine #TurboJet …”. The frozen lowercase model scored it `0.526536` and predicted 1; preserving case reduced the score to `0.436493` and predicted 0. This supports the mechanism that case-fragmented vocabulary can weaken sparse evidence.
2. **Word bigrams, fixed false positive — ID 241.** “My phone looks like it was in a car ship airplane accident. Terrible” is figurative/non-disaster. Its score moved from `0.594712` to `0.467062`, repairing the error. However, the same probe created more new false positives and false negatives than it fixed overall, so this appealing example does not establish a net improvement.
3. **Solver substitution, new false negative — ID 5282.** A tweet about living in fear of being shot had frozen score `0.500826`. `liblinear` moved it narrowly below the decision boundary to `0.498597`. The only changed label in this probe illustrates how solver-level numerical differences can matter at 0.5 even when aggregate metrics look nearly identical.
4. **Stronger regularization, new false negative — ID 137.** A concrete traffic-accident report scored `0.531763` at `C=1.0` and `0.497301` at `C=0.5`. Stronger shrinkage removed a correct positive; across dev it created 17 new false negatives while fixing only 3.
5. **Balanced class weights, fixed false negative — ID 105.** “BigRigRadio Live Accident Awareness” moved from `0.467047` to `0.521118`, repairing a positive. But ID 110, the non-disaster phrase “‘By accident’ they knew what was gon happen,” moved from `0.486671` to `0.547238`, becoming a false positive. This pair concretely shows the recall/precision tradeoff behind the higher F1.
6. **Leaky vocabulary/IDF fit, new false negative — ID 97.** A Nashville traffic-accident report moved from `0.521890` to `0.478395` when dev text was included in the unsupervised TF-IDF fit. Leakage changed the representation and harmed this case; access to evaluation text is neither valid nor guaranteed to improve performance.

Representative final held-out errors are retained separately. For example, held-out ID 198 is a false positive: a general statement about lifetime airplane-accident odds scored `0.712980`. Held-out ID 17 is a false negative: a first-person report of South Tampa flooding scored `0.340040`. These illustrate that the frozen model can over-weight disaster vocabulary in generic statements and under-score informal, conversational reports. They were inspected only after the baseline had been frozen and evaluated and were not used to change it.

## Reproducibility and validation commands

The original dev-only baseline command was:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_baselines --data data\train.csv --split starter\data\split_indices.json --output-dir experiments\step-4-baselines --repetitions 2
```

The saved dev-only diagnostic command was:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket1_probes --data data\train.csv --split starter\data\split_indices.json --plan experiments\ticket-1\probes\probe_plan.json --output-dir experiments\ticket-1\probes
```

The recovery-only held-out command is preserved in `experiments/ticket-1/heldout/run_command.txt` for provenance. **It must not be run again.** The runner itself refuses ordinary primary execution and requires the explicit recovery flag, but the current recovery directory is non-empty and therefore also prevents overwrite.

Validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Conclusion

The independently implemented baseline does **not** reproduce the contract within tolerance. The local result is deterministic, converged, ID-aligned, train-only, raw-text-only, and stable across repeated dev fits. Its held-out F1 is about 0.00824 below the reference.

No observed evidence supports split corruption, row-position indexing, accidental leakage in the frozen pipeline, inadequate `max_iter`, or seed instability as the cause. The dev probes show that lowercasing, n-gram range, solver, regularization strength, class weighting, and feature-fit scope can each change predictions; several shift F1 by more than the contract gap or tolerance. Yet similarity in effect size is not causal identification. Because the contract omits the reference's full parameters and software versions, the discrepancy is best classified as an unresolved reference-implementation/defaults difference rather than assigned to one unverified cause.

The Ticket 1 decision remains the frozen minimal baseline. The summary transition counts are all zero because the Ticket 1 row is the baseline compared with itself; later candidate rows must also compare against this same baseline.

## Limitations

1. The reference contract exposes only a scalar held-out F1 and tolerance, not reference predictions, exact TF-IDF/logistic parameters, preprocessing code, solver diagnostics, or package versions. A unique causal attribution is therefore impossible from the available evidence.
2. Probe evidence is from one fixed dev split. It demonstrates sensitivity and mechanisms but does not estimate uncertainty across resampled splits and cannot prove what the external reference used.
3. The probe plan and regenerated diagnostic artifacts were created after the mismatch, so they are diagnostic rather than pre-registered model-selection evidence. They did not revise the frozen decision.
4. The original Ticket 1 files were deleted during a requested rollback. Although the freeze was reconstructed byte-for-byte and the authorized deterministic replay reproduced the historical scalar metrics exactly, the provenance includes one recovery replay in addition to the single historical primary comparison.
5. Concrete errors are illustrative rather than an exhaustive qualitative taxonomy. Broader normalization, shortcut, decision-rule, and data-quality investigations belong to Tickets 2–5 and were deliberately not started here.
