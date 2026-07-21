# Project Status and Continuation Handoff

Last updated: 2026-07-21 (Asia/Shanghai)

## Current task status

The full project and report workflow is complete through Step 5 final verification. The final IEEE conference source is `report/main.tex`; the final 12-page PDF is `report.pdf`; the final checklist is `report_verification_checklist.md`; and the machine-readable verification record is `report/final_verification.json`. The verified author names are LAI Jiaxing and HE Jiarui. Unsupplied affiliation, location, and email fields are omitted rather than fabricated.

Step 10 consolidated all five immutable decisions in `configs/frozen_decisions.json`, replayed each final configuration in a distinct clean Python process, reproduced archived dev and held-out prediction cores, regenerated every active result table, recalculated all error transitions against Ticket 1, and published `predictions/final-heldout-predictions.csv`. The audit result is PASS. This was an explicitly requested audit replay after every decision was already frozen; it did not reopen model selection and is not a new primary held-out comparison.

Ticket 1 content was temporarily removed by a user-requested rollback and then explicitly restored. Because the deleted stable held-out predictions could not be recovered from scalar metrics, the exact frozen model was replayed once solely for artifact recovery. This was not a new primary comparison, tuning run, or decision.

The historical Ticket 1 primary comparison count remains 1; the artifact-recovery rerun count is 1. Both produced target-1 F1 `0.749185667752443`, accuracy `0.7977675640183848`, and TN/FP/FN/TP = `755/114/194/460`. The result does not match reference `0.7574221578566256`: absolute difference `0.008236490104182592` exceeds tolerance `0.001`. Step 10 additionally performed one explicitly authorized audit-only clean replay per ticket; do not run held-out again unless a later instruction explicitly requires another audit.

Current boundary:

- Completed: repository audit, environment, modular infrastructure, dataset validation, floor model, untuned baseline, Tickets 1-5, Step 10 freeze/reproduction audit, final README/chat documentation, `report_evidence_audit.md`, `report_outline.md`, `results/report_evidence_matrix.csv`, and reproducible report tables/figures/case materials under `report_assets/`.
- Completed: final citation review, pinned PDF compilation, inspection of all 12 rendered pages, 209-check final report verification, project tests, submission validation, and final checklist.
- Next: stop. The requested finalization is complete; do not repeat any completed held-out run, audit replay, or experiment.

## Read these files before continuing

These files are authoritative and must be read before any new implementation or evaluation:

1. `topic-a-handout.md`
2. `teacher_clarifications.md`
3. `starter/README.md`
4. `starter/data/README_DATA.md`
5. `starter/data/split_indices.json`
6. `starter/configs/project_contract.json`
7. `experiments/step-4-baselines/run_config.json`
8. `experiments/step-4-baselines/results/dev_metrics.csv`
9. This `PROJECT_STATUS.md`

Instructor-provided files have not been modified.

## Non-negotiable constraints

1. Never regenerate, modify, reorder, or replace `starter/data/split_indices.json`.
2. Match every row by the Kaggle `id` column, never dataframe position.
3. Fit only on `train_ids`.
4. Use only `dev_ids` for preprocessing, model, threshold, and hyperparameter selection.
5. Never use held-out results to tune or revise a decision.
6. Run held-out evaluation only after the corresponding ticket decision is explicitly frozen in an artifact/document.
7. Never edit held-out labels or remove held-out examples.
8. Preserve stable IDs and fixed split order in prediction and audit artifacts.
9. Keep execution CPU-compatible, deterministic, seeded, versioned, and configuration-driven.
10. Change one intended lever at a time unless an interaction is explicitly justified.
11. Report only real executed outputs; never invent metrics, examples, logs, confusion matrices, or conclusions.
12. Keep code modular; do not move the whole project into a notebook.
13. Every experiment must save configuration, exact command, dev metrics, predictions or a change table, confusion matrix, FP/FN examples, interpretation, and limitation.
14. `fixed_fp`, `fixed_fn`, `new_fp`, and `new_fn` must compare candidates with the same frozen TF-IDF + Logistic Regression baseline, not the immediately previous model.
15. At the end of every requested step, run relevant validation, list changed files and actual results, document unresolved issues, update this file, and stop.

## Completed work

### Initial repository audit

- Read all project instructions, clarification files, starter metadata, fixed split, and reference contract.
- Confirmed the starter intentionally contained no baseline code.
- Identified the required five tickets and final deliverable schemas.
- Confirmed there was initially no dataset, source code, notebook, test suite, result artifact, or root submission README.
- Preserved all instructor-provided files unchanged.

### Python environment

- Created an isolated `.venv/` from the available CPython 3.13.0 AMD64 interpreter. The project does not prescribe a Python version.
- Installed and pinned CPU-compatible dependencies.
- Created:
  - `requirements.txt` for direct runtime packages.
  - `requirements-dev.txt` for pytest.
  - `requirements-lock.txt` for the complete installed environment.
- Current direct versions:
  - Python 3.13.0
  - NumPy 2.5.1
  - pandas 3.0.3
  - scikit-learn 1.9.0
  - Matplotlib 3.11.1
  - pytest 9.1.1
- `pip check` currently reports `No broken requirements found.`
- `requirements-lock.txt` was previously verified to match `pip freeze` exactly.
- In the restricted execution sandbox, set `MPLCONFIGDIR` to a writable temporary directory for plotting commands.

### Reproducible modular skeleton

- `configs/reproducibility.json` is the saved source of seed `3102` and `n_jobs=1`.
- `pipeline/reproducibility.py` seeds Python/NumPy, constrains common CPU thread pools, and applies exposed `random_state`/`n_jobs` estimator parameters.
- `pipeline/splits.py` parses the immutable split, rejects malformed/duplicate/overlapping IDs, and partitions data by stable IDs in fixed JSON order.
- `pipeline/data.py` validates the labeled source schema and exact split-ID coverage, then selects named splits by ID.
- `pipeline/modeling.py` creates a generic scikit-learn `Pipeline`, ensuring text feature fitting occurs inside the train-only model fit.
- `pipeline/metrics.py` provides target-1 precision, recall, F1, accuracy, and TN/FP/FN/TP counts.
- `pipeline/artifacts.py` defines required schemas and validates/writes stable-ID CSV plus deterministic JSON/text artifacts.
- `pipeline/versions.py` captures Python, platform, and package versions.
- Tests cover split integrity/order, ID-based partitioning, deterministic predictions, required artifact columns, missing prediction IDs, metric semantics, version capture, floor behavior, effective baseline defaults, and real train-to-dev baseline reproducibility.

### Public labeled dataset

- Downloaded `data/train.csv` from the exact public GitHub mirror named in `starter/README.md`.
- Dataset SHA-256: `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`.
- Validation results:
  - Rows: 7,613.
  - Columns: `id`, `keyword`, `location`, `text`, `target`.
  - Unique IDs: 7,613.
  - Missing fixed-split IDs: 0.
  - Unexpected IDs: 0.
  - Target values: 0 and 1 only.
  - Train: 4,567 rows; 2,605 class 0 and 1,962 class 1.
  - Dev: 1,523 rows; 868 class 0 and 655 class 1.
- Kaggle `test.csv` and `sample_submission.csv` remain absent, but they are not used by the course split or reference contract.
- `data/train.csv` is ignored by the root `.gitignore`; it is a local public input, not a generated or modified split.

### Fixed split and contract verification

- Split seed: `3102`.
- Split policy: stratified 60/20/20 over Kaggle `train.csv`, with membership stored by Kaggle ID.
- Counts: 4,567 train; 1,523 dev; 1,523 held-out; 7,613 total.
- Unique IDs: 7,613; within-split duplicates: 0; cross-split overlaps: 0.
- All three stored ID lists are strictly ascending and remain unchanged.
- Split SHA-256: `db2fd1fdcc24043dd40ed202efe2c6cc19183d75de2becb2bb8645e99d8988f1`.
- Contract SHA-256: `84de6c87a21df0bc352110edafa73891fa1e2eabb380dc648e1af2e99bf4f646`.
- Contract metric: `heldout_f1_target_1`.
- Reference floor F1: `0.0`.
- Reference TF-IDF + Logistic Regression held-out F1: `0.7574221578566256`.
- Allowed tolerance: `0.001`.
- The reference value was read from the contract. The frozen primary comparison and its later artifact-recovery replay are recorded at the top of this file; the reconstructed stable artifacts are present under `experiments/ticket-1/`, `predictions/`, and `results/`.

## Important model decisions

### Instructor floor model

`teacher_clarifications.md` defines the floor exactly:

1. Determine the majority class using only `train_ids`.
2. The fixed training majority is `target=0`.
3. Predict `target=0` for every dev and held-out row.
4. Its target-1 F1 is expected to be `0.0` because it never predicts class 1.
5. It is only a data/split/export/evaluation wiring check, not a competitive model.

The implementation derives the majority label from training labels rather than hard-coding it.

### Minimal reference baseline

The project specifies raw-text TF-IDF + Logistic Regression but no tuned parameter set. The chosen baseline is therefore the most literal version-locked sklearn implementation:

- Feature input: raw `text` only.
- Excluded inputs: `keyword`, `location`, text length, and every other metadata or engineered feature.
- No manual normalization, URL handling, mention handling, hashtag handling, emoji handling, threshold tuning, class weighting, or hyperparameter search.
- Model is a scikit-learn `Pipeline`; the vectorizer is fitted only through the train-only pipeline fit.
- Only the project seed is supplied explicitly to Logistic Regression.

Effective TF-IDF parameters:

- `analyzer='word'`
- `binary=False`
- `decode_error='strict'`
- `dtype=numpy.float64`
- `encoding='utf-8'`
- `input='content'`
- `lowercase=True`
- `max_df=1.0`
- `max_features=None`
- `min_df=1`
- `ngram_range=(1, 1)`
- `norm='l2'`
- `preprocessor=None`
- `smooth_idf=True`
- `stop_words=None`
- `strip_accents=None`
- `sublinear_tf=False`
- `token_pattern='(?u)\\b\\w\\w+\\b'`
- `tokenizer=None`
- `use_idf=True`
- `vocabulary=None`

Effective Logistic Regression parameters:

- `C=1.0`
- `class_weight=None`
- `dual=False`
- `fit_intercept=True`
- `intercept_scaling=1`
- `l1_ratio=0.0`
- `max_iter=100`
- `n_jobs=None`
- `penalty='deprecated'` in the scikit-learn 1.9.0 parameter representation
- Effective regularization is pure L2 because `l1_ratio=0.0`.
- `random_state=3102`
- `solver='lbfgs'`
- `tol=0.0001`
- `verbose=0`
- `warm_start=False`

Prediction behavior:

- Labels come from `Pipeline.predict`; no threshold was selected.
- Exported scores are `predict_proba` probabilities for classifier class 1.
- The complete effective dictionaries and rationale are saved in `experiments/step-4-baselines/run_config.json`.
- Ticket 1 held-out status in that configuration is `not_frozen_and_not_evaluated`.

## Verified dev-only results

Exact metrics are in `experiments/step-4-baselines/results/dev_metrics.csv`.

| Model | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train-majority floor | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 | 0.5699277741300066 | 868 | 0 | 655 | 0 |
| Raw-text TF-IDF + Logistic Regression | 0.7909407665505227 | 0.6931297709923664 | 0.7388120423108218 | 0.7892317793827971 | 748 | 120 | 201 | 454 |

Fit diagnostics:

- Reference baseline convergence: PASS.
- Iterations used: 15 of `max_iter=100`.
- Warnings in floor runs 1 and 2: none.
- Warnings in reference runs 1 and 2: none.

Reproducibility:

- The floor evaluation was run twice and matched exactly.
- The real reference pipeline was independently fitted twice on the fixed train rows.
- Both reference runs produced identical 1,523 dev prediction rows, class-1 probabilities, metrics, convergence state, and iteration count.
- Reproducibility check: PASS.

Stable-ID and error-artifact checks:

- Each dev prediction CSV has 1,523 rows, 1,523 unique IDs, the exact fixed dev ID order, and columns `id,y_true,y_pred,score,model_name,ticket`.
- Floor false positives: 0.
- Floor false negatives: 655.
- Reference false positives: 120.
- Reference false negatives: 201.
- No held-out-named artifact exists under `experiments/step-4-baselines/`.
- Held-out predictions created: false.
- Held-out metrics computed: false.

## Baseline execution and artifacts

Exact completed command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_baselines --data data\train.csv --split starter\data\split_indices.json --output-dir experiments\step-4-baselines --repetitions 2
```

The command is also saved verbatim in `experiments/step-4-baselines/run_command.txt`.

All 13 Step 4 files are present:

- `experiments/step-4-baselines/run_command.txt`
- `experiments/step-4-baselines/run_config.json`
- `experiments/step-4-baselines/run_notes.json`
- `experiments/step-4-baselines/software_versions.json`
- `experiments/step-4-baselines/warnings.json`
- `experiments/step-4-baselines/predictions/train_majority_floor_dev_predictions.csv`
- `experiments/step-4-baselines/predictions/raw_text_tfidf_logistic_regression_dev_predictions.csv`
- `experiments/step-4-baselines/results/dev_metrics.csv`
- `experiments/step-4-baselines/results/dev_confusion_matrices.csv`
- `experiments/step-4-baselines/results/train_majority_floor_false_positives.csv`
- `experiments/step-4-baselines/results/train_majority_floor_false_negatives.csv`
- `experiments/step-4-baselines/results/raw_text_tfidf_logistic_regression_false_positives.csv`
- `experiments/step-4-baselines/results/raw_text_tfidf_logistic_regression_false_negatives.csv`

The Step 4 notes include a hypothesis, interpretation, limitation, and explicit held-out non-access flags.

## Restored Ticket 1 evidence

The frozen baseline does not reproduce the reference contract:

- Actual held-out target-1 F1: `0.749185667752443`.
- Reference: `0.7574221578566256`.
- Signed difference: `-0.008236490104182592`.
- Absolute difference: `0.008236490104182592`.
- Tolerance: `0.001`.
- Match: false.
- Held-out precision/recall/accuracy: `0.8013937282229965` / `0.7033639143730887` / `0.7977675640183848`.
- Held-out TN/FP/FN/TP: `755/114/194/460`.
- Stable held-out prediction rows: 1,523 unique IDs in exact fixed held-out order.
- Historical primary comparison count: 1.
- Authorized artifact-recovery rerun count: 1.

The nine regenerated dev-only probes contain the frozen baseline plus nine one-lever comparisons:

| Probe | Dev F1 | Delta vs baseline | Prediction changes |
|---|---:|---:|---:|
| Frozen baseline | 0.738812 | 0.000000 | 0 |
| Lowercase disabled | 0.731148 | -0.007665 | 59 |
| Word bigrams | 0.728745 | -0.010067 | 64 |
| `liblinear` solver | 0.737785 | -0.001027 | 1 |
| `C=0.5` | 0.726672 | -0.012140 | 32 |
| `max_iter=1000` | 0.738812 | 0.000000 | 0 |
| `class_weight='balanced'` | 0.752085 | +0.013273 | 90 |
| `random_state=9999` | 0.738812 | 0.000000 | 0 |
| Explicit historical L2 representation | 0.738812 | 0.000000 | 0 |
| Leaky train+dev TF-IDF fit | 0.725237 | -0.013575 | 68 |

These probes are diagnostic only and did not revise the frozen baseline. `max_iter=1000`, alternate seed, and explicit L2 were prediction-equivalent in the current environment. Lowercasing, n-grams, regularization, class weighting, solver, and leakage changed dev predictions. Full metrics, transitions, confusion matrices, warnings, FP/FN rows, and stable-ID predictions are under `experiments/ticket-1/probes/`.

Restored Ticket 1 artifact locations:

- Freeze: `experiments/ticket-1/frozen_baseline_config.json`.
- Recovery chronology: `experiments/ticket-1/heldout/heldout_evaluation_started.json` and `heldout_evaluation_completed.json`.
- Held-out evidence: `experiments/ticket-1/heldout/`.
- Stable final prediction interface: `predictions/heldout_predictions.csv`.
- Ticket summary row: `results/summary.csv`.
- Probe plan and 45 probe files: `experiments/ticket-1/probes/`.
- Total files under `experiments/ticket-1/`: 59.

## Completed Ticket 2 evidence

Ticket 2 evaluated a raw control and six one-lever normalization variants on dev: URL placeholdering, mention placeholdering, hashtag-marker stripping, punctuation-to-space, Unicode casefolding, and emoji placeholdering. The implementations are composable, but every scored variant enabled at most one switch.

The dev-selected decision was `normalize_urls_placeholder`, which replaces complete HTTP/HTTPS/WWW URLs with `URLTOKEN` before the otherwise unchanged frozen TF-IDF + Logistic Regression pipeline.

Dev evidence for the selected variant versus the frozen raw control:

- F1: `0.7403132728771641` versus `0.7388120423108218` (`+0.0015012305663423264`).
- Precision: `0.8046594982078853` versus `0.7909407665505227`.
- Recall: `0.6854961832061068` versus `0.6931297709923664`.
- Accuracy: `0.7931713722915299` versus `0.7892317793827971`.
- Selected TN/FP/FN/TP: `759/109/206/449`.
- Changes relative to the baseline: 50 predictions; 22 fixed FP, 6 fixed FN, 11 new FP, and 11 new FN.
- URL perturbation: 767 affected dev rows; the selected normalizer changed 0 predictions and 0 scores, while the raw control changed 275 predictions.
- All seven dev fits converged without warnings.

The decision was frozen at `2026-07-21T13:00:54+08:00` before Ticket 2 held-out evaluation. Freeze SHA-256: `51a9ed0c07d092fa194a1f1399ae502eb0c3662bfdfb2b661ef22dfcbc5376cf`.

The single frozen held-out evaluation produced:

- Precision/recall/F1/accuracy: `0.825136612021858` / `0.6926605504587156` / `0.7531172069825436` / `0.8049901510177282`.
- TN/FP/FN/TP: `773/96/201/453`.
- Changes relative to the same frozen Ticket 1 baseline: 49 predictions; 22 fixed FP, 8 fixed FN, 4 new FP, and 15 new FN.
- Convergence: 22 of 100 iterations, with no warnings.
- Ticket 2 held-out evaluation count: 1.
- Selection reopened: false.

Stable Ticket 2 held-out predictions contain 1,523 unique IDs in exact fixed order and have SHA-256 `486bd8af3f808ed29a4471b384cfa82d6080670f657193c6717f1d0c4305acb4`.

Ticket 2 artifact locations:

- Implementation: `pipeline/normalization.py`, `pipeline/ticket2.py`, `pipeline/run_ticket2_dev.py`, `pipeline/freeze_ticket2.py`, and `pipeline/run_ticket2_heldout.py`.
- Dev plan and 39 dev artifacts: `experiments/ticket-2/dev/`.
- Freeze: `experiments/ticket-2/frozen_decision.json` and `freeze_decision.md`.
- Twelve held-out artifacts: `experiments/ticket-2/heldout/`.
- Total files under `experiments/ticket-2/`: 53.
- Stable predictions: `predictions/ticket-2-heldout-predictions.csv`.
- Ticket report: `tickets/ticket-2-normalization.md`.
- Summary: the second row of `results/summary.csv`.

## Completed Ticket 3 evidence

Ticket 3 evaluated ten controlled dev variants: majority floor, exact text control, keyword-only, length-only, keyword+length, location-only, keyword+location, selected shallow-only, text+keyword, and text+keyword+selected shallow features. Missing keyword and location values are explicitly imputed inside train-fitted pipelines; numeric extraction is stateless and scaling is train-only.

The best visible candidate was `text_plus_selected_shallow_features`:

- Dev precision/recall/F1/accuracy: `0.7935153583617748` / `0.7099236641221374` / `0.7493956486704271` / `0.7957977675640184`.
- Dev TN/FP/FN/TP: `747/121/190/465`.
- F1 delta versus frozen text baseline: `+0.0105836063596053`.
- Changes: 164 predictions; 41 fixed FP, 46 fixed FN, 42 new FP, and 35 new FN.

It was rejected because its improvement was shortcut-sensitive:

- Keyword masking changed 702 predictions and reduced F1 to `0.6423057128152342`.
- Masking keyword and location changed 696 predictions and produced F1 `0.6432627774909654`.
- Superficial-text neutralization changed 82 predictions and reduced F1 to `0.7330677290836654`.
- Text+keyword without the shallow block was worse than baseline at F1 `0.7350835322195705`.
- Location-only F1 was `0.22441430332922319`; 591 dev rows had nonmissing locations unseen in train.
- Length-only F1 was `0.4376130198915009`; selected-shallow-only F1 was `0.5302897278314311`.

The final dev-only decision was to reject all Ticket 3 shortcut additions and retain `raw_text_tfidf_logistic_regression`. Keyword was classified as mixed evidence; length and missingness as dataset artifacts; surface counts as mixed evidence; and full location categories as dataset artifacts.

The decision was frozen at `2026-07-21T13:13:53+08:00`; freeze SHA-256 is `651905972507d06427d427e3e3f0a219faba48d8dff2183dbd86e4c194cf0ac6`. Because the selected model is exactly the Ticket 1 baseline, Ticket 3 reused the already validated stable held-out predictions after freeze. It performed no new held-out fit or prediction pass.

Ticket 3 held-out report:

- Precision/recall/F1/accuracy: `0.8013937282229965` / `0.7033639143730887` / `0.749185667752443` / `0.7977675640183848`.
- TN/FP/FN/TP: `755/114/194/460`.
- Transitions versus frozen baseline: all zero.
- Ticket 3 held-out reporting count: 1.
- New held-out model fit: false.
- Selection reopened: false.

Ticket 3 artifacts:

- Implementation: `pipeline/shortcut_features.py`, `pipeline/run_ticket3_dev.py`, `pipeline/freeze_ticket3.py`, and `pipeline/run_ticket3_heldout.py`.
- Tests: `tests/test_shortcut_features.py` and `tests/test_ticket3.py`.
- Dev plan and 52 dev artifacts: `experiments/ticket-3/dev/`.
- Freeze: `experiments/ticket-3/frozen_decision.json` and `freeze_decision.md`.
- Eleven held-out reporting artifacts: `experiments/ticket-3/heldout/`.
- Total files under `experiments/ticket-3/`: 65.
- Stable prediction copy: `predictions/ticket-3-heldout-predictions.csv`.
- Report: `tickets/ticket-3-shortcuts.md`.
- Summary: the third row of `results/summary.csv`.

## Completed Ticket 4 evidence

Ticket 4 held raw-text default TF-IDF and the fixed ID split constant while testing a bounded decision-rule/model plan on dev only:

- 61 baseline probability thresholds from `0.20` through `0.80` in increments of `0.01`;
- `class_weight=None` versus `class_weight='balanced'` independently at `C=1.0`, threshold `0.50`;
- unweighted Logistic Regression `C ∈ {0.25, 0.5, 1.0, 2.0, 4.0}` at threshold `0.50`;
- one CPU-compatible second classifier, `LinearSVC(C=1.0, dual='auto')`, at decision threshold `0.0`.

The dev command loaded zero held-out rows. Its refitted control reproduced every frozen baseline dev prediction; score differences were only floating-point roundoff, with maximum absolute difference `1.1102230246251565e-16` and none above `1e-12`.

Key dev evidence:

- Frozen control precision/recall/F1/accuracy: `0.7909407665505227` / `0.6931297709923664` / `0.7388120423108218` / `0.7892317793827971`.
- Best threshold-only choice: threshold `0.47`, precision/recall/F1/accuracy `0.7770491803278688` / `0.7236641221374046` / `0.7494071146245059` / `0.7918581746552856`; 20 fixed FN and 16 new FP.
- Best regularization-only choice: `C=2.0`, F1 `0.750402576489533`; 8 fixed FP, 16 fixed FN, 9 new FP, and 4 new FN.
- Balanced Logistic Regression: precision/recall/F1/accuracy `0.7469879518072289` / `0.7572519083969466` / `0.7520849128127369` / `0.7852921864740644`; 42 fixed FN and 48 new FP.
- Linear SVM: precision/recall/F1/accuracy `0.7728026533996684` / `0.7114503816793893` / `0.7408585055643879` / `0.7859487852921865`.

The predeclared target-1 dev F1 criterion selected balanced Logistic Regression with raw-text default TF-IDF, `C=1.0`, and threshold `0.50`. This intentionally accepts lower precision and accuracy for higher recall. The decision was frozen at `2026-07-21T13:26:17+08:00`; freeze SHA-256 is `14e3c474f15a04a9997bada41d477447ea2e3c7e49533b8980edfac7896b371e`. No Ticket 4 held-out evidence was used at selection time.

The single frozen held-out evaluation produced:

- Precision/recall/F1/accuracy: `0.7443609022556391` / `0.7568807339449541` / `0.7505686125852918` / `0.7839789888378201`.
- TN/FP/FN/TP: `699/170/159/495`.
- Versus the frozen baseline: 91 changes; 0 fixed FP, 35 fixed FN, 56 new FP, and 0 new FN.
- F1 delta versus baseline: `+0.0013829448328488425`.
- Precision delta: `-0.05703282596735737`; recall delta: `+0.05351681957186549`; accuracy delta: `-0.013788575180564644`.
- Evaluation count: 1; selection reopened: false.

The recall-oriented mechanism generalized, but the held-out F1 benefit was small and the false-positive cost exceeded the number of fixed misses. The report therefore describes the frozen result as a defensible F1-oriented operating choice, not an unequivocally superior classifier.

Ticket 4 artifact locations:

- Implementation: `pipeline/decision_rule.py`, `pipeline/run_ticket4_dev.py`, `pipeline/freeze_ticket4.py`, and `pipeline/run_ticket4_heldout.py`.
- Tests: `tests/test_decision_rule.py` and `tests/test_ticket4.py`.
- Dev plan and controlled artifacts: `experiments/ticket-4/dev/`.
- Required 61-row sweep: `results/threshold_sweep.csv`.
- Freeze: `experiments/ticket-4/frozen_decision.json` and `freeze_decision.md`.
- Held-out evaluation artifacts: `experiments/ticket-4/heldout/`.
- Total files under `experiments/ticket-4/`: 56.
- Stable predictions: `predictions/ticket-4-heldout-predictions.csv`, SHA-256 `d45d3f0b53c29d304bb571a3e9c01dc48d4c742060b410e2fcb8a6095190feeb`.
- Report: `tickets/ticket-4-decision-rule.md`.
- Summary: the fourth row of `results/summary.csv`.

## Completed Ticket 5 evidence

Ticket 5 separated train/dev decision evidence from post-freeze held-out inspection. The predeclared audit used three deterministic relationship definitions:

- Raw exact: unmodified text equality.
- Canonical: NFKC normalization, HTML entity decoding, casefolding, URL placeholdering, and whitespace collapse, while retaining words, punctuation, mentions, and hashtags.
- Near: canonical char-word-boundary TF-IDF 3-5 grams, brute cosine k-NN with eight neighbors including self, and similarity at least `0.88`; canonical-equal pairs are excluded.

Train/dev-only findings:

| Relationship | Groups/pairs | Member rows or unique IDs | Conflicting | Cross train/dev |
|---|---:|---:|---:|---:|
| Raw exact | 50 | 124 members | 13 | 19 |
| Canonical | 214 | 677 members | 46 | 98 |
| Near | 693 pairs | 538 unique IDs | 104 | 265 |

The frozen Ticket 4 dev model made 168 FP and 159 FN. Its complete errors were joined to duplicate flags and reviewed as likely mislabels, ambiguous cases, legitimate hard negatives, or model weakness. Model disagreement alone was never treated as label evidence.

Eight high-confidence train corrections were preserved with original label, proposed label, evidence, and confidence, then applied only to an in-memory training copy. The controlled model kept Ticket 4 preprocessing/classifier settings fixed:

- Control precision/recall/F1/accuracy: `0.7469879518072289` / `0.7572519083969466` / `0.7520849128127369` / `0.7852921864740644`.
- Corrected-copy precision/recall/F1/accuracy: `0.7421289355322339` / `0.7557251908396947` / `0.7488653555219364` / `0.7820091923834537`.
- F1 delta: `-0.003219557290800479`, failing the predeclared `-0.002` noninferiority margin.
- Eleven dev prediction changes: 3 fixed FP, 0 fixed FN, 7 new FP, and 1 new FN.
- Original source labels changed: 0; dev labels changed: 0; held-out rows loaded: 0.

The dev-only decision was to reject the corrected-training model, retain `lr_c1_balanced_default`, and apply no source corrections. It was frozen at `2026-07-21T13:42:00+08:00`; freeze SHA-256 is `9423deaf6f30659f55547176a5ddb2191ed4895738cf7fe17b9d6138df693afa`.

Post-freeze full-dataset findings:

| Relationship | Groups/pairs | Member rows or unique IDs | Conflicting | Cross-split |
|---|---:|---:|---:|---:|
| Raw exact | 69 | 179 members | 18 | 41 |
| Canonical | 292 | 932 members | 64 | 194 |
| Near | 960 pairs | 715 unique IDs | 186 | 546 |

The final required `results/data_quality_audit.csv` has 64 unique valid IDs: 30 `fix`, 15 `ambiguous`, 6 `keep_but_flag`, and 13 `reject_false_positive`. Confidence is explicitly confidence in the disposition, not the model or stored label; submitted values range from `0.86` to `0.99`. The audit contains 9 train, 16 dev, and 39 held-out IDs. `fix` entries are future data-governance recommendations only and were not applied.

Because Ticket 5 retained the exact Ticket 4 model, its single post-freeze report reused validated Ticket 4 held-out predictions without a new fit or prediction pass:

- Precision/recall/F1/accuracy: `0.7443609022556391` / `0.7568807339449541` / `0.7505686125852918` / `0.7839789888378201`.
- TN/FP/FN/TP: `699/170/159/495`.
- Versus the frozen Ticket 1 baseline: 0 fixed FP, 35 fixed FN, 56 new FP, and 0 new FN.
- Ticket 5 reporting count: 1; new model fit: false; selection reopened: false.
- Held-out labels modified: false; held-out rows removed: 0.

Integrity and stable artifacts:

- `data/train.csv` remains SHA-256 `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`.
- Audit SHA-256: `9387710cc13d24ab914e61514da4d2c6a23a2d1bbd98beb3017e0faf88c39421`.
- Stable Ticket 5 prediction SHA-256: `b4ca0706ca355e4ba05a9c09f4c3fb8fd701e07c3ef211d3a9ee93d4a1c465d2`.

Ticket 5 artifact locations:

- Implementation: `pipeline/data_quality.py`, `pipeline/run_ticket5_dev.py`, `pipeline/run_ticket5_corrections.py`, `pipeline/freeze_ticket5.py`, `pipeline/run_ticket5_heldout.py`, and `pipeline/finalize_ticket5_audit.py`.
- Tests: `tests/test_data_quality.py` and `tests/test_ticket5.py`.
- Train/dev audit plans, duplicate candidates, error reviews, and correction experiment: `experiments/ticket-5/dev/`.
- Freeze: `experiments/ticket-5/frozen_decision.json` and `freeze_decision.md`.
- Full cross-split and held-out audit/reporting artifacts: `experiments/ticket-5/heldout/`.
- Final curated records and manifest: `experiments/ticket-5/final_audit_records.json` and `final_audit_manifest.json`.
- Total files under `experiments/ticket-5/`: 46.
- Required audit: `results/data_quality_audit.csv`.
- Stable predictions: `predictions/ticket-5-heldout-predictions.csv`.
- Report: `tickets/ticket-5-data-quality.md`.
- Summary: the fifth row of `results/summary.csv`.

## Completed Step 10 frozen-decision reproducibility audit

The Step 10 audit was executed only after all five ticket decisions were frozen. `configs/frozen_decisions.json` records each recipe, source freeze path and SHA-256, archived dev/held-out prediction paths and SHA-256 values, expected metrics, result-table inputs, and selection provenance. Manifest SHA-256: `4eabf0830dfa088c38cc698df7488bbd9cca4628f381873f14a093faf4da0b39`.

Decision-provenance verification:

- All five freezes predate their ticket held-out action.
- Tickets 1-4 were selected from `dev_ids` only; Ticket 5 used train audit evidence plus `dev_ids` for its controlled correction decision.
- Every freeze records zero held-out evaluations at decision time, `heldout_used_for_selection=false`, and `selection_reopening_permitted=false`.
- Step 10 did not alter a decision, threshold, feature, normalization, class weighting setting, training label, dev label, or held-out label.

Clean-process reproduction:

- Five ticket configurations were re-fit in five distinct Python interpreter processes under `experiments/final-reproducibility-audit/replays/`.
- All five replays matched archived dev and held-out IDs, ID order, true labels, and predicted labels exactly.
- Prediction changes across all ten archived comparisons: 0.
- Maximum absolute score difference: `1.1102230246251565e-16`, below the audit tolerance of `1e-12`.
- Every metric reproduced within `1e-15`; all five fits converged without warnings.

Reproduced final metrics and consistent Ticket 1 baseline transitions:

| Ticket | Dev F1 | Held-out F1 | Held-out accuracy | Fixed FP | Fixed FN | New FP | New FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7388120423108218 | 0.7491856677524430 | 0.7977675640183848 | 0 | 0 | 0 | 0 |
| 2 | 0.7403132728771641 | 0.7531172069825436 | 0.8049901510177282 | 22 | 8 | 4 | 15 |
| 3 | 0.7388120423108218 | 0.7491856677524430 | 0.7977675640183848 | 0 | 0 | 0 | 0 |
| 4 | 0.7520849128127369 | 0.7505686125852918 | 0.7839789888378201 | 0 | 35 | 56 | 0 |
| 5 | 0.7520849128127369 | 0.7505686125852918 | 0.7839789888378201 | 0 | 35 | 56 | 0 |

Result and artifact verification:

- `results/summary.csv` reproduced byte-for-byte from the five clean replay prediction sets and has the exact required 11-column schema with Tickets 1-5 exactly once.
- `results/threshold_sweep.csv` reproduced byte-for-byte from Ticket 1 raw-text dev scores and contains the exact five-column schema and 61 thresholds.
- `results/data_quality_audit.csv` reproduced byte-for-byte from `experiments/ticket-5/final_audit_records.json`; all 64 rows have valid stable IDs, dispositions, evidence, and confidences.
- `predictions/final-heldout-predictions.csv` contains exactly `id,y_true,y_pred,score,model_name,ticket`, 1,523 rows, 1,523 unique IDs, and the fixed held-out order. SHA-256: `8e3644db78fea2a9986a0962e8723eb115c82666ed33461578217021f9a66192`.
- Ticket 1 and Ticket 3 have intentionally identical prediction cores. Ticket 4, Ticket 5, and the final artifact have intentionally identical prediction cores. Their full CSV hashes differ because the ticket provenance field differs.
- No unexplained stale, contradictory, byte-duplicated, or manually edited active result file was found. The generic `predictions/heldout_predictions.csv` is explicitly documented as the historical Ticket 1 file, not the final prediction artifact.

Audit artifacts:

- Human-readable audit: `experiments/final-reproducibility-audit/reproducibility_audit.md`.
- Machine verification: `experiments/final-reproducibility-audit/reproducibility_verification.json`.
- Recreated tables: `reproduced_summary.csv`, `reproduced_threshold_sweep.csv`, and `reproduced_data_quality_audit.csv` in the same directory.
- Consistent transitions: `experiments/final-reproducibility-audit/transition_recalculation.csv`.
- Active artifact inventory: `experiments/final-reproducibility-audit/artifact_inventory.csv`.
- Five replay directories plus root audit evidence total 50 files.
- No final report, chat log, or final submission assembly was written in Step 10.

## Completed report evidence audit

`report_evidence_audit.md` is complete. The audit read all requested project instructions and documentation, all five ticket documents, every file under `experiments/`, `results/`, and `predictions/`, all pipeline source and tests, all frozen decisions/configurations, validation evidence, and the accessible curated AI-interaction record. At the time of the audit it found no existing report source or draft and no accessible verbatim earlier Codex transcript; the report source was created later in Step 4.

The audit independently recomputed selected dev and held-out precision, recall, F1, accuracy, confusion counts, and Ticket 1-relative transitions from prediction CSVs. All selected quantitative results match the generated metric/evaluation artifacts. It also verified every representative stable ID cited in the ticket documents.

One material narrative conflict was found during report evidence review: `tickets/ticket-4-decision-rule.md` originally gave stale held-out ID `767` scores. The final objective consistency audit corrected that narrative to the machine-generated values `0.4876230020837692 -> 0.5448153095708489`; the ID, label, transition, interpretation, report, and ticket now agree.

The audit also identified stale handoff statements that incorrectly listed `logs/chat.md` and the final README commands as missing; this status update corrects them. The repository was ready for report planning with the caveats recorded in `report_evidence_audit.md`; structure planning, asset generation, and report-source drafting are now complete.

## Completed IEEE conference report-structure design

`report_outline.md` now defines the complete evidence-bound structure for an official IEEE Conference LaTeX paper using `\documentclass[conference]{IEEEtran}`. It does not create or download a template. The planned narrative is coherent across tickets: protocol and baseline, reproducibility diagnosis, isolated normalization, shortcut audit, operating-point analysis, data-quality intervention, cross-ticket synthesis, case analysis, difficulties, AI usage, limitations, and conclusion.

For every planned subsection, the outline records its purpose, main verified argument, evidence to include, exact artifact sources, planned table/figure, representative cases, and required limitation. It plans 17 tables across the main paper and appendix and four data-backed figures. Confusion matrices are planned as an exact table rather than a redundant figure unless the later drafting step explicitly substitutes a three-panel figure.

`results/report_evidence_matrix.csv` has the exact columns `section,claim,evidence_file,table_or_figure,status,notes`, 73 evidence rows, and no blank required fields. The final objective consistency audit resolved the former Ticket 4 ID `767` narrative conflict using the machine CSV values; no `CONFLICTING` or `MISSING` row remains. Its earlier recorded SHA-256 is historical because the matrix was updated to record that resolution.

Appendix-only content is explicitly bounded: exhaustive parameters and candidate grids, correction proposal detail, extended stable-ID cases, commands/hashes/provenance, and long negative-result tables. Raw terminal output, screenshots, full logs, and verbatim CSV dumps are excluded even from the appendix.

## Completed verified report-asset generation

`scripts/generate_report_assets.py` reproducibly generates the evidence package under `report_assets/` without fitting a model, changing a label, or rerunning held-out evaluation. It reads the frozen decision manifest, archived dev/held-out prediction files, controlled experiment results, threshold sweep, duplicate summaries, audit records, and stable-ID change tables.

Generated outputs:

- Fourteen tables, each with a source CSV and an IEEE/LaTeX-ready `.tex` snippet.
- Four figures, each in 300-DPI PNG and SVG, with a dedicated source-data CSV.
- A 16-row representative case ledger containing successes, failures, ambiguity, a potential annotation issue, validated model errors, and the machine-correct Ticket 4 ID `767` scores.
- `report_assets/asset_manifest.csv`, which maps all 40 report assets to exact source artifacts, split, comparison baseline, and interpretation notes.
- `report_assets/validation_report.json`, which records 196 passing assertions and the known ID `767` narrative conflict.

The generator verifies source and split hashes, expected schemas and row counts, fixed split membership, unique stable IDs, source-label equality, every selected dev/held-out metric, Ticket 1-relative transitions, threshold-grid completeness, audit-disposition totals, case text/labels, shared prediction cores, and final replay-summary agreement. It confirms `heldout_labels_modified=false` and `heldout_rows_removed=0`. All requested assets were generated; none was omitted.

## Completed IEEE conference report-source draft (historical Step 4 state, superseded by Step 5)

`report/main.tex` is a complete standard IEEE conference source using `\documentclass[conference]{IEEEtran}`. It contains the title, abstract, IEEE keywords, Project Problem and Goal, Methodology, Main Evidence and Results, Case Analysis, Difficulties and Solutions, AI Usage Declaration, Discussion and Limitations, Conclusion, references, and four appendices. It imports all 14 generated LaTeX tables and all four verified PNG figures, interprets every major asset, uses stable-ID cases, preserves failed experiments and the Ticket 1 recovery qualification, and retains the frozen Ticket 5/Ticket 4 final model rather than selecting Ticket 2 post hoc.

`report/references.bib` contains only repository-supported dataset, course-material, software-version, and AI-tool entries. `report/README.md` records the later compilation sequence and the finalization blockers. `scripts/validate_report_source.py` performs non-compiling source validation, and `report/source_validation.json` records 122 passing checks over IEEE structure, required sections, 14 table inputs, four figure inputs, source encoding and structural balance, cross-references, citations, quantitative anchors, ID `767`, held-out integrity, final-model identity, author placeholder, and absence of a PDF. The source contains approximately 5,990 words before imported tables.

At Step 4, the source deliberately retained metadata placeholders and no PDF was compiled. Step 5 superseded that state: the supplied author names were inserted, unsupported affiliation/email fields were omitted, public bibliography URLs were checked, Tectonic 0.16.9 was pinned, and all 12 PDF pages were compiled and inspected.

## Current test status

Most recent validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Most recent real output:

```text
collected 56 items
56 passed, 0 failed, 0 warnings in 2.25s
```

Test files:

- `tests/test_artifacts.py`
- `tests/test_baselines.py`
- `tests/test_data_quality.py`
- `tests/test_decision_rule.py`
- `tests/test_final_reproducibility.py`
- `tests/test_metrics.py`
- `tests/test_normalization.py`
- `tests/test_reproducibility.py`
- `tests/test_shortcut_features.py`
- `tests/test_splits.py`
- `tests/test_ticket1.py`
- `tests/test_ticket2.py`
- `tests/test_ticket3.py`
- `tests/test_ticket4.py`
- `tests/test_ticket5.py`
- `tests/test_versions.py`

`pip check` also passes. The workspace is not a valid Git worktree (`git rev-parse` fails), so do not rely on `git status` or `git diff` for change reporting.

## Files created or modified by completed work

Environment/setup:

- Created `.venv/`.
- Created `requirements.txt`.
- Created `requirements-dev.txt`.
- Created `requirements-lock.txt`.

Project infrastructure:

- Created root `.gitignore`.
- Created root `README.md`.
- Created `configs/reproducibility.json`.
- Created `pipeline/__init__.py`.
- Created `pipeline/reproducibility.py`.
- Created `pipeline/splits.py`.
- Created `pipeline/data.py`.
- Created `pipeline/modeling.py`.
- Created `pipeline/metrics.py`.
- Created `pipeline/artifacts.py`; later extended with general CSV/text writers for baseline artifacts.
- Created `pipeline/versions.py`.
- Created `pipeline/baselines.py`.
- Created `pipeline/run_baselines.py`.
- Created `pipeline/normalization.py` and `pipeline/ticket2.py` for composable Ticket 2 transformations and shared evaluation logic.
- Created `pipeline/run_ticket2_dev.py`, `pipeline/freeze_ticket2.py`, and `pipeline/run_ticket2_heldout.py`.
- Created `pipeline/shortcut_features.py` for leakage-safe keyword, location, length, and selected shallow feature construction.
- Created `pipeline/run_ticket3_dev.py`, `pipeline/freeze_ticket3.py`, and `pipeline/run_ticket3_heldout.py`.
- Created `pipeline/decision_rule.py` for bounded thresholds, controlled classifier specifications, thresholding, and Ticket 4 evaluation.
- Created `pipeline/run_ticket4_dev.py`, `pipeline/freeze_ticket4.py`, and `pipeline/run_ticket4_heldout.py`.
- Created `pipeline/data_quality.py` for deterministic text canonicalization, duplicate/near-duplicate discovery, and audit validation.
- Created `pipeline/run_ticket5_dev.py`, `pipeline/run_ticket5_corrections.py`, `pipeline/freeze_ticket5.py`, `pipeline/run_ticket5_heldout.py`, and `pipeline/finalize_ticket5_audit.py`.
- Created `pipeline/build_frozen_decisions_manifest.py` and `pipeline/reproduce_frozen_ticket.py` for consolidated immutable decision provenance and isolated one-ticket replay.
- Created `pipeline/verify_final_reproducibility.py` for table regeneration, stable-ID validation, consistent transition recalculation, active-artifact auditing, and final prediction publication.
- Created `pytest.ini`.

Tests:

- Created all sixteen files listed in the test section above.

Data and executed artifacts:

- Downloaded `data/train.csv` without changing its content.
- Created all 13 files listed in the baseline artifact section.
- Restored the 59 files under `experiments/ticket-1/`, including the frozen configuration, held-out evidence, structural audit, and nine dev-only probe suites.
- Restored `predictions/heldout_predictions.csv` and the Ticket 1 row in `results/summary.csv`.
- Created all 53 Ticket 2 experiment files, including predeclared hypotheses, seven dev prediction sets, complete error/change tables, perturbation evidence, the frozen decision, and the single held-out evidence set.
- Created `predictions/ticket-2-heldout-predictions.csv` and appended the Ticket 2 row to `results/summary.csv`.
- Created all 65 Ticket 3 experiment files, including the predeclared plan, ten controlled dev variants, coefficient tables, complete transition/error tables, robustness probes, the frozen decision, and held-out reporting evidence.
- Created `predictions/ticket-3-heldout-predictions.csv` and appended the Ticket 3 row to `results/summary.csv`.
- Created all 56 Ticket 4 experiment files, including the predeclared bounded plan, eight reported dev candidates, complete predictions/error/change tables, coefficient evidence, the frozen decision, and the single held-out evaluation.
- Created the required 61-row `results/threshold_sweep.csv`, `predictions/ticket-4-heldout-predictions.csv`, and the fourth `results/summary.csv` row.
- Created all 46 Ticket 5 experiment files, including predeclared duplicate/confidence policy, exact/canonical/near candidates, complete dev/held-out error reviews, the eight-label controlled probe, freeze, full cross-split audit, and final audit manifest.
- Created `results/data_quality_audit.csv`, `predictions/ticket-5-heldout-predictions.csv`, and the fifth `results/summary.csv` row.
- Created `configs/frozen_decisions.json`, all 50 files under `experiments/final-reproducibility-audit/`, and `predictions/final-heldout-predictions.csv` for Step 10.

Documentation/status:

- Created and updated root `README.md` with environment, test, and Step 4 reproduction commands.
- Created the detailed `tickets/ticket-1-baseline.md` diagnosis.
- Created the detailed `tickets/ticket-2-normalization.md` investigation.
- Created the detailed `tickets/ticket-3-shortcuts.md` feature and shortcut audit.
- Created the detailed `tickets/ticket-4-decision-rule.md` precision-recall and model audit.
- Created the detailed `tickets/ticket-5-data-quality.md` duplicate, annotation, and error audit.
- Created `logs/chat.md` as a curated, non-verbatim AI interaction and verification record.
- Created `report_evidence_audit.md` after reading all requested repository evidence; no report outline, report draft, IEEE template, or PDF was created.
- Created `report_outline.md` with the complete evidence-bound IEEE conference paper structure, table/figure register, appendix boundary, and subsection-level provenance requirements.
- Created and updated `results/report_evidence_matrix.csv`; it now contains 73 section/claim and generated-asset evidence rows.
- Created `scripts/generate_report_assets.py` and generated 14 source/LaTeX table pairs, four PNG/SVG figures with source CSVs, a 16-row stable-ID case ledger, `report_assets/asset_manifest.csv`, `report_assets/validation_report.json`, and `report_assets/README.md`.
- Created the complete IEEE conference source `report/main.tex`, repository-supported `report/references.bib`, `report/README.md`, `scripts/validate_report_source.py`, and the 122-check `report/source_validation.json`; no PDF was generated.
- Step 5 inserted the verified authors, checked public citation URLs, regenerated all 40 report assets, updated the evidence matrix, and reformatted only report presentation where required for legibility.
- Step 5 created `scripts/build_report.ps1`, `scripts/verify_final_report.py`, `report/final_verification.json`, `report_verification_checklist.md`, and the final root `report.pdf`.
- The final source validator records 123 passing checks; the final verifier records 209 passing checks; the final PDF has 12 visually inspected US-letter pages and SHA-256 `a1b81e2a29963ca755f00b2c63170e087f3468dda05a49a177c4d23782384275`.
- Created and repeatedly updated `PROJECT_STATUS.md`; this file is now the authoritative current handoff.

Instructor files that remain unchanged:

- `topic-a-handout.md`
- `teacher_clarifications.md`
- `starter/README.md`
- `starter/data/README_DATA.md`
- `starter/data/split_indices.json`
- `starter/configs/project_contract.json`
- Hidden starter metadata such as `starter/.gitignore` and `starter/.DS_Store`

## Final deliverables

- `report/main.tex` and `report/references.bib`
- `report.pdf` (12 pages; SHA-256 `a1b81e2a29963ca755f00b2c63170e087f3468dda05a49a177c4d23782384275`)
- `report_verification_checklist.md`
- `report/final_verification.json`
- `scripts/build_report.ps1` and `scripts/verify_final_report.py`

All requested report deliverables are present. Remaining warnings are limited to unsupplied affiliation/location/email, repository-local course references without public author/date metadata, non-visible LaTeX underfull/font-substitution warnings, and the substantive unresolved Ticket 1 reference discrepancy.

## Exact next steps

Final submission validation is complete. Stop here. Do not execute any completed held-out runner, clean audit replay, or new experiment.

The completed Step 10 validation established:

1. All five final recipes, freeze hashes, decision evidence, and archived artifact hashes are consolidated in `configs/frozen_decisions.json`.
2. Every decision was frozen from train/dev evidence without held-out tuning, and selection remained closed during audit replay.
3. Five distinct clean processes reproduced all archived dev/held-out labels and predictions; maximum score drift was `1.1102230246251565e-16`.
4. All scalar metrics reproduced within `1e-15`, and all five replay fits converged without warnings.
5. `results/summary.csv`, `results/threshold_sweep.csv`, and `results/data_quality_audit.csv` each regenerated byte-for-byte from preserved evidence.
6. All fixed/new FP/FN counts were recomputed against the same Ticket 1 held-out baseline and match the summary.
7. `predictions/final-heldout-predictions.csv` has the exact six-column schema, 1,523 unique stable IDs in fixed order, and SHA-256 `8e3644db78fea2a9986a0962e8723eb115c82666ed33461578217021f9a66192`.
8. No unexplained stale, contradictory, duplicate, or manually edited active result artifact was detected; intentional semantic equivalences are documented.
9. The audit report is `experiments/final-reproducibility-audit/reproducibility_audit.md`; it is an audit record, not the final project report.
10. `pytest` collected 56 tests and all 56 passed; `pip check`, Python bytecode compilation, requirements-lock comparison, and explicit final schema/ID checks passed. The latest audit-time run completed in 2.37 seconds.

### Later stages

All five investigation tickets, the freeze/reproduction audit, report evidence audit, report structure/evidence matrix, verified report assets, IEEE source, final PDF, citation review, page inspection, and final submission validation are complete. Do not reopen ticket decisions using held-out outcomes.

The machine-checkable tables, static figures, stable-ID case ledger, prediction interface, README commands, AI-use log, IEEE report prose, build script, checklist, and final PDF now exist. No experimental work remains in scope.

## Unresolved issues and cautions

- Ticket 1 is complete, but the reference discrepancy remains causally unresolved because the contract does not disclose the reference implementation's full effective configuration or package versions.
- One historical Ticket 1 primary comparison and one explicitly authorized Ticket 1 artifact-recovery replay occurred before Step 10. Step 10 then performed one explicitly requested audit-only clean replay for each frozen ticket; these replays did not reopen selection or change the historical primary counts. Do not run held-out again without explicit authorization.
- The restored held-out prediction rows, metrics, confusion matrix, FP/FN tables, freeze, summary row, and dev probes are present.
- Ticket 2 is complete. Its held-out evaluation count is 1; do not rerun it or use its held-out result to revise URL normalization.
- URL placeholdering's dev F1 gain is small and was selected using combined dev error-transition and robustness evidence, not score alone.
- The declared emoji ranges matched zero dataset rows, so Ticket 2 makes no dataset-level claim about emoji usefulness.
- Ticket 3 is complete. Its selected decision retains the frozen Ticket 1 baseline; its one post-freeze held-out action reused validated predictions and did not refit or repredict.
- Ticket 3's visible shallow-feature dev gain was rejected because keyword masking caused severe degradation; do not use its held-out result to reopen that decision.
- Ticket 4 is complete. Its held-out evaluation count is 1; do not rerun it or use its small held-out F1 gain to revise the balanced-class decision.
- Ticket 4's selected model improves recall while lowering precision and accuracy. The held-out F1 delta is only `+0.0013829448328488425`, so the report does not claim broad superiority.
- Ticket 4's narrative now uses the machine-verified held-out ID `767` scores `0.4876230020837692 -> 0.5448153095708489` from `experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv`.
- Ticket 5 is complete. Its one post-freeze report reused Ticket 4 predictions without fitting or repredicting; do not rerun it or reopen the no-correction decision.
- Ticket 5 `fix` dispositions are recommendations only. No source, dev, or held-out label was changed, and no held-out row was removed.
- The eight-row controlled training correction set was rejected because dev F1 fell by `0.003219557290800479` and it created more errors than it fixed; this does not prove the original labels are semantically correct.
- The retained Step 4 `run_config.json` says `not_frozen_and_not_evaluated` because it truthfully describes the earlier Step 4 state; later Ticket 1 chronology is recorded separately.
- The local `train.csv` is ignored by Git and must be downloaded again in a fresh clone using the starter source.
- The workspace has no usable Git history for recovering or comparing changes.
- Matplotlib may require a writable `MPLCONFIGDIR` in this restricted sandbox.
- All five tickets, the Step 10 reproducibility audit, and the final report/submission verification are complete.
- The final prediction core is Ticket 5/Ticket 4 by design even though Ticket 2 has the highest observed held-out F1. Choosing Ticket 2 after seeing held-out would violate the dev-only selection rule, so the manifest correctly retains Ticket 5's frozen decision.
- Floating-point scores reproduced within `1e-12`, not byte-for-byte inside replay CSVs; labels, stable IDs, order, metrics, and active result tables reproduced exactly under their stated tolerances.
- Stop here. Do not begin a new experiment.

## Rollback and recovery record for Ticket 1

The following content was removed during the user-requested rollback:

- `experiments/ticket-1/` in full, including freeze, held-out, and dev-probe artifacts.
- `pipeline/run_ticket1_heldout.py`.
- `pipeline/run_ticket1_probes.py`.
- `tests/test_ticket1.py`.
- `predictions/heldout_predictions.csv`; the now-empty `predictions/` directory was removed.
- `results/summary.csv`; the now-empty `results/` directory was removed.
- Step 5-specific Python and pytest cache files.

Retained unchanged:

- All instructor files and the fixed split.
- `data/train.csv`.
- The complete reusable pipeline skeleton from Step 3.
- The floor and raw-text baseline implementation from Step 4.
- All `experiments/step-4-baselines/` dev artifacts.
- The original 12-test Step 4 test suite.

It was subsequently restored after explicit user authorization:

- The original freeze JSON was reconstructed byte-for-byte; SHA-256 is again `3b1f589fb5b445cf146e63ad176dd5255e99aa342cc0502c8b8df657945ee3e8`.
- `pipeline/run_ticket1_heldout.py`, `pipeline/run_ticket1_probes.py`, and `tests/test_ticket1.py` were recreated.
- The frozen model was replayed once solely to reconstruct stable held-out artifacts. Recovery ledgers explicitly record `historical_primary_comparison_count=1` and `artifact_recovery_rerun=1`.
- `predictions/heldout_predictions.csv` and `results/summary.csv` were restored.
- All nine dev-only controlled probes and their 45 probe artifacts were regenerated; no held-out evaluation occurred in the probe command.
- `PROJECT_STATUS.md` was updated to preserve the full chronology.
