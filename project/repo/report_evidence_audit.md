# Report Evidence Audit

Audit date: 2026-07-21 (Asia/Shanghai)

Scope: evidence reconstruction only. This file is not a report outline, report draft, IEEE template, or held-out rerun. No experimental result, prediction, label, frozen decision, or held-out artifact was modified during this audit.

## Evidence-state definitions

- **VERIFIED**: supported by a machine-generated artifact, independently recomputed from prediction rows during this audit, or confirmed by a current non-refitting validation command.
- **UNCERTAIN**: documented or plausible, but not independently established by preserved machine evidence; this includes semantic label judgments without external adjudication.
- **MISSING**: required or useful evidence is absent.
- **CONFLICTING**: two preserved sources disagree. The machine-generated, reproducible source is preferred where one exists.

## Audit coverage and evidence precedence

### Files and context inspected

- **VERIFIED**: Read the project handout, root `README.md`, starter `README.md`, starter data README, `teacher_clarifications.md`, `PROJECT_STATUS.md`, `logs/chat.md`, the fixed split, project contract, reproducibility configuration, consolidated frozen-decision manifest, requirements files, and all five ticket documents.
- **VERIFIED**: Read, parsed, and hashed every file under `experiments/`, `results/`, and `predictions/`: 351 files, 14,077,108 bytes, comprising 240 CSV files, 86 JSON files, 19 TXT files, and 6 Markdown files. The CSVs contained 124,220 data rows in total. No file was empty; all CSV and JSON files parsed successfully.
- **VERIFIED**: Read all 35 Python source files under `pipeline/` and all 16 Python test files under `tests/` (51 files, 306,679 bytes, 6,891 lines). No source or test file was empty.
- **VERIFIED**: Searched the full repository, including hidden files outside `.git`, `.venv`, and cache directories, for report sources or drafts. No `report.pdf`, report source, draft, `.tex`, `.docx`, `.qmd`, or notebook exists.
- **VERIFIED**: The only previous-conversation evidence preserved in the repository is the curated `logs/chat.md` plus chronology in `PROJECT_STATUS.md` and generated run/freeze ledgers.
- **MISSING**: No verbatim earlier Codex conversation transcript or recoverable Git history is accessible in this task. The current user request is accessible, but earlier raw messages cannot be audited directly.

### Evidence precedence used in this audit

1. Prediction CSVs and the immutable fixed ID split.
2. Machine-generated metrics, confusion matrices, transition tables, selection results, freeze JSON, run configurations, hashes, and completion ledgers.
3. Source code and tests that generate or validate those artifacts.
4. Ticket narratives, README, status file, and AI-use log.
5. Unsupported comments or recollections.

When narrative and generated evidence disagree, this audit uses the generated evidence and records the conflict.

## 1. Project-wide verified workflow

1. **VERIFIED - assignment contract**: The task is binary disaster-tweet classification with five required forensic tickets. Decisions must be made on the fixed dev split; held-out is for post-freeze reporting. Sources: `topic-a-handout.md`, `starter/README.md`, `starter/configs/project_contract.json`.
2. **VERIFIED - instructor clarifications**: The floor predicts the train-majority class (`0`) everywhere. All `fixed_fp`, `fixed_fn`, `new_fp`, and `new_fn` values must compare with the same frozen Ticket 1 baseline, not the previous ticket. Source: `teacher_clarifications.md`.
3. **VERIFIED - input integrity**: `data/train.csv` has 7,613 rows and 7,613 unique IDs; SHA-256 is `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`. The fixed split has 4,567 train, 1,523 dev, and 1,523 held-out IDs; all 7,613 IDs are unique across the three lists and overlaps are zero. Split SHA-256 is `db2fd1fdcc24043dd40ed202efe2c6cc19183d75de2becb2bb8645e99d8988f1`. Sources: `data/train.csv`, `starter/data/split_indices.json`, `pipeline/data.py`, `pipeline/splits.py`, `tests/test_splits.py`.
4. **VERIFIED - reproducibility settings**: Seed `3102`, one job/thread where supported, locked environment hash `5bcff454cf416c8759e4e2a46beb1f9aedbf2637dd2b455da0d2b3242682a4d5`. Sources: `configs/reproducibility.json`, `requirements-lock.txt`, per-run `software_versions.json`, `pipeline/reproducibility.py`.
5. **VERIFIED - baseline wiring**: The majority floor and raw-text TF-IDF plus Logistic Regression were fitted on train only and evaluated on dev. Two baseline dev repetitions were identical. Sources: `experiments/step-4-baselines/run_config.json`, `run_notes.json`, `results/dev_metrics.csv`, `predictions/*.csv`, `warnings.json`.
6. **VERIFIED - Ticket 1**: The minimal raw-text baseline was frozen, then compared with the reference. It did not match. Diagnostic probes were dev-only and did not reopen the freeze. A later artifact-recovery replay reconstructed deleted row-level evidence under an explicit recovery mode. Sources: `experiments/ticket-1/frozen_baseline_config.json`, `heldout/primary_contract_comparison.json`, `heldout/heldout_evaluation_started.json`, `heldout/heldout_evaluation_completed.json`, `probes/`.
7. **VERIFIED - Ticket 2**: Six one-lever normalization variants and a raw control were evaluated on dev. URL placeholdering was selected using dev metrics, error transitions, and URL perturbation invariance, frozen, and then evaluated once on held-out. Sources: `experiments/ticket-2/dev/`, `experiments/ticket-2/frozen_decision.json`, `experiments/ticket-2/heldout/`.
8. **VERIFIED - Ticket 3**: Ten text/metadata/shallow-feature variants were evaluated on dev. The best visible candidate was rejected because of shortcut sensitivity. The frozen choice retained Ticket 1 text-only; held-out reporting reused the validated Ticket 1 prediction core without refitting. Sources: `experiments/ticket-3/dev/`, `experiments/ticket-3/frozen_decision.json`, `experiments/ticket-3/heldout/`.
9. **VERIFIED - Ticket 4**: A bounded 61-threshold sweep, class weighting, five `C` values, and one LinearSVC comparator were evaluated on dev. Balanced Logistic Regression at threshold 0.50 won the predeclared dev F1 rule, was frozen, and was evaluated once on held-out. Sources: `experiments/ticket-4/dev/`, `results/threshold_sweep.csv`, `experiments/ticket-4/frozen_decision.json`, `experiments/ticket-4/heldout/`.
10. **VERIFIED - Ticket 5**: Exact, canonical, and near-duplicate audits were separated into train/dev decision evidence and post-freeze full-dataset inspection. Eight proposed train-label corrections were tested only in memory; the candidate failed the dev gate. Ticket 5 retained Ticket 4, changed no source/dev/held-out label, removed no held-out row, and reused Ticket 4 predictions. Sources: `experiments/ticket-5/dev/`, `experiments/ticket-5/frozen_decision.json`, `experiments/ticket-5/heldout/`, `results/data_quality_audit.csv`.
11. **VERIFIED - final reproducibility audit**: Five clean processes reproduced archived dev and held-out prediction labels with maximum score drift `1.1102230246251565e-16`; active result tables reproduced byte-for-byte; selection remained closed. Sources: `configs/frozen_decisions.json`, `experiments/final-reproducibility-audit/reproducibility_verification.json`, `comparison.json` in each replay directory, and the reproduced result tables.
12. **VERIFIED - final frozen submission identity**: The final prediction core is Ticket 5, which is intentionally the unchanged Ticket 4 balanced model with no label corrections. Ticket 2 has the highest observed held-out F1, but selecting it after held-out inspection would violate the decision protocol. Sources: `configs/frozen_decisions.json`, `predictions/final-heldout-predictions.csv`, `results/summary.csv`.

## 2. Verified quantitative results

All values below were independently recomputed during this audit from the indicated saved prediction CSVs. They agree with the corresponding metric CSVs, confusion matrices, completion JSON, frozen manifest, and `results/summary.csv`.

| Ticket / split | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP | Fixed FP | Fixed FN | New FP | New FN | State |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ticket 1 dev selected | 0.7909407666 | 0.6931297710 | 0.7388120423 | 0.7892317794 | 748 | 120 | 201 | 454 | 0 | 0 | 0 | 0 | **VERIFIED** |
| Ticket 1 held-out | 0.8013937282 | 0.7033639144 | 0.7491856678 | 0.7977675640 | 755 | 114 | 194 | 460 | 0 | 0 | 0 | 0 | **VERIFIED** |
| Ticket 2 dev selected | 0.8046594982 | 0.6854961832 | 0.7403132729 | 0.7931713723 | 759 | 109 | 206 | 449 | 22 | 6 | 11 | 11 | **VERIFIED** |
| Ticket 2 held-out | 0.8251366120 | 0.6926605505 | 0.7531172070 | 0.8049901510 | 773 | 96 | 201 | 453 | 22 | 8 | 4 | 15 | **VERIFIED** |
| Ticket 3 dev selected | 0.7909407666 | 0.6931297710 | 0.7388120423 | 0.7892317794 | 748 | 120 | 201 | 454 | 0 | 0 | 0 | 0 | **VERIFIED** |
| Ticket 3 held-out | 0.8013937282 | 0.7033639144 | 0.7491856678 | 0.7977675640 | 755 | 114 | 194 | 460 | 0 | 0 | 0 | 0 | **VERIFIED** |
| Ticket 4 dev selected | 0.7469879518 | 0.7572519084 | 0.7520849128 | 0.7852921865 | 700 | 168 | 159 | 496 | 0 | 42 | 48 | 0 | **VERIFIED** |
| Ticket 4 held-out | 0.7443609023 | 0.7568807339 | 0.7505686126 | 0.7839789888 | 699 | 170 | 159 | 495 | 0 | 35 | 56 | 0 | **VERIFIED** |
| Ticket 5 dev selected | 0.7469879518 | 0.7572519084 | 0.7520849128 | 0.7852921865 | 700 | 168 | 159 | 496 | 0 | 42 | 48 | 0 | **VERIFIED** |
| Ticket 5 held-out | 0.7443609023 | 0.7568807339 | 0.7505686126 | 0.7839789888 | 699 | 170 | 159 | 495 | 0 | 35 | 56 | 0 | **VERIFIED** |

Notes:

- **VERIFIED**: Dev transitions in this table use the Ticket 1 dev prediction comparator; held-out transitions use the Ticket 1 held-out comparator. Sources: each ticket's saved `changes` tables and `experiments/final-reproducibility-audit/transition_recalculation.csv`.
- **VERIFIED**: Ticket 5's rejected correction candidate had precision `0.7421289355`, recall `0.7557251908`, F1 `0.7488653555`, accuracy `0.7820091924`, and TN/FP/FN/TP `696/172/160/495`. Versus Ticket 4 it fixed 3 FP and 0 FN, introduced 7 FP and 1 FN, and changed 11 predictions. Sources: `experiments/ticket-5/dev/correction_experiment/candidate_dev_predictions.csv`, `dev_metrics.csv`, `changes_vs_ticket4.csv`, `selection_result.json`.
- **VERIFIED**: The all-zero floor had dev precision/recall/F1 `0/0/0`, accuracy `0.5699277741`, and TN/FP/FN/TP `868/0/655/0`. Sources: `experiments/step-4-baselines/results/dev_metrics.csv`, floor prediction CSV.

## 3. Ticket 1 - baseline discrepancy diagnosis

1. **Required question - VERIFIED**: Does the independently implemented raw-text TF-IDF plus Logistic Regression baseline reproduce the reference; if not, what explains the gap? Source: `topic-a-handout.md`, `tickets/ticket-1-baseline.md`.
2. **Hypothesis - VERIFIED as pre-freeze record**: The literal version-locked raw-text default pipeline, with only seed `3102` supplied, would match the reference within tolerance. Source: `experiments/ticket-1/frozen_baseline_config.json`, `experiments/ticket-1/freeze_decision.md`.
3. **Controlled setup - VERIFIED**: Stable-ID fixed split; train-only fit; dev-only pre-freeze evidence; raw `text` only; default word-unigram `TfidfVectorizer`; `LogisticRegression(C=1.0, class_weight=None, solver='lbfgs', max_iter=100, random_state=3102)`; no manual normalization, metadata, threshold tuning, or class weighting; one CPU job at project level. Sources: freeze JSON, `pipeline/modeling.py`, `pipeline/baselines.py`, `experiments/step-4-baselines/run_config.json`.
4. **Intended lever - VERIFIED**: Ticket 1 freezes the minimal baseline; subsequent discrepancy probes change one of casing, n-grams, solver, `C`, iterations, class weighting, seed, explicit L2 representation, or feature-fit scope. Source: `experiments/ticket-1/probes/probe_plan.json`.
5. **Baseline/comparison - VERIFIED**: Train-majority floor for wiring and the frozen minimal baseline for every probe. Source: baseline dev metrics and probe metrics.
6. **Dev results - VERIFIED**: Selected baseline metrics are in the master table. Negative/diagnostic probe results: lowercase off F1 `0.7311475410`; bigrams `0.7287449393`; liblinear `0.7377850163`; `C=0.5` `0.7266721718`; leaky train+dev TF-IDF `0.7252368648`; `max_iter=1000`, seed `9999`, and explicit L2 produced no label or metric change; balanced weights increased dev F1 to `0.7520849128` but were diagnostic and not adopted. Source: `experiments/ticket-1/probes/dev_probe_metrics.csv` and prediction/change/error files.
7. **Dev decision - VERIFIED**: Retain the frozen minimal baseline; do not use post-mismatch probes to revise it. Source: freeze JSON and `configs/frozen_decisions.json`.
8. **Frozen configuration - VERIFIED**: `experiments/ticket-1/frozen_baseline_config.json`, SHA-256 `3b1f589fb5b445cf146e63ad176dd5255e99aa342cc0502c8b8df657945ee3e8`.
9. **Held-out results - VERIFIED**: Master table values. Reference F1 `0.7574221578566256`; actual `0.749185667752443`; absolute gap `0.008236490104182592`, exceeding tolerance `0.001`; `matches_reference=false`. Sources: `starter/configs/project_contract.json`, `experiments/ticket-1/heldout/primary_contract_comparison.json`, held-out predictions/metrics/confusion.
10. **Metrics and confusion - VERIFIED**: See master table; independently recomputed from `predictions/heldout_predictions.csv`.
11. **Fixed/new errors - VERIFIED**: The selected Ticket 1 model is the comparator, so selected-model transitions are all zero. Probe-specific transition counts are in `experiments/ticket-1/probes/dev_probe_metrics.csv` and `changes/`.
12. **Representative examples - VERIFIED for IDs, scores, labels, and transitions**: dev ID `237` (lowercasing off, new FN, `0.5265357370 -> 0.4364927077`); ID `241` (bigrams, fixed FP); ID `5282` (liblinear, new FN, `0.5008256985 -> 0.4985974780`); ID `137` (`C=0.5`, new FN); IDs `105`/`110` (balanced weighting, fixed FN/new FP); ID `97` (invalid leaky fit, new FN). Held-out ID `198` is a baseline FP at `0.7129798248`; ID `17` is a baseline FN at `0.3400404344`. Exact sources: the corresponding files under `experiments/ticket-1/probes/changes/` and `experiments/ticket-1/heldout/heldout_false_{positives,negatives}.csv`.
13. **Semantic interpretation - UNCERTAIN in part**: The examples support sensitivity mechanisms, but they do not identify which undisclosed reference setting caused the external gap.
14. **Final conclusion - VERIFIED**: The local baseline is deterministic, converged, stable-ID aligned, train-only, and does not reproduce the contract. The unique cause is unresolved.
15. **Main limitation - VERIFIED**: The reference exposes only a scalar F1 and tolerance, not predictions, full effective parameters, preprocessing code, solver diagnostics, or package versions. Sources: project contract and absence from the starter package.
16. **Artifact support - VERIFIED**: `experiments/step-4-baselines/`; `experiments/ticket-1/frozen_baseline_config.json`; `experiments/ticket-1/probes/`; `experiments/ticket-1/heldout/`; `predictions/heldout_predictions.csv`; Ticket 1 row of `results/summary.csv`.

## 4. Ticket 2 - text normalization lever

1. **Required question - VERIFIED**: Do URL, mention, hashtag, punctuation, casing, or emoji decisions help or hurt, and do moved errors match the hypothesis?
2. **Hypotheses - VERIFIED as pre-execution records**: One separate hypothesis for the raw control and each of six surface transformations. Source: `experiments/ticket-2/dev/experiment_plan.json`.
3. **Controlled setup - VERIFIED**: Same train/dev IDs, TF-IDF, classifier, threshold, and seed as Ticket 1; identity control or exactly one normalization switch; held-out rows not loaded by dev command. Sources: plan, `run_config.json`, `pipeline/normalization.py`, `pipeline/ticket2.py`.
4. **Intended lever - VERIFIED**: URL placeholdering, mention placeholdering, hashtag-marker removal, Unicode punctuation-to-space, Unicode casefolding, or emoji placeholdering, one at a time.
5. **Baseline/comparison - VERIFIED**: Raw identity normalizer reproduces Ticket 1 dev predictions and metrics.
6. **Dev results - VERIFIED**: URL F1 `0.7403132729`; mention `0.7367563162`; hashtag `0.7388120423`; punctuation `0.7394136808`; casefold `0.7388120423`; emoji `0.7388120423`; raw `0.7388120423`. Source: `experiments/ticket-2/dev/results/dev_metrics.csv` and all seven dev prediction files.
7. **Dev decision - VERIFIED**: Select URL placeholdering because it had the highest normalization F1, improved precision/accuracy, fixed 28 errors while creating 22, and was invariant on 767 URL-perturbed dev rows while raw changed 275 predictions. Source: `experiments/ticket-2/frozen_decision.json`, robustness metrics.
8. **Frozen configuration - VERIFIED**: Replace `(?i)\b(?:https?://|www\.)\S+` with `URLTOKEN`; all other Ticket 1 settings unchanged. Freeze SHA-256 `51a9ed0c07d092fa194a1f1399ae502eb0c3662bfdfb2b661ef22dfcbc5376cf`.
9. **Held-out results - VERIFIED**: See master table. The post-freeze pattern is precision up, recall down, F1 and accuracy up relative to Ticket 1.
10. **Metrics/confusion - VERIFIED**: See master table; prediction-row recomputation matches `heldout_metrics.csv` and confusion CSV.
11. **Fixed/new errors - VERIFIED**: Dev `22 FP + 6 FN` fixed, `11 FP + 11 FN` introduced. Held-out `22 FP + 8 FN` fixed, `4 FP + 15 FN` introduced.
12. **Representative examples - VERIFIED for ID/score/transition**: Dev fixed FP IDs `174` and `971`; new FN `237`; new FP `110`; mention new FN `2352`; punctuation boundary changes `3903` and `3097`. Held-out fixed FP IDs `773` and `936`; new FN `902`; new FP `2569`. Sources: exact files under `experiments/ticket-2/dev/changes/` and `experiments/ticket-2/heldout/heldout_changes_vs_frozen_baseline.csv`.
13. **Semantic interpretation - UNCERTAIN in part**: URL canonicalization demonstrably removes link-string sensitivity, but the natural deployment frequency and semantic value of link destinations were not measured.
14. **Final conclusion - VERIFIED**: Adopt URL placeholdering for Ticket 2 only; do not claim a generic cleaning bundle.
15. **Main limitation - VERIFIED**: Small single-split gain with no resampling uncertainty; deterministic perturbations are stress tests; recognized emoji affected zero dataset rows.
16. **Artifact support - VERIFIED**: `experiments/ticket-2/dev/experiment_plan.json`; `dev/results/`; `dev/predictions/`; `dev/changes/`; `dev/errors/`; `dev/robustness/`; freeze files; `heldout/`; `predictions/ticket-2-heldout-predictions.csv`; Ticket 2 summary row.

## 5. Ticket 3 - feature and shortcut audit

1. **Required question - VERIFIED**: How much signal comes from keyword, length, location, and shallow artifacts, and is it legitimate task information, a dataset artifact, or both?
2. **Hypotheses - VERIFIED as pre-execution records**: Ten variant hypotheses and a rule rejecting score-only selection. Source: `experiments/ticket-3/dev/experiment_plan.json`.
3. **Controlled setup - VERIFIED**: Train-fitted pipelines; stable split; unchanged classifier rule; isolated keyword/location/length/shallow blocks and controlled text combinations; explicit missing-value sentinels; held-out not loaded by the dev command. Sources: plan, run config, `pipeline/shortcut_features.py`.
4. **Intended lever - VERIFIED**: Addition or isolation of keyword, location, length, selected shallow counts, and their declared combinations with raw text.
5. **Baseline/comparison - VERIFIED**: Exact Ticket 1 text-only dev predictions.
6. **Dev results - VERIFIED**: keyword-only F1 `0.6598586017`; length-only `0.4376130199`; keyword+length `0.6650602410`; location-only `0.2244143033`; keyword+location `0.6586345382`; shallow-only `0.5302897278`; text+keyword `0.7350835322`; text+keyword+shallow `0.7493956487`; text-only `0.7388120423`. Source: `experiments/ticket-3/dev/results/dev_metrics.csv`.
7. **Dev decision - VERIFIED**: Reject the best visible rich candidate and retain text-only because keyword masking changed 702 predictions and reduced F1 to `0.6423057128`; superficial neutralization reduced it to `0.7330677291`; text+keyword alone was worse than baseline. Source: robustness metrics and frozen decision.
8. **Frozen configuration - VERIFIED**: Ticket 1 raw-text baseline, no Ticket 3 feature additions. Freeze SHA-256 `651905972507d06427d427e3e3f0a219faba48d8dff2183dbd86e4c194cf0ac6`.
9. **Held-out results - VERIFIED**: Same prediction core and metrics as Ticket 1; copied post-freeze without a new fit or prediction pass.
10. **Metrics/confusion - VERIFIED**: See master table.
11. **Fixed/new errors - VERIFIED**: Selected model has zero transitions. The rejected rich candidate changed 164 dev predictions: fixed 41 FP/46 FN; introduced 42 FP/35 FN.
12. **Representative examples - VERIFIED for ID/score/transition**: keyword-only IDs `7` (fixed FN) and `25` (same sentinel score, new FP); length-only ID `4` (new FN); location-only IDs `86` (fixed FP) and `16` (new FN); text+keyword IDs `105`/`110`; rich candidate IDs `18` (fixed FN), `331` (new FP), `519` (new FN). Sources: exact files under `experiments/ticket-3/dev/changes/`.
13. **Signal interpretation - VERIFIED as an evidence judgment, not universal fact**: Raw text is legitimate task information; keyword is mixed evidence; length/missingness/full locations are dataset artifacts in this benchmark; surface counts are mixed. Source: plan, metrics, robustness, metadata profile, coefficient table.
14. **Final conclusion - VERIFIED**: The visible rich-model dev gain is not trustworthy enough to adopt because it is heavily keyword-availability dependent.
15. **Main limitation - VERIFIED**: One dev split, severe mask-to-missing stress tests, correlated coefficient magnitudes, and no external deployment contract for metadata availability.
16. **Artifact support - VERIFIED**: `experiments/ticket-3/dev/experiment_plan.json`; `results/`; `predictions/`; `changes/`; `errors/`; `robustness/`; `interpretability/top_coefficients.csv`; freeze files; held-out copy/ledgers; stable prediction CSV; summary row.

## 6. Ticket 4 - decision rule and model

1. **Required question - VERIFIED**: Can threshold tuning, class weighting, regularization, or a second CPU classifier improve the operating point, and what precision-recall tradeoff results?
2. **Hypotheses - VERIFIED as pre-execution records**: Lower threshold and balanced weighting should raise recall at a precision cost; bounded `C` changes may alter sparse-feature separation; LinearSVC may learn a different boundary. Source: `experiments/ticket-4/dev/experiment_plan.json`.
3. **Controlled setup - VERIFIED**: Raw text and Ticket 1 TF-IDF fixed; 61 thresholds from 0.20 to 0.80 applied only to baseline scores; `C` in `{0.25,0.5,1,2,4}`; balanced comparison at `C=1`, threshold 0.50; one `LinearSVC(C=1)`; dev-only selection.
4. **Intended lever - VERIFIED**: One bounded decision-rule or classifier lever at a time.
5. **Baseline/comparison - VERIFIED**: Ticket 1 raw-text baseline at 0.50.
6. **Dev results - VERIFIED**: Best threshold-only `0.47`, F1 `0.7494071146`; best regularization-only `C=2`, F1 `0.7504025765`; balanced LR F1 `0.7520849128`; LinearSVC F1 `0.7408585056`; full metrics in `experiments/ticket-4/dev/results/dev_model_metrics.csv`. Threshold sweep has exactly 61 rows and matches `results/threshold_sweep.csv`.
7. **Dev decision - VERIFIED**: Balanced Logistic Regression wins the predeclared dev target-1 F1 criterion.
8. **Frozen configuration - VERIFIED**: Raw word-unigram TF-IDF; Logistic Regression; `C=1.0`; `class_weight='balanced'`; inclusive threshold `0.50`; seed `3102`. Freeze SHA-256 `14e3c474f15a04a9997bada41d477447ea2e3c7e49533b8980edfac7896b371e`.
9. **Held-out results - VERIFIED**: See master table. F1 delta `+0.0013829448`, precision delta `-0.0570328260`, recall delta `+0.0535168196`, accuracy delta `-0.0137885752` relative to Ticket 1.
10. **Metrics/confusion - VERIFIED**: See master table.
11. **Fixed/new errors - VERIFIED**: Dev fixed 42 FN and introduced 48 FP. Held-out fixed 35 FN and introduced 56 FP; no fixed FP or new FN.
12. **Representative examples - VERIFIED except noted conflict**: Dev IDs `105`, `110`, `353`, `390`, `394`, `402`, `840`; held-out IDs `556`, `907`, `59`, `767`, `996`, `1645` are all present with the stated transitions in the change ledgers. Sources: `experiments/ticket-4/dev/changes/lr_c1_balanced_default_changes.csv`, `experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv`.
13. **CONFLICTING - held-out ID 767 scores**: `tickets/ticket-4-decision-rule.md` states `0.481794 -> 0.544153`; the generated transition CSV states baseline `0.4876230020837692` and candidate `0.5448153095708489`. The ID, target `0`, and new-FP transition are verified. Only the CSV scores may be used in the final report.
14. **Final conclusion - VERIFIED**: Balanced LR is a defensible F1-oriented, recall-heavy operating choice under the declared criterion, not an unequivocally better classifier.
15. **Main limitation - VERIFIED**: One fixed dev split, selection over a finite grid, only one second classifier, and no explicit deployment cost model; the held-out gain is small and accuracy falls.
16. **Artifact support - VERIFIED**: plan; `dev/results/dev_model_metrics.csv`; detailed and contract threshold sweeps; all candidate predictions/changes/errors; coefficient table; selection JSON; freeze files; held-out metrics/confusion/changes/errors/ledgers; stable prediction CSV; summary row.

## 7. Ticket 5 - data quality and error analysis

1. **Required question - VERIFIED**: Which duplicates, likely mislabels, ambiguous tweets, hard negatives, or rejected audit suspicions limit the score, and should a data-quality intervention change the frozen model?
2. **Hypotheses - VERIFIED as pre-execution records**: Raw-exact conflicts indicate annotation inconsistency; canonical/near relationships surface reposts but need review; model disagreement alone is insufficient; high-confidence train corrections may help or may reduce agreement with the noisy benchmark. Source: `experiments/ticket-5/dev/audit_plan.json`.
3. **Controlled setup - VERIFIED**: Train/dev-only duplicate and error audit before freeze; exact, deterministic canonical, and bounded near-duplicate definitions; eight train-label proposals tested as one in-memory set; same Ticket 4 model otherwise; dev labels untouched; held-out rows not loaded by the correction runner.
4. **Intended lever - VERIFIED**: The only model lever is the eight-row in-memory train-label correction set. Duplicate discovery and final dispositions are audits, not hidden training changes.
5. **Baseline/comparison - VERIFIED**: Frozen Ticket 4 balanced LR with original training labels.
6. **Dev results - VERIFIED**: Train/dev duplicate counts: raw exact `50 groups/124 members/13 conflicting/19 cross-split`; canonical `214/677/46/98`; near `693 pairs/538 unique IDs/104 conflicting/265 cross-split`. Correction candidate metrics and transitions are recorded above.
7. **Dev decision - VERIFIED**: Reject corrections because F1 delta `-0.0032195573` fails the `-0.002` noninferiority margin and 3 errors fixed is fewer than 8 introduced.
8. **Frozen configuration - VERIFIED**: Retain Ticket 4 balanced LR at `C=1`, threshold `0.50`, with `training_label_corrections=[]`. Freeze SHA-256 `9423deaf6f30659f55547176a5ddb2191ed4895738cf7fe17b9d6138df693afa`.
9. **Held-out results - VERIFIED**: Same prediction core as Ticket 4, no new fit or prediction pass. Full duplicate counts after freeze: raw exact `69/179/18/41`; canonical `292/932/64/194`; near `960 pairs/715 unique IDs/186 conflicting/546 cross-split`.
10. **Metrics/confusion - VERIFIED**: See master table.
11. **Fixed/new errors - VERIFIED**: Final model versus Ticket 1: 0 fixed FP, 35 fixed FN, 56 new FP, 0 new FN. Correction candidate versus Ticket 4: 3 fixed FP, 0 fixed FN, 7 new FP, 1 new FN.
12. **Representative correction changes - VERIFIED**: Dev ID `9470` was fixed (`0.6330437169 -> 0.4652778089`); IDs `4014`, `3633`, and `9943` became new errors. Source: `experiments/ticket-5/dev/correction_experiment/changes_vs_ticket4.csv`.
13. **Representative data-quality cases - VERIFIED for stored IDs, labels, relationships, errors, and submitted dispositions; UNCERTAIN as external ground-truth adjudication**: train proposal IDs `4076`, `6566`, `8698`, `8739`, `1723`, `1760`, `6097`, `9472`; ambiguous dev pair `353/390`; leakage flags `5140`, `8183`; dev cases `5247`, `6407`, `6325`, `805`, `6002`, `9341`, `1208`, `7761`; held-out cases `6132`, `6112`, `6220/6223`, `7613`, `7804`, `1409`, `5228`, `10795`, `5760`, `783`, `4003`, `546`, `2619`, `4043`. Sources: duplicate relationship tables, dev/held-out model-error reviews, correction proposal CSV, and `results/data_quality_audit.csv`.
14. **Final conclusion - VERIFIED**: Duplicate leakage and inconsistent labels constrain score interpretation; the correction set is not justified as a model intervention; preserve labels/rows and retain Ticket 4.
15. **Main limitation - VERIFIED**: Near-duplicate search examines at most seven non-self neighbors at one threshold/representation; canonical URLs can hide destination context; the 64-row audit is curated, not exhaustive; no expert multi-annotator adjudication exists.
16. **Artifact support - VERIFIED**: audit plan; dev duplicate/review tables; correction plan and experiment; freeze; full held-out duplicate/review tables; final audit records and manifest; `results/data_quality_audit.csv`; stable prediction CSV; summary row.

## 8. Available qualitative cases and evidence boundaries

- **VERIFIED**: Every representative ID named in Sections 3-7 was located in its claimed prediction-change, error, duplicate, or audit file, except no missing ID was found.
- **VERIFIED**: Score and transition statements were checked directly against row-level CSVs. Ticket 4 ID `767` is the sole detected narrative score conflict.
- **VERIFIED**: `results/data_quality_audit.csv` has 64 unique valid IDs with exact disposition counts: 30 `fix`, 15 `ambiguous`, 6 `keep_but_flag`, 13 `reject_false_positive`.
- **VERIFIED**: All `fix` rows are recommendations only. Source hash before/after the correction probe is identical; source labels changed `0`; dev labels changed `0`; held-out labels changed `0`; held-out rows removed `0`.
- **UNCERTAIN**: Semantic dispositions are informed human/AI judgments based on stored text and duplicate evidence. They have not been externally adjudicated, and link destinations were not fetched.

## 9. Verified difficulties and solutions

| Difficulty | State | How it was addressed | Supporting evidence |
|---|---|---|---|
| The literal baseline missed the reference F1 by `0.00823649`, but the reference exposes insufficient implementation detail to isolate one cause. | **VERIFIED** | Audited split/ID discipline, train-only fitting, convergence, effective parameters, seeds, versions, and nine one-lever dev probes; retained uncertainty rather than inventing a cause. | Ticket 1 freeze, contract comparison, probe metrics/configuration audit, ticket conclusion. |
| Floating-point score replay differed by at most `1.1102230246251565e-16`, so exact byte/float equality would be too strict despite identical labels. | **VERIFIED** | Used an explicit score tolerance `1e-12`, required identical IDs/labels/order, and required metrics within `1e-15`; final audit passed. | `experiments/final-reproducibility-audit/reproducibility_verification.json`; Ticket 4 `run_config.json`; replay comparison JSONs. |
| The best visible Ticket 3 dev score depended strongly on keyword availability and superficial style. | **VERIFIED** | Added explicit masking/neutralization probes and rejected the higher-F1 candidate under the predeclared trust rule. | Ticket 3 plan, robustness metrics, frozen decision. |
| Class weighting improved recall but lowered precision and accuracy, making “improvement” operationally ambiguous. | **VERIFIED** | Preserved the full threshold sweep, confusion matrices, error transitions, and examples; described the choice as F1-oriented rather than universally superior. | Ticket 4 dev/held-out artifacts and threshold sweep. |
| Duplicate leakage, conflicting labels, and task-scope ambiguity prevent a simple model-error-equals-mislabel rule. | **VERIFIED** | Separated exact/canonical/near candidate generation from four-way dispositions; tested corrections in memory; rejected the correction set when it failed dev gates; changed no source label. | Ticket 5 duplicate tables, audit plan, correction experiment, audit table. |
| Ticket 2 emoji analysis had no in-dataset examples under the declared detector. | **VERIFIED** | Kept transformer unit tests but marked the dataset-level emoji hypothesis unresolved. | Ticket 2 robustness metrics (`affected_rows=0`), `tests/test_normalization.py`. |
| Ticket 1 row-level artifacts required a recovery-mode reconstruction after a documented rollback. | **VERIFIED for the recovery replay and counts; UNCERTAIN for the complete deletion history** | Used the exact frozen configuration under an explicit artifact-recovery flag; recorded one historical primary comparison and one recovery replay; did not retune. | Ticket 1 started/completed ledgers and contract comparison. The exact prior deletion set is described only in `PROJECT_STATUS.md`/`logs/chat.md` and cannot be independently reconstructed without Git history. |
| A first Ticket 4 dev invocation allegedly stopped on bitwise-equality guarding before writing artifacts. | **UNCERTAIN** | The final run records tolerant score comparison and passed; no preserved failure log independently verifies the first attempt. | Final run config verifies the solution, but the failed attempt exists only in narrative prose. |
| Matplotlib allegedly needed a writable `MPLCONFIGDIR` in the restricted sandbox. | **UNCERTAIN** | Status says a temporary writable directory was used, but no saved command/output proves the incident. | `PROJECT_STATUS.md` only. Do not present as a verified report difficulty. |

## 10. Verified AI usage and validation process

- **VERIFIED as repository-documented, not as a verbatim transcript**: AI assisted with repository inspection, implementation planning, Python/test drafting, experiment orchestration, artifact review, ticket documentation, and the final reproducibility audit. Source: `logs/chat.md`.
- **VERIFIED through artifacts**: AI-supported outputs were not accepted solely as prose. They were checked through fixed-ID validation, train-only pipelines, stable prediction CSVs, complete metrics/confusion/error tables, source and artifact hashes, frozen decision ledgers, tests, and clean-process replay. Sources: code/tests, per-ticket run/freeze outputs, final reproducibility audit.
- **VERIFIED**: The latest non-refitting validation during this audit collected 56 tests and passed all 56 in 2.37 seconds; `pip check` reported no broken requirements; `pipeline.validate_submission` returned `PASS` with 5 tickets, 5 summary rows, 61 threshold rows, 64 audit rows, 1,523 final prediction rows, and `refit_performed=false`.
- **UNCERTAIN**: `logs/chat.md` is explicitly reconstructed rather than verbatim. It cannot prove the exact wording, timing, or full set of earlier AI interactions, and no raw prior conversation transcript is present.
- **UNCERTAIN**: The log's generic statement that outputs received “visual inspection” is not backed by a saved visual-review artifact. The stronger machine checks are verified; the visual-inspection claim should not be emphasized.

## 11. Reproducibility checks

- **VERIFIED**: Fixed data, split, dependency, freeze, source, plan, prediction, and result hashes are recorded and validated.
- **VERIFIED**: Every held-out prediction file contains 1,523 unique IDs in exact fixed held-out order with required columns.
- **VERIFIED**: Independent metric recomputation from all selected dev and held-out prediction files matches saved metrics/confusion matrices.
- **VERIFIED**: Held-out transition counts for Tickets 1-5 recompute against the same Ticket 1 comparator and match `results/summary.csv`.
- **VERIFIED**: Five final recipes were replayed in five distinct processes; predicted labels were identical; maximum score drift was below `1e-12`; no selection reopened.
- **VERIFIED**: `results/summary.csv`, `results/threshold_sweep.csv`, and `results/data_quality_audit.csv` reproduced byte-for-byte in the final audit.
- **VERIFIED**: Ticket 1 equals Ticket 3 by deliberate model retention; Ticket 4 equals Ticket 5 and final by deliberate model retention. Their full CSV hashes differ because ticket provenance differs.
- **VERIFIED**: Current `pytest`, `pip check`, and non-refitting submission validation pass.
- **UNCERTAIN**: Bit-identical scores on other operating systems or future library versions are not established.

## 12. Failed experiments and negative results

- **VERIFIED - Ticket 1**: Reference reproduction failed. Lowercase-off, word bigrams, liblinear, `C=0.5`, and intentionally leaky TF-IDF all reduced dev F1. Increasing `max_iter`, changing seed, and spelling L2 explicitly produced no prediction change. Balanced weights improved dev F1 but were not eligible to rewrite the already frozen Ticket 1 baseline.
- **VERIFIED - Ticket 2**: Mention placeholdering reduced F1; punctuation improvement was only `+0.00060164`; hashtag stripping, casefolding, and emoji placeholdering produced no label/metric change; recognized emoji affected zero rows.
- **VERIFIED - Ticket 3**: Keyword-only, length-only, location-only, shallow-only, and text+keyword underperformed text-only. The rich candidate improved visible F1 but failed the trust/robustness rule and was rejected.
- **VERIFIED - Ticket 4**: LinearSVC did not win; `C=0.25` and `C=0.5` hurt; threshold `0.47` and `C=2` improved dev F1 but lost to balanced LR under the predeclared rule. On held-out, balanced LR's F1 gain shrank to `+0.00138294` while accuracy fell and 56 new FPs were introduced.
- **VERIFIED - Ticket 5**: The eight-row correction candidate failed both noninferiority and error-balance gates.

## 13. Missing, conflicting, stale, or unverifiable evidence

1. **MISSING**: No final report, report source, draft, IEEE template, figures, or bibliography exists.
2. **MISSING**: No verbatim earlier Codex transcript is available; `logs/chat.md` is a curated reconstruction.
3. **MISSING**: The reference baseline's predictions, exact code, effective parameter dictionary, preprocessing, solver diagnostics, and package versions are not supplied, so Ticket 1's causal explanation cannot be completed.
4. **MISSING**: No resampling, confidence interval, external test set, or duplicate-group-aware evaluation quantifies uncertainty around small metric changes.
5. **MISSING**: No expert/multi-annotator ground-truth adjudication validates Ticket 5 semantic recommendations.
6. **MISSING**: No saved output file preserves historical pytest runs. Current validation was rerun non-refitting during this audit.
7. **CONFLICTING**: Ticket 4 ID `767` scores disagree between narrative and generated transition CSV; use `0.4876230020837692 -> 0.5448153095708489` from the CSV.
8. **CONFLICTING / STALE**: `PROJECT_STATUS.md` (last modified before README/log) says `logs/chat.md` and final README commands are still missing, but both now exist. This audit update must correct that handoff state.
9. **STALE, not substantively conflicting**: Prior documentation reports pytest durations of 1.94 seconds and 2.30 seconds from different runs. The latest audit run is 2.37 seconds. Only the pass count is durable; execution time is environment/run dependent and should not be a report claim.
10. **STALE but intentionally historical**: `experiments/step-4-baselines/run_config.json` says Ticket 1 was `not_frozen_and_not_evaluated`; this accurately describes that earlier stage, not current project state.
11. **UNCERTAIN**: The detailed rollback deletion list, the alleged first Ticket 4 guard failure, removed report scaffolding, and MPL configuration incident are narrative-only without independently preserved before/after evidence.
12. **UNCERTAIN**: Near-duplicate pairs are candidates, not proof of semantic equivalence. The method is bounded and may miss or falsely connect pairs.

## 14. Claims that must not appear in the final report

- **MUST NOT CLAIM** that a specific solver, seed, version, preprocessing setting, or TF-IDF parameter caused the Ticket 1 reference discrepancy. The cause is unresolved.
- **MUST NOT CLAIM** that Ticket 1 matched the reference or that the gap was within tolerance.
- **MUST NOT CLAIM** that Ticket 2 is the final submitted model because it has the highest visible held-out F1. The final frozen submission is Ticket 5/Ticket 4; post-held-out selection would be leakage.
- **MUST NOT CLAIM** that emoji are useless. The detector matched zero dataset rows.
- **MUST NOT CLAIM** that the Ticket 3 rich feature model gives a trustworthy generalization improvement; it was rejected for shortcut sensitivity.
- **MUST NOT CLAIM** that balanced Logistic Regression is universally superior, improves accuracy, or has a large held-out advantage. It improves recall, lowers precision/accuracy, and has only a small held-out F1 gain.
- **MUST NOT USE** the stale Ticket 4 ID `767` scores from the ticket narrative.
- **MUST NOT CLAIM** that Ticket 5 corrections were applied, that `fix` dispositions changed labels, that held-out labels were edited, or that held-out rows were removed.
- **MUST NOT CLAIM** that rejection of the correction model proves the original eight labels are semantically correct.
- **MUST NOT CLAIM** that every duplicate is leakage, every conflicting duplicate has an obvious correct label, or every model error is a mislabel.
- **MUST NOT CLAIM** exhaustive label audit coverage; only 64 curated IDs were submitted, and near-duplicate search was bounded.
- **MUST NOT PRESENT** `logs/chat.md` as a verbatim conversation transcript or present unavailable raw AI reasoning as evidence.
- **MUST NOT CLAIM** bit-identical floating-point scores across arbitrary platforms. The verified local guarantee is identical labels/order and score drift at most `1e-12`.
- **MUST NOT PRESENT AS VERIFIED DIFFICULTIES** the detailed rollback deletion list, first Ticket 4 guard failure, removed report scaffolding, or MPL incident without qualification.

## 15. Source map for report claims, tables, and figures

This is a provenance map only, not a report outline.

| Planned report element | State | Primary artifact source(s) | Required caveat |
|---|---|---|---|
| Assignment question, split discipline, five-ticket method | **VERIFIED** | `topic-a-handout.md`; `starter/README.md`; `teacher_clarifications.md`; `starter/data/split_indices.json` | Held-out is post-freeze only. |
| Dataset/split integrity table | **VERIFIED** | `data/train.csv`; split JSON; `pipeline/data.py`; `pipeline/splits.py`; `tests/test_splits.py` | Data must be downloaded in a fresh clone. |
| Baseline floor and dev table | **VERIFIED** | `experiments/step-4-baselines/results/dev_metrics.csv`; confusion CSV; dev predictions | Floor is a wiring check, not competitive. |
| Ticket 1 contract comparison claim/table | **VERIFIED** | `starter/configs/project_contract.json`; `experiments/ticket-1/heldout/primary_contract_comparison.json`; held-out metrics/predictions | Cause remains unresolved. |
| Ticket 1 probe sensitivity table | **VERIFIED** | `experiments/ticket-1/probes/dev_probe_metrics.csv`; confusion CSV; change/error files | Diagnostic, not post-held-out model selection. |
| Ticket 2 normalization comparison table | **VERIFIED** | `experiments/ticket-2/dev/results/dev_metrics.csv`; confusion CSV | Small single-split deltas. |
| Ticket 2 robustness figure/table | **VERIFIED** | `experiments/ticket-2/dev/robustness/robustness_metrics.csv`; `prediction_changes.csv` | Perturbations are deterministic stress tests. |
| Ticket 2 held-out result and change examples | **VERIFIED** | `experiments/ticket-2/heldout/heldout_metrics.csv`; confusion/change/error CSVs; stable predictions | Held-out did not select the lever. |
| Ticket 3 feature-family result table | **VERIFIED** | `experiments/ticket-3/dev/results/dev_metrics.csv`; confusion CSV; metadata profile | Highest F1 candidate was rejected. |
| Ticket 3 shortcut robustness figure/table | **VERIFIED** | `experiments/ticket-3/dev/robustness/robustness_metrics.csv`; prediction changes | Masking is severe, not a deployment-frequency estimate. |
| Ticket 3 coefficient examples | **VERIFIED as associations** | `experiments/ticket-3/dev/interpretability/top_coefficients.csv` | Coefficients are not causal importance. |
| Ticket 4 model comparison table | **VERIFIED** | `experiments/ticket-4/dev/results/dev_model_metrics.csv`; selection JSON | Criterion is dev target-1 F1. |
| Ticket 4 precision-recall/threshold figure | **VERIFIED** | `results/threshold_sweep.csv`; detailed threshold sweep | No adaptive threshold refinement was run. |
| Ticket 4 held-out tradeoff table | **VERIFIED** | held-out metrics/confusion/change CSVs; stable predictions | Small F1 gain, lower precision/accuracy. |
| Ticket 4 qualitative ID `767` | **CONFLICTING narrative** | Use `experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv` | Use CSV scores, not ticket prose scores. |
| Ticket 5 duplicate-count table | **VERIFIED** | `experiments/ticket-5/dev/results/duplicate_summary.csv`; `experiments/ticket-5/heldout/full_duplicate_summary.csv` | Relationship categories overlap; do not sum them. |
| Ticket 5 correction comparison table | **VERIFIED** | correction `dev_metrics.csv`, `selection_result.json`, `changes_vs_ticket4.csv`, proposal CSV | Corrections were in-memory and rejected. |
| Ticket 5 disposition table/figure | **VERIFIED for records; UNCERTAIN for external truth** | `results/data_quality_audit.csv`; `final_audit_records.json`; `final_audit_manifest.json` | `fix` means recommendation only. |
| Cross-ticket headline result table | **VERIFIED** | `results/summary.csv`; all stable prediction CSVs; `transition_recalculation.csv` | All transitions use Ticket 1. |
| Final-model claim | **VERIFIED** | `configs/frozen_decisions.json`; `predictions/final-heldout-predictions.csv` | Ticket 5 retains Ticket 4; do not select Ticket 2 post hoc. |
| Reproducibility workflow figure | **VERIFIED** | manifest; per-ticket freezes/run commands; final reproducibility verification/replay comparisons | Audit replay did not reopen selection. |
| Difficulties and solutions table | **VERIFIED with exclusions** | Section 9 sources | Omit or qualify narrative-only incidents. |
| AI usage declaration | **VERIFIED as documented; transcript MISSING** | `logs/chat.md`; tests; run/freeze/audit artifacts | Do not call the log verbatim. |

## 16. Readiness decision

- **VERIFIED**: The repository has enough machine-supported evidence to begin report planning after this audit: all five ticket questions, hypotheses, controlled setups, dev selections, frozen configurations, held-out results, metrics, confusion counts, transition counts, negative results, stable-ID examples, limitations, reproducibility evidence, and AI-use documentation are present.
- **CONDITIONAL**: Report planning must use the machine-generated Ticket 4 ID `767` scores, correct the stale status statements, preserve Ticket 1 causal uncertainty, and distinguish Ticket 5 audit recommendations from applied changes.
- **MISSING but not blocking planning**: No report source/template, citation bibliography, external label adjudication, or uncertainty interval exists. These must not be invented.

Repository readiness for report planning: **YES, WITH DOCUMENTED CAVEATS**. Stop here until a separate instruction explicitly requests report planning or drafting.
