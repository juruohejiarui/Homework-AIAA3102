# IEEE Conference Report Structure

Planning status: evidence-bound outline only. No report prose, LaTeX source, IEEE template files, plots, or PDF are created in this step.

Evidence rule: every conclusion, number, case, table, and figure below must come from `report_evidence_audit.md` and the exact verified artifacts listed here. Items marked as limitations or uncertainty must remain qualified in the report.

## Format and narrative strategy

- Use the official IEEE Conference LaTeX template with `\documentclass[conference]{IEEEtran}`. Do not use journal mode or a society-specific class.
- Use the normal IEEE two-column conference layout, numbered Roman-numeral sections, compact tables, bracketed citations, and figure/table captions in IEEE style.
- Keep author names, affiliations, acknowledgments, and course metadata as placeholders until the user supplies them; do not invent them.
- Build one forensic narrative: establish the protocol, test progressively stronger explanations and interventions, preserve the dev/held-out boundary, and end with the conservative final decision. Do not present five disconnected activity logs.
- Keep exact commands, exhaustive configurations, long grids, extended case lists, hashes, and artifact inventories in the appendix or repository references. Do not include terminal screenshots or raw command output.
- There is no strict page limit, but the main paper should remain concise and self-contained. The appendix may carry reproducibility detail without displacing the main argument.

## Proposed title

**Text Classification Pipeline Forensics: Reproducibility, Shortcut Sensitivity, and Data Quality in Disaster Tweets**

Title evidence boundary: “reproducibility,” “shortcut sensitivity,” and “data quality” are all supported by the frozen-decision replay, Ticket 3 robustness evidence, and Ticket 5 audit. Do not add claims such as “state of the art,” “robust,” or “generalizable.”

## Front matter

### Abstract

1. **Purpose**: Summarize the problem, controlled methodology, central findings, final frozen decision, and key limitation in approximately 150–220 words when drafted.
2. **Main argument/conclusion**: Small score changes are not trustworthy without prediction-level transitions, robustness checks, freeze discipline, and data-quality analysis. The final model is Ticket 5/Ticket 4 balanced Logistic Regression with no label corrections, retained by dev evidence rather than post-hoc held-out ranking.
3. **Verified evidence to include**: Fixed split sizes; Ticket 1 reference mismatch; Ticket 2 precision-oriented URL result; Ticket 3 shortcut rejection; Ticket 4 recall/precision tradeoff; Ticket 5 correction rejection; final held-out F1 and principal limitation from duplicate/label inconsistency.
4. **Exact sources**: `report_evidence_audit.md`; `results/summary.csv`; `configs/frozen_decisions.json`; `experiments/final-reproducibility-audit/reproducibility_verification.json`; `results/data_quality_audit.csv`.
5. **Tables/figures**: None in the abstract.
6. **Representative cases**: None; avoid anecdotal detail in the abstract.
7. **Required limitation/uncertainty**: Do not imply that the reference discrepancy has a known cause, that Ticket 2 is the final model, or that the final balanced model is universally superior.

### Keywords

1. **Purpose**: Provide 4–6 searchable technical terms.
2. **Planned terms**: text classification, reproducibility, error analysis, dataset shortcuts, decision thresholds, data quality.
3. **Evidence**: These terms describe verified project activities.
4. **Exact sources**: Handout; five ticket documents as qualified by `report_evidence_audit.md`.
5. **Tables/figures**: None.
6. **Representative cases**: None.
7. **Limitation**: Do not use unsupported terms such as causal inference, domain adaptation, or uncertainty quantification.

## I. Project Problem and Goal

### A. Problem Context and Forensic Goal

1. **Purpose**: Introduce disaster-tweet binary classification and explain why a score-only comparison is insufficient.
2. **Main argument/conclusion**: The project goal is to determine which pipeline changes are reproducible and trustworthy by connecting controlled levers to metric changes, prediction transitions, and concrete cases.
3. **Verified evidence to include**: Task definition (`target=1` real disaster, `target=0` not); five investigation questions; requirement to freeze decisions before held-out reporting.
4. **Exact sources**: `topic-a-handout.md`; `starter/README.md`; `teacher_clarifications.md`.
5. **Planned tables/figures**: No standalone visual; refer forward to Table III (cross-ticket summary).
6. **Representative cases**: None here.
7. **Required limitation/uncertainty**: The benchmark label definition contains scope ambiguity; do not imply every row has an objectively recoverable label.

### B. Project Goal and Evidence Contributions

1. **Purpose**: State the report’s unified questions and contributions without narrating implementation chronology.
2. **Main argument/conclusion**: The work contributes a deterministic baseline audit, one-lever normalization study, shortcut stress audit, bounded operating-point study, data-quality audit, and frozen-decision reproducibility verification.
3. **Verified evidence to include**: Five tickets and final five-process replay; stable-ID predictions and common baseline transition rule.
4. **Exact sources**: `configs/frozen_decisions.json`; `experiments/final-reproducibility-audit/reproducibility_verification.json`; `results/summary.csv`.
5. **Planned tables/figures**: Table III previews the five-ticket evidence chain; no workflow figure is necessary.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: These are project contributions within one fixed benchmark, not claims of novel algorithms or external generalization.

## II. Methodology

### A. Dataset and Fixed Split

1. **Purpose**: Define the source data, fields, stable-ID partition, class counts, and integrity checks.
2. **Main argument/conclusion**: All experiments use the same immutable, disjoint ID split, preventing split drift and row-position ambiguity.
3. **Verified evidence to include**: 7,613 total unique IDs; 4,567 train, 1,523 dev, 1,523 held-out; full class balance and source/split hashes if space permits.
4. **Exact sources**: `data/train.csv`; `starter/data/README_DATA.md`; `starter/data/split_indices.json`; `pipeline/data.py`; `pipeline/splits.py`; `tests/test_splits.py`.
5. **Planned tables/figures**: **Table I — Dataset and fixed-split statistics**: rows, class 0, class 1, use, and access rule for full/train/dev/held-out.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: The local data file is ignored by Git and must be downloaded from the documented public source; the fixed split contains cross-split text duplicates even though ID sets are disjoint.

### B. Evaluation Metrics and Common Comparator

1. **Purpose**: Define target-1 precision, recall, F1, accuracy, TN/FP/FN/TP, and prediction-transition counts.
2. **Main argument/conclusion**: Aggregate metrics and row-level transitions answer different questions; both are required. Every ticket’s fixed/new FP/FN counts use Ticket 1 as the common comparator.
3. **Verified evidence to include**: Metric definitions; inclusive threshold rule where applicable; transition definitions; stable-ID prediction schema.
4. **Exact sources**: `pipeline/metrics.py`; `pipeline/ticket2.py`; `pipeline/artifacts.py`; `teacher_clarifications.md`; `tests/test_metrics.py`; `tests/test_artifacts.py`.
5. **Planned tables/figures**: Definitions can remain in prose/equations; Table VIII reports transition counts.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: F1 encodes no deployment cost model and does not establish semantic correctness of individual predictions.

### C. Selection, Freeze, and Held-Out Usage Policy

1. **Purpose**: Explain the chronological protection against held-out selection and distinguish primary evaluation, prediction reuse, artifact recovery, and audit replay.
2. **Main argument/conclusion**: Ticket decisions are train/dev based and immutable before ticket-specific held-out reporting; the final replay verifies rather than reselects.
3. **Verified evidence to include**: Zero ticket-specific held-out count at freeze; `heldout_used_for_selection=false`; `selection_reopening_permitted=false`; Tickets 3 and 5 reuse earlier prediction cores; Ticket 1 recovery and Step 10 audit are explicitly qualified.
4. **Exact sources**: All five freeze JSON files; all five held-out completion JSON files; `configs/frozen_decisions.json`; final reproducibility verification.
5. **Planned tables/figures**: A compact “decision basis / held-out action” column in Table III; no chronology figure unless space permits.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: Earlier tickets’ held-out artifacts existed before later ticket work; protection is procedural provenance, not literal ignorance of all prior scores. The detailed rollback deletion history is not independently recoverable.

### D. Baseline Pipeline and Reproducibility Controls

1. **Purpose**: Describe the majority floor, raw-text TF-IDF + Logistic Regression baseline, train-only feature fitting, seed/thread controls, and artifact schemas.
2. **Main argument/conclusion**: The local baseline implementation is deterministic, converged, stable-ID aligned, and leakage-controlled in the locked environment.
3. **Verified evidence to include**: Word-unigram TF-IDF, lowercase, no manual normalization/metadata, Logistic Regression `C=1`, unweighted, threshold 0.50, seed 3102; two identical dev fits; no warnings; 15 iterations.
4. **Exact sources**: `experiments/step-4-baselines/run_config.json`; `results/dev_metrics.csv` under that directory; baseline predictions; `warnings.json`; `pipeline/modeling.py`; `pipeline/baselines.py`; `tests/test_baselines.py`; `tests/test_reproducibility.py`.
5. **Planned tables/figures**: Baseline configuration summarized in prose; exhaustive parameter dictionary moves to Appendix A.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: Locked local reproducibility does not prove bit-identical scores on other platforms or future versions.

### E. Controlled Ticket Design

1. **Purpose**: Explain how each ticket isolates a lever while sharing the fixed split and baseline comparator.
2. **Main argument/conclusion**: Trust is assessed through a recurring pattern: predeclared hypothesis, controlled dev experiment, row-level transitions/robustness, freeze, post-freeze held-out evidence, and limitation.
3. **Verified evidence to include**: Ticket plans and selection rules for normalization, shortcut features, model/threshold, and data correction.
4. **Exact sources**: `experiments/ticket-2/dev/experiment_plan.json`; `experiments/ticket-3/dev/experiment_plan.json`; `experiments/ticket-4/dev/experiment_plan.json`; `experiments/ticket-5/dev/audit_plan.json`; `experiments/ticket-5/dev/label_correction_plan.json`.
5. **Planned tables/figures**: Table III organizes tickets by lever, dev decision, held-out action, and conclusion.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: Predeclared grids are bounded, not exhaustive; the project uses one fixed dev split.

## III. Main Evidence and Results

### A. Ticket 1: Baseline Discrepancy Diagnosis

1. **Purpose**: Establish whether the independently implemented reference baseline matches the contract and delimit what can be diagnosed.
2. **Main argument/conclusion**: The baseline does not match the reference within tolerance; local split, train-only fitting, seed path, convergence, and effective settings are verified, but the unique external cause is unresolved.
3. **Verified evidence to include**: Dev baseline metrics; held-out precision/recall/F1/accuracy and TN/FP/FN/TP; reference F1 `0.7574221578566256`; actual F1 `0.749185667752443`; absolute gap `0.008236490104182592`; tolerance `0.001`; selected one-lever probe outcomes.
4. **Exact sources**: `experiments/step-4-baselines/results/dev_metrics.csv`; `experiments/ticket-1/heldout/primary_contract_comparison.json`; `heldout_metrics.csv`; `heldout_confusion_matrix.csv`; `experiments/ticket-1/probes/dev_probe_metrics.csv`; frozen baseline JSON.
5. **Planned tables/figures**: **Table II — Baseline and reference comparison**. Put the full nine-probe table in Appendix A; main text reports only the most diagnostic negative/no-change probes.
6. **Representative cases**: In main text, use ID `5282` (solver boundary movement) and IDs `105`/`110` (balanced-weight recall/precision pair). Optional Appendix C: IDs `237`, `241`, `137`, `97`, held-out FP `198`, FN `17`.
7. **Required limitation/uncertainty**: Never identify one parameter or version as the cause; reference predictions and complete configuration are missing. Qualify the artifact-recovery replay.

### B. Ticket 2: Text Normalization

1. **Purpose**: Compare six isolated surface-normalization levers and connect dev movements to perturbation robustness.
2. **Main argument/conclusion**: URL placeholdering is the only adopted Ticket 2 lever. It yields a small dev gain, higher precision/accuracy, lower recall, and exact invariance to the paired URL substitution; other levers are neutral, weak, or negative.
3. **Verified evidence to include**: Seven-variant dev metrics; URL dev transitions (`22/6` fixed FP/FN, `11/11` new FP/FN); robustness results (767 affected rows, raw 275 changed, normalized 0); held-out metrics and transitions (`22/8` fixed, `4/15` new).
4. **Exact sources**: `experiments/ticket-2/dev/results/dev_metrics.csv`; `dev_confusion_matrices.csv`; `dev/robustness/robustness_metrics.csv`; `frozen_decision.json`; `heldout/heldout_metrics.csv`; held-out confusion/change CSVs.
5. **Planned tables/figures**: **Table IV — Normalization comparison and paired robustness**. No separate normalization figure; the table is more compact and preserves exact values.
6. **Representative cases**: Main text: fixed FP ID `773` and new FN ID `902`, or dev fixed FP `174` and new FP `110`. Appendix C may include IDs `971`, `237`, `2352`, `3903`, `3097`, `936`, `2569`.
7. **Required limitation/uncertainty**: The gain is small and from one dev split; perturbations are stress tests, not deployment frequencies; intact recognized emoji affected zero rows, so emoji usefulness is unresolved.

### C. Ticket 3: Shortcut and Shallow-Feature Audit

1. **Purpose**: Determine how much predictive signal comes from keyword, location, length, missingness, and surface statistics, and whether it is trustworthy.
2. **Main argument/conclusion**: Keyword carries real benchmark signal but is mixed with acquisition artifacts; length/location/shallow families are weak or brittle; the best visible rich candidate is rejected because it depends heavily on keyword availability.
3. **Verified evidence to include**: Ten dev F1 values; rich-candidate F1 `0.7493956486704271`; text baseline `0.7388120423108218`; rich transitions `41/46` fixed FP/FN and `42/35` new; keyword mask 702 changes and F1 `0.6423057128152342`; metadata sparsity (591 unseen nonmissing dev locations).
4. **Exact sources**: `experiments/ticket-3/dev/results/dev_metrics.csv`; `dev_confusion_matrices.csv`; `metadata_profile.csv`; `dev/robustness/robustness_metrics.csv`; `interpretability/top_coefficients.csv`; `frozen_decision.json`.
5. **Planned tables/figures**: **Table V — Shallow-feature and shortcut audit**, including original F1, key perturbation F1, and decision. Coefficient details remain textual or in Appendix A, not a main figure.
6. **Representative cases**: Main text: IDs `7`/`25` as identical missing-keyword score contrast and ID `331` as a rich-model figurative FP. Appendix C: IDs `4`, `86`, `16`, `105`, `110`, `18`, `519`.
7. **Required limitation/uncertainty**: Mask-to-missing is severe and activates learned missing categories; coefficients are associations, not causal importance; feature conclusions are benchmark-specific.

### D. Ticket 4: Decision Rule and Model

1. **Purpose**: Compare threshold tuning, regularization, class weighting, and LinearSVC over the same raw-text representation.
2. **Main argument/conclusion**: Balanced Logistic Regression at `C=1`, threshold 0.50 wins the predeclared dev F1 rule and increases recall, but lowers precision and accuracy. Its held-out F1 gain is small.
3. **Verified evidence to include**: Best threshold 0.47; best regularization `C=2`; balanced model dev and held-out metrics; LinearSVC result; 61-threshold tradeoff; dev transitions (42 FN fixed, 48 FP new); held-out transitions (35 FN fixed, 56 FP new).
4. **Exact sources**: `experiments/ticket-4/dev/results/dev_model_metrics.csv`; `results/threshold_sweep.csv`; detailed threshold sweep; `selection_result.json`; `frozen_decision.json`; held-out metrics/confusion/change CSVs.
5. **Planned tables/figures**: **Table VI — Bounded model and operating-point comparison**. **Figure 1 — Threshold versus precision, recall, and F1 on dev**, generated directly from `results/threshold_sweep.csv`.
6. **Representative cases**: Main text: dev IDs `105`/`110`; held-out IDs `556` (fixed FN) and `59` or `996` (new FP). Appendix C: IDs `353/390`, `394`, `402`, `840`, `907`, `767`, `1645`.
7. **Required limitation/uncertainty**: F1 has no explicit cost model; one split and bounded grid can exploit dev noise; only one alternative classifier was tested; held-out ID `767` must use machine CSV scores `0.4876230020837692 -> 0.5448153095708489`, not stale ticket-prose scores.

### E. Ticket 5: Data-Quality and Error Audit

1. **Purpose**: Quantify duplicate/label inconsistency, distinguish data defects from hard cases, and test whether high-confidence train-label corrections should change the model.
2. **Main argument/conclusion**: Cross-split duplication and conflicting annotations materially limit score interpretation, but the eight-row correction set fails both dev adoption gates; the conservative decision is no source change and retention of Ticket 4.
3. **Verified evidence to include**: Train/dev and full exact/canonical/near counts; eight correction proposals; control/candidate metrics; 3 fixed versus 8 new errors; final 64-row disposition counts; no label or held-out-row modification.
4. **Exact sources**: `experiments/ticket-5/dev/results/duplicate_summary.csv`; `dev/label_correction_plan.json`; correction `dev_metrics.csv`, `selection_result.json`, `changes_vs_ticket4.csv`; `heldout/full_duplicate_summary.csv`; `results/data_quality_audit.csv`; final audit manifest; frozen decision.
5. **Planned tables/figures**: **Table IX — Data-quality relationship and disposition summary**. **Figure 4 — Data-quality disposition counts**, generated from `results/data_quality_audit.csv`. Detailed correction proposals go to Appendix B.
6. **Representative cases**: Main text: correction case `9470`; ambiguous duplicate pair `6220/6223`; valid hard negative `5760`; likely mislabel recommendation `2619` or `546`; leakage flag `5228`. Appendix C contains the broader verified ID set.
7. **Required limitation/uncertainty**: Dispositions are not expert ground truth; `fix` is a recommendation only; duplicate categories overlap; near-duplicate search is bounded; links were not fetched; the 64-row audit is curated, not exhaustive.

### F. Cross-Ticket Comparison and Frozen Final Choice

1. **Purpose**: Synthesize the evidence chain and justify the final frozen model without post-hoc held-out ranking.
2. **Main argument/conclusion**: Ticket 2 has the highest observed held-out F1, but the valid final choice remains Ticket 5/Ticket 4 because each ticket decision is local and frozen from dev evidence; selecting Ticket 2 after inspecting held-out would violate the protocol.
3. **Verified evidence to include**: Five rows of dev F1, held-out F1, accuracy, decision, and Ticket 1-relative transitions; intentional prediction-core equivalences (T1=T3, T4=T5=final); final reproducibility PASS.
4. **Exact sources**: `results/summary.csv`; `experiments/final-reproducibility-audit/transition_recalculation.csv`; `configs/frozen_decisions.json`; final prediction CSV; reproducibility verification JSON.
5. **Planned tables/figures**: **Table III — Five-ticket summary**. **Figure 2 — Dev versus held-out F1 by ticket** from `results/summary.csv`, with annotation that Tickets 1/3 and 4/5 share prediction cores. **Table VII — Selected confusion-matrix comparison** for Ticket 1, Ticket 2, and Ticket 4/final. **Table VIII — Ticket 1-relative prediction transitions**. Optional **Figure 3 — Fixed versus newly introduced held-out errors by ticket**, from transition recalculation.
6. **Representative cases**: No new cases; reference Section IV.
7. **Required limitation/uncertainty**: Do not rank final configurations by held-out after the fact. Ticket labels denote ticket-specific frozen decisions, not a cumulative monotonic pipeline.

## IV. Case Analysis

### A. Prediction Movements that Match Intended Levers

1. **Purpose**: Show that selected metric movements have identifiable mechanisms and costs.
2. **Main argument/conclusion**: URL canonicalization mainly removes false positives at some recall cost, while balanced weighting recovers borderline positives at a larger false-positive cost.
3. **Verified evidence to include**: Before/after scores, labels, transition types, and short text excerpts for a balanced set of fixed and new errors.
4. **Exact sources**: Ticket 2 dev/held-out change CSVs; Ticket 4 dev/held-out change CSVs; Ticket 1 probe change CSVs.
5. **Planned tables/figures**: Part of **Table X — Representative prediction and data-quality cases** with columns ticket, ID, split, lever/audit type, truth, baseline/candidate scores where available, transition/disposition, and interpretation.
6. **Representative cases**: URL: `773` fixed FP and `902` new FN. Balanced weighting: `556` fixed FN and `59` new FP. Diagnostic boundary: `5282`. Limit to 4–6 main rows.
7. **Required limitation/uncertainty**: Examples illustrate mechanisms but do not estimate prevalence or causal generalization.

### B. Hard Negatives, Likely Mislabels, and Ambiguous Duplicates

1. **Purpose**: Separate model weakness from annotation/data-quality problems.
2. **Main argument/conclusion**: Error direction alone does not determine whether a row should be corrected; the same error set mixes negation/metaphor failures, plausible mislabels, leakage, and irreconcilable duplicate conflicts.
3. **Verified evidence to include**: Submitted dispositions and duplicate evidence for selected stable IDs.
4. **Exact sources**: `results/data_quality_audit.csv`; Ticket 5 dev/held-out error-review CSVs; exact/canonical/near relationship CSVs.
5. **Planned tables/figures**: Remaining rows of Table X; Figure 4 gives disposition counts.
6. **Representative cases**: `5760` hard negative/negation; `2619` likely target-1 mislabel; `6220/6223` ambiguous exact pair; `5228` cross-split conflict/leakage; optional `10795` model weakness.
7. **Required limitation/uncertainty**: Semantic dispositions are human/AI judgments without independent expert adjudication; preserve original labels in all wording.

### C. Why Aggregate Improvement Is Not Sufficient

1. **Purpose**: Tie the cases back to the report thesis.
2. **Main argument/conclusion**: A higher F1 can coexist with new semantic errors, shortcut fragility, or contradictory labels; trust requires convergent quantitative, robustness, provenance, and case evidence.
3. **Verified evidence to include**: Ticket 2 small gain with mixed transitions; Ticket 3 rejected higher dev score; Ticket 4 recall/precision exchange; Ticket 5 failed correction set.
4. **Exact sources**: Tables IV–IX source artifacts; relevant frozen decisions.
5. **Planned tables/figures**: No new visual; synthesize existing tables and Figure 1.
6. **Representative cases**: Refer to Table X rather than adding new anecdotes.
7. **Required limitation/uncertainty**: This is an interpretation of verified project evidence, not a formal statistical proof of trustworthiness.

## V. Difficulties and Solutions

### A. Reproducibility and Provenance Challenges

1. **Purpose**: Satisfy the instructor requirement with concrete, verifiable challenges rather than generic development statements.
2. **Main argument/conclusion**: The reference mismatch and floating-point replay tolerance required explicit provenance and calibrated verification.
3. **Verified evidence to include**: Unresolved reference gap; clean-process replay; identical labels with `1.1102230246251565e-16` maximum score drift; Ticket 1 recovery replay qualification.
4. **Exact sources**: Ticket 1 contract comparison and probe audit; final reproducibility verification; replay comparison JSONs; Ticket 1 recovery ledgers.
5. **Planned tables/figures**: Rows in **Table XI — Verified difficulties, solutions, and residual risk**.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: The detailed rollback deletion history and alleged first Ticket 4 failed guard are not independently preserved; either omit them or qualify them as narrative-only.

### B. Analytical and Operating-Point Challenges

1. **Purpose**: Explain why score maximization did not automatically determine trustworthy choices.
2. **Main argument/conclusion**: Shortcut-sensitive gains and precision-recall tradeoffs required robustness probes and explicit selection criteria.
3. **Verified evidence to include**: Ticket 3 keyword mask collapse; Ticket 4 class-weight tradeoff and small held-out delta; Ticket 2 small normalization gain.
4. **Exact sources**: Ticket 2/3 robustness CSVs; Ticket 4 selection and held-out files.
5. **Planned tables/figures**: Rows in Table XI; cross-reference Figure 1 and Tables IV–VIII.
6. **Representative cases**: `7/25`, `105/110`, or `556/59` only if not already overused.
7. **Required limitation/uncertainty**: No deployment cost model or resampling interval exists.

### C. Data-Quality Challenges

1. **Purpose**: Explain the difficulty of separating model error, label error, scope ambiguity, and duplicate leakage.
2. **Main argument/conclusion**: Conservative four-way dispositions and an in-memory correction gate prevent unsupported relabeling.
3. **Verified evidence to include**: Duplicate counts, 64-row disposition counts, correction rejection, unchanged source/held-out hashes.
4. **Exact sources**: Ticket 5 audit/correction artifacts and final audit manifest.
5. **Planned tables/figures**: Rows in Table XI; cross-reference Table IX and Figure 4.
6. **Representative cases**: `6220/6223`, `5760`, `2619`.
7. **Required limitation/uncertainty**: No expert adjudication; near-duplicate candidate generation is bounded.

## VI. AI Usage Declaration

### A. Scope of AI Assistance

1. **Purpose**: State how AI supported the project in a transparent, course-compliant manner.
2. **Main argument/conclusion**: AI assisted inspection, planning, code/test drafting, experiment orchestration, artifact review, and documentation; it did not provide independent ground truth.
3. **Verified evidence to include**: Categories of assistance recorded in the curated interaction log.
4. **Exact sources**: `logs/chat.md`; `report_evidence_audit.md`, Section 10.
5. **Planned tables/figures**: No figure. A compact paragraph is preferred; Table XI may mention AI-verification challenge only if needed.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: `logs/chat.md` is a reconstructed summary, not a verbatim transcript; no raw earlier Codex transcript is available.

### B. Verification and Human Responsibility

1. **Purpose**: Explain how AI-generated code and interpretations were checked.
2. **Main argument/conclusion**: Outputs were accepted only after stable-ID validation, prediction-level metric checks, tests, frozen hashes, source/label integrity checks, and clean-process replay; semantic recommendations remain reviewable judgments.
3. **Verified evidence to include**: 56 passing tests; submission validator PASS with no refit; five-process replay PASS; exact result-table regeneration; no source/held-out label modification.
4. **Exact sources**: `tests/`; `pipeline.validate_submission` behavior in `pipeline/validate_submission.py`; final reproducibility verification; Ticket 5 final audit manifest; `logs/chat.md`.
5. **Planned tables/figures**: No new visual; optional short verification checklist in prose.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: Internal reproducibility does not replace external label adjudication or human responsibility for the final narrative and citations.

## VII. Discussion and Limitations

### A. What Evidence Is Trustworthy

1. **Purpose**: Rank evidence types and explain why some score gains were adopted while others were rejected.
2. **Main argument/conclusion**: Strong evidence combines controlled dev selection, expected directional transitions, robustness, stable-ID cases, freeze provenance, and post-freeze confirmation; isolated score maxima are weaker.
3. **Verified evidence to include**: Ticket 2 adoption, Ticket 3 rejection, Ticket 4 cautious adoption, Ticket 5 rejection.
4. **Exact sources**: Four frozen decisions and their dev/held-out evidence; final manifest.
5. **Planned tables/figures**: Cross-reference Tables III–IX; no additional visual.
6. **Representative cases**: Refer to Table X.
7. **Required limitation/uncertainty**: “Trustworthy” is an evidence judgment within this benchmark, not a guarantee of deployment robustness.

### B. Remaining Uncertainty and Risks

1. **Purpose**: Consolidate unresolved evidence gaps.
2. **Main argument/conclusion**: The primary remaining risks are the unknown Ticket 1 reference implementation, single-split selection noise, duplicate dependence, label ambiguity, bounded near-duplicate search, absent cost model, and platform-specific floating-point behavior.
3. **Verified evidence to include**: Audit gap list and relevant ticket limitations.
4. **Exact sources**: `report_evidence_audit.md`, Sections 13–14; ticket limitation sections; reproducibility audit limitation.
5. **Planned tables/figures**: No new visual; optionally a compact risk paragraph after Table XI.
6. **Representative cases**: `6220/6223` may anchor label ambiguity; no additional examples.
7. **Required limitation/uncertainty**: Explicitly state that no confidence intervals, resampling, external test set, or expert adjudication are available.

### C. Final Model Interpretation and External Validity

1. **Purpose**: Explain what the final model choice means and does not mean.
2. **Main argument/conclusion**: The final balanced model is the last valid dev-based frozen choice with no corrections. It is recall-oriented and procedurally valid, not the highest post-hoc held-out score or a universal deployment optimum.
3. **Verified evidence to include**: Ticket 4/5 identical prediction core; Ticket 2 higher held-out F1; final metrics; precision/recall/accuracy tradeoff.
4. **Exact sources**: `configs/frozen_decisions.json`; `results/summary.csv`; final predictions; Ticket 4/5 held-out metrics.
5. **Planned tables/figures**: Cross-reference Table III, Table VII, Figure 2.
6. **Representative cases**: `556` and `59` may illustrate the final model’s benefit/cost.
7. **Required limitation/uncertainty**: A real deployment needs explicit error costs, temporal/external validation, and duplicate-aware splitting.

## VIII. Conclusion

### A. Findings and Final Decision

1. **Purpose**: Answer the project goal concisely and close the argument.
2. **Main argument/conclusion**: The baseline is locally reproducible but misses the undisclosed reference; URL normalization is a defensible surface lever; shallow-feature gains are shortcut-sensitive; balanced weighting changes the operating point toward recall; data inconsistency limits score meaning; the final frozen model retains balanced LR with no label corrections.
3. **Verified evidence to include**: Only the headline metrics/decisions already established in Tables II–IX.
4. **Exact sources**: `results/summary.csv`; all frozen decisions; final reproducibility verification; data-quality audit.
5. **Planned tables/figures**: None new.
6. **Representative cases**: None new.
7. **Required limitation/uncertainty**: End with a cautious statement about benchmark noise and external validity, not a claim of universal superiority.

## References

### A. Sources to Cite

1. **Purpose**: Credit the dataset, assignment-provided source/contract, and software/method references used in the final paper.
2. **Main argument/conclusion**: References support provenance and methods; they are not substitutes for repository evidence.
3. **Verified evidence to include**: Kaggle Disaster Tweets dataset/public mirror; scikit-learn components; NumPy/pandas if substantively discussed; IEEE template/documentation; course handout/starter materials as permitted by course citation conventions.
4. **Exact sources**: Root `README.md` data-source section; `requirements-lock.txt`; per-run `software_versions.json`; handout/starter files.
5. **Planned tables/figures**: None.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: A complete bibliography does not yet exist. Do not fabricate author names, publication metadata, access dates, DOIs, or course citation format. Verify each citation during report drafting.

## Appendix

### Appendix A. Extended Configurations and Negative Results

1. **Purpose**: Preserve technical detail without turning the main report into a log.
2. **Main argument/conclusion**: The main conclusions are robust to the documented controlled comparisons, including failed and no-change experiments.
3. **Verified evidence to include**: Effective baseline parameters; full Ticket 1 probe table; all Ticket 2 variants; full Ticket 3 variants and selected coefficient examples; all Ticket 4 model candidates.
4. **Exact sources**: Per-ticket run configs, plan JSONs, metric/confusion CSVs, warnings, and coefficient tables.
5. **Planned tables/figures**: **Table A1 — Full Ticket 1 discrepancy probes**; **Table A2 — Complete Ticket 4 candidate metrics** if Table VI is abbreviated.
6. **Representative cases**: None or brief cross-references to Appendix C.
7. **Required limitation/uncertainty**: These are controlled observations on one split, not exhaustive hyperparameter searches.

### Appendix B. Data-Quality and Correction Detail

1. **Purpose**: Preserve the exact audit definitions, correction gate, proposal list, and overlapping duplicate counts.
2. **Main argument/conclusion**: Data-governance recommendations were explicitly separated from model intervention and source mutation.
3. **Verified evidence to include**: Exact/canonical/near definitions; eight proposals with original/proposed labels; correction result; final disposition schema/counts.
4. **Exact sources**: Ticket 5 audit plan; label-correction plan/proposals; correction selection; duplicate summaries; final audit manifest and CSV.
5. **Planned tables/figures**: **Table B1 — Eight proposed training-label corrections and decision outcome**; optional **Table B2 — Audit disposition definitions**.
6. **Representative cases**: Proposal IDs `4076`, `6566`, `8698`, `8739`, `1723`, `1760`, `6097`, `9472`.
7. **Required limitation/uncertainty**: Do not imply corrections were applied or externally adjudicated.

### Appendix C. Extended Stable-ID Case Ledger

1. **Purpose**: Give traceable examples without overcrowding the two-column main paper.
2. **Main argument/conclusion**: Each reported mechanism has both favorable and unfavorable row-level evidence.
3. **Verified evidence to include**: Selected extra IDs from the verified audit with artifact path, split, truth, scores, transition/disposition, and one-sentence interpretation.
4. **Exact sources**: Ticket change/error CSVs and `results/data_quality_audit.csv` listed in `report_evidence_audit.md`, Sections 3–8.
5. **Planned tables/figures**: **Table C1 — Extended representative-case ledger**.
6. **Representative cases**: Use only verified IDs from the audit. Include Ticket 4 ID `767` only with machine-generated scores.
7. **Required limitation/uncertainty**: Cases are illustrative, not prevalence estimates; semantic dispositions remain unadjudicated.

### Appendix D. Reproducibility and Artifact Provenance

1. **Purpose**: Make the work reproducible without placing raw commands in the main narrative.
2. **Main argument/conclusion**: Frozen recipes, hashes, stable-ID artifacts, and non-refitting validation establish internal reproducibility in the locked environment.
3. **Verified evidence to include**: Seed/thread settings; data/split/dependency hashes; five freeze hashes; clean replay summary; exact commands by reference to README/run-command files; validation results.
4. **Exact sources**: Root `README.md`; `configs/reproducibility.json`; `configs/frozen_decisions.json`; final reproducibility audit directory; per-run `run_command.txt`; tests; validator source.
5. **Planned tables/figures**: **Table D1 — Frozen decisions and artifact provenance**. Do not paste command output; list commands compactly or cite repository paths.
6. **Representative cases**: None.
7. **Required limitation/uncertainty**: No verbatim prior AI transcript or Git recovery history exists; cross-platform bit identity is not claimed.

## Planned table register

| ID | Planned title | Placement | Direct source artifacts | Main content |
|---|---|---|---|---|
| Table I | Dataset and Fixed-Split Statistics | Methodology | `data/train.csv`; split JSON; data README | Rows, class counts, purpose, access policy |
| Table II | Baseline and Reference Comparison | Ticket 1 | Contract JSON; Ticket 1 dev/held-out metrics and confusion | Floor, local baseline, reference F1, tolerance/gap |
| Table III | Five-Ticket Summary | Cross-ticket synthesis | `results/summary.csv`; frozen manifest | Lever, dev F1, held-out F1/accuracy, decision, held-out action |
| Table IV | Normalization Comparison and Paired Robustness | Ticket 2 | Ticket 2 dev metrics and robustness CSV | Seven variants, precision/recall/F1, changes, perturbation result |
| Table V | Shallow-Feature and Shortcut Audit | Ticket 3 | Ticket 3 dev and robustness CSVs | Variant F1, key robustness result, classification, decision |
| Table VI | Bounded Model and Operating-Point Comparison | Ticket 4 | Ticket 4 dev model metrics/selection | Baseline, best threshold, best C, balanced LR, LinearSVC |
| Table VII | Selected Held-Out Confusion-Matrix Comparison | Cross-ticket synthesis | Ticket 1/2/4 held-out metrics/confusion CSVs | TN/FP/FN/TP plus precision/recall/F1/accuracy |
| Table VIII | Ticket 1-Relative Prediction Transitions | Cross-ticket synthesis | Transition recalculation; summary CSV | Fixed/new FP/FN for all tickets |
| Table IX | Data-Quality Relationship and Disposition Summary | Ticket 5 | Duplicate summaries; audit CSV/manifest | Exact/canonical/near counts and four disposition counts |
| Table X | Representative Prediction and Data-Quality Cases | Case analysis | Verified change/error/audit CSVs | Balanced fixed/new errors and audit categories with stable IDs |
| Table XI | Verified Difficulties, Solutions, and Residual Risk | Difficulties | Evidence audit Section 9 and cited machine artifacts | Challenge, evidence, action, remaining uncertainty |
| Table A1 | Full Ticket 1 Discrepancy Probes | Appendix A | Ticket 1 probe metrics | All probes and transitions |
| Table A2 | Complete Ticket 4 Candidate Metrics | Appendix A if needed | Ticket 4 dev model metrics | All bounded candidates |
| Table B1 | Proposed Training-Label Corrections | Appendix B | Correction plan/proposal CSV | Eight IDs, original/proposed labels, evidence, outcome |
| Table B2 | Audit Disposition Definitions | Appendix B optional | Audit plan and result schema | `fix`, `keep_but_flag`, `ambiguous`, `reject_false_positive` |
| Table C1 | Extended Stable-ID Case Ledger | Appendix C | Per-ticket change/error CSVs; audit CSV | Additional verified cases |
| Table D1 | Frozen Decisions and Artifact Provenance | Appendix D | Frozen manifest and verification JSON | Recipe, freeze hash, prediction artifact, replay status |

## Planned figure register

Only generate these figures later from the listed existing CSV/JSON artifacts. Do not draw values manually.

| ID | Planned title | Placement | Direct source | Plot design and caveat |
|---|---|---|---|---|
| Figure 1 | Dev Threshold Tradeoff for the Frozen Raw-Text Baseline | Ticket 4 | `results/threshold_sweep.csv` | Three-line plot: threshold vs precision, recall, F1; mark 0.47 and 0.50. Dev only. |
| Figure 2 | Dev and Held-Out F1 Across Frozen Ticket Decisions | Cross-ticket synthesis | `results/summary.csv` | Paired bars or connected dots; annotate T1=T3 and T4=T5 prediction cores. Do not use the plot for post-hoc model selection. |
| Figure 3 | Fixed and Newly Introduced Held-Out Errors Relative to Ticket 1 | Cross-ticket synthesis, optional | `experiments/final-reproducibility-audit/transition_recalculation.csv` | Grouped or diverging bars for fixed FP/FN and new FP/FN; avoid implying categories cancel semantically. |
| Figure 4 | Data-Quality Audit Disposition Counts | Ticket 5 | `results/data_quality_audit.csv`; final audit manifest | Four-category bar chart using verified counts 30/6/15/13; state that `fix` is recommendation only. |

Confusion matrices are planned as Table VII rather than a heatmap figure to preserve exact counts and avoid redundancy. If the report later needs a visual matrix comparison, a three-panel Ticket 1/Ticket 2/Ticket 4 figure may replace—not duplicate—Table VII, using the held-out confusion CSVs.

## Content restricted to the appendix

- Full effective estimator parameter dictionaries and package-version detail.
- The complete Ticket 1 probe grid and complete Ticket 4 candidate grid when abbreviated in the main text.
- Exact reproduction commands, hashes, run ledgers, and artifact-path inventories.
- The eight-row label-correction proposal table with full evidence text.
- Extended stable-ID cases beyond the small balanced main-paper selection.
- Detailed coefficient rankings and long normalization/robustness results if they do not fit in compact main tables.
- Audit disposition definitions and additional duplicate examples.
- No raw terminal output, screenshots, full logs, or verbatim CSV dumps, even in the appendix.

## Unresolved evidence gaps to preserve

- The Ticket 1 reference implementation, predictions, full parameters, and software versions are missing; the discrepancy cause remains unresolved.
- No resampling intervals, external test set, temporal validation, or duplicate-group-aware reevaluation exists.
- No expert or multi-annotator adjudication validates Ticket 5 semantic dispositions.
- The near-duplicate search is bounded and cannot establish exhaustive equivalence.
- No deployment error-cost model justifies F1 as the operational optimum.
- Cross-platform bit-identical probabilities are not established.
- The AI interaction log is curated rather than verbatim; no earlier raw Codex transcript is available.
- A complete verified bibliography and user-supplied author/affiliation metadata are still missing.
- Ticket 4 ID `767` narrative scores conflict with the generated CSV; only the CSV scores may be used.

## Readiness and stopping point

The verified repository evidence supports this report structure and every planned main table/figure. Report drafting may begin only on a new explicit instruction. The final prose must remain within the evidence boundaries above, and figure/table generation must read directly from the cited artifacts.
