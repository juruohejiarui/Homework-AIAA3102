# AI Interaction and Verification Record

## Scope and format

This file is an honest, human-readable record of how AI assistance was used in the project and how its outputs were verified. It is a structured chronology reconstructed from the repository's saved commands, status record, freeze ledgers, artifacts, and tests; it is **not** presented as a verbatim export of every message or hidden model-reasoning trace. Omitting a full raw transcript avoids falsely implying that a curated summary is a literal conversation and keeps private/internal reasoning out of the submission.

AI assistance was used throughout repository inspection, implementation planning, Python and test drafting, experiment orchestration, artifact review, and documentation. The human user supplied the assignment steps, authorized each major stage, and explicitly authorized the final clean replay and final documentation assembly. Empirical claims were accepted only after execution against local data and inspection of saved outputs.

## AI usage chronology

### Repository and contract audit

The AI read the handout, starter README, fixed split, project contract, instructor clarification, and existing workspace state. It identified the required stable-ID schemas, the prohibition on held-out model selection, the floor-model definition, and the rule that every ticket's fixed/new FP/FN counts must use Ticket 1 as the common comparator.

Verification performed:

- checked the fixed split for disjoint train/dev/held-out IDs and exact coverage;
- validated the labeled CSV schema and ID coverage;
- recorded the source data, split, and locked-environment hashes;
- created synthetic unit tests for reusable components before relying on real-data output.

### Reusable baseline infrastructure

The AI assisted in drafting deterministic split loading, modeling, metric, artifact, and version-capture modules. It then assisted with the majority-floor and raw-text TF-IDF plus Logistic Regression baseline runners.

Verification performed:

- the floor predicted target 0 for every dev row and produced target-1 F1 0.0;
- two independent baseline dev fits produced identical IDs, labels, scores, metrics, warnings, and iteration counts;
- all learned text state was fit on train IDs only;
- saved configuration files captured effective estimator parameters and software versions.

### Ticket 1 - baseline discrepancy diagnosis

The AI helped formalize the literal baseline, structural checks, and one-lever dev probes. The baseline was frozen before its primary held-out comparison. Its held-out F1, 0.749185667752443, fell outside the contract's 0.001 tolerance around 0.7574221578566256.

A later user-requested rollback removed Ticket 1 row-level artifacts. Because scalar metrics cannot reconstruct predictions, the user explicitly authorized one deterministic replay of the exact frozen configuration for artifact recovery. The restored files record that this was recovery, not a new selection round. The AI did not claim that any single probe caused the external reference discrepancy; the contract lacks reference predictions, full parameters, and package versions.

Verification performed:

- split, train-only fitting, stable ID order, seed, convergence, and effective defaults were audited;
- nine dev-only probes saved complete predictions, errors, transitions, metrics, and warnings;
- the recovery replay exactly reproduced the historical aggregate metrics;
- frozen choice and chronology remained unchanged.

### Ticket 2 - normalization

The AI helped implement composable URL, mention, hashtag, punctuation, case, and emoji transforms, but each scored variant enabled only one lever. URL placeholdering was selected on dev because it had the strongest normalization F1 and a matching robustness mechanism.

Verification performed:

- the raw control reproduced Ticket 1 dev predictions exactly;
- all variants shared split, vectorizer, classifier, seed, and decision threshold;
- URL placeholdering changed zero predictions on all 767 URL-perturbed dev rows while the raw model changed 275;
- complete FP/FN and stable-ID change tables were inspected;
- the freeze file recorded zero held-out evaluations at selection time;
- the single post-freeze held-out report did not reopen the decision.

### Ticket 3 - feature and shortcut audit

The AI helped construct leakage-safe keyword, location, length, and selected surface-count features and controlled combinations with text. Although text plus selected shallow features improved visible dev F1, targeted masking showed severe keyword dependence, so the feature addition was rejected.

Verification performed:

- imputers/scalers were fit inside train-only pipelines;
- ten controlled dev variants saved metrics and stable-ID transitions;
- coefficient tables and metadata profiles were inspected;
- keyword masking and superficial-text neutralization were applied as explicit stress probes;
- the frozen choice retained the exact Ticket 1 model;
- post-freeze Ticket 3 reporting reused validated Ticket 1 predictions without a new fit.

### Ticket 4 - decision rule and classifier

The AI helped define the bounded threshold, regularization, class-weighting, and LinearSVC experiment. Balanced Logistic Regression at threshold 0.50 won the predeclared dev target-1 F1 criterion. Documentation explicitly treats it as a recall-oriented operating point, not a universal improvement.

Verification performed:

- the baseline threshold sweep contained exactly 61 values from 0.20 through 0.80;
- classifier candidates changed only their stated lever;
- convergence, iteration count, confusion counts, and full error transitions were saved;
- the selected dev gain was decomposed into 42 fixed false negatives and 48 new false positives;
- the freeze preceded the single held-out evaluation;
- the held-out result preserved the recall mechanism but showed only a small F1 gain and lower accuracy.

### Ticket 5 - data quality and label-correction probe

The AI helped implement deterministic raw-exact, URL-canonical, and character n-gram near-duplicate discovery; join model errors to duplicate evidence; curate four allowed dispositions; and preserve proposed corrections beside original labels. Model disagreement alone was never accepted as mislabel evidence.

Eight high-confidence training corrections were tested only in an in-memory copy under a predeclared non-inferiority/error-transition gate. The candidate worsened dev F1 and created more errors than it fixed, so corrections were rejected as a model intervention. This does not prove that the original labels are semantically correct.

Verification performed:

- source data hash was checked before and after the correction probe;
- no dev or held-out label was edited or removed;
- proposal CSV retained stable ID, original label, proposed label, evidence, confidence, and split;
- full duplicate counts and model-error tables were saved;
- the final audit table was validated for exact schema, valid unique IDs, allowed dispositions, nonempty evidence, and confidence range;
- Ticket 5 reporting reused Ticket 4 predictions without fitting or repredicting.

### Final frozen-decision reproducibility audit

At the user's explicit request, the AI helped consolidate the five immutable freezes and execute one audit-only replay per ticket in five distinct Python processes. This replay occurred after all decisions were frozen and did not permit selection to reopen.

Verified audit outputs:

- audit result: `PASS`;
- distinct clean processes: 5;
- archived dev/held-out label changes: 0;
- maximum archived score difference: `1.1102230246251565e-16` against a `1e-12` limit;
- all five fits converged without warnings;
- `results/summary.csv`, `results/threshold_sweep.csv`, and `results/data_quality_audit.csv` regenerated byte-for-byte;
- every fixed/new FP/FN count recalculated against Ticket 1;
- final predictions contained 1,523 unique held-out IDs in the instructor's exact order;
- no unexplained stale, contradictory, duplicate, or manually edited active result artifact was found.

### Final documentation

The AI inspected all five detailed ticket documents and added an explicit traceability map for the required hypothesis, intended lever, controlled setup, dev evidence, frozen decision, held-out evidence, concrete prediction changes, interpretation, and limitation. It replaced the stale infrastructure-stage README with exact setup, download, reproduction, final-artifact regeneration, audit, and non-refitting validation commands.

At the user's clarification, report/PDF creation was explicitly excluded from this step. The AI removed the briefly scaffolded report-generation work and kept the deliverable scope to README, ticket documents, and this log. It did not rerun a held-out experiment or use held-out outcomes to revise a ticket.

### Subsequent report preparation and finalization

In later explicitly scoped work, AI assistance was used to audit report evidence, design the IEEE conference structure, generate report tables and figures from saved artifacts, draft and validate the LaTeX source, check citations, compile the PDF with a pinned Tectonic binary, and inspect all rendered pages. This later work did not contradict the earlier scope boundary: no report was created during the README/ticket/log-only step, and report work began only in the later report stages recorded in `PROJECT_STATUS.md`.

The two author names were supplied directly; affiliation, location, and email were omitted rather than invented. Quantitative tables, figures, and stable-ID cases were regenerated from repository CSV/JSON evidence. The final report verifier recomputed metrics from archived predictions, checked freeze and artifact hashes, confirmed no held-out label mutation or decision reopening, extracted the PDF, and validated 12 nonblank pages. A stale narrative score pair for Ticket 4 held-out ID `767` was rejected in favor of the machine transition CSV; the final objective submission audit also corrected the ticket narrative so the documents now agree. No experimental conclusion or frozen model decision changed during report preparation.

### Final submission audit

The final audit tested all complete prediction artifacts against the fixed split and source labels, installed the exact locked dependencies in a new temporary virtual environment, reran the dev-only baseline and full tests there, revalidated the report source and PDF, and inspected the rendered pages. Temporary audit environments, build directories, compiler downloads, caches, and duplicate archives were removed after verification. The audit did not execute a held-out runner or reopen selection.

## Verification record

The final assembly used these non-refitting checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pipeline.validate_submission
```

Final documentation validation completed without refitting: pytest collected 56 tests and all 56 passed in 2.30 seconds; `pip check` reported no broken requirements; Python bytecode compilation completed successfully; and `pipeline.validate_submission` returned `PASS` with 5 ticket documents, 5 summary rows, 61 threshold rows, 64 audit rows, and 1,523 final held-out prediction rows. The earlier Step 10 five-process reproducibility audit also remained `PASS`.

During the later strict final submission audit, the same 56 tests passed both in the established environment and in a newly created environment installed from `requirements-lock.txt`. The clean environment reproduced the dev floor and raw-text baseline exactly and confirmed `heldout_evaluated=False`. The non-refitting submission validator passed, and the report verifier passed 209 checks over 12 pages with PDF SHA-256 `a1b81e2a29963ca755f00b2c63170e087f3468dda05a49a177c4d23782384275`.

## Human judgment and responsibility

The AI proposed structures, code, and interpretations; it did not supply independent ground-truth labels or external expert adjudication. Semantic label recommendations in Ticket 5 remain recommendations. The repository deliberately distinguishes:

- evidence generated by executable code;
- a model decision frozen from train/dev evidence;
- post-freeze held-out reporting;
- human/AI semantic audit judgments;
- audit-only reproducibility replays.

The human submitter remains responsible for reviewing the final narrative, commands, citations, and course-policy compliance.

## Limitations of the AI interaction record

This is a faithful summary supported by saved repository evidence, not a word-for-word chat export. It cannot reproduce messages or transient tool output that were not retained. AI assistance may introduce coding or interpretation errors; mitigation consisted of deterministic tests, schema checks, stable-ID artifacts, hashes, controlled transitions, freeze chronology, clean-process reproduction, and visual inspection. These checks establish internal reproducibility in the locked environment but do not provide external label adjudication or prove bit-identical floating-point scores on every future platform.
