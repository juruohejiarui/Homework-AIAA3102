# Final Report Verification Checklist

Date: 2026-07-21 (Asia/Shanghai)

Final source: `report/main.tex`  
Final PDF: `report.pdf`  
Overall result: **PASS** - the report and PDF pass all technical and evidence checks. Course-internal references have no public author/date metadata.

No experiment, model fit, held-out evaluation, ticket decision, source label, or held-out label was changed or rerun during finalization.

## Required checklist

| Area | Status | Verification evidence |
|---|---|---|
| Template compliance | PASS | `report/main.tex` uses `\documentclass[conference]{IEEEtran}`. The compiled PDF is US letter, 10-point, two-column IEEE conference layout; it is not a journal or society-specific class. |
| Section completeness | PASS | Title, Abstract, Keywords, Project Problem and Goal, Methodology, Main Evidence and Results, Case Analysis, Difficulties and Solutions, AI Usage Declaration, Discussion and Limitations, Conclusion, References, and four appendices are present in source and extracted PDF text. |
| Metric consistency | PASS | Ticket 1-5 dev and held-out precision, recall, F1, accuracy, and TN/FP/FN/TP were independently recomputed from the archived prediction CSVs and matched the frozen manifest and generated tables within `1e-12`. |
| Table consistency | PASS | All 14 source CSVs and report-ready LaTeX tables regenerated from verified artifacts. `report_assets/validation_report.json` records 196 passing assertions; the final verifier checks summary, metric, confusion, transition, case, and source relationships. |
| Figure consistency | PASS | All four figures regenerated from their dedicated CSV data. The final verifier checks the 61-row threshold curve, 10 dev/held-out F1 values, five held-out transition rows, and four audit-disposition counts against their source artifacts. |
| Case consistency | PASS | All 16 stable IDs in Table XII were checked for source text, true label, prediction(s), score(s), transition/disposition, and source path. Both successes and failures are retained. ID 767 uses only machine CSV scores `0.4876230020837692 -> 0.5448153095708489`. |
| Ticket coverage | PASS | Tickets 1-5 each include the question/hypothesis and lever, controlled dev setup, dev decision, frozen configuration, held-out report, error transitions, representative evidence, conclusion, and limitations. |
| Frozen-decision consistency | PASS | All five data, split, requirement, dev-prediction, held-out-prediction, freeze, and held-out-completion hashes match `configs/frozen_decisions.json`. The final identity remains Ticket 5 retaining Ticket 4's balanced Logistic Regression with no label corrections. |
| Held-out integrity | PASS | Every decision records `heldout_used_for_selection=false` and `selection_reopening_permitted=false`; `heldout_labels_modified=false`, zero held-out rows were removed, and the submission validator reports `refit_performed=false`. |
| Reproducibility | PASS | Pinned `scripts/build_report.ps1` verified Tectonic 0.16.9 archive SHA-256 `131A24604785A9600989A3D91225F597DF52AC06F00AEFFE86FD529F99EE5CDD`; the PDF compiled from a workspace without a system LaTeX installation. Project tests, package checks, bytecode compilation, artifact generation, source validation, submission validation, and final verification all pass. |
| Citation quality | WARNING | All 11 citation keys resolve and the dataset, public mirror, software, and Codex entries use real relevant URLs. Course handout/clarification/starter entries are repository-local sources without public author/date metadata. |
| Difficulty verification | PASS | Table XIII contains six artifact-supported difficulties and documented responses. The narrative treats rollback reconstruction as a qualified recovery event and does not claim independently recoverable Git history. |
| AI usage accuracy | PASS | The declaration matches `logs/chat.md`: Codex assisted with repository work, implementation/testing, authorized orchestration, auditing, and report preparation; outputs were accepted, changed, or rejected only after tests and artifact checks. It does not claim an independent labeler or expert service. |
| Limitation coverage | PASS | The report retains the unresolved Ticket 1 reference cause, single-split uncertainty, no confidence intervals/external testing, duplicate dependence, shortcut risk, unadjudicated labels, finite Ticket 4 search, bounded Ticket 5 audit, hidden-test risk, and cross-platform floating-point caveat. |
| PDF rendering quality | PASS | `report.pdf` has readable extracted text, US-letter page dimensions, complete table, figure, and section coverage, and no placeholder text. |

## Automated validation results

| Command or artifact | Result |
|---|---|
| `scripts/generate_report_assets.py` | PASS - 206 assertions; 15 tables; six PNG and six SVG figures; 40 manifest assets |
| `scripts/validate_report_source.py` | PASS - 132 checks; 15 table inputs; six figure inputs; approximately 6,480 source words |
| `scripts/verify_final_report.py` | PASS - 186 checks; 16 cases; current PDF SHA-256 recorded in `report/final_verification.json` |
| `python -m pytest -q` | PASS - 56 passed in 2.25 s |
| `python -m pipeline.validate_submission` | PASS - five tickets, five summary rows, 61 thresholds, 64 audit rows, 1,523 held-out predictions, no refit |
| `python -m pip check` | PASS - no broken requirements |
| `python -m compileall -q pipeline scripts tests` | PASS |
| Final Tectonic compilation | PASS - no errors, overfull boxes, unresolved citations, or unresolved references |

## Remaining warnings

1. Course-internal references are real repository artifacts but lack public bibliographic author/date metadata.
2. Tectonic reports non-fatal underfull-box and font-substitution warnings.
3. The Ticket 1 supplied-reference discrepancy remains causally unresolved; this is a scientific limitation, not a verification failure.

Final PDF SHA-256: `99677fe6ea573942d558bd3eda96d88453b930b9f68d3678848a107b987abec7`.
