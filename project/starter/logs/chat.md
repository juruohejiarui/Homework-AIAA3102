# AI-Assisted Work Record

This is a concise, truthful summary of the AI-assisted work performed for this submission. It is not a reconstructed verbatim transcript.

## Audit Request

The project draft was compared with the Topic A handout and the clarification that all `fixed_fp`, `fixed_fn`, `new_fp`, and `new_fn` counts in `summary.csv` must be measured against the same frozen original baseline.

The review verified that the implementation already used Ticket 1 predictions as that common baseline. It also identified missing deliverables (`report.pdf` and this record), insufficient baseline discrepancy evidence, a Ticket 2 stress test that did not evaluate the selected model, a Ticket 3 metadata stress test applied to a text-only model, incomplete Ticket 4 attribution, an incorrect regularization phrase, and insufficient concrete Ticket 5 examples.

## Changes Proposed and Verified

The following suggestions were implemented and independently checked against regenerated artifacts:

- Evaluate Ticket 2 stress perturbations on both the raw baseline and the selected `url_replace` model.
- Apply Ticket 3 metadata perturbations only to models that consume keyword or location fields.
- Add `results/decision_ablation.csv` to separate threshold-only, class-weight, regularization, and combined Ticket 4 comparisons.
- Correct the statement that increasing Logistic Regression `C` strengthens regularization; it weakens L2 regularization.
- Expand the baseline forensic probes without changing the frozen submitted baseline.
- Add artifact validation that reconstructs all four error-transition counts from the Ticket 1 frozen baseline predictions.
- Add stable-ID data-quality cases and a formal report source.

## Verification Performed

All commands were run from `project/starter` after activating the standard `.venv` created with Python 3.12.9:

```bash
source .venv/bin/activate
python -m pytest -q
python -m pipeline.cli run-all
python -m pipeline.cli validate-artifacts
```

The final test run reported 60 passing tests. Artifact validation reported 7,615 held-out prediction rows, 5 summary rows, 81 threshold-sweep rows, and 1,688 audit rows. The report only uses metrics and examples from these regenerated artifacts.

## Human Verification Boundary

AI output was treated as a proposal, not evidence. Numerical claims were checked against final CSV files, data-quality examples were checked by stable ID, and code changes were checked through the automated tests and artifact reconstruction. No held-out labels were changed, and no chat content was fabricated as a verbatim record.

## Follow-up Protocol Repair

A later audit found that the previous `run-ticket --split dev` command rebuilt all artifacts through `run-all`, which made the advertised dev/freeze/held-out interface inaccurate. The pipeline was changed so that dev runs create only pending decisions and dev predictions, `freeze-ticket` promotes a pending decision, and held-out commands require a frozen decision and write only that ticket's held-out prediction file. A direct command check confirmed that a dev run leaves the aggregate held-out prediction artifact unchanged.

The follow-up audit also added a conservative `reject_false_positive` disposition for manually reviewed ID 198. Ticket 1 retained its fixed, post-freeze diagnostic matrix: held-out probe scores are forensic evidence only and are checked for agreement with dev deltas, never used to replace the baseline or select later settings. The final regeneration passed 60 tests and artifact validation; `REPORT.md` and `report.pdf` were regenerated from the final artifacts.
