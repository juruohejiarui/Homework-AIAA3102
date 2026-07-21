# Step 10 reproducibility audit

This is the reproducibility audit requested before final report assembly. It is not the final project report and it does not reopen any ticket decision.

## Audit conclusion

**PASS.** All five frozen decisions were independently re-fit from the unchanged source data and fixed ID split in five distinct Python processes. Dev and held-out labels, prediction order, predictions, and scores reproduce the archived artifacts. Every scalar metric reproduces within `1e-15`. The three active result tables reproduce from source/replay evidence, and the final Ticket 5 held-out artifact has the required six-column schema.

## Decision provenance

- The consolidated manifest contains five decisions and five immutable freeze hashes.
- Every freeze records a dev-only decision basis, zero held-out evaluations at freeze time, `heldout_used_for_selection=false`, and `selection_reopening_permitted=false`.
- Step 10 held-out runs are audit replays only. They are not additional primary evaluations and were not used to alter any model, feature, normalization, label, or threshold choice.

## Clean-process reproduction

- Distinct process IDs: `85798, 85871, 85954, 86042, 86120`.
- Archived prediction changes across all ten dev/held-out comparisons: `0`.
- Maximum score discrepancy from an archived artifact: `1.1102230246251565e-16` (acceptance limit `1e-12`).
- Convergence: all five replays completed without convergence warnings.

| Decision | Dev F1 | Held-out F1 | Held-out accuracy | fixed FP / fixed FN / new FP / new FN vs Ticket 1 |
|---|---:|---:|---:|---:|
| Ticket 1 | 0.7388120423108218 | 0.7491856677524430 | 0.7977675640183848 | 0 / 0 / 0 / 0 |
| Ticket 2 | 0.7403132728771641 | 0.7531172069825436 | 0.8049901510177282 | 22 / 8 / 4 / 15 |
| Ticket 3 | 0.7388120423108218 | 0.7491856677524430 | 0.7977675640183848 | 0 / 0 / 0 / 0 |
| Ticket 4 | 0.7520849128127369 | 0.7505686125852918 | 0.7839789888378201 | 0 / 35 / 56 / 0 |
| Ticket 5 | 0.7520849128127369 | 0.7505686125852918 | 0.7839789888378201 | 0 / 35 / 56 / 0 |

## Result-table and stable-ID checks

- `results/summary.csv`: exact required schema, tickets 1–5 exactly once, and semantic reproduction from the five clean replays.
- `results/threshold_sweep.csv`: exact required schema and all 61 thresholds reproduced from the Ticket 1 raw-text dev scores.
- `results/data_quality_audit.csv`: exact required schema and all curated records reproduced from the preserved Ticket 5 audit-record source; IDs are valid and dispositions/confidences pass validation.
- All historical held-out prediction artifacts and the final artifact contain `1523` unique stable held-out IDs in the fixed instructor order, with unchanged true labels.

## Consistent error-transition comparison

Every ticket was recalculated against the same comparator: the clean Ticket 1 held-out replay. These counts reproduce `results/summary.csv`. Ticket 1 and Ticket 3 are intentionally the same prediction core; Ticket 4 and Ticket 5 are intentionally the same prediction core because Ticket 5 retained Ticket 4 without label corrections.

## Stale, duplicate, contradictory, and manual-edit audit

- No unexplained stale active result file was found: each active result table has a source-to-reproduction comparison in this audit.
- No contradictory active metric or transition count was found.
- No unexplained manual edit was found. The summary and threshold sweep are regenerated from replay predictions; the data-quality table is regenerated from its preserved structured record source.
- Two semantic duplicate groups are intentional and documented: Ticket 1 = Ticket 3, and Ticket 4 = Ticket 5 = the final submission core. Their CSV byte hashes differ because the `ticket` provenance field differs.
- `predictions/heldout_predictions.csv` is the historical Ticket 1 baseline despite its generic filename. It remains immutable; the unambiguous final path is `predictions/final-heldout-predictions.csv`.
- Historical pre-freeze and completion ledgers are retained as time-stamped provenance. They are not interpreted as current model state, and none was overwritten during this audit.

## Limitations

This audit demonstrates deterministic reproduction in the locked local environment and verifies internal provenance. It does not claim bit-identical floating-point scores across arbitrary operating systems or future dependency versions; the stronger cross-artifact requirement used here is identical class predictions and score differences no greater than `1e-12`.
