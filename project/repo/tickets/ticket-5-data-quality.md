# Ticket 5 — Data Quality and Error Analysis

## Required evidence map

The ticket's duplicate, error, and correction **hypothesis** is stated under "Hypotheses." Its **intended lever** is the eight-row, in-memory training-label correction set; duplicate discovery and manual dispositions are audits, not hidden training mutations. The **controlled setup** is documented under "Audit chronology and leakage controls," "Deterministic duplicate definitions," and "Controlled training-label correction experiment." **Dev evidence** includes the train/dev duplicate counts, complete dev-error review, and correction-probe transitions. The **frozen decision** is explicit under "Frozen Ticket 5 decision." Post-freeze **held-out evidence** is confined to "Full cross-split and held-out duplicate audit" and "Held-out model evidence." **Concrete prediction changes** are given by stable ID in the correction and held-out discussions. The **interpretation** is summarized in "Conclusion," and the **limitation** appears under "Limitations." No source, dev, or held-out label was edited.

## Required question and answer

Which duplicates, likely mislabels, ambiguous tweets, hard negatives, and false-positive audit findings limit the classifier, and should any data-quality intervention change the frozen model?

The dataset contains substantial duplicate and annotation inconsistency. In the train/dev-only decision phase, 50 raw-exact duplicate groups contained 124 rows; 13 groups had conflicting labels and 19 crossed the train/dev boundary. A conservative canonicalization expanded this to 214 groups and 677 member rows; 46 groups conflicted and 98 crossed train/dev. The bounded near-duplicate audit found 693 pairs involving 538 unique IDs, including 104 label-conflicting pairs and 265 cross-train/dev pairs.

After the model/data decision was frozen, the required full train/dev/held-out relationship audit found 69 raw-exact groups (179 members, 18 conflicting, 41 cross-split), 292 canonical groups (932 members, 64 conflicting, 194 cross-split), and 960 high-similarity non-canonical-equal pairs (715 unique IDs, 186 conflicting pairs, 546 cross-split pairs). These counts show that evaluation rows are not always independent from training rows and that identical evidence can receive opposite labels.

However, neither duplication nor model disagreement makes every error a mislabel. Some conflicts are clear annotation problems, some are scope ambiguities, and many false positives are valid hard negatives caused by disaster words used in titles, metaphors, negations, products, sports, or historical references. A controlled eight-row training-label correction probe was run on dev while preserving every original and proposed label. It reduced target-1 F1 from `0.7520849128127369` to `0.7488653555219364`, a delta of `-0.003219557290800479`; it fixed 3 Ticket 4 dev errors and created 8. It failed both predeclared adoption gates. The frozen Ticket 5 decision was therefore to retain the Ticket 4 balanced Logistic Regression model and apply no label correction to the source data.

The final `results/data_quality_audit.csv` contains 64 unique stable IDs: 30 `fix` recommendations, 15 `ambiguous`, 6 `keep_but_flag`, and 13 `reject_false_positive`. A `fix` row is a recommendation for future data governance, not a mutation performed in this project. `data/train.csv` retains its original SHA-256, no dev or held-out label was edited, and no held-out row was removed.

## Audit chronology and leakage controls

The audit was deliberately split into two phases.

First, `pipeline.run_ticket5_dev` loaded only `train_ids` and `dev_ids`. It generated raw-exact, canonical, and near-duplicate candidates; profiled cross-train/dev relationships; and attached duplicate flags to every frozen Ticket 4 dev error. Its saved configuration records 4,567 train rows, 1,523 dev rows, zero held-out rows loaded, zero held-out labels inspected, zero source rows modified, and zero labels changed.

The correction plan was then written from those train/dev candidates before executing the correction comparison. `pipeline.run_ticket5_corrections` validated that all proposed IDs belonged to train, that stored original labels matched the proposal artifact, and that every proposal confidence was at least `0.90`. It changed labels only in an in-memory copy. The original data hash was checked before and after and remained `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`. Dev labels were never changed.

The no-correction model decision was frozen at `2026-07-21T13:42:00+08:00`. Freeze SHA-256 is `9423deaf6f30659f55547176a5ddb2191ed4895738cf7fe17b9d6138df693afa`. The freeze records that prior-ticket held-out artifacts already existed but that no Ticket 5 held-out artifact or label informed the choice. It prohibits held-out label modification, row removal, and reopening selection.

Only after this freeze did `pipeline.run_ticket5_heldout` load the full dataset to investigate held-out duplicate relationships and frozen-model errors. Because Ticket 5 retained the exact Ticket 4 model, the runner reused the validated Ticket 4 held-out predictions without a new fit or prediction pass. The post-freeze audit could add evidence to the final data-quality table but could not change the frozen model decision.

## Hypotheses

### Exact duplicates

Hypothesis: raw-identical tweets should normally receive the same label. Conflicting labels within a raw-exact group are direct annotation inconsistency unless hidden context outside the stored fields changes interpretation. Same-label cross-split duplicates may leak content from train into evaluation and make examples non-independent even when labels are internally consistent.

### Canonical duplicates

Hypothesis: many tweets differ only in URL, case, HTML entity encoding, Unicode artifacts, or whitespace. Replacing URLs with a common token while retaining words, punctuation, mentions, and hashtags should reveal reposts and templates without erasing most semantic differences. Conflicts in these groups are strong review candidates but not automatically fixes because the linked URL could contain unavailable context.

### Near duplicates

Hypothesis: small edits, source prefixes, handle changes, punctuation, or truncated text create semantically equivalent tweets that raw/canonical equality misses. High character n-gram similarity can surface these pairs. Similarity alone is candidate-generation evidence, not proof of equivalence or a label error.

### Model errors

Hypothesis: high-confidence false positives will include hard negatives containing disaster vocabulary in figurative, negated, historical, product, sports, entertainment, or non-event contexts. Some false negatives will be model weaknesses, while others will reveal target-1 annotations on clearly non-disaster text. Model score and error type are prioritization signals only; neither is used as independent evidence that a label is wrong.

### Controlled label correction

Hypothesis: correcting a small set of very high-confidence train inconsistencies may improve or preserve dev performance. Because conflicting annotations are widespread, a correction may instead make the model less aligned with the noisy benchmark. Adoption therefore requires both independent evidence and acceptable controlled dev behavior.

## Deterministic duplicate definitions

### Raw exact

Raw exact means Python string equality on the unmodified `text` column. IDs, labels, and split membership do not participate in grouping. Each group has a stable SHA-256-derived group ID. The audit records group size, member IDs, labels present, splits present, conflict status, and every member row.

### Canonical exact

Canonicalization performs these deterministic steps:

1. Unicode NFKC normalization.
2. HTML entity decoding, such as `&amp;` to `&`.
3. Unicode-aware casefolding.
4. Replacement of every HTTP, HTTPS, or WWW URL with `<url>`.
5. Whitespace collapse.

Words, hashtags, mentions, and punctuation remain. This is intentionally narrower than a general cleaning pipeline. It detects URL-varied reposts while limiting false equivalence.

### Near duplicate

Canonical strings are represented with `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, sublinear_tf=True, norm='l2')`. A brute-force cosine nearest-neighbor search uses one CPU job and eight neighbors including self, so at most seven non-self neighbors per row are inspected. Canonical-equal pairs are excluded from this table. Pairs with cosine similarity at least `0.88` are retained and deduplicated by ordered stable ID pair.

This is bounded candidate discovery rather than exhaustive clustering. The method is documented in `experiments/ticket-5/dev/audit_plan.json` and implemented in `pipeline/data_quality.py`.

## Confidence scale and disposition semantics

Confidence is a number in `[0,1]` measuring confidence that the assigned disposition is appropriate. It is not the model's probability and not a probability that the stored label is true.

- `0.90–1.00`: very strong direct evidence, normally exact/canonical contradiction plus clear semantics, obvious leakage, or an obvious valid hard negative.
- `0.75–0.89`: strong evidence with limited contextual or task-scope uncertainty.
- `0.55–0.74`: moderate evidence; ambiguity remains and no correction is justified.
- Below `0.55`: exploratory only and omitted from the final table until stronger evidence exists.

The submitted audit's confidence range is `0.86–0.99`, with mean `0.96640625`.

The dispositions are used as follows:

- `fix`: evidence strongly supports a future label correction or duplicate-record repair. The row remains unchanged in this project.
- `keep_but_flag`: retain the row/label but flag leakage, duplication, or unusual evaluation dependence.
- `ambiguous`: multiple reasonable readings, unavailable link context, or irreconcilable identical-text labels prevent a confident correction.
- `reject_false_positive`: reject the suspicion that the row is a data defect. The example is a legitimate hard negative or a credible target-1 row the model failed to understand.

## Train/dev duplicate evidence

| Relationship | Groups/pairs | Member rows or unique IDs | Conflicting groups/pairs | Cross-split groups/pairs |
|---|---:|---:|---:|---:|
| Raw exact | 50 | 124 members | 13 | 19 |
| Canonical exact | 214 | 677 members | 46 | 98 |
| Near | 693 pairs | 538 unique IDs | 104 | 265 |

The categories overlap. A raw-exact group is also usually canonical-equal, so these counts must not be summed as unique affected rows.

Representative contradictions include:

- IDs `4656`, `4659`, `4669`, `4672`, and `4684` have identical Madinah tribal-war/peace text, yet labels are `0,1,1,0,0` across train/dev. The text mentions war historically but emphasizes peace. This is annotation uncertainty, not a safe majority-vote correction.

- IDs `6087`, `6090`, `6097`, `6111`, and `6118` have identical religious text about saving oneself from Hellfire through charity. Four are target 0; train ID `6097` is target 1. Religious metaphor and duplicate consensus independently support recommending `6097: 1→0`.

- IDs `4068`, `4072`, `4076`, and `4077` have exact text about genocide, refugees, and internally displaced people, with labels `1,1,0,1`. This supports recommending train ID `4076: 0→1`.

- IDs `6537`, `6548`, and `6566` share a literal cleared traffic-incident-with-injury text. Two are target 1 and ID `6566` is target 0, supporting `6566: 0→1`.

- Train IDs `8698`, `8702`, `8714`, and `8739` share a figurative “sinking feeling” about discovering a phone used 3G, with labels `1,0,0,1`. The semantics support target 0, even though a canonical dev copy also has target 1. This demonstrates why a semantically plausible correction may conflict with benchmark labels.

- Dev IDs `353` and `390` become identical after URL replacement but have opposite labels. Both refer to “World Annihilation vs Self Transformation” and aliens. The URL is not available as evidence and the content is borderline, so both are classified `ambiguous`, not fixed by voting.

Same-label duplicates also matter. Dev ID `5140` exactly duplicates six target-1 train rows about a manslaughter charge. Dev ID `8183` duplicates three target-1 train rows about migrant rescue bodies. These labels are not disputed, but train/dev content leakage makes their evaluation evidence non-independent; both are `keep_but_flag`.

## Model-error analysis: mislabel, ambiguity, or model weakness

The frozen Ticket 4 dev model made 168 false positives and 159 false negatives. Every error is saved in `experiments/ticket-5/dev/review/dev_model_errors.csv`, ordered by error type and model confidence and annotated with duplicate membership when applicable.

### Likely mislabels

Some rows have semantics that strongly contradict their stored label independently of the prediction:

- Dev ID `5247`, target 0, literally reports police investigating a pedestrian fatality caused by a train. A canonical held-out copy, ID `5228`, is target 1. The model's false positive aligns with the content; the audit recommends `0→1` at confidence `0.97`.

- Dev ID `6407`, target 1, says only “My back is so sunburned :(”. This is a minor personal condition rather than a disaster report. The false negative does not establish model failure; the audit recommends `1→0`.

- Dev ID `6325`, target 1, says a bartender held change “hostage” to obtain a phone number. The use is plainly figurative, so `1→0` is recommended at `0.99`.

- Dev ID `805`, target 1, explicitly describes a virtual space battle with fleets and destroyed ships. The model's low score reflects the non-real context; `1→0` is recommended.

### Valid hard negatives and rejected audit suspicions

Other false positives are model weaknesses rather than bad data:

- Dev ID `6002`, target 0, contains hazardous-weather vocabulary but explicitly says no hazardous weather is expected. The false positive is rejected as a data issue at confidence `0.99`.

- Dev ID `9341`, target 0, is a book listing titled “Suicide of a Superpower.” The model responds to “suicide,” but the negative label is supported.

- Dev ID `1208`, target 0, contains “Lizard Wizard in a Blizzard,” which is title-like wordplay rather than a disaster report.

- Dev ID `7761`, target 0, reports a police search for a missing pregnant woman. This is serious news, but whether it belongs to “disaster” is a task-scope question. Model disagreement alone does not justify relabeling; the audit rejects the automatic mislabel hypothesis with lower confidence `0.86`.

These examples distinguish lexical model weakness from annotation uncertainty. The model overweights topical words without reliably resolving negation, metaphor, titles, or event scope.

## Controlled training-label correction experiment

Eight train corrections met the predeclared evidence gate and were tested as one controlled set:

| ID | Original | Proposed | Principal evidence | Confidence |
|---:|---:|---:|---|---:|
| 4076 | 0 | 1 | Exact group is 3-to-1 for target 1; literal genocide/refugee text | 0.98 |
| 6566 | 0 | 1 | Two exact target-1 copies; literal injury incident | 0.99 |
| 8698 | 1 | 0 | Exact target-0 copies; figurative phone/3G “sinking” | 0.98 |
| 8739 | 1 | 0 | Same figurative phone/3G exact group | 0.98 |
| 1723 | 1 | 0 | Exact target-0 copy; lyric-like “burning buildings” | 0.96 |
| 1760 | 1 | 0 | Exact target-0 copy; hypothetical music-video request | 0.96 |
| 6097 | 1 | 0 | Four exact target-0 copies; religious Hellfire metaphor | 0.99 |
| 9472 | 1 | 0 | Exact dev target-0 copy; terrorism is explicitly negated | 0.98 |

The proposal artifact `label_correction_proposals.csv` preserves stable ID, original label, proposed label, confidence, evidence, split, and explicit flags that changes occurred only in memory and the source was not modified.

The control was the frozen Ticket 4 raw-text default TF-IDF plus balanced Logistic Regression with `C=1.0` and threshold `0.50`. The candidate used the identical configuration and split, changing only those eight copied training labels. Dev labels were untouched.

| Variant | Precision 1 | Recall 1 | F1 1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Ticket 4 labels | 0.746988 | 0.757252 | 0.752085 | 0.785292 | 700 | 168 | 159 | 496 |
| Eight in-memory corrections | 0.742129 | 0.755725 | 0.748865 | 0.782009 | 696 | 172 | 160 | 495 |

The candidate changed 11 dev predictions relative to Ticket 4: 3 fixed false positives, 0 fixed false negatives, 7 new false positives, and 1 new false negative. It failed the `-0.002` F1 non-inferiority margin and the requirement to fix at least as many errors as it created.

The change to train ID `9472` did correct dev ID `9470`, its exact target-0 duplicate: the score fell from `0.633044` to `0.465278`. That local movement matches the proposed correction. But other small boundary movements created errors, including ID `4014` (“My first staining attempt was a disaster”), ID `3633` (a song title containing “I See Fire”), and ID `9943` (a house being “trashed”). This is concrete evidence that a few plausible corrections alter global coefficients and can worsen unrelated borderline examples.

The experiment does not prove the original eight labels are correct. It proves that adopting the correction set as a benchmark model intervention was not justified by the predeclared dev rule. Data governance and benchmark-score optimization are different questions.

## Frozen Ticket 5 decision

The final train/dev-only decision was:

1. Retain `lr_c1_balanced_default` from Ticket 4.
2. Apply no correction to `data/train.csv`.
3. Preserve the eight proposals as explicit evidence, not hidden mutations.
4. Preserve all dev and held-out labels and rows.
5. Use post-freeze held-out inspection only to complete the audit, never to reopen model selection.

This decision is saved in `experiments/ticket-5/frozen_decision.json` with source and artifact hashes. The freeze explicitly records correction rejection, no source mutation, no held-out evidence used, and Ticket 5 reporting count zero.

## Full cross-split and held-out duplicate audit

| Relationship | Groups/pairs | Member rows or unique IDs | Conflicting groups/pairs | Cross-split groups/pairs |
|---|---:|---:|---:|---:|
| Raw exact | 69 | 179 members | 18 | 41 |
| Canonical exact | 292 | 932 members | 64 | 194 |
| Near | 960 pairs | 715 unique IDs | 186 | 546 |

Concrete held-out relationships include:

- Held-out ID `6132`, target 1, exactly duplicates the religious charity/Hellfire text whose four other non-heldout copies are target 0 and one is target 1. The audit recommends `1→0`, but the held-out label remains unchanged.

- Held-out ID `6112`, target 1, exactly matches train IDs `6088` and `6125`, both target 0. The text is explicitly about Hellfire and the afterlife, supporting a likely mislabel recommendation.

- Held-out IDs `6220` and `6223` are exact copies of the same historical D.B. Cooper image tweet but have targets 0 and 1. Because both are inside held-out, the same model evidence is scored once as correct and once as incorrect. Both are `ambiguous`, not silently reconciled.

- Held-out ID `7613`, target 1, belongs to a large set of near/canonical copies of the “woman delivers baby without face” headline with both labels across all splits and similarity up to `0.9985`. URL/punctuation differences do not explain the disagreement.

- Held-out ID `7804`, target 1, is near/canonical identical to five train/dev/heldout copies about Reddit quarantining offensive content that are target 0. This is a high-confidence `1→0` recommendation.

- Held-out ID `1409`, target 1, is a handbag product listing. Canonical train ID `1420` and held-out ID `1402` are target 0. It is a likely mislabel, not evidence that the model should learn handbags as disasters.

- Held-out ID `5228`, target 1, has the same pedestrian-fatality report as dev ID `5247`, target 0. The held-out label is semantically plausible and retained; the pair is flagged for cross-split conflict/leakage.

Same-label cross-split duplicates can inflate apparent generalization because a model may see nearly the same lexical instance during training. Conflicting duplicates can depress or destabilize measured performance because no deterministic text classifier can satisfy both labels for identical text.

## Held-out model evidence

Ticket 5 selected no new model. After freeze, the existing validated Ticket 4 predictions were copied with Ticket 5 provenance. No model was fitted and no new prediction pass occurred.

| Metric | Ticket 5 frozen report |
|---|---:|
| Precision target 1 | 0.7443609022556391 |
| Recall target 1 | 0.7568807339449541 |
| F1 target 1 | 0.7505686125852918 |
| Accuracy | 0.7839789888378201 |
| TN / FP / FN / TP | 699 / 170 / 159 / 495 |

Relative to the original Ticket 1 baseline, the retained Ticket 4 model has 0 fixed false positives, 35 fixed false negatives, 56 new false positives, and 0 new false negatives. These are the Ticket 5 summary transitions because the final model is unchanged.

Held-out error review reinforces the taxonomy:

- ID `10795`, target 1, “Israel wrecked my home. Now it wants my land,” is a plausible real destruction report missed by the model. The audit rejects a mislabel suspicion and classifies this as model weakness.

- ID `5760`, target 0, explicitly says “no forest fires.” Its false positive is a negation failure and is rejected as a data defect.

- ID `783`, target 0, uses Avalanche as a sports-team name. This is a valid hard negative.

- ID `4003`, target 0, says “I'm a disaster,” a figurative self-description. This is a valid hard negative.

- ID `546`, target 0, reports terrorists charged in a church arson. Here the model's false positive plausibly identifies a likely target-0 annotation error; the audit recommends `0→1`.

- ID `2619`, target 1, says “My iPod crashed.” The model's false negative is not a semantic failure; the row is a likely `1→0` mislabel.

- ID `4043`, target 0, reports police killing a cinema gunman while saying “disaster averted.” It is a real violent event but its task scope is debatable, so it is `ambiguous` rather than automatically fixed.

These examples demonstrate that false positives and false negatives contain mixtures of model weakness, annotation noise, and scope uncertainty. Error direction alone does not determine disposition.

## Required audit artifact

`results/data_quality_audit.csv` uses the exact required schema:

```text
id,issue_type,evidence,disposition,confidence
```

It has 64 rows and 64 unique valid dataset IDs: 9 train, 16 dev, and 39 held-out. Disposition counts are:

| Disposition | Rows |
|---|---:|
| `fix` | 30 |
| `keep_but_flag` | 6 |
| `ambiguous` | 15 |
| `reject_false_positive` | 13 |

Audit SHA-256 is `9387710cc13d24ab914e61514da4d2c6a23a2d1bbd98beb3017e0faf88c39421`. The JSON source record hash, finalizer command, artifact hash, disposition counts, source-mutation flags, and chronology are saved in `experiments/ticket-5/final_audit_manifest.json`. The finalizer validates the exact columns, nonempty evidence, known stable IDs, confidence range, unique IDs, and allowed dispositions.

## Stable artifacts and exact commands

Train/dev discovery command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket5_dev --data data\train.csv --split starter\data\split_indices.json --plan experiments\ticket-5\dev\audit_plan.json --output-dir experiments\ticket-5\dev
```

Controlled correction command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket5_corrections --data data\train.csv --split starter\data\split_indices.json --plan experiments\ticket-5\dev\label_correction_plan.json --output-dir experiments\ticket-5\dev\correction_experiment
```

Post-freeze held-out/reporting command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket5_heldout --data data\train.csv --split starter\data\split_indices.json --freeze experiments\ticket-5\frozen_decision.json --output-dir experiments\ticket-5\heldout --confirm-single-ticket5-report
```

Final audit command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.finalize_ticket5_audit --data data\train.csv --split starter\data\split_indices.json --records experiments\ticket-5\final_audit_records.json --output results\data_quality_audit.csv
```

Important artifacts include:

- `experiments/ticket-5/dev/audit_plan.json`: definitions, hypotheses, confidence scale, and correction gate;
- `experiments/ticket-5/dev/duplicates/`: train/dev exact, canonical, and near-duplicate candidates;
- `experiments/ticket-5/dev/review/`: every Ticket 4 dev error plus prioritized duplicate conflicts;
- `experiments/ticket-5/dev/curated_dev_review.csv`: pre-freeze dispositions with original/proposed labels where applicable;
- `experiments/ticket-5/dev/label_correction_plan.json`: eight pre-execution proposals;
- `experiments/ticket-5/dev/correction_experiment/`: preserved proposals, metrics, predictions, transitions, FP/FN rows, selection result, configuration, and command;
- `experiments/ticket-5/frozen_decision.json` and `freeze_decision.md`;
- `experiments/ticket-5/heldout/duplicates/`: full and heldout-related exact/canonical/near evidence;
- `experiments/ticket-5/heldout/review/`: complete frozen-model held-out error candidates;
- `predictions/ticket-5-heldout-predictions.csv`: 1,523 stable IDs in exact held-out order, SHA-256 `b4ca0706ca355e4ba05a9c09f4c3fb8fd701e07c3ef211d3a9ee93d4a1c465d2`;
- `results/data_quality_audit.csv` and the fifth row of `results/summary.csv`.

## Conclusion

Duplicate leakage and inconsistent annotation materially constrain what this benchmark score means. Exact and near-identical tweets cross split boundaries frequently, and conflicting labels sometimes make correct text-only prediction logically impossible. The error audit also shows many hard negatives where disaster vocabulary appears without a disaster, while several target-1 false negatives are likely annotation errors rather than genuine misses.

The correct Ticket 5 action is conservative. The eight proposed train corrections are well supported as data-governance candidates, but their controlled dev model failed the predeclared adoption rule. The source dataset therefore remains untouched and the Ticket 4 model is retained. The audit separates recommendations from applied changes and preserves uncertainty instead of converting every disagreement into a relabeling claim.

## Limitations

Near-duplicate discovery inspects only seven non-self neighbors per row and uses one character n-gram representation and a fixed `0.88` threshold. It can miss duplicate families whose nearest-neighbor lists are crowded, and it can falsely connect templated but semantically distinct tweets. Canonical URL replacement treats different links as equivalent even though link destinations may provide missing context; links were not fetched or used as external verification.

The final 64-row table is a curated actionable/representative audit, not an exhaustive manual adjudication of all 7,613 labels or all 329 model errors per split. Semantic dispositions reflect the stored text and documented task definition; expert annotators with a formal labeling manual might reasonably disagree. Confidence measures confidence in disposition, not empirical correctness.

The correction experiment tests one conservative eight-row set, not every possible relabeling combination. Its lower dev score may partly reflect noisy dev labels that conflict with corrected train duplicates, so rejection as a model intervention does not validate the original labels. Conversely, post-freeze held-out inspection can identify likely problems but cannot be used to retune or relabel the submitted model. Finally, duplicate relationships mean ordinary point estimates assume more independence than the data actually provides; a future evaluation should group duplicate families before splitting and use multi-annotator adjudication for conflicts.
