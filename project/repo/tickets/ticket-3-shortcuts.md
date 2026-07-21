# Ticket 3 — Feature and Shortcut Audit

## Required evidence map

The **hypothesis** and audit questions appear under "Audit questions and predeclared hypotheses." The **intended lever** is the isolated inclusion of keyword, location, length, selected shallow counts, or their controlled combinations with text. The **controlled setup** is specified under "Leakage-safe implementation." **Dev evidence** is reported in "Dev metrics," "Interpretable coefficient evidence," and "Perturbation robustness." The **frozen decision** and post-freeze **held-out evidence** are separated under "Frozen decision and held-out reporting." **Concrete prediction changes** are listed by stable ID under "Concrete changed examples." The **interpretation** is made explicit under "Signal classification" and "Conclusion," and the **limitation** appears under "Limitations." The evidence map does not treat the highest visible dev score as automatically trustworthy.

## Decision and answer

The audit found substantial predictive signal in `keyword`, weak signal in text length and selected surface counts, and highly sparse, brittle signal in `location`. The highest visible dev F1 came from text plus keyword and selected shallow features (`0.7493956486704271`, versus `0.7388120423108218` for text-only). That gain was **not adopted**: masking keyword changed 702 predictions and reduced F1 to `0.6423057128152342`; neutralizing superficial text reduced F1 to `0.7330677290836654`; and text plus keyword without the shallow numeric block was already worse than text-only (`0.7350835322195705`).

The frozen Ticket 3 decision is therefore to reject the keyword, location, length, and selected shallow additions and retain the frozen raw-text TF-IDF + Logistic Regression baseline. This is deliberately not the maximum visible dev score. It is the simplest representation whose evidence is primarily tweet content rather than benchmark metadata availability, sparse user identities, or style statistics.

Because this exact text-only model already had validated stable Ticket 1 held-out predictions, Ticket 3 reused those rows after freezing the decision. It did not refit the model or execute another held-out prediction pass. The held-out report is F1 `0.749185667752443`, accuracy `0.7977675640183848`, and zero transitions against the baseline itself.

## Audit questions and predeclared hypotheses

The plan was saved in `experiments/ticket-3/dev/experiment_plan.json` before dev execution. The audit asked how well each feature family predicts alone, whether it complements text, whether its coefficients are interpretable, and whether its apparent benefit survives a relevant perturbation.

| Variant | Predeclared hypothesis |
|---|---|
| Majority floor | No-signal wiring check; target-1 F1 should be zero. |
| Text-only control | Frozen legitimate-content comparator. |
| Keyword-only | Strong topical signal, but mixed evidence because the field is benchmark-provided retrieval metadata rather than tweet prose. |
| Length-only | Weak style signal, primarily a dataset artifact. |
| Keyword + length | Length may complement keyword, but the combination remains shortcut-prone. |
| Location-only | Sparse user-entered categories should be weak and brittle; apparent signal is likely an artifact. |
| Keyword + location | Location may add memorized identity-like categories but should not be trusted without robust evidence. |
| Selected shallow-only | Length, counts, ratios, and missingness may expose dataset construction patterns but should remain weaker than semantic text. |
| Text + keyword | Keyword may complement content, but eligibility requires acceptable behavior under keyword masking. |
| Text + keyword + selected shallow features | A visible gain is possible, but adoption requires that it not be dominated by metadata availability or superficial formatting. |

The predeclared selection rule explicitly rejected “take the highest dev F1.” A candidate needed favorable error movement, an interpretable mechanism, and acceptable behavior when the feature source was masked or superficial text was neutralized. Sparse location identity was presumptively an artifact unless contrary evidence emerged.

## Leakage-safe implementation

All models fit exclusively on the 4,567 fixed train IDs and were selected using the 1,523 fixed dev IDs. IDs were joined by Kaggle `id` and emitted in fixed JSON order. Held-out rows were not loaded by the dev command.

### Text

The text-only control is the exact frozen `TfidfVectorizer` + `LogisticRegression` baseline. Its dev IDs, labels, predictions, and class-1 probabilities reproduce the retained Step 4 artifact exactly (scores at absolute tolerance `1e-15`). Combined models use the same default word-unigram vectorizer inside a train-fitted `ColumnTransformer`.

### Keyword and location

Categorical fields are processed inside the scikit-learn pipeline:

```text
column -> train-fitted SimpleImputer(constant sentinel)
       -> train-fitted OneHotEncoder(handle_unknown='ignore')
```

Missing keyword becomes `__MISSING_KEYWORD__`; missing location becomes `__MISSING_LOCATION__`. This is explicit rather than silently converting missing values to the string `nan`. Unknown dev categories produce an all-zero one-hot block rather than failing or learning from dev.

Metadata profile:

| Split | Rows | Missing keyword | Missing location | Unique nonmissing keywords | Unique nonmissing locations |
|---|---:|---:|---:|---:|---:|
| Train | 4,567 | 39 | 1,492 | 221 | 2,152 |
| Dev | 1,523 | 11 | 536 | 221 | 807 |

All dev keyword values occur in train, but 591 dev rows have a nonmissing location string unseen in train. This high-cardinality mismatch makes full location one-hot features structurally fragile.

### Length features

The length-only block contains character count, whitespace-token count, and mean token length. The extraction is stateless; `StandardScaler` learns means and scales only from train.

### Selected shallow features

The shallow block contains:

- character count;
- whitespace-token count;
- mean token length;
- URL count;
- mention count;
- hashtag count;
- digit count;
- Unicode punctuation count;
- uppercase-letter ratio;
- keyword-missing indicator;
- location-missing indicator.

These functions use only the current row, never labels or split-wide statistics. Scaling is fitted on train. The richest candidate combines raw text TF-IDF, one-hot keyword, and this numeric block; it does not include full location categories.

Every non-floor model uses the same default logistic-regression decision rule, `C=1.0`, `solver='lbfgs'`, `class_weight=None`, `max_iter=100`, and `random_state=3102`. All ten variants converged without warnings.

## Dev metrics

| Variant | Precision | Recall | F1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority floor | 0.000000 | 0.000000 | 0.000000 | 0.569928 | 868 | 0 | 655 | 0 |
| Text-only baseline | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 |
| Keyword-only | 0.679612 | 0.641221 | 0.659859 | 0.715693 | 670 | 198 | 235 | 420 |
| Length-only | 0.536585 | 0.369466 | 0.437613 | 0.591596 | 659 | 209 | 413 | 242 |
| Keyword + length | 0.701695 | 0.632061 | 0.665060 | 0.726198 | 692 | 176 | 241 | 414 |
| Location-only | 0.583333 | 0.138931 | 0.224414 | 0.586999 | 803 | 65 | 564 | 91 |
| Keyword + location | 0.694915 | 0.625954 | 0.658635 | 0.720946 | 688 | 180 | 245 | 410 |
| Selected shallow-only | 0.623967 | 0.461069 | 0.530290 | 0.648720 | 686 | 182 | 353 | 302 |
| Text + keyword | 0.767442 | 0.705344 | 0.735084 | 0.781353 | 728 | 140 | 193 | 462 |
| Text + keyword + shallow | 0.793515 | 0.709924 | 0.749396 | 0.795798 | 747 | 121 | 190 | 465 |

The majority floor confirms wiring. Keyword-only is far stronger than length, location, or the numeric shallow block, establishing real benchmark signal, but remains nearly 0.079 F1 below semantic text. Length adds 0.00520 F1 to keyword-only, yet both remain much weaker than text. Adding location to keyword slightly lowers F1. Text+keyword alone lowers F1 by `0.0037285100912513`: recall rises, but 20 extra false positives more than offset eight fewer false negatives.

The rich candidate gains `0.0105836063596053` F1 and ten net correct predictions. It changes 164 baseline labels, fixing 41 false positives and 46 false negatives while creating 42 false positives and 35 false negatives. The movement is balanced rather than a pure threshold shift, but perturbation evidence shows that it depends heavily on keyword availability.

## Interpretable coefficient evidence

Coefficients are technically valid here because every scored non-floor model ends in binary logistic regression. Numeric features were standardized, making their within-block coefficient magnitudes more comparable. One-hot and TF-IDF magnitudes remain affected by frequency and regularization, so they are associations, not causal importance.

### Keyword

Keyword-only's strongest positive coefficients include `derailment` (`+2.5983`), `debris` (`+2.4783`), `rescuers` (`+2.4098`), `wreckage` (`+2.3730`), and `oil spill` (`+2.3730`). Strong negatives include `aftershock` (`-2.0018`), `body bags` (`-2.0018`), `bloody` (`-1.8664`), `body bag` (`-1.8664`), and `blazing` (`-1.8281`).

The directions are revealing. Some positives plausibly encode actual disaster topics, but strongly negative disaster-looking keywords show that the field also captures how the benchmark was collected and how particular search terms attract figurative tweets. In the rich model, `derailment` remains strongly positive (`+2.0361`) and `body bags` strongly negative (`-1.8108`). The model is learning category-specific label prevalence, not merely the lexical meaning of the word.

### Length and shallow counts

In length-only, standardized character count is positive (`+1.1553`) while whitespace-token count is negative (`-0.8958`). The model distinguishes long character-heavy strings from texts with many separated words, a style relationship rather than disaster semantics.

In the rich model, character count remains the dominant shallow coefficient (`+0.8022`), followed by digit count (`+0.1547`) and keyword-missing (`+0.1281`). Whitespace-token count is negative (`-0.5382`); punctuation (`-0.1251`), URL count (`-0.1087`), mention count (`-0.0816`), and uppercase ratio (`-0.0676`) are also negative. These directions are plausible dataset-style correlations but are not stable definitions of a real disaster.

### Location

Location-only assigns large positives to `India` (`+1.5135`), `Mumbai` (`+1.3439`), and `Nigeria` (`+1.3153`), while `304` (`-1.1935`) and `NYC` (`-1.1085`) are strongly negative. Many locations occur too rarely to distinguish geography from individual users or source clusters. The 591 unseen-location dev rows and collapse under masking confirm that these coefficients should not be interpreted as geographic disaster risk.

### Text

In the rich model, positive text coefficients include `hiroshima`, `fire`, `train`, `california`, and `crash`; negatives include `you`, `new`, `my`, `full`, and `or`. Text itself also contains benchmark artifacts—`http` is strongly positive in text+keyword—yet it remains direct task content and substantially outperforms every shallow-only family.

The full saved coefficient table contains the top 25 positive and negative features per available group and model.

## Perturbation robustness

Perturbations were applied only on dev after train fitting. They are stress tests, not selectable variants.

### Metadata masking

| Model | Perturbation | Changed predictions | Perturbed F1 | Delta from original |
|---|---|---:|---:|---:|
| Keyword-only | Mask keyword | 905 | 0.601469 | -0.058389 |
| Keyword + length | Mask keyword | 929 | 0.601656 | -0.063404 |
| Location-only | Mask location | 156 | 0.000000 | -0.224414 |
| Keyword + location | Mask keyword | 902 | 0.603263 | -0.055371 |
| Keyword + location | Mask location | 44 | 0.657028 | -0.001606 |
| Text + keyword | Mask keyword | 468 | 0.680352 | -0.054732 |
| Text + keyword + shallow | Mask keyword | 702 | 0.642306 | -0.107090 |
| Text + keyword + shallow | Mask location | 5 | 0.745955 | -0.003441 |
| Text + keyword + shallow | Mask both | 696 | 0.643263 | -0.106133 |

Masking sets the field to the explicit missing sentinel. This is deliberately a missing-field stress test, not a permutation importance estimate. The rich candidate's behavior is particularly concerning: keyword masking changes almost half the dev predictions and pushes recall to `0.952672` with precision `0.484472`. The learned missing-keyword category and redistributed classifier weights turn the model into an over-positive rule. A system that may not receive Kaggle `keyword` consistently would fail badly.

Location-only becomes the majority floor when location is masked. This confirms that its already-low visible signal resides entirely in brittle user-entered categories. Location masking barely changes the rich model because it uses only a location-missing indicator, not full location identities.

### Superficial text neutralization

The stress transformation canonicalizes URLs and mentions, removes hashtag markers, converts punctuation to spaces, and casefolds. It intentionally combines superficial changes only for robustness auditing; it is not a candidate cleaning pipeline.

| Model | Changed predictions | Perturbed F1 | Delta from original |
|---|---:|---:|---:|
| Text-only | 99 | 0.721053 | -0.017759 |
| Length-only | 297 | 0.249110 | -0.188503 |
| Keyword + length | 125 | 0.620567 | -0.044493 |
| Selected shallow-only | 242 | 0.520158 | -0.010132 |
| Text + keyword | 77 | 0.714044 | -0.021040 |
| Text + keyword + shallow | 82 | 0.733068 | -0.016328 |

The raw baseline is not perfectly surface-robust, a limitation already motivating Ticket 2. The rich model retains a small edge over perturbed text-only in this particular stress test, but it falls below the unperturbed baseline and remains catastrophically keyword-dependent. No shallow feature set provides a generally robust improvement.

## Concrete changed examples

1. **Keyword-only fixed a false negative but exposed the missing-keyword artifact.** Dev ID 7, a real wildfire-smoke report, moved from baseline score `0.412138` to `0.663488`. But non-disaster ID 25 (“Summer is lovely”) received the identical `0.663488` because both have missing keyword. One sentinel category overrides entirely different tweet meanings.
2. **Length-only discarded short, clear disasters.** ID 4 (“Forest fire near La Ronge Sask. Canada”) moved from a correct `0.833987` positive to `0.236998` and became a false negative. Its brevity is not evidence against a disaster.
3. **Location-only memorized metadata rather than content.** ID 86, a non-disaster publication message with location “Inang Pamantasan,” was fixed, but ID 4 and ID 16 (“Three people died from the heat wave so far”) became false negatives. Rows with missing/unseen locations receive nearly content-independent scores.
4. **Text+keyword fixed a real short report and created a keyword-driven false positive.** ID 105 (“BigRigRadio Live Accident Awareness”) moved from `0.467047` to `0.538284`, while non-disaster ID 110 (“‘By accident’ they knew what was gon happen”) moved from `0.486671` to `0.569292` because both share `accident` metadata.
5. **The rich candidate fixed weak-text positives.** ID 18, an informal flooding report, moved from `0.311628` to `0.563404`. Keyword missingness, length, digits, hashtags, and text jointly rescue it, but the same missingness mechanism also makes the model unstable when keyword is systematically absent.
6. **The rich candidate created a figurative false positive.** ID 331 describes a concert that “annihilated the place”; the score rose from `0.418088` to `0.521747`. The keyword and style block overrode the figurative content.
7. **The rich candidate created a false negative on an unusual product-like positive.** ID 519, labeled 1, moved from `0.643889` to `0.432717`. Its uppercase, URLs, and retail style resemble benchmark negatives, demonstrating why shallow style correlations are mixed evidence.

These examples were selected from complete stable-ID change tables. They illustrate both fixes and new errors; none was used to alter the frozen decision after held-out reporting.

## Signal classification

| Signal | Classification | Evidence and rationale |
|---|---|---|
| Raw tweet text | **Legitimate task information** | It directly describes the event and outperforms every shallow-only family. It still contains superficial artifacts, so “legitimate” does not mean perfectly robust. |
| Keyword | **Mixed evidence** | Disaster-topic categories are meaningful, but category-specific label prevalence, missing-sentinel behavior, and 702 masked-field changes in the rich model show benchmark/acquisition dependence. |
| Character and token length | **Dataset artifact** | Weak alone; short factual disasters are rejected; coefficient directions describe formatting style rather than event truth. |
| URL/mention/hashtag/punctuation/case/digit counts | **Mixed evidence** | Some may correlate with live reporting behavior, but the coefficients largely capture platform and collection style and do not provide robust standalone performance. |
| Keyword missingness | **Dataset artifact** | A missing field assigns diverse texts the same strong category effect and causes extreme behavior under systematic masking. |
| Full location categories | **Dataset artifact** | Very high cardinality, 591 unseen dev rows, user-entered noise, low F1, and total collapse when masked. |
| Location missingness | **Dataset artifact** | Tiny coefficient and five changed predictions when masked in the rich model; it reflects data collection completeness rather than disaster content. |

The categories are evidence judgments for this dataset and pipeline, not universal claims. Keyword could be legitimate in a production system where it is reliably generated by an audited upstream process; that deployment contract is absent here.

## Frozen decision and held-out reporting

The decision was frozen at `2026-07-21T13:13:53+08:00` with SHA-256 `651905972507d06427d427e3e3f0a219faba48d8dff2183dbd86e4c194cf0ac6`. The freeze records the dev metrics, rejected best-visible candidate, perturbation failures, exact source/data/artifact hashes, and the rule that selection cannot reopen.

This ticket occurs after Ticket 1, so prior held-out artifacts necessarily existed. The freeze states this explicitly rather than claiming held-out was globally unseen. It also records `ticket3_heldout_artifact_used_in_decision=false` and a Ticket 3 held-out reporting count of zero.

After freeze, the reporting command validated and reused `predictions/heldout_predictions.csv`. There was no new fit and no new prediction pass. The Ticket 3 copy changes only the `ticket` provenance column.

| Precision | Recall | F1 | Accuracy | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.801394 | 0.703364 | 0.749186 | 0.797768 | 755 | 114 | 194 | 460 |

Transitions versus the frozen baseline are necessarily zero. This is not an omitted comparison; it is the direct consequence of rejecting all shortcut additions.

## Artifacts and reproducibility

Dev audit:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket3_dev --data data\train.csv --split starter\data\split_indices.json --plan experiments\ticket-3\dev\experiment_plan.json --output-dir experiments\ticket-3\dev
```

Freeze:

```powershell
.\.venv\Scripts\python.exe -m pipeline.freeze_ticket3 --dev-dir experiments\ticket-3\dev --output experiments\ticket-3\frozen_decision.json
```

The exact held-out reporting command is retained in `experiments/ticket-3/heldout/run_command.txt`. **Do not run it again.** It is a one-time provenance/reporting step and not a model evaluation rerun.

Key evidence includes dev metrics and confusion matrices, ten stable prediction files, ten change tables, twenty error tables, top coefficients, metadata profile, perturbation metrics and changed rows, freeze ledgers, held-out copies, `predictions/ticket-3-heldout-predictions.csv`, and the Ticket 3 row in `results/summary.csv`.

## Conclusion

Keyword contains genuine predictive signal, but its strongest behavior is entangled with benchmark construction and field availability. Length and surface statistics capture style, not a reliable definition of disaster truth. Full locations are too sparse and identity-like to trust. The richest shortcut model improves visible dev F1, but its failure under keyword masking is large enough to reject the gain.

Ticket 3 therefore retains text-only. This decision answers the audit's real question—what evidence is trustworthy—rather than treating every dev improvement as a feature request. Ticket 2's separately frozen URL-normalization finding remains valid for its own lever, but Ticket 3 does not silently combine ticket decisions or start model/threshold work reserved for Ticket 4.

## Limitations

1. One fixed dev split cannot quantify uncertainty in coefficient ranks or small metric differences.
2. Mask-to-missing is a severe availability shift, not a permutation test; it also activates the learned missing category. That severity is intentional because missing metadata is a plausible deployment state.
3. The combined superficial-text perturbation changes several related surface forms at once and cannot attribute its score drop to one transformation. Ticket 2 provides the one-lever normalization evidence.
4. Logistic coefficients reflect regularized associations and feature scaling; correlated TF-IDF, keyword, and numeric features can redistribute weight.
5. Full location one-hot encoding cannot generalize to unseen location strings; richer geocoding was not attempted because it would introduce external data and a different hypothesis.
6. “Retain text-only” means zero Ticket 3 feature additions. It does not reverse or combine decisions from other tickets.
7. Prior held-out metrics existed from earlier completed tickets. Procedural protection here is a hashed dev-only decision and no use of prior held-out artifacts in the freeze rationale, not literal ignorance of all earlier results.
