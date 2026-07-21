# Ticket 2 — Text Normalization Lever

## Required evidence map

The ticket's **hypothesis** is decomposed by surface type under "Separate hypotheses." Its **intended lever** is exactly one of URL, mention, hashtag, punctuation, casing, or emoji normalization at a time. The **controlled setup** is documented under "Experimental discipline and chronology," "Composable implementation," and "Raw control verification." **Dev evidence** is reported under "Dev results" and "Surface-perturbation robustness." The immutable **frozen decision** appears under "Frozen decision." **Held-out evidence** appears only under "Final held-out evidence." **Concrete prediction changes** are given by stable ID in both representative-change sections. The **interpretation** is separated by lever under "Per-lever interpretation" and summarized in "Conclusion." The **limitation** is explicit under "Limitations." This map adds no post-hoc selection rule.

## Decision and result

Ticket 2 tested six independently switchable normalization levers against the exact frozen raw-text control: URL placeholdering, mention placeholdering, hashtag-marker stripping, Unicode punctuation-to-space conversion, Unicode casefolding, and emoji placeholdering.

The dev-only decision was to adopt **URL placeholder normalization**: replace each complete URL matching `(?i)\b(?:https?://|www\.)\S+` with the literal token `URLTOKEN`, then apply the otherwise unchanged frozen `TfidfVectorizer` and `LogisticRegression` pipeline. On dev, this variant increased target-1 F1 from `0.7388120423108218` to `0.7403132728771641`, increased precision and accuracy, repaired 28 baseline errors, created 22, and was exactly invariant under deterministic URL substitutions on all 767 affected dev rows. The raw control changed 275 predictions under the same URL perturbation.

The choice was frozen before Ticket 2 held-out access. Its single held-out evaluation produced F1 `0.7531172069825436` and accuracy `0.8049901510177282`. Relative to the same frozen Ticket 1 baseline, it fixed 22 false positives and 8 false negatives while creating 4 false positives and 15 false negatives. The selection was not reopened after seeing these results.

## Experimental discipline and chronology

1. The exact Ticket 1 text-only baseline remained the comparator and model template.
2. Separate hypotheses and a selection rule were saved in `experiments/ticket-2/dev/experiment_plan.json` before the dev command ran.
3. The normalizations were implemented as boolean switches in one stateless scikit-learn transformer. Every scored variant enabled zero switches (control) or exactly one switch.
4. The raw control and six variants were fitted only on the 4,567 fixed train IDs and evaluated only on the 1,523 fixed dev IDs.
5. Each variant retained the frozen word-unigram TF-IDF and default logistic-regression settings. No threshold, class weight, metadata feature, or classifier setting changed.
6. Each candidate was compared directly with the frozen raw baseline, not with another normalization candidate.
7. After metrics, error transitions, changed examples, and perturbation behavior were inspected, URL placeholdering was selected using dev evidence only.
8. The decision was frozen at `2026-07-21T13:00:54+08:00` in `experiments/ticket-2/frozen_decision.json`. At freeze time, `heldout_observed_at_freeze=false` and `ticket2_heldout_evaluation_count_at_freeze=0`.
9. The frozen URL variant was evaluated once on held-out. The start/completion ledgers record evaluation number 1 and `selection_reopened=false`.

No Ticket 3 experiment was created or run.

## Separate hypotheses

| Variant | One normalization lever | Pre-execution hypothesis |
|---|---|---|
| Raw control | None | The frozen baseline may already be invariant to several surface forms because default TF-IDF lowercases words and ignores much punctuation and emoji. It is the required control. |
| URL placeholder | Replace complete URLs with `URLTOKEN` | Collapsing unstable domains and paths while preserving URL presence should reduce memorization of link fragments and improve robustness with little loss of disaster-semantic evidence. |
| Mention placeholder | Replace `@handle` with `MENTIONTOKEN` | Collapsing identities should remove user-specific vocabulary while preserving mention presence, improving robustness with little score cost. |
| Hashtag-marker stripping | Remove `#` while retaining the following word | The topical word should remain available and hash-mark formatting should become irrelevant; the baseline token pattern may already make this equivalent. |
| Punctuation-to-space | Replace each Unicode punctuation character with a space | Canonical boundaries may reduce stylistic noise, but apostrophes, underscores, and URL punctuation may carry tokenization effects; movements may be mixed. |
| Unicode casefold | Apply `str.casefold()` before the vectorizer | Case perturbations should become harmless. Because the frozen vectorizer already lowercases, visible results should mostly match except for non-ASCII case mappings. |
| Emoji placeholder | Replace recognized emoji spans with `EMOJITOKEN` | A common presence token may preserve affect without relying on a particular glyph. The frozen word tokenizer ignores most emoji, so visible movement may be small. |

## Composable implementation

`pipeline/normalization.py` defines a `TextNormalizer` with six explicit boolean parameters:

- `replace_urls`
- `replace_mentions`
- `strip_hashtag_markers`
- `punctuation_to_space`
- `casefold_text`
- `replace_emoji`

The registry maps every experiment name to at most one enabled switch. The transformer has no learned state, so all vocabulary and inverse-document-frequency learning still occurs inside the train-only TF-IDF fit. The pipeline sequence is:

```text
raw text -> selected stateless normalizer -> frozen TfidfVectorizer -> frozen LogisticRegression
```

The replacement strings are ordinary word tokens (`URLTOKEN`, `MENTIONTOKEN`, and `EMOJITOKEN`) so the frozen token pattern retains the fact that the surface object existed. Transformations are composable in code, but combinations were deliberately not evaluated for this ticket.

The exact frozen downstream settings remain word unigrams, `lowercase=True`, no stop-word list, no feature cap, L2 normalization, `C=1.0`, `solver='lbfgs'`, `class_weight=None`, `max_iter=100`, default prediction at 0.5 through `Pipeline.predict`, and `random_state=3102`. Only the selected text-normalization switch differs from Ticket 1.

## Raw control verification

The Ticket 2 raw control includes an identity normalizer but is otherwise the frozen Ticket 1 pipeline. Its 1,523 dev IDs, labels, predictions, and model name exactly match the retained Step 4 baseline; its probabilities match at absolute tolerance `1e-15`. This assertion is enforced by the dev runner before it writes results.

Control dev metrics:

- Precision: `0.7909407665505227`
- Recall: `0.6931297709923664`
- F1: `0.7388120423108218`
- Accuracy: `0.7892317793827971`
- TN/FP/FN/TP: `748/120/201/454`
- Convergence: 15 iterations of 100, no warnings

This exact reproduction makes subsequent differences attributable to the enabled normalization rather than a changed model template.

## Dev results

| Variant | Precision | Recall | F1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw control | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 |
| URL placeholder | 0.804659 | 0.685496 | 0.740313 | 0.793171 | 759 | 109 | 206 | 449 |
| Mention placeholder | 0.790210 | 0.690076 | 0.736756 | 0.787919 | 748 | 120 | 203 | 452 |
| Strip hashtag marker | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 |
| Punctuation to space | 0.792321 | 0.693130 | 0.739414 | 0.789888 | 749 | 119 | 201 | 454 |
| Unicode casefold | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 |
| Emoji placeholder | 0.790941 | 0.693130 | 0.738812 | 0.789232 | 748 | 120 | 201 | 454 |

All fits converged without warnings. Iteration counts were 15 for the control, 22 for URL normalization, 14 for mention normalization, and 15 for hashtag, punctuation, casefold, and emoji variants.

### Error transitions against the frozen baseline

| Variant | F1 delta | Changed labels | Fixed FP | Fixed FN | New FP | New FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| URL placeholder | +0.001501 | 50 | 22 | 6 | 11 | 11 |
| Mention placeholder | -0.002056 | 8 | 2 | 1 | 2 | 3 |
| Strip hashtag marker | 0.000000 | 0 | 0 | 0 | 0 | 0 |
| Punctuation to space | +0.000602 | 5 | 1 | 2 | 0 | 2 |
| Unicode casefold | 0.000000 | 0 | 0 | 0 | 0 | 0 |
| Emoji placeholder | 0.000000 | 0 | 0 | 0 | 0 | 0 |

URL placeholdering produced the clearest precision-oriented change: false positives fell from 120 to 109, while false negatives rose from 201 to 206. It fixed six more total errors than it created, increasing accuracy by about 0.00394. Its F1 gain is modest, so robustness and the semantics of the changed cases are important parts of the decision.

## Surface-perturbation robustness

Each lever had a deterministic paired perturbation applied to dev at inference time. The perturbation changed only the associated surface form. For every affected row, the saved table compares the original and perturbed scores and labels for both the raw control and the matching normalized model.

| Perturbation | Affected rows | Raw changed labels | Raw max score shift | Matching variant changed labels | Matching max score shift |
|---|---:|---:|---:|---:|---:|
| Replace URLs with an unseen deterministic URL | 767 | 275 | 0.577568 | 0 | 0.000000 |
| Replace handles with an unseen deterministic handle | 399 | 3 | 0.098463 | 0 | 0.000000 |
| Remove hashtag markers | 330 | 0 | 0.036076 | 0 | 0.000000 |
| Substitute Unicode punctuation | 1,445 | 0 | 0.098205 | 0 | 0.000000 |
| Swap casing | 1,523 | 0 | 0.000000 | 0 | 0.000000 |
| Replace recognized emoji spans | 0 | 0 | 0.000000 | 0 | 0.000000 |

The matching transformation is exactly invariant by construction: the original and perturbed forms map to the same normalized string. This is strongest practical evidence for URL normalization because the raw model was unusually sensitive to URL-token changes. Hashtag stripping and casefolding add no label-level benefit because the default vectorizer already provides the relevant invariance. Mention normalization removes a smaller but real sensitivity, yet its visible dev errors worsened. Punctuation normalization makes scores exactly invariant but the raw model already retained every label under the paired perturbation. Emoji robustness could not be measured on this split because the declared Unicode ranges matched zero train and dev rows; only the synthetic unit test establishes transformer behavior.

These are stress checks, not estimates of production-shift frequency. The URL perturbation deliberately replaces link text and is more severe than changing only a short-link suffix. It demonstrates a failure mode but does not claim that 35.9% label instability will occur naturally.

## Per-lever interpretation

### URLs

The URL transformation changed 2,420 training texts and 767 dev texts. Its direction partly matches the hypothesis: precision increased from `0.790941` to `0.804659`, false positives fell by 11 net, accuracy improved, and URL substitutions became exactly irrelevant. Recall fell from `0.693130` to `0.685496`, showing that some original link fragments supplied useful or correlated positive evidence. The mechanism is therefore mixed but trustworthy enough to adopt: the representation trades a small amount of recall for higher precision and much stronger invariance to unstable link strings.

### Mentions

Mention replacement changed 1,156 train and 399 dev texts. It made handle substitutions exactly invariant, as hypothesized, but dev F1 fell by `0.002056` and two net correct predictions were lost. The result suggests some handles or handle-derived tokens correlate with the label, but relying on them is unlikely to transfer reliably. The robustness hypothesis is supported; the claim of little visible-score cost is only partially supported, and this lever was not selected.

### Hashtags

Removing hashtag markers changed the raw strings of 1,055 train and 330 dev rows but changed no dev labels or metrics. This strongly matches the hypothesis: the default word token pattern does not treat `#` as part of a word, so `#FloodAlert` and `FloodAlert` ordinarily generate the same token. Raw scores did move slightly under the perturbation, with maximum shift `0.036076`, because edge cases can change token boundaries or interactions, but no decision crossed 0.5. Explicit stripping is unnecessary for this model.

### Punctuation

Punctuation-to-space changed 4,362 train and 1,451 dev texts. It changed only five dev labels, fixed three errors, and created two, yielding a small F1 increase of `0.000602`. The observed cases are near the 0.5 boundary, and the broad transformation also modifies apostrophes, underscores, URL punctuation, and mojibake punctuation. This matches the mixed-effect hypothesis but does not provide enough benefit to justify such a wide intervention.

### Casing

Casefolding changed the strings of 4,397 train and 1,471 dev rows yet produced exactly the same dev labels, scores, confusion matrix, and iteration count as the control. Both the raw and casefold models were also exactly invariant to `swapcase()` across all 1,523 dev texts. The hypothesis is confirmed: default TF-IDF lowercasing already handles the observed case variation, so an additional casefold stage is redundant on this dataset and model.

### Emoji

The emoji placeholder changed zero train and zero dev texts under the declared emoji ranges and was prediction-equivalent to control. This does not prove that emoji are irrelevant. It means the source representation used here contains no characters recognized by this explicit detector; some visually unusual material is mojibake rather than intact emoji. The synthetic tests verify that fire emoji are replaced and that glyph substitution becomes invariant, but there is no in-dataset evidence with which to judge accuracy effects. The emoji hypothesis remains unresolved.

## Representative dev changes

1. **URL normalization fixed a false positive, ID 174.** A non-disaster book advertisement—“Aftershock: Protect Yourself and Profit in the Next Global Financial Meltdown …”—moved from `0.506346` to `0.420190`. Removing unique URL fragments made the figurative/commercial language less likely to cross the disaster boundary, matching the hypothesis.
2. **URL normalization fixed a false positive, ID 971.** “The mixtape is coming …” was labeled non-disaster and moved from `0.516670` to `0.448604`. Again, URL canonicalization prevented sparse link tokens from helping a keyword-heavy but non-disaster tweet.
3. **URL normalization created a false negative, ID 237.** A real-disaster airplane-accident tweet with multiple topical hashtags moved from `0.526536` to `0.471299`. This contradicts any claim that URL fragments are purely noise: removing them and adding a common token changed feature normalization enough to suppress a correct positive.
4. **URL normalization created a false positive, ID 110.** The non-disaster phrase “‘By accident’ they knew what was gon happen” moved from `0.486671` to `0.572534`. A shared URL-presence token can itself acquire weight; canonicalization does not merely delete noise.
5. **Mention normalization created a false negative, ID 2352.** “The @POTUS economy continues to collapse” moved from `0.505759` to `0.461160`. Removing a high-profile handle removed potentially meaningful context, illustrating the transferability-versus-visible-signal tradeoff.
6. **Punctuation normalization fixed one error but created another near the boundary.** ID 3903 moved from `0.499044` to `0.500822`, while ID 3097 moved from `0.500172` to `0.499711`. These tiny shifts reinforce that its apparent gain is fragile rather than a strong semantic improvement.

Variants with zero label changes have no changed-prediction examples to inspect. That absence is itself evidence of redundancy for the current tokenizer, not evidence that the raw strings were identical.

## Frozen decision

URL placeholdering was selected because it was the highest-F1 normalization candidate on dev, improved precision and accuracy, produced favorable total error movement, and directly eliminated the largest measured superficial sensitivity. The decision did not depend only on its `+0.001501` F1 delta.

The freeze records:

- selected variant `normalize_urls_placeholder`;
- exactly one enabled switch, `replace_urls=true`;
- all downstream effective model parameters;
- complete raw-control and selected dev evidence;
- paired URL robustness evidence;
- data, split, Ticket 1 freeze, source, plan, dev-result, prediction, robustness, and dependency hashes;
- held-out unseen and Ticket 2 held-out count zero;
- selection reopening prohibited.

The freeze SHA-256 used by the held-out ledgers is `51a9ed0c07d092fa194a1f1399ae502eb0c3662bfdfb2b661ef22dfcbc5376cf`.

## Final held-out evidence

| Model | Precision | Recall | F1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Ticket 1 baseline | 0.801394 | 0.703364 | 0.749186 | 0.797768 | 755 | 114 | 194 | 460 |
| Frozen URL normalizer | 0.825137 | 0.692661 | 0.753117 | 0.804990 | 773 | 96 | 201 | 453 |

Held-out F1 increased by approximately `0.0039315392301006`, and accuracy increased by approximately `0.0072225869993434`. As on dev, precision rose and recall fell. The candidate changed 49 labels relative to the frozen baseline:

- fixed false positives: 22;
- fixed false negatives: 8;
- new false positives: 4;
- new false negatives: 15.

Thus 30 baseline errors were repaired and 19 correct baseline predictions were lost, a net gain of 11 correct rows. This held-out pattern is consistent with the dev mechanism: URL canonicalization mainly reduces false positives, with a smaller recall cost. It supports the frozen decision but was not used to make or revise it.

The stable Ticket 2 held-out file contains 1,523 unique IDs in exact fixed order and has SHA-256 `486bd8af3f808ed29a4471b384cfa82d6080670f657193c6717f1d0c4305acb4`.

## Representative held-out changes

1. **Fixed false positive, ID 773.** A product advertisement for a “FUEL/GAS SAVER CHEVY TAHOE/BLAZER/AVALANCHE” with two links moved from `0.533045` to `0.460680`. The disaster-associated word “avalanche” was a product name; URL canonicalization reduced the distracting link-specific evidence.
2. **Fixed false positive, ID 936.** A real-estate virtual-tour listing on “Fir St Cannon Beach” moved from `0.519723` to `0.485321`. This is aligned with the hypothesis that arbitrary listing URLs should not determine a disaster label.
3. **New false negative, ID 902.** A real report about possible bioterrorism using genetically modified organisms moved from `0.508666` to `0.499356`, just below the default threshold. The body remains semantically relevant, but the original link features supplied enough marginal evidence to retain the positive.
4. **New false positive, ID 2569.** The short non-disaster text “am boy @Crash_______” plus a link moved from `0.349065` to `0.508014`. This exposes a limitation of the common `URLTOKEN`: URL presence can acquire global positive weight and dominate very short ambiguous texts.

These held-out examples were inspected only after evaluation and did not reopen selection.

## Artifacts and reproducibility

Dev-only command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.run_ticket2_dev --data data\train.csv --split starter\data\split_indices.json --plan experiments\ticket-2\dev\experiment_plan.json --output-dir experiments\ticket-2\dev
```

Freeze command:

```powershell
.\.venv\Scripts\python.exe -m pipeline.freeze_ticket2 --dev-dir experiments\ticket-2\dev --output experiments\ticket-2\frozen_decision.json
```

The exact held-out command is retained in `experiments/ticket-2/heldout/run_command.txt` for provenance. **Do not run it again.** The non-empty output directory and existing Ticket 2 summary row provide additional rerun guards.

Key artifacts:

- plan and hypotheses: `experiments/ticket-2/dev/experiment_plan.json`;
- all dev metrics and confusion matrices: `experiments/ticket-2/dev/results/`;
- stable dev predictions: `experiments/ticket-2/dev/predictions/`;
- changed predictions and complete FP/FN tables: `experiments/ticket-2/dev/changes/` and `errors/`;
- perturbation metrics and label changes: `experiments/ticket-2/dev/robustness/`;
- frozen decision: `experiments/ticket-2/frozen_decision.json` and `freeze_decision.md`;
- held-out metrics, confusion, changes, errors, ledgers, and versions: `experiments/ticket-2/heldout/`;
- stable Ticket 2 prediction interface: `predictions/ticket-2-heldout-predictions.csv`;
- final Ticket 2 row: `results/summary.csv`.

## Conclusion

URL placeholdering is the only normalization adopted for Ticket 2. It provides a small but real dev improvement, a much larger and directly relevant robustness benefit under URL substitutions, understandable error movement, and consistent precision-oriented behavior on the frozen held-out evaluation. Mentions also benefited in robustness but lost dev F1; punctuation provided only a fragile boundary-level gain; hashtag stripping and casefolding were redundant with frozen tokenization; and the dataset provided no intact recognized emoji evidence.

The adopted normalization is not an unexplained cleaning bundle. It changes one surface lever, preserves URL presence, and leaves every TF-IDF, classifier, feature, threshold, and split decision frozen. Its held-out outcome is reported as confirmation of the already closed decision, not as a basis for further tuning.

## Limitations

1. Results come from one fixed dev split, so the small F1 differences—especially URL `+0.001501` and punctuation `+0.000602`—have no resampling-based uncertainty estimate.
2. The robustness perturbations are deterministic stress tests, not a sampled deployment distribution. Exact invariance is guaranteed by canonicalization and does not by itself establish better semantic generalization.
3. The URL regex consumes non-whitespace through the end of a URL and can include adjacent trailing punctuation. It recognizes `http`, `https`, and `www` forms but not every possible obfuscated or scheme-free link.
4. The mention regex recognizes ASCII letters, digits, and underscores. It does not attempt platform-specific internationalized handle rules.
5. Punctuation-to-space is deliberately broad and can alter contractions, underscores, mojibake, and link structure; its five changed labels do not support a universal punctuation policy.
6. The declared emoji ranges matched no source rows. The candidate is unit-tested on synthetic emoji, but no dataset-level conclusion about emoji usefulness is possible.
7. No normalization combinations were tested. This preserves causal clarity but does not answer whether carefully justified interactions might help.
8. Qualitative examples are representative mechanisms, not an exhaustive error taxonomy. Data quality and ambiguous-label analysis remain reserved for Ticket 5.
