# Ticket 1 Baseline Freeze Decision

Frozen at: 2026-07-21T12:08:53+08:00

The exact baseline recorded in `frozen_baseline_config.json` was frozen for Ticket 1 before the original local held-out comparison. The decision was based only on the project specification, fixed train/dev split, successful convergence, dev evidence, and two-run reproducibility.

The frozen model uses raw `text` only, default `TfidfVectorizer`, default `LogisticRegression` behavior under scikit-learn 1.9.0 with `random_state=3102`, the default prediction rule, no class weights, no threshold tuning, no metadata, and no added normalization.

Held-out performance observed at original freeze: **no**.

The current copy was reconstructed after a user-requested rollback; its configuration is identical to the original freeze.
