# Ticket 1 Baseline Freeze Decision

Frozen at: 2026-07-21T23:30:08+08:00

The exact baseline recorded in `frozen_baseline_config.json` was frozen for Ticket 1 before the held-out comparison. The decision was based only on the project specification, fixed train/dev split, successful convergence, dev evidence, and two-run reproducibility.

The frozen model uses raw `text` only, default `TfidfVectorizer`, default `LogisticRegression` behavior with `random_state=3102`, the default prediction rule, no class weights, no threshold tuning, no metadata, and no added normalization.

Held-out performance observed at freeze: **no**.
