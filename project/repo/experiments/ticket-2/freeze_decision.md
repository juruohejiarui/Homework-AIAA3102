# Ticket 2 Freeze Decision

Frozen at: 2026-07-21T20:13:37+08:00

Selected variant: `normalize_urls_placeholder`.

Selected on dev only: URL placeholdering had the highest normalization-variant F1 (0.7403132728771641 versus raw 0.7388120423108218), increased precision and accuracy, fixed 28 baseline errors while creating 22, and was exactly invariant on all 767 URL-perturbed dev rows while the raw control changed 275 predictions.

Held-out observed for Ticket 2 at freeze: **no**.

The selection is closed. Held-out may be evaluated once and must not reopen the choice.
