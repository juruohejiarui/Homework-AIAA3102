# Ticket 3 Freeze Decision

Frozen at: 2026-07-21T20:14:15+08:00

Selected decision: retain the frozen raw-text baseline and reject shortcut additions.

Retain the frozen text-only baseline. Although text plus keyword and selected shallow features raised dev F1 to 0.7493956486704271, its gain was shortcut-sensitive: masking keyword changed 702 predictions and reduced F1 to 0.6423057128152342, while superficial-text neutralization reduced F1 to 0.7330677290836654. Text plus keyword alone was worse than baseline (0.7350835322195705). Sparse location, length-only, and shallow-only variants were substantially weaker. The visible gain is therefore rejected rather than adopted automatically.

No Ticket 3 held-out artifact or metric was used in this decision. Prior-ticket held-out artifacts already existed, so the Ticket 3 report will reuse the validated baseline predictions without a new held-out fit or prediction pass.
