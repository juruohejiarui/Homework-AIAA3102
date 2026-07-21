# Ticket 5 Freeze Decision

Frozen at: 2026-07-21T23:34:47+08:00

Selected: retain the frozen Ticket 4 model (lr_c1_balanced_default); apply no label corrections.

Retain the frozen Ticket 4 balanced Logistic Regression model and do not apply training-label corrections. The eight-row correction probe preserved all original/proposed labels and source data, but dev F1 fell from 0.7520849128127369 to 0.7488653555219364 (delta -0.003219557290800479), failing the predeclared -0.002 noninferiority margin. It fixed 3 Ticket 4 dev errors while creating 8. Duplicate conflicts remain important audit findings, but benchmark inconsistency and annotation ambiguity make silent relabeling unjustified.

No Ticket 5 held-out artifact or label informed this decision. Held-out labels and rows are immutable, and held-out reporting cannot reopen selection.
