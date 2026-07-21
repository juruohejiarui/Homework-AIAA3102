# Ticket 4 Freeze Decision

Frozen at: 2026-07-21T13:26:17+08:00

Selected: balanced Logistic Regression, C=1.0, raw-text default TF-IDF, probability threshold 0.50.

Select balanced Logistic Regression with the frozen raw-text TF-IDF preprocessing, C=1.0, and probability threshold 0.50. It maximized target-1 dev F1 under the predeclared bounded criterion (0.7520849128127369 versus baseline 0.7388120423108218). The gain came from fixing 42 baseline false negatives while creating 48 new false positives: recall rose from 0.6931297709923664 to 0.7572519083969466 while precision fell from 0.7909407665505227 to 0.7469879518072289. The best baseline threshold-only candidate was 0.47 at F1 0.7494071146245059, and the best regularization-only candidate was C=2.0 at F1 0.7504025764895334.

No Ticket 4 held-out artifact or metric was used. Model, preprocessing, hyperparameters, and threshold are locked; held-out cannot reopen the decision.
