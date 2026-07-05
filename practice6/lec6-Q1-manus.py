import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import loguniform

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Enable the experimental HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

# ---------------------------------------------------------------
# Fixed Setup
# ---------------------------------------------------------------
# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split: 70% Train / 30% Test, stratified, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print("--- Dataset ---")
print(f"Total samples: {len(X)}, Features: {X.shape[1]}, Classes: {np.unique(y)}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

# Build the pipeline: StandardScaler + SVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True))
])

# Define the search space: C and gamma sampled log-uniformly
param_dist = {
    'svc__C':     loguniform(1e-2, 1e2),
    'svc__gamma': loguniform(1e-4, 1e0)
}

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------
# Step 1: Fit HalvingRandomSearchCV on the Train split
# ---------------------------------------------------------------
search = HalvingRandomSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    factor=3,
    resource='n_samples',
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    refit=False   # We will manually refit with best params below
)

print("--- Step 1: Fitting HalvingRandomSearchCV ---")
search.fit(X_train, y_train)
print("Search complete.\n")

# ---------------------------------------------------------------
# Step 2: Refit the pipeline on the full Train split using best hyperparameters
# ---------------------------------------------------------------
best_params = search.best_params_
print("--- Step 2: Refit with Best Hyperparameters ---")
print(f"Best hyperparameters found:")
print(f"  C     = {best_params['svc__C']:.6f}")
print(f"  gamma = {best_params['svc__gamma']:.6f}\n")

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True,
                C=best_params['svc__C'],
                gamma=best_params['svc__gamma']))
])
best_pipeline.fit(X_train, y_train)
print("Refit on full Train split complete.\n")

# ---------------------------------------------------------------
# Step 3: Evaluate once on the Test split
# ---------------------------------------------------------------
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_prob)

# ---------------------------------------------------------------
# Step 4: Print all required results
# ---------------------------------------------------------------
print("--- Step 4: Results ---")

# (a) Best hyperparameters
print(f"Best Hyperparameters:")
print(f"  C     = {best_params['svc__C']:.6f}")
print(f"  gamma = {best_params['svc__gamma']:.6f}")

# (b) Number of candidates evaluated at each halving iteration
import pandas as pd
cv_results = pd.DataFrame(search.cv_results_)
print(f"\nCandidates evaluated at each halving iteration:")
for iteration, group in cv_results.groupby('iter'):
    print(f"  Iteration {iteration}: {len(group)} candidates")

# (c) Best CV ROC-AUC: mean ± std over 5 folds for the selected configuration
#     at its final halving iteration
best_index = search.best_index_
best_mean_cv = search.cv_results_['mean_test_score'][best_index]
best_std_cv  = search.cv_results_['std_test_score'][best_index]
print(f"\nBest CV ROC-AUC (final halving iteration, 5-fold):")
print(f"  {best_mean_cv:.4f} ± {best_std_cv:.4f}")

# (d) Test ROC-AUC
print(f"\nTest ROC-AUC: {test_roc_auc:.4f}")
