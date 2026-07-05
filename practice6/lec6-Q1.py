# ignore warning
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split, StratifiedKFold

TEST_RATIO = 0.3
RANDOM_SEED = 42

if __name__ == "__main__" :
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True))
    ])
    
    raw_dataset = load_breast_cancer()
    X, y = raw_dataset.data, raw_dataset.target
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    search_space = {
        'svc__C': loguniform(1e-2, 1e2),
        'svc__gamma': loguniform(1e-4, 1e0)
    }
    
    # fit HalvingRandomSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    search = HalvingRandomSearchCV(
        estimator=pipe,
        param_distributions=search_space,
        factor=3,
        resource='n_samples',
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=False
    )
    
    search.fit(X_tr, y_tr)
    
    bst_params = search.best_params_
    print(f"Best hyperparameters:\n\t{'C':<5s} = {bst_params['svc__C']:.6f}\n\t{'gamma':<5s} = {bst_params['svc__gamma']:.6f}")
    
    # refit with full training set
    bst_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, C=bst_params['svc__C'], gamma=bst_params['svc__gamma']))
    ])
    
    bst_pipe.fit(X_tr, y_tr)
    
    # Evaluate one the test set
    y_pred = bst_pipe.predict_proba(X_te)[:, 1]
    
    cv_res = pd.DataFrame(search.cv_results_)
    
    # print result
    print("Number of candidates evaluated at each halving iteration:")
    for iter, grp in cv_res.groupby('iter') :
        print(f"\tIteration {iter}: {len(grp)}")
    
    bst_idx = search.best_index_
    print(f"Best CV ROC-AUC: {cv_res['mean_test_score'][bst_idx]:.6f}±{cv_res['std_test_score'][bst_idx]:.6f}")
    
    print(f"Test ROC-AUC: {roc_auc_score(y_te, y_pred):.6f}")
