"""Five-fold AdaBoost benchmark on corrected encounter features.

Run after the features_v4 chain. Preprocessing and target encoding are fit
within each training fold; saves OOF and full-train test probabilities.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from cv_v2 import add_te, prep
from lgbm_corrected_v2 import add_fold_counts, align_categories


ROOT = Path(__file__).resolve().parents[1]


def make_model(train):
    cats = [c for c in train if isinstance(train[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in train if c not in cats]
    prepper = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=300)), cats),
    ], sparse_threshold=1.0)
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=100),
        n_estimators=60, learning_rate=0.1, random_state=0,
    )
    return prepper, model


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    oof = np.zeros(len(y))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    for fold, (a, b) in enumerate(folds):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = add_te(A, y[a], [B], seed=fold)
        A, B = prep(A), prep(B)
        A, B = align_categories(A, B)
        transform, model = make_model(A)
        oof[b] = make_pipeline(transform, model).fit(A, y[a]).predict_proba(B)[:, 1]
        print(f"fold {fold}: {roc_auc_score(y[b], oof[b]):.6f}", flush=True)
    auc = roc_auc_score(y, oof)
    print("OOF AUC", round(auc, 6), flush=True)
    np.save(ROOT / "artifacts" / "oof_adaboost_corrected_v1.npy", oof)

    A, T = add_fold_counts(Xtr, Xte)
    A, (T,) = add_te(A, y, [T], seed=0)
    A, T = prep(A), prep(T)
    A, T = align_categories(A, T)
    transform, model = make_model(A)
    test_pred = make_pipeline(transform, model).fit(A, y).predict_proba(T)[:, 1]
    np.save(ROOT / "artifacts" / "test_adaboost_corrected_v1.npy", test_pred)
    print("test mean", float(test_pred.mean()), flush=True)


if __name__ == "__main__":
    main()
