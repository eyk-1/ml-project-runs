"""Lower-budget NGBoost fold ensemble on corrected features.

Uses 40 NGBoost iterations and one model per outer fold for both OOF and test
predictions. All preprocessing and target encodings are fold-local.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from ngboost.scores import LogScore
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

import cv_v2 as c
from lgbm_corrected_v2 import add_fold_counts, align_categories


ROOT = Path(__file__).resolve().parents[1]


def preprocess(train, *others):
    train = train.replace([np.inf, -np.inf], np.nan).copy()
    others = [x.replace([np.inf, -np.inf], np.nan).copy() for x in others]
    cats = [col for col in train if isinstance(train[col].dtype, pd.CategoricalDtype)]
    nums = [col for col in train if col not in cats]
    for frame in [train, *others]:
        for col in cats:
            frame[col] = frame[col].astype(object)
    transformer = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=300)), cats),
    ], sparse_threshold=1.0)
    A = transformer.fit_transform(train)
    transformed = [transformer.transform(frame) for frame in others]
    if sparse.issparse(A):
        A = A.tocsr()
        transformed = [x.tocsr() if sparse.issparse(x) else x for x in transformed]
    return A, transformed


def make_model(seed):
    return NGBClassifier(
        Dist=Bernoulli, Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=2, min_samples_leaf=200, random_state=seed),
        n_estimators=40, learning_rate=0.03, minibatch_frac=0.5,
        col_sample=0.8, verbose=False, random_state=seed,
    )


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 50
    oof, test_pred = np.zeros(len(y)), np.zeros(len(te))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    for fold, (a, b) in enumerate(folds):
        A, B, T = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b], Xte)
        A, (B, T) = c.add_te(A, y[a], [B, T], seed=fold)
        A, B, T = c.prep(A), c.prep(B), c.prep(T)
        A, B, T = align_categories(A, B, T)
        A, (Bmat, Tmat) = preprocess(A, B, T)
        model = make_model(fold)
        model.fit(A, y[a])
        oof[b] = model.predict_proba(Bmat)[:, 1]
        test_pred += model.predict_proba(Tmat)[:, 1] / 5
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}", flush=True)
    auc = roc_auc_score(y, oof)
    print("OOF AUC", round(auc, 6), "test mean", float(test_pred.mean()), flush=True)
    np.save(ROOT / "artifacts" / "oof_ngboost_corrected_v2.npy", oof)
    np.save(ROOT / "artifacts" / "test_ngboost_corrected_v2.npy", test_pred)


if __name__ == "__main__":
    main()
