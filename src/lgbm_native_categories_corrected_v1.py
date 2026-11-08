"""Test native attending/procedure categories alongside fold-local target encodings.

Uses the corrected feature table and fixed five folds; saves OOF and test
predictions under artifacts/ for later OOF diversity checks.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import cv_v2 as c
from lgbm_corrected_v2 import add_fold_counts, align_categories


ROOT = Path(__file__).resolve().parents[1]


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 25
    params = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                  lambda_l2=60, learning_rate=0.04)
    oof, rounds = np.zeros(len(y)), []
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    for fold, (a, b) in enumerate(folds):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = c.add_te(A, y[a], [B], seed=fold)
        A, B = align_categories(A, B)
        native_levels = {col: A[col].cat.categories for col in ("attending_id", "proc_first")}
        # Keep these raw categoricals alongside their target-encoded versions.
        A, B = c.prep(A), c.prep(B)
        for col in ("attending_id", "proc_first"):
            A[col] = pd.Categorical(Xtr.iloc[a][col].astype(object), categories=native_levels[col])
            B[col] = pd.Categorical(Xtr.iloc[b][col].astype(object), categories=native_levels[col])
        cats = [col for col in A if isinstance(A[col].dtype, pd.CategoricalDtype)]
        model = lgb.train(params, lgb.Dataset(A, y[a], categorical_feature=cats), 3000,
                          valid_sets=[lgb.Dataset(B, y[b], categorical_feature=cats)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = model.predict(B)
        rounds.append(model.best_iteration)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={rounds[-1]}", flush=True)
    print("OOF AUC", round(roc_auc_score(y, oof), 6), flush=True)
    np.save(ROOT / "artifacts" / "oof_lgbm_native_categories_corrected_v1.npy", oof)

    A, T = add_fold_counts(Xtr, Xte)
    A, (T,) = c.add_te(A, y, [T], seed=0)
    A, T = align_categories(A, T)
    native_levels = {col: A[col].cat.categories for col in ("attending_id", "proc_first")}
    A, T = c.prep(A), c.prep(T)
    for col in ("attending_id", "proc_first"):
        A[col] = pd.Categorical(Xtr[col].astype(object), categories=native_levels[col])
        T[col] = pd.Categorical(Xte[col].astype(object), categories=native_levels[col])
    cats = [col for col in A if isinstance(A[col].dtype, pd.CategoricalDtype)]
    final = lgb.train(params, lgb.Dataset(A, y, categorical_feature=cats), max(1, int(np.mean(rounds) * 1.1)))
    pred = final.predict(T)
    np.save(ROOT / "artifacts" / "test_lgbm_native_categories_corrected_v1.npy", pred)
    print("test mean", float(pred.mean()), flush=True)


if __name__ == "__main__":
    main()
