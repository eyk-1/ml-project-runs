"""Test clinically plausible positive monotonic constraints on corrected features.

Constraints are applied only to numeric risk-burden features; categorical
features and target encodings remain unconstrained. Saves OOF/test predictions.
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
POSITIVE_RISK = {
    "age", "dx_n", "dx_chronic_n", "charlson", "pu_all", "pu_acute_90", "pu_acute_365",
    "pu_ED_30d", "pu_ED_90d", "pu_ED_180d", "pu_ED_365d", "pu_IP_30d", "pu_IP_90d",
    "pu_IP_180d", "pu_IP_365d", "prior_ip_recent",
}


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 25
    base = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                lambda_l2=60, learning_rate=0.04)
    oof, rounds = np.zeros(len(y)), []
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    for fold, (a, b) in enumerate(folds):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = c.add_te(A, y[a], [B], seed=fold)
        A, B = c.prep(A), c.prep(B)
        A, B = align_categories(A, B)
        constraints = [1 if col in POSITIVE_RISK else 0 for col in A.columns]
        model = lgb.train(dict(base, monotone_constraints=constraints), lgb.Dataset(A, y[a]), 3000,
                          valid_sets=[lgb.Dataset(B, y[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = model.predict(B)
        rounds.append(model.best_iteration)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={rounds[-1]}", flush=True)
    auc = roc_auc_score(y, oof)
    print("OOF AUC", round(auc, 6), "constrained features", sorted(POSITIVE_RISK), flush=True)
    np.save(ROOT / "artifacts" / "oof_lgbm_monotone_corrected_v1.npy", oof)

    A, T = add_fold_counts(Xtr, Xte)
    A, (T,) = c.add_te(A, y, [T], seed=0)
    A, T = c.prep(A), c.prep(T)
    A, T = align_categories(A, T)
    constraints = [1 if col in POSITIVE_RISK else 0 for col in A.columns]
    final_rounds = max(1, int(np.mean(rounds) * 1.1))
    final = lgb.train(dict(base, monotone_constraints=constraints), lgb.Dataset(A, y), final_rounds)
    pred = final.predict(T)
    np.save(ROOT / "artifacts" / "test_lgbm_monotone_corrected_v1.npy", pred)
    print("test mean", float(pred.mean()), flush=True)


if __name__ == "__main__":
    main()
