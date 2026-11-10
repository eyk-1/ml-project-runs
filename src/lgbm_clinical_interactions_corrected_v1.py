"""Test four remaining clinical pairwise features on corrected v4 data.

Adds ICU-by-length-of-stay, discharge-planning-by-prior-use, medication-count
by-diagnosis-burden, and lab-abnormality-by-length-of-stay interactions. Uses
the fixed folds and fold-local preprocessing; saves OOF and test probabilities.
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


def add_interactions(frame):
    x = frame.copy()
    los = np.log1p(x.los_days.fillna(0).clip(lower=0))
    prior = np.log1p(x.pu_acute_365.fillna(0).clip(lower=0))
    meds = np.log1p(x.med_n.fillna(0).clip(lower=0))
    dx = np.log1p(x.dx_n.fillna(0).clip(lower=0))
    abnormal = (x.lab_abn_H.fillna(0) + x.lab_abn_L.fillna(0)).clip(lower=0)
    x["icu_x_log_los"] = x.tr_icu.fillna(0) * los
    x["dp_x_log_prior_acute"] = x.dp_present.fillna(0) * prior
    x["log_med_x_log_dx"] = meds * dx
    x["lab_abnormal_x_log_los"] = abnormal * los
    return x


def main():
    X = add_interactions(pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl"))
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 50
    params = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                  lambda_l2=60, learning_rate=0.04)
    oof, rounds = np.zeros(len(y)), []
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = c.add_te(A, y[a], [B], seed=fold)
        A, B = c.prep(A), c.prep(B)
        A, B = align_categories(A, B)
        model = lgb.train(params, lgb.Dataset(A, y[a]), 3000,
                          valid_sets=[lgb.Dataset(B, y[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = model.predict(B)
        rounds.append(model.best_iteration)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={rounds[-1]}", flush=True)
    auc = roc_auc_score(y, oof)
    print("OOF AUC", round(auc, 6), "mean rounds", int(np.mean(rounds)), flush=True)
    np.save(ROOT / "artifacts" / "oof_lgbm_clinical_interactions_corrected_v1.npy", oof)

    A, T = add_fold_counts(Xtr, Xte)
    A, (T,) = c.add_te(A, y, [T], seed=0)
    A, T = c.prep(A), c.prep(T)
    A, T = align_categories(A, T)
    final = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
    pred = final.predict(T)
    np.save(ROOT / "artifacts" / "test_lgbm_clinical_interactions_corrected_v1.npy", pred)
    print("test mean", float(pred.mean()), flush=True)


if __name__ == "__main__":
    main()
