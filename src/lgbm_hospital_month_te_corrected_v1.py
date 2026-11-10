"""Test nested-OOF hospital-by-admission-month target encoding.

Uses corrected v4 features and the standard five folds; fits the interaction
encoding inside each outer training fold and saves OOF/test probabilities.
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


def keys(frame):
    hosp = frame.hospital_id.astype(object).where(frame.hospital_id.notna(), "<NA>").astype(str)
    month = frame.admit_month.astype("Int64").astype(str)
    return hosp + "|" + month


def fit_map(key, y, smoothing=50):
    prior = float(np.mean(y))
    table = pd.DataFrame({"key": key.to_numpy(), "y": y}).groupby("key").y.agg(["sum", "count"])
    return (table["sum"] + smoothing * prior) / (table["count"] + smoothing), prior


def apply_map(key, mapping):
    values, prior = mapping
    return key.map(values).fillna(prior).to_numpy(dtype=float)


def add_interaction_te(train, y, others, seed, smoothing=50):
    train = train.copy()
    others = [frame.copy() for frame in others]
    k = keys(train)
    oof = np.zeros(len(train))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y):
        oof[b] = apply_map(k.iloc[b], fit_map(k.iloc[a], y[a], smoothing))
    train["te_hospital_admit_month"] = oof
    full = fit_map(k, y, smoothing)
    for frame in others:
        frame["te_hospital_admit_month"] = apply_map(keys(frame), full)
    return train, others


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 50
    params = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                  lambda_l2=60, learning_rate=0.04)
    oof, rounds = np.zeros(len(y)), []
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    for fold, (a, b) in enumerate(folds):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = c.add_te(A, y[a], [B], seed=fold)
        A, (B,) = add_interaction_te(A, y[a], [B], seed=fold)
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
    np.save(ROOT / "artifacts" / "oof_lgbm_hospital_month_te_corrected_v1.npy", oof)

    A, T = add_fold_counts(Xtr, Xte)
    A, (T,) = c.add_te(A, y, [T], seed=0)
    A, (T,) = add_interaction_te(A, y, [T], seed=0)
    A, T = c.prep(A), c.prep(T)
    A, T = align_categories(A, T)
    model = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
    test_pred = model.predict(T)
    np.save(ROOT / "artifacts" / "test_lgbm_hospital_month_te_corrected_v1.npy", test_pred)
    print("test mean", float(test_pred.mean()), flush=True)


if __name__ == "__main__":
    main()
