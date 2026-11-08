"""Test fold-local target encoding for the hospital x service-line interaction."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import cv_v2 as c
from lgbm_corrected_v2 import add_fold_counts, align_categories

ROOT = Path(__file__).resolve().parents[1]


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    for col in ("hospital_id", "service_line"):
        X[col] = X[col].astype(object).fillna("__MISSING__").astype(str)
    X["hospital_service"] = X.hospital_id + "__X__" + X.service_line
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 50
    c.TE_COLS = tuple(c.TE_COLS) + ("hospital_service",)
    params = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                  lambda_l2=60, learning_rate=0.04, n_jobs=2)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    oof, test_pred, rounds = np.zeros(len(y)), np.zeros(len(te)), []
    for fold, (a, b) in enumerate(folds):
        A, B, T = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b], Xte)
        A, (B, T) = c.add_te(A, y[a], [B, T], seed=fold)
        A, B, T = c.prep(A), c.prep(B), c.prep(T)
        A, B, T = (frame.drop(columns=["hospital_service"]) for frame in (A, B, T))
        A, B, T = align_categories(A, B, T)
        model = lgb.train(
            params, lgb.Dataset(A, label=y[a]), num_boost_round=3000,
            valid_sets=[lgb.Dataset(B, label=y[b])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        rounds.append(model.best_iteration)
        oof[b] = model.predict(B, num_iteration=model.best_iteration)
        test_pred += model.predict(T, num_iteration=model.best_iteration) / len(folds)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={rounds[-1]}", flush=True)
    auc = roc_auc_score(y, oof)
    np.save(ROOT / "artifacts" / "oof_lgbm_hospital_service_te_corrected_v1.npy", oof)
    np.save(ROOT / "artifacts" / "test_lgbm_hospital_service_te_corrected_v1.npy", test_pred)
    print(f"OOF AUC={auc:.6f}; mean rounds={np.mean(rounds):.1f}; test mean={test_pred.mean():.6f}", flush=True)


if __name__ == "__main__":
    main()
