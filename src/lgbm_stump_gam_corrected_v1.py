"""Test a LightGBM boosted-stump GAM with the corrected fold-local pipeline."""
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
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 50
    params = dict(
        objective="binary", learning_rate=0.1, num_leaves=2, max_depth=1,
        min_child_samples=100, feature_fraction=0.5, lambda_l2=20,
        max_bin=63, bagging_freq=1, cat_smooth=50, max_cat_to_onehot=8,
        verbose=-1, n_jobs=2,
    )
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    oof, test_pred, rounds = np.zeros(len(y)), np.zeros(len(te)), []
    for fold, (a, b) in enumerate(folds):
        A, B, T = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b], Xte)
        A, (B, T) = c.add_te(A, y[a], [B, T], seed=fold)
        A, B, T = c.prep(A), c.prep(B), c.prep(T)
        A, B, T = align_categories(A, B, T)
        model = lgb.train(
            params, lgb.Dataset(A, label=y[a]), num_boost_round=5000,
            valid_sets=[lgb.Dataset(B, label=y[b])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        rounds.append(model.best_iteration)
        oof[b] = model.predict(B, num_iteration=model.best_iteration)
        test_pred += model.predict(T, num_iteration=model.best_iteration) / len(folds)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={rounds[-1]}", flush=True)
    auc = roc_auc_score(y, oof)
    np.save(ROOT / "artifacts" / "oof_lgbm_stump_gam_corrected_v1.npy", oof)
    np.save(ROOT / "artifacts" / "test_lgbm_stump_gam_corrected_v1.npy", test_pred)
    print(f"OOF AUC={auc:.6f}; mean rounds={np.mean(rounds):.1f}; test mean={test_pred.mean():.6f}", flush=True)


if __name__ == "__main__":
    main()
