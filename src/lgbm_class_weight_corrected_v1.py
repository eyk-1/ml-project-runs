"""Screen positive-class weights on corrected v4 features and fixed folds.

Tests three scale_pos_weight values while keeping fold-local encodings,
provider counts, category alignment, and early stopping. Saves OOF/test arrays.
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
    c.M = 50
    base = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                lambda_l2=60, learning_rate=0.04)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    for weight in (0.75, 1.25, 1.5):
        params = dict(base, scale_pos_weight=weight)
        oof, rounds = np.zeros(len(y)), []
        for fold, (a, b) in enumerate(folds):
            A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
            A, (B,) = c.add_te(A, y[a], [B], seed=fold)
            A, B = c.prep(A), c.prep(B)
            A, B = align_categories(A, B)
            model = lgb.train(params, lgb.Dataset(A, y[a]), 3000,
                              valid_sets=[lgb.Dataset(B, y[b])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[b] = model.predict(B)
            rounds.append(model.best_iteration)
            print(f"weight {weight} fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}", flush=True)
        auc = roc_auc_score(y, oof)
        A, T = add_fold_counts(Xtr, Xte)
        A, (T,) = c.add_te(A, y, [T], seed=0)
        A, T = c.prep(A), c.prep(T)
        A, T = align_categories(A, T)
        final = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
        test_pred = final.predict(T)
        tag = str(weight).replace(".", "p")
        np.save(ROOT / "artifacts" / f"oof_lgbm_class_weight_{tag}_corrected_v1.npy", oof)
        np.save(ROOT / "artifacts" / f"test_lgbm_class_weight_{tag}_corrected_v1.npy", test_pred)
        print("weight", weight, "OOF AUC", round(auc, 6), "rounds", rounds,
              "test mean", float(test_pred.mean()), flush=True)


if __name__ == "__main__":
    main()
