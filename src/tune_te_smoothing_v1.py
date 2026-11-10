"""Compare target-encoding smoothing strengths on corrected features.

Run from the repository root after features_v4*.py have built artifacts/X4_3.pkl:
    python src/tune_te_smoothing_v1.py
Uses the same five stratified folds for each value and fold-local counts,
categories, and target encodings. Saves OOF/test .npy files per smoothing value.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv_v2 as c
from lgbm_corrected_v2 import add_fold_counts, align_categories


def run_one(smoothing, Xtr, Xte, y, folds, params):
    c.M = smoothing
    oof = np.zeros(len(y))
    rounds = []
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
    A, B = add_fold_counts(Xtr, Xte)
    A, (B,) = c.add_te(A, y, [B], seed=0)
    A, B = c.prep(A), c.prep(B)
    A, B = align_categories(A, B)
    model = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
    test_pred = model.predict(B)
    return oof, test_pred, rounds


def main():
    X = pd.read_pickle('artifacts/X4_3.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    params = dict(c.P, num_leaves=4, min_child_samples=400,
                  feature_fraction=0.2, lambda_l2=60, learning_rate=0.04)
    for smoothing in (10, 25, 100, 200):
        oof, test_pred, rounds = run_one(smoothing, Xtr, Xte, y, folds, params)
        auc = roc_auc_score(y, oof)
        np.save(f'artifacts/oof_te_m{smoothing}_corrected_v1.npy', oof)
        np.save(f'artifacts/test_te_m{smoothing}_corrected_v1.npy', test_pred)
        print('smoothing', smoothing, 'OOF AUC', round(auc, 6),
              'rounds', [int(x) for x in rounds],
              'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
