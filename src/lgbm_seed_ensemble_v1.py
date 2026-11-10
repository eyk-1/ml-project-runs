"""Evaluate a three-seed LightGBM ensemble on the standard five folds.

Run from the repository root after features_v2.py has built artifacts/X2.pkl:
    python src/lgbm_seed_ensemble_v1.py
Target encoding is recomputed inside every outer fold. Writes OOF and full-train
test probabilities under artifacts/ for later blend evaluation.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv_v2 as c


def main():
    X = pd.read_pickle('artifacts/X2.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    params = dict(c.P, num_leaves=4, min_child_samples=400,
                  feature_fraction=0.2, lambda_l2=60, learning_rate=0.04)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    oof = np.zeros(len(y))
    for fold, (a, b) in enumerate(folds):
        fold_pred = np.zeros(len(b))
        for seed in range(3):
            A, (B,) = c.add_te(Xtr.iloc[a], y[a], [Xtr.iloc[b]], seed=seed)
            A, B = c.prep(A), c.prep(B)
            model = lgb.train(dict(params, seed=seed, feature_fraction_seed=seed,
                                   bagging_seed=seed, data_random_seed=seed),
                              lgb.Dataset(A, y[a]), 3000,
                              valid_sets=[lgb.Dataset(B, y[b])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            fold_pred += model.predict(B) / 3
        oof[b] = fold_pred
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5), flush=True)

    test_pred = np.zeros(len(te))
    for seed in range(3):
        A, (B,) = c.add_te(Xtr, y, [Xte], seed=seed)
        A, B = c.prep(A), c.prep(B)
        model = lgb.train(dict(params, seed=seed, feature_fraction_seed=seed,
                               bagging_seed=seed, data_random_seed=seed),
                          lgb.Dataset(A, y), 700)
        test_pred += model.predict(B) / 3
        print('full-data seed', seed, 'done', flush=True)

    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_lgbm_seed_ensemble_v1.npy', oof)
    np.save('artifacts/test_lgbm_seed_ensemble_v1.npy', test_pred)
    print('OOF AUC', round(auc, 6), 'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
