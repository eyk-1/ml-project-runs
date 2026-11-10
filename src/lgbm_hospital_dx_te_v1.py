"""Add nested-OOF hospital-by-primary-diagnosis target encoding to LightGBM.

Run from the repository root after src/features_v4*.py:
    python src/lgbm_hospital_dx_te_v1.py
The interaction encoding is smoothed with strength 25 and computed inside each
outer fold (and out-of-fold for its training rows). Saves OOF/test .npy files.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv_v2 as c
from lgbm_corrected_v2 import add_fold_counts, align_categories


def pair_keys(X):
    a = X.hospital_id.astype(object).where(X.hospital_id.notna(), '<NA>').astype(str)
    b = X.dx_primary3.astype(object).where(X.dx_primary3.notna(), '<NA>').astype(str)
    return a + '|' + b


def fit_map(keys, y, smoothing):
    global_mean = float(np.mean(y))
    tab = pd.DataFrame({'key': keys.to_numpy(), 'y': y}).groupby('key').y.agg(['sum', 'count'])
    values = (tab['sum'] + smoothing * global_mean) / (tab['count'] + smoothing)
    return values, global_mean


def apply_map(keys, mapping):
    values, global_mean = mapping
    return keys.map(values).fillna(global_mean).to_numpy(dtype=float)


def add_pair_te(train, y, others, seed, smoothing=25):
    train = train.copy()
    others = [frame.copy() for frame in others]
    keys = pair_keys(train)
    oof = np.zeros(len(train))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y):
        mp = fit_map(keys.iloc[a], y[a], smoothing)
        oof[b] = apply_map(keys.iloc[b], mp)
    train['te_hospital_dx_primary3'] = oof
    full_map = fit_map(keys, y, smoothing)
    for frame in others:
        frame['te_hospital_dx_primary3'] = apply_map(pair_keys(frame), full_map)
    return train, others


def main():
    X = pd.read_pickle('artifacts/X4_3.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 25
    params = dict(c.P, num_leaves=4, min_child_samples=400,
                  feature_fraction=0.2, lambda_l2=60, learning_rate=0.04)
    oof, rounds = np.zeros(len(y)), []
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = c.add_te(A, y[a], [B], seed=fold)
        A, (B,) = add_pair_te(A, y[a], [B], seed=fold)
        A, B = c.prep(A), c.prep(B)
        A, B = align_categories(A, B)
        model = lgb.train(params, lgb.Dataset(A, y[a]), 3000,
                          valid_sets=[lgb.Dataset(B, y[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = model.predict(B)
        rounds.append(model.best_iteration)
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5),
              'rounds', model.best_iteration, flush=True)

    A, B = add_fold_counts(Xtr, Xte)
    A, (B,) = c.add_te(A, y, [B], seed=0)
    A, (B,) = add_pair_te(A, y, [B], seed=0)
    A, B = c.prep(A), c.prep(B)
    A, B = align_categories(A, B)
    model = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
    test_pred = model.predict(B)
    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_lgbm_hospital_dx_te_v1.npy', oof)
    np.save('artifacts/test_lgbm_hospital_dx_te_v1.npy', test_pred)
    print('OOF AUC', round(auc, 6), 'mean rounds', int(np.mean(rounds)),
          'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
