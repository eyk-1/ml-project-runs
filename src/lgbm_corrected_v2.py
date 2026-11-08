"""Corrected LightGBM with provider/hospital counts fit inside each fold.

Run from the repository root after:
    python src/features_v4.py
    python src/features_v4_v2.py
    python src/features_v4_v3.py
Target encodings and categorical levels are fitted within each fold. Outputs
artifacts/oof_lgbm_corrected_v2.npy and artifacts/test_lgbm_corrected_v2.npy.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from cv_v2 import add_te, prep, P as BASE_P


def align_categories(train, *others):
    """Use levels observed in this training partition; unknowns become missing."""
    train = train.copy()
    others = [frame.copy() for frame in others]
    cat_cols = [c for c in train.columns
                if str(train[c].dtype) == 'category' or not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        levels = pd.Index(train[col].dropna().astype(object).unique())
        train[col] = pd.Categorical(train[col].astype(object), categories=levels)
        for frame in others:
            frame[col] = pd.Categorical(frame[col].astype(object), categories=levels)
    return (train, *others)


def add_fold_counts(train, *others):
    """Map counts learned from train-fold encounters to train/validation/test."""
    train = train.copy()
    others = [frame.copy() for frame in others]
    for col in ('attending_id', 'hospital_id'):
        counts = train[col].astype(object).value_counts(dropna=True)
        train[col + '_n_fold'] = train[col].astype(object).map(counts).fillna(0).astype(float)
        for frame in others:
            frame[col + '_n_fold'] = frame[col].astype(object).map(counts).fillna(0).astype(float)
    return (train, *others)


def main():
    X = pd.read_pickle('artifacts/X4_3.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    params = dict(BASE_P, num_leaves=4, min_child_samples=400,
                  feature_fraction=0.2, lambda_l2=60, learning_rate=0.04)
    oof = np.zeros(len(y))
    rounds = []
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = add_te(A, y[a], [B], seed=fold)
        A, B = prep(A), prep(B)
        A, B = align_categories(A, B)
        model = lgb.train(params, lgb.Dataset(A, y[a]), 3000,
                          valid_sets=[lgb.Dataset(B, y[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = model.predict(B)
        rounds.append(model.best_iteration)
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5),
              'rounds', model.best_iteration, flush=True)

    A, B = add_fold_counts(Xtr, Xte)
    A, (B,) = add_te(A, y, [B], seed=0)
    A, B = prep(A), prep(B)
    A, B = align_categories(A, B)
    final = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
    test_pred = final.predict(B)
    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_lgbm_corrected_v2.npy', oof)
    np.save('artifacts/test_lgbm_corrected_v2.npy', test_pred)
    print('rows/features', X.shape, 'OOF AUC', round(auc, 6),
          'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
