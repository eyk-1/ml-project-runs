"""Drop one corrected feature block at a time with five-fold LightGBM CV.

Run from the repository root after src/features_v4*.py:
    python src/ablate_feature_blocks_v1.py
Target encodings, provider/hospital counts, and category levels are fit inside
each fold. Writes OOF/test .npy predictions for each ablation under artifacts/.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv_v2 as c
from lgbm_corrected_v2 import align_categories


def add_fold_counts(train, *others):
    train = train.copy()
    others = [frame.copy() for frame in others]
    for col in ('attending_id', 'hospital_id'):
        if col not in train.columns:
            continue
        counts = train[col].astype(object).value_counts(dropna=True)
        train[col + '_n_fold'] = train[col].astype(object).map(counts).fillna(0).astype(float)
        for frame in others:
            frame[col + '_n_fold'] = frame[col].astype(object).map(counts).fillna(0).astype(float)
    return (train, *others)


def block_columns(X):
    return {
        'labs': [x for x in X if x.startswith(('lab_', 'l3_')) or x == 'bun_creat'],
        'vitals': [x for x in X if x.startswith(('vit_', 'v3_')) or x == 'shock_index'],
        'medications': [x for x in X if x.startswith(('med_', 'm3_')) or x in ('med_cont_frac', 'med_per_day')],
        'procedures': [x for x in X if x.startswith(('proc_', 'p3_'))],
        'transfers': [x for x in X if x.startswith(('tr_', 't3_')) or x == 'icu_hours'],
        'prior_utilization': [x for x in X if x.startswith(('pu_', 'u3_')) or x == 'prior_ip_recent'],
        'diagnoses': [x for x in X if x.startswith(('dx_', 'dxch_', 'dx3_', 'dx3b_', 'charlson', 'lace'))
                      or x in ('dx_distinct3', 'dx_per_day', 'age_x_charlson', 'los_x_charlson')],
        'discharge_planning': [x for x in X if x.startswith('dp_') or x in (
            'followup_scheduled_days', 'followup_missing', 'home_health_referral',
            'education_done', 'med_reconciliation_done')],
        'hospital_provider': [x for x in X if x in (
            'hospital_id', 'attending_id', 'attending_specialty', 'bed_size',
            'teaching_flag', 'urban_rural', 'ownership')],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['labs', 'vitals', 'medications', 'procedures',
                                           'transfers', 'prior_utilization', 'diagnoses',
                                           'discharge_planning', 'hospital_provider'])
    args = parser.parse_args()
    X = pd.read_pickle('artifacts/X4_3.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    Xtr_full, Xte_full = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    groups = block_columns(X)
    if args.only:
        groups = {args.only: groups[args.only]}
    params = dict(c.P, num_leaves=4, min_child_samples=400,
                  feature_fraction=0.2, lambda_l2=60, learning_rate=0.04)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr_full, y))
    original_te_cols = list(c.TE_COLS)

    for block, dropped in groups.items():
        Xtr = Xtr_full.drop(columns=dropped)
        Xte = Xte_full.drop(columns=dropped)
        c.TE_COLS = [col for col in original_te_cols if col in Xtr.columns]
        c.M = 25
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

        A, B = add_fold_counts(Xtr, Xte)
        A, (B,) = c.add_te(A, y, [B], seed=0)
        A, B = c.prep(A), c.prep(B)
        A, B = align_categories(A, B)
        model = lgb.train(params, lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
        test_pred = model.predict(B)
        auc = roc_auc_score(y, oof)
        np.save(f'artifacts/oof_ablation_drop_{block}_v1.npy', oof)
        np.save(f'artifacts/test_ablation_drop_{block}_v1.npy', test_pred)
        print('drop', block, 'columns', len(dropped), 'OOF AUC', round(auc, 6),
              'rounds', [int(r) for r in rounds], 'test mean',
              round(float(test_pred.mean()), 6), flush=True)

    c.TE_COLS = original_te_cols


if __name__ == '__main__':
    main()
