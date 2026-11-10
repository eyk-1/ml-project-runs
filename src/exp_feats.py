"""Feature experiments on corrected pipeline: same folds/params as lgbm_corrected_v2."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from cv_v2 import add_te, prep, P as BASE_P
from lgbm_corrected_v2 import align_categories, add_fold_counts

HOSP_AGG = ['los_days', 'charlson', 'lace', 'age', 'dx_n', 'pu_acute_365', 'pu_all', 'followup_missing',
            'tr_icu', 'dx_chronic_n']


def rowfeats(X):
    X = X.copy()
    g = lambda c: X[c].astype(float)
    X['x_acute30'] = g('pu_ED_30d') + g('pu_IP_30d')
    X['x_acute180'] = g('pu_ED_180d') + g('pu_IP_180d')
    X['x_acute30_frac'] = X.x_acute30 / (g('pu_acute_365') + 1)
    X['x_acute90_frac'] = g('pu_acute_90') / (g('pu_acute_365') + 1)
    X['x_ip30_frac'] = g('pu_IP_30d') / (g('pu_IP_365d') + 1)
    X['x_ed90_frac'] = g('pu_ED_90d') / (g('pu_ED_365d') + 1)
    X['x_gap_min'] = np.fmin(g('pu_ED_lastgap'), g('pu_IP_lastgap'))
    X['x_samehosp_ip_frac'] = g('u3_same_ip_all') / (g('pu_IP_all') + 1)
    X['x_age_lace'] = g('age') * g('lace')
    X['x_los_lace'] = g('los_days') * g('lace')
    X['x_lace_acute90'] = g('lace') * g('pu_acute_90')
    X['x_charl_acute365'] = g('charlson') * g('pu_acute_365')
    X['x_age_acute365'] = g('age') * g('pu_acute_365')
    X['x_los_dxn'] = g('los_days') * g('dx_n')
    fu = g('followup_scheduled_days')
    X['x_fu_bin'] = pd.cut(fu, [-1, 3, 7, 14, 30, 1e9], labels=False).astype(float)
    X['x_fu_disp'] = (X.discharge_disposition.astype(object).fillna('NA') + '|' +
                      X.x_fu_bin.fillna(-1).astype(int).astype(str))
    return X


def hosp_aggs(train, *others):
    train = train.copy(); others = [o.copy() for o in others]
    cols = [c for c in HOSP_AGG if c in train.columns]
    m = train.groupby(train.hospital_id.astype(object))[cols].mean()
    for c in cols:
        mp = m[c]
        train['hm_' + c] = train.hospital_id.astype(object).map(mp).astype(float)
        for o in others:
            o['hm_' + c] = o.hospital_id.astype(object).map(mp).astype(float)
        for fr in [train] + others:   # difference from hospital mean
            fr['hd_' + c] = fr[c].astype(float) - fr['hm_' + c]
    return (train, *others)


def run(name, use_row, use_hosp, seeds=(0,), final=False, lr=0.04, over=None):
    X = pd.read_pickle('artifacts/X4_3.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    if use_row: X = rowfeats(X)
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    params = dict(BASE_P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                  lambda_l2=60, learning_rate=lr)
    params.update(over or {})
    oof = np.zeros(len(y)); rounds = []
    def prepare(A, others, yy, seed):
        A, *others = add_fold_counts(A, *others)
        if use_hosp: A, *others = hosp_aggs(A, *others)
        A, others = add_te(A, yy, list(others), seed=seed)
        A, *others = [prep(f) for f in [A] + list(others)]
        return align_categories(A, *others)
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        A, B = prepare(Xtr.iloc[a], [Xtr.iloc[b]], y[a], fold)
        p = np.zeros(len(b)); rs = []
        for s in seeds:
            m = lgb.train(dict(params, seed=s), lgb.Dataset(A, y[a]), 4000 if lr < .04 else 3000,
                          valid_sets=[lgb.Dataset(B, y[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            p += m.predict(B) / len(seeds); rs.append(m.best_iteration)
        oof[b] = p; rounds.append(int(np.mean(rs)))
        print(name, 'fold', fold, round(roc_auc_score(y[b], p), 5), 'rounds', rounds[-1], flush=True)
    print(name, 'OOF AUC', round(roc_auc_score(y, oof), 6), 'ncols', A.shape[1], flush=True)
    np.save(f'artifacts/oof_{name}.npy', oof)
    if final:
        A, T = prepare(Xtr, [Xte], y, 0)
        pt = np.zeros(len(T))
        for s in range(5):
            m = lgb.train(dict(params, seed=s), lgb.Dataset(A, y), max(1, int(np.mean(rounds) * 1.1)))
            pt += m.predict(T) / 5
        np.save(f'artifacts/test_{name}.npy', pt)
        print(name, 'final done', flush=True)
    return oof


if __name__ == '__main__':
    name = sys.argv[1]
    cfg = {'A': (True, False), 'B': (True, True), 'H': (False, True)}[name]
    run('exp_' + name, *cfg)
