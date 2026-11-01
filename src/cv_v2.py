import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TE_COLS = ['hospital_id', 'attending_id', 'dx_primary3', 'proc_first']
M = 50  # smoothing strength


def te_fit(col, y):
    """category -> smoothed mean of y, plus global mean"""
    g = y.mean()
    s = pd.DataFrame({'c': col.astype(object).values, 'y': y}).groupby('c').y.agg(['sum', 'count'])
    return ((s['sum'] + M * g) / (s['count'] + M)), g


def te_apply(col, mp):
    m, g = mp
    return col.astype(object).map(m).astype(float).fillna(g).values


def add_te(Xtr, ytr, Xva_list, seed=0):
    """OOF target encoding for the train part, full-train encoding for the others."""
    Xtr = Xtr.copy(); outs = [x.copy() for x in Xva_list]
    for c in TE_COLS:
        oof = np.zeros(len(Xtr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr):
            oof[b] = te_apply(Xtr[c].iloc[b], te_fit(Xtr[c].iloc[a], ytr[a]))
        mp = te_fit(Xtr[c], ytr)
        Xtr['te_' + c] = oof
        for o in outs:
            o['te_' + c] = te_apply(o[c], mp)
    return Xtr, outs


P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, cat_smooth=50, min_data_per_group=100, verbose=-1, n_jobs=-1)


def prep(X):
    # attending_id and proc_first are high-cardinality: drop raw, keep target-encoded version
    return X.drop(columns=['attending_id', 'proc_first'], errors='ignore')


def run(path, use_te, name, over=None):
    P.update(over or {})
    X = pd.read_pickle(path)
    tr = pd.read_csv('data/train.csv'); y = tr.readmit_30d.values
    Xt = X.loc[tr.encounter_id]
    oof = np.zeros(len(y)); its = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xt, y):
        A, B = Xt.iloc[a], Xt.iloc[b]
        if use_te:
            A, (B,) = add_te(A, y[a], [B])
        A, B = prep(A), prep(B)
        m = lgb.train(P, lgb.Dataset(A, y[a]), 3000, valid_sets=[lgb.Dataset(B, y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(B); its.append(m.best_iteration)
        print(name, 'fold', round(roc_auc_score(y[b], oof[b]), 4), m.best_iteration, flush=True)
    print(name, 'OOF AUC', round(roc_auc_score(y, oof), 4), 'mean iters', int(np.mean(its)), flush=True)
    np.save(f'artifacts/oof_{name}.npy', oof)


if __name__ == '__main__':
    import json
    run(sys.argv[1], sys.argv[2] == '1', sys.argv[3], json.loads(sys.argv[4]) if len(sys.argv) > 4 else None)
