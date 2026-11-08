"""Five-fold LightGBM LambdaRank experiment for binary readmission ranking.

Run from the repository root after features_v2.py has created artifacts/X2.pkl:
    python src/pairwise_lgb_v1.py
Each fold is one ranking group; target encoding is fitted inside its training
fold. Writes OOF ranking scores and full-training test scores under artifacts/.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv_v2 as c


def grouped_order(n, seed, group_size=1024):
    order = np.random.default_rng(seed).permutation(n)
    sizes = [group_size] * (n // group_size)
    if n % group_size:
        sizes.append(n % group_size)
    return order, sizes


def main():
    X = pd.read_pickle('artifacts/X2.pkl')
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    params = dict(c.P, objective='lambdarank', metric='auc', num_leaves=4,
                  min_child_samples=400, feature_fraction=0.2, lambda_l2=60,
                  learning_rate=0.04, lambdarank_truncation_level=30,
                  label_gain=[0, 1])
    oof = np.zeros(len(y))
    rounds = []
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        A, (B,) = c.add_te(Xtr.iloc[a], y[a], [Xtr.iloc[b]])
        A, B = c.prep(A), c.prep(B)
        oa, ga = grouped_order(len(a), fold)
        ob, gb = grouped_order(len(b), 100 + fold)
        da = lgb.Dataset(A.iloc[oa], y[a][oa], group=ga)
        db = lgb.Dataset(B.iloc[ob], y[b][ob], group=gb)
        model = lgb.train(params, da, 3000, valid_sets=[db],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict(B.iloc[ob])
        inverse = np.empty(len(ob), dtype=int)
        inverse[ob] = np.arange(len(ob))
        oof[b] = pred[inverse]
        rounds.append(model.best_iteration)
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5),
              'rounds', model.best_iteration, flush=True)

    test_pred = np.zeros(len(te))
    A, (B,) = c.add_te(Xtr, y, [Xte])
    A, B = c.prep(A), c.prep(B)
    order, groups = grouped_order(len(y), 500)
    model = lgb.train(params, lgb.Dataset(A.iloc[order], y[order], group=groups),
                      max(1, int(np.mean(rounds) * 1.1)))
    test_pred[:] = model.predict(B)
    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_pairwise_lgb_v1.npy', oof)
    np.save('artifacts/test_pairwise_lgb_v1.npy', test_pred)
    print('OOF AUC', round(auc, 6), 'mean rounds', int(np.mean(rounds)), flush=True)


if __name__ == '__main__':
    main()
