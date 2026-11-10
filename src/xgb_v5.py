"""v5: XGBoost on v2 features with in-fold target encoding (same recipe as the v2 LightGBM).
5-fold CV (saves OOF), then fit on all training data (3 seeds) and write a submission."""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv_v2 as c  # reuse add_te() and prep()

X = pd.read_pickle('artifacts/X2.pkl')
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
y = tr.readmit_30d.values
Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]

P = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', eta=0.04, max_depth=3, min_child_weight=20,
         subsample=0.8, colsample_bytree=0.3, reg_lambda=30, reg_alpha=1, max_cat_to_onehot=8, max_bin=128, nthread=-1)


def dm(A, label=None):
    return xgb.DMatrix(A, label=label, enable_categorical=True)


t0 = time.time(); oof = np.zeros(len(y)); its = []
for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
    A, (B,) = c.add_te(Xtr.iloc[a], y[a], [Xtr.iloc[b]])
    A, B = c.prep(A), c.prep(B)
    dA, dB = dm(A, y[a]), dm(B, y[b])
    m = xgb.train(P, dA, 3000, evals=[(dB, 'val')], early_stopping_rounds=100, verbose_eval=False)
    oof[b] = m.predict(dB, iteration_range=(0, m.best_iteration + 1)); its.append(m.best_iteration + 1)
    print('fold', i, round(roc_auc_score(y[b], oof[b]), 4), its[-1], round(time.time() - t0), 's', flush=True)
print('OOF AUC', round(roc_auc_score(y, oof), 4), 'mean rounds', int(np.mean(its)), flush=True)
np.save('artifacts/oof_xgb_v5.npy', oof)

n = int(np.mean(its) * 1.1); pred = np.zeros(len(te))
for s in range(3):
    A, (B,) = c.add_te(Xtr, y, [Xte], seed=s)
    A, B = c.prep(A), c.prep(B)
    m = xgb.train(dict(P, seed=s), dm(A, y), n)
    pred += m.predict(dm(B)) / 3
    print('final seed', s, 'done', round(time.time() - t0), 's', flush=True)
np.save('artifacts/test_xgb_v5.npy', pred)
pd.DataFrame({'encounter_id': te.encounter_id, 'readmit_30d': pred}).to_csv('submissions/v5_xgboost.csv', index=False)
print('written', pred.mean(), flush=True)
