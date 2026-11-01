"""v3: CatBoost with native categorical handling. 5-fold CV (saves OOF), then fit on all train and write submission."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X = pd.read_pickle('artifacts/X2.pkl')
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
y = tr.readmit_30d.values

cats = [c for c in X.columns if str(X[c].dtype) == 'category']
for c in cats:
    X[c] = X[c].astype(object).where(X[c].notna(), 'missing').astype(str)
Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]

P = dict(iterations=2000, learning_rate=0.06, depth=5, l2_leaf_reg=10, rsm=0.3, border_count=64,
         eval_metric='AUC', od_type='Iter', od_wait=100, random_seed=0, thread_count=-1, verbose=0,
         cat_features=cats)

oof = np.zeros(len(y)); its = []
for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
    m = CatBoostClassifier(**P)
    m.fit(Pool(Xtr.iloc[a], y[a], cat_features=cats), eval_set=Pool(Xtr.iloc[b], y[b], cat_features=cats))
    oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]; its.append(m.get_best_iteration())
    print('fold', i, round(roc_auc_score(y[b], oof[b]), 4), its[-1], flush=True)
print('OOF AUC', round(roc_auc_score(y, oof), 4), 'mean iters', int(np.mean(its)), flush=True)
np.save('artifacts/oof_catboost_v3.npy', oof)

n = int(np.mean(its) * 1.1)
pred = np.zeros(len(te))
for s in range(2):
    Q = dict(P, iterations=n, od_type=None, od_wait=None, random_seed=s)
    Q.pop('od_type'); Q.pop('od_wait')
    m = CatBoostClassifier(**Q)
    m.fit(Pool(Xtr, y, cat_features=cats))
    pred += m.predict_proba(Xte)[:, 1] / 2
    print('final seed', s, 'done', flush=True)
sub = pd.DataFrame({'encounter_id': te.encounter_id, 'readmit_30d': pred})
sub.to_csv('submissions/v3_catboost.csv', index=False)
np.save('artifacts/test_catboost_v3.npy', pred)
print('written', sub.shape, sub.readmit_30d.mean(), flush=True)
