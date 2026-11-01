"""v4: logistic regression on v2 features, then blend with the v2 LightGBM using out-of-fold predictions."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

warnings.filterwarnings('ignore')
X = pd.read_pickle('artifacts/X2.pkl').drop(columns=['attending_id'])
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
y = tr.readmit_30d.values
cat = [c for c in X if str(X[c].dtype) == 'category']
num = [c for c in X if c not in cat]
X[cat] = X[cat].astype(object)
Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]


def make(C, scaler):
    sc = QuantileTransformer(n_quantiles=200, output_distribution='normal') if scaler == 'quantile' else StandardScaler()
    pre = ColumnTransformer([
        ('n', Pipeline([('i', SimpleImputer(strategy='median', add_indicator=True)), ('s', sc)]), num),
        ('c', Pipeline([('i', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('o', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=300, sparse_output=False))]), cat)])
    return Pipeline([('p', pre), ('m', LogisticRegression(C=C, max_iter=500))])


cv = StratifiedKFold(5, shuffle=True, random_state=0)  # same folds as the boosting models
best = (0, None)
for scaler in ['standard', 'quantile']:
    for C in [0.005, 0.02]:
        oof = cross_val_predict(make(C, scaler), Xtr, y, cv=cv, method='predict_proba')[:, 1]
        a = roc_auc_score(y, oof)
        print(scaler, C, round(a, 4), flush=True)
        if a > best[0]:
            best = (a, (C, scaler), oof)
auc, (C, scaler), oof_lr = best
print('best LR', C, scaler, round(auc, 4), flush=True)
np.save('artifacts/oof_logreg_v4.npy', oof_lr)

# ---- blend test on OOF ----
r = lambda v: rankdata(v) / len(v)
lgb = np.load('artifacts/oof_reg_b.npy'); cb = np.load('artifacts/oof_catboost_v3.npy')
print('corr lgb-lr', round(np.corrcoef(r(lgb), r(oof_lr))[0, 1], 3))
res = {}
for w in np.arange(0, 1.01, 0.1):
    res[round(w, 1)] = roc_auc_score(y, w * r(lgb) + (1 - w) * r(oof_lr))
for w, v in res.items():
    print('w_lgb', w, round(v, 4))
wb = max(res, key=res.get); print('best w_lgb', wb, round(res[wb], 4), flush=True)
for w in [0.4, 0.5]:
    print('3-way lgb/lr/cb', w, round(roc_auc_score(y, w * r(lgb) + (0.9 - w) * r(oof_lr) + 0.1 * r(cb)), 4))

# ---- final fit on full train, test predictions ----
m = make(C, scaler).fit(Xtr, y)
p_lr = m.predict_proba(Xte)[:, 1]
np.save('artifacts/test_logreg_v4.npy', p_lr)
sub_lr = pd.DataFrame({'encounter_id': te.encounter_id, 'readmit_30d': p_lr})
sub_lr.to_csv('submissions/v4_logreg.csv', index=False)
v2 = pd.read_csv('submissions/v2_lgbm_reg_te.csv').set_index('encounter_id').loc[te.encounter_id, 'readmit_30d'].values
blend = wb * r(v2) + (1 - wb) * r(p_lr)
pd.DataFrame({'encounter_id': te.encounter_id, 'readmit_30d': blend}).to_csv('submissions/v4_blend_lgb_lr.csv', index=False)
print('done; blend weight on LGB =', wb, flush=True)
