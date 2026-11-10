"""Five-fold Explainable Boosting Machine with fold-local LightGBM selection.

Run from the repository root after features_v2.py has created artifacts/X2.pkl:
    python src/ebm_top150_v1.py
Each fold selects the top 150 columns by LightGBM gain using only that fold's
training rows, then fits EBM. Writes OOF and fold-ensemble test probabilities.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier
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
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)
    oof, test_pred = np.zeros(len(y)), np.zeros(len(te))

    for fold, (a, b) in enumerate(folds):
        A, (B,) = c.add_te(Xtr.iloc[a], y[a], [Xtr.iloc[b]])
        A, (T,) = c.add_te(Xtr.iloc[a], y[a], [Xte], seed=0)
        A, B, T = c.prep(A), c.prep(B), c.prep(T)
        selector = lgb.train(dict(params, seed=0), lgb.Dataset(A, y[a]), 700)
        gain = selector.feature_importance(importance_type='gain')
        chosen = np.argsort(gain)[-min(150, len(gain)):]
        names = A.columns[chosen].tolist()
        ebm_A, ebm_B, ebm_T = A[names].copy(), B[names].copy(), T[names].copy()
        cats = [col for col in names if str(ebm_A[col].dtype) == 'category']
        for col in cats:
            levels = pd.Index(ebm_A[col].dropna().astype(object).unique())
            ebm_A[col] = ebm_A[col].astype(object).where(ebm_A[col].notna(), 'missing')
            ebm_B[col] = pd.Categorical(ebm_B[col].astype(object).where(ebm_B[col].notna(), 'missing'),
                                        categories=list(levels) + ['missing'])
            ebm_T[col] = pd.Categorical(ebm_T[col].astype(object).where(ebm_T[col].notna(), 'missing'),
                                        categories=list(levels) + ['missing'])
        ebm_A = ebm_A.replace([np.inf, -np.inf], np.nan)
        ebm_B = ebm_B.replace([np.inf, -np.inf], np.nan)
        ebm_T = ebm_T.replace([np.inf, -np.inf], np.nan)
        model = ExplainableBoostingClassifier(
            max_bins=64, interactions=10, outer_bags=2, inner_bags=0,
            learning_rate=0.015, max_rounds=2000, early_stopping_rounds=100,
            min_samples_leaf=50, n_jobs=-1, random_state=fold)
        model.fit(ebm_A, y[a])
        oof[b] = model.predict_proba(ebm_B)[:, 1]
        test_pred += model.predict_proba(ebm_T)[:, 1] / 5
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5),
              'selected', len(names), flush=True)

    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_ebm_top150_v1.npy', oof)
    np.save('artifacts/test_ebm_top150_v1.npy', test_pred)
    print('OOF AUC', round(auc, 6), 'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
