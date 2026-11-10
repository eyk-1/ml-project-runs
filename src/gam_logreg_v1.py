"""Five-fold spline-logistic model for the readmission competition.

Run from the repository root after features_v2.py has created artifacts/X2.pkl:
    python src/gam_logreg_v1.py
Fits imputers, splines, and one-hot encoding inside each outer training fold.
Writes OOF and fold-ensemble test probabilities to artifacts/oof_gam_logreg_v1.npy
and artifacts/test_gam_logreg_v1.npy.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


def make_preprocessor(numeric, categorical):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('splines', SplineTransformer(n_knots=4, degree=3, knots='quantile',
                                      extrapolation='linear', sparse_output=True)),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 min_frequency=300, sparse_output=True)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipe, numeric),
        ('cat', categorical_pipe, categorical),
    ], sparse_threshold=1.0)


def main():
    X = pd.read_pickle('artifacts/X2.pkl').drop(columns=['attending_id'])
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    categorical = [c for c in X if str(X[c].dtype) == 'category']
    numeric = [c for c in X if c not in categorical]
    X[categorical] = X[categorical].astype(object)
    Xtr = X.loc[tr.encounter_id].reset_index(drop=True)
    Xte = X.loc[te.encounter_id].reset_index(drop=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    for fold, (a, b) in enumerate(cv.split(Xtr, y)):
        prep = make_preprocessor(numeric, categorical)
        A = prep.fit_transform(Xtr.iloc[a])
        B = prep.transform(Xtr.iloc[b])
        T = prep.transform(Xte)
        model = LogisticRegression(C=0.02, max_iter=800, solver='lbfgs')
        model.fit(A, y[a])
        oof[b] = model.predict_proba(B)[:, 1]
        test_pred += model.predict_proba(T)[:, 1] / 5
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5),
              'matrix', A.shape, flush=True)

    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_gam_logreg_v1.npy', oof)
    np.save('artifacts/test_gam_logreg_v1.npy', test_pred)
    print('OOF AUC', round(auc, 6), 'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
