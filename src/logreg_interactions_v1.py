"""Five-fold logistic regression with fold-selected pairwise numeric interactions.

Run from the repository root after features_v2.py has created artifacts/X2.pkl:
    python src/logreg_interactions_v1.py
The numeric selector, imputers, scalers, and categorical encoder fit inside each
outer fold. Writes OOF and fold-ensemble test probabilities under artifacts/.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def make_preprocessor(numeric, categorical):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
        ('select', SelectKBest(f_classif, k=min(40, len(numeric)))),
        ('interactions', PolynomialFeatures(degree=2, interaction_only=True,
                                            include_bias=False)),
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
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    categorical = [c for c in X if str(X[c].dtype) == 'category']
    numeric = [c for c in X if c not in categorical]
    X[categorical] = X[categorical].astype(object)
    Xtr, Xte = X.loc[tr.encounter_id].reset_index(drop=True), X.loc[te.encounter_id].reset_index(drop=True)
    oof, test_pred = np.zeros(len(y)), np.zeros(len(te))
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        prep = make_preprocessor(numeric, categorical)
        A, B, T = prep.fit_transform(Xtr.iloc[a], y[a]), prep.transform(Xtr.iloc[b]), prep.transform(Xte)
        model = LogisticRegression(C=0.02, max_iter=800, solver='lbfgs')
        model.fit(A, y[a])
        oof[b] = model.predict_proba(B)[:, 1]
        test_pred += model.predict_proba(T)[:, 1] / 5
        print('fold', fold, 'AUC', round(roc_auc_score(y[b], oof[b]), 5),
              'matrix', A.shape, flush=True)
    auc = roc_auc_score(y, oof)
    np.save('artifacts/oof_logreg_interactions_v1.npy', oof)
    np.save('artifacts/test_logreg_interactions_v1.npy', test_pred)
    print('OOF AUC', round(auc, 6), 'test mean', round(float(test_pred.mean()), 6), flush=True)


if __name__ == '__main__':
    main()
