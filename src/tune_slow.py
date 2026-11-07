"""Random-search tuning + full 5-fold CV for algorithms we have not tried yet.

Usage (from the repo root, after features_v2.py has built artifacts/X2.pkl):
    python src/tune_slow.py rf  --n_iter 20 --jobs 4
    python src/tune_slow.py hgb --n_iter 30 --submit
    python src/tune_slow.py --list

Models: rf, et, ada, hgb, mlp, logreg, lda, linsvc

What it does for the chosen model
  1. Random search: n_iter random configs, scored with --folds-fold CV on a --subsample-row stratified subsample.
  2. Re-scores the best config with 5-fold CV on the full training set (same folds as every other script).
  3. Saves  artifacts/oof_<model>_tuned.npy   (out-of-fold predictions, usable for blend tests)
            experiments/search_<model>.csv    (every config tried and its score - useful for the report)
     and appends a CV-only row to experiments/results.csv.
  4. With --submit, also fits on all training data and writes submissions/<model>_tuned.csv.
All imputers / scalers / encoders are fit on the training part of each fold only.
"""
import argparse
import ast
import csv
import datetime
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------- search spaces
def pick(r, opts):
    return opts[int(r.integers(len(opts)))]


def logu(r, lo, hi):
    return float(np.exp(r.uniform(np.log(lo), np.log(hi))))


SPACES = {
    'rf': lambda r: dict(n_estimators=pick(r, [200, 300, 400]), max_depth=pick(r, [8, 12, 16]),
                         min_samples_leaf=pick(r, [20, 40, 80]), max_features=pick(r, [0.1, 0.2, 0.3, 'sqrt']),
                         max_samples=pick(r, [0.5, 0.8])),
    'et': lambda r: dict(n_estimators=pick(r, [150, 250, 400]), max_depth=pick(r, [10, 14, 18]),
                         min_samples_leaf=pick(r, [20, 40, 80]), max_features=pick(r, [0.2, 0.3, 'sqrt'])),
    'ada': lambda r: dict(depth=pick(r, [1, 2, 3, 4]), n_estimators=pick(r, [100, 200, 400]),
                          learning_rate=pick(r, [0.03, 0.05, 0.1, 0.2, 0.5])),
    'hgb': lambda r: dict(learning_rate=pick(r, [0.03, 0.05, 0.1]), max_depth=pick(r, [3, 4, 6]),
                          max_leaf_nodes=pick(r, [8, 15, 31]), min_samples_leaf=pick(r, [50, 100, 200, 400]),
                          l2_regularization=pick(r, [1.0, 10.0, 30.0]), max_features=pick(r, [0.2, 0.3, 0.5])),
    'mlp': lambda r: dict(hidden=pick(r, [(64,), (128, 64), (256, 128), (128, 64, 32)]),
                          alpha=pick(r, [1e-3, 1e-2, 1e-1, 1.0]), lr=pick(r, [5e-4, 1e-3, 3e-3]),
                          batch=pick(r, [256, 512])),
    'logreg': lambda r: dict(C=logu(r, 1e-2, 1.0), l1_ratio=pick(r, [0.0, 0.2, 0.5, 0.8, 1.0])),
    'lda': lambda r: dict(shrinkage=pick(r, [None, 'auto', 0.1, 0.3, 0.5, 0.8])),
    'linsvc': lambda r: dict(C=logu(r, 1e-4, 1e-1)),
}
KIND = {'rf': 'tree', 'et': 'tree', 'ada': 'tree', 'hgb': 'hgb', 'mlp': 'scaled', 'logreg': 'scaled', 'lda': 'scaled',
        'linsvc': 'scaled'}
DEFAULT_SCALER = {'mlp': 'quantile'}  # everything else that is scaled uses 'standard'
NAMES = {'rf': 'Random Forest', 'et': 'ExtraTrees', 'ada': 'AdaBoost', 'hgb': 'HistGradientBoosting', 'mlp': 'MLP',
         'logreg': 'Logistic Regression (elastic-net, saga)', 'lda': 'Linear Discriminant Analysis',
         'linsvc': 'Linear SVM (sigmoid-calibrated)'}


def make(model, p, nj, seed, hgb_mask=None):
    if model == 'rf':
        return RandomForestClassifier(**p, n_jobs=nj, random_state=seed)
    if model == 'et':
        return ExtraTreesClassifier(**p, n_jobs=nj, random_state=seed)
    if model == 'ada':
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=p['depth']),
                                  n_estimators=p['n_estimators'], learning_rate=p['learning_rate'], random_state=seed)
    if model == 'hgb':
        return HistGradientBoostingClassifier(**p, max_iter=2000, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=50, categorical_features=hgb_mask, random_state=seed)
    if model == 'mlp':
        return MLPClassifier(hidden_layer_sizes=tuple(p['hidden']), alpha=p['alpha'], learning_rate_init=p['lr'],
                             batch_size=p['batch'], early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                             max_iter=80, random_state=seed)
    if model == 'logreg':  # l1_ratio 0 = ridge, 1 = lasso, in between = elastic-net
        return LogisticRegression(C=p['C'], l1_ratio=p['l1_ratio'], solver='saga', max_iter=300)
    if model == 'lda':
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=p['shrinkage'])
    if model == 'linsvc':
        return CalibratedClassifierCV(LinearSVC(C=p['C'], dual=False, max_iter=5000), cv=3, method='sigmoid')
    raise ValueError(model)


# ---------------------------------------------------------------- preprocessing
def build_pre(kind, scaler, num, cat):
    if kind == 'hgb':  # HistGradientBoosting handles NaN natively; categoricals -> ordinal codes
        return ColumnTransformer([
            ('n', 'passthrough', num),
            ('c', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan,
                                 max_categories=100), cat)])
    if kind == 'tree':
        ni = Pipeline([('i', SimpleImputer(strategy='median', add_indicator=True))])
    else:
        sc = QuantileTransformer(n_quantiles=200, output_distribution='normal') if scaler == 'quantile' else StandardScaler()
        ni = Pipeline([('i', SimpleImputer(strategy='median', add_indicator=True)), ('s', sc)])
    return ColumnTransformer([
        ('n', ni, num),
        ('c', Pipeline([('i', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('o', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=300, sparse_output=False))]), cat)])


def parse_params(txt):
    """JSON ('{"a": 1}') or quote-free 'a=1,b=sqrt,c=None' (easier on Windows shells). Lists/tuples use [] or ()."""
    txt = txt.strip()
    if txt.startswith('{'):
        try:
            return json.loads(txt)
        except json.JSONDecodeError:  # shell stripped the quotes: {a: 1, b: sqrt} -> fall through
            txt = txt.strip('{}').replace(':', '=')
    parts, depth, cur = [], 0, ''
    for ch in txt:
        depth += ch in '[(' ; depth -= ch in '])'
        if ch == ',' and depth == 0:
            parts.append(cur); cur = ''
        else:
            cur += ch
    parts.append(cur)
    out = {}
    for part in parts:
        k, v = part.split('=', 1); k, v = k.strip().strip('"\''), v.strip().strip('"\'')
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def score_cfg(model, p, fold_data, nj, hgb_mask):
    t = time.time(); aucs = []
    for A, yA, B, yB in fold_data:
        m = make(model, p, nj, 0, hgb_mask).fit(A, yA)
        aucs.append(roc_auc_score(yB, m.predict_proba(B)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), round(time.time() - t, 1), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?', choices=list(SPACES))
    ap.add_argument('--n_iter', type=int, default=15, help='random configs to try')
    ap.add_argument('--subsample', type=int, default=30000, help='rows used during the search')
    ap.add_argument('--folds', type=int, default=3, help='CV folds during the search')
    ap.add_argument('--jobs', type=int, default=1, help='configs evaluated in parallel (joblib)')
    ap.add_argument('--scaler', choices=['standard', 'quantile'], default=None)
    ap.add_argument('--seeds', type=int, default=3, help='seeds averaged in the final fit (with --submit)')
    ap.add_argument('--submit', action='store_true', help='also fit on all training data and write a submission file')
    ap.add_argument('--quick', action='store_true', help='tiny smoke test; writes nothing to experiments/ or submissions/')
    ap.add_argument('--params', default=None, help='JSON config: skip the search and just run the full 5-fold CV with it')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list or not a.model:
        print('models:', ', '.join(f'{k} ({NAMES[k]})' for k in SPACES)); return
    model = a.model; kind = KIND[model]; scaler = a.scaler or DEFAULT_SCALER.get(model, 'standard')
    if a.quick:
        a.n_iter, a.subsample, a.folds, a.seeds = 2, 4000, 2, 1

    X = pd.read_pickle('artifacts/X2.pkl').drop(columns=['attending_id'])
    tr = pd.read_csv('data/train.csv'); y = tr.readmit_30d.values
    cat = [c for c in X if str(X[c].dtype) == 'category']; num = [c for c in X if c not in cat]
    X[cat] = X[cat].astype(object)
    Xall = X.loc[tr.encounter_id].reset_index(drop=True)
    hgb_mask = [False] * len(num) + [True] * len(cat) if kind == 'hgb' else None
    if model in ('rf', 'et', 'hgb') and a.jobs > 1:
        print(f'note: {model} already uses all cores per fit, so --jobs {a.jobs} is ignored (running 1 config at a time)', flush=True)
        a.jobs = 1
    nj = 1 if a.jobs > 1 else -1
    t0 = time.time()

    idx, _ = train_test_split(np.arange(len(y)), train_size=min(a.subsample, len(y)), stratify=y, random_state=0)
    Xs, ys = Xall.iloc[idx].reset_index(drop=True), y[idx]
    if a.params:  # reuse a config found earlier (e.g. after a crash) - no search
        best_p = parse_params(a.params); cfgs = [best_p]
        print('using given config:', best_p, flush=True)
    else:
        # ---- 1. random search on a subsample
        fold_data = []
        for tri, vai in StratifiedKFold(a.folds, shuffle=True, random_state=0).split(Xs, ys):
            pre = build_pre(kind, scaler, num, cat).fit(Xs.iloc[tri])
            fold_data.append((pre.transform(Xs.iloc[tri]), ys[tri], pre.transform(Xs.iloc[vai]), ys[vai]))
        rng = np.random.default_rng(0); cfgs = []
        for _ in range(a.n_iter * 20):  # sample unique configs (small spaces like LDA have few)
            p = SPACES[model](rng)
            if p not in cfgs:
                cfgs.append(p)
            if len(cfgs) == a.n_iter:
                break
        print(f'{model}: searching {len(cfgs)} configs on {len(idx)} rows, {a.folds}-fold, jobs={a.jobs}', flush=True)
        res = []
        gen = Parallel(n_jobs=a.jobs, return_as='generator_unordered')(
            delayed(score_cfg)(model, p, fold_data, nj, hgb_mask) for p in cfgs)
        for k, r_ in enumerate(gen, 1):
            res.append(r_)
            print(f'  [{k}/{len(cfgs)}] auc {r_[0]:.4f}  {r_[2]}s  {r_[3]}  (total {round(time.time() - t0)}s)', flush=True)
        rows = sorted(((r_[:3], r_[3]) for r_ in res), key=lambda t: -t[0][0])
        print('top 5:', flush=True)
        for (m_, s_, sec), p in rows[:5]:
            print(f'  {m_:.4f} +-{s_:.4f}  {sec}s  {p}', flush=True)
        best_p = rows[0][1]; print('best config:', best_p, f'({round(time.time() - t0)}s elapsed)', flush=True)
        if not a.quick:
            os.makedirs('experiments', exist_ok=True)
            with open(f'experiments/search_{model}.csv', 'w', newline='') as f:
                w = csv.writer(f, lineterminator='\n'); w.writerow(['mean_auc', 'std_auc', 'seconds', 'params', 'search_rows', 'search_folds'])
                for (m_, s_, sec), p in rows:
                    w.writerow([round(m_, 4), round(s_, 4), sec, json.dumps(p), len(idx), a.folds])

    # ---- 2. full 5-fold CV of the best config (same folds as all other models)
    ycv = y if not a.quick else ys; Xcv = Xall if not a.quick else Xs
    oof = np.zeros(len(ycv))
    for i, (tri, vai) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xcv, ycv)):
        pre = build_pre(kind, scaler, num, cat).fit(Xcv.iloc[tri])
        m = make(model, best_p, -1, 0, hgb_mask).fit(pre.transform(Xcv.iloc[tri]), ycv[tri])
        oof[vai] = m.predict_proba(pre.transform(Xcv.iloc[vai]))[:, 1]
        print(f'  fold {i}: {roc_auc_score(ycv[vai], oof[vai]):.4f}  ({round(time.time() - t0)}s)', flush=True)
    cv_auc = roc_auc_score(ycv, oof)
    print(f'FULL 5-fold CV AUC {model}: {cv_auc:.4f}', flush=True)
    if a.quick:
        print('quick mode: nothing written'); return
    os.makedirs('artifacts', exist_ok=True)
    np.save(f'artifacts/oof_{model}_tuned.npy', oof)

    # ---- 3. optional submission
    sub_file = ''
    if a.submit:
        te = pd.read_csv('data/test.csv'); Xte = X.loc[te.encounter_id]
        pre = build_pre(kind, scaler, num, cat).fit(Xall)
        A, B = pre.transform(Xall), pre.transform(Xte); pred = np.zeros(len(te))
        for s in range(a.seeds):
            pred += make(model, best_p, -1, s, hgb_mask).fit(A, y).predict_proba(B)[:, 1] / a.seeds
        os.makedirs('submissions', exist_ok=True); sub_file = f'submissions/{model}_tuned.csv'
        pd.DataFrame({'encounter_id': te.encounter_id, 'readmit_30d': pred}).to_csv(sub_file, index=False)
        np.save(f'artifacts/test_{model}_tuned.npy', pred)
        print('wrote', sub_file, 'mean prob', round(pred.mean(), 4), flush=True)

    # ---- 4. log to experiments/results.csv (CV-only row; fill the LB column after you submit)
    path = 'experiments/results.csv'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            f.seek(-1, 2); nl = f.read(1) == b'\n'
        with open(path, 'a', newline='') as f:
            if not nl:
                f.write('\n')
            csv.writer(f, lineterminator='\n').writerow(['-', datetime.date.today().isoformat(), NAMES[model], json.dumps(best_p), 'features v2',
                                    round(cv_auc, 4), '', sub_file,
                                    ('CV only; config taken from an earlier search (--params)' if a.params else f'CV only; random search {len(cfgs)} configs on {len(idx)} rows ({a.folds}-fold)')])
        print('appended row to', path)


if __name__ == '__main__':
    main()