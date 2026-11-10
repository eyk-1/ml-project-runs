"""Compare OOF rank blends of the new models and rescale the best to LGBM probs.

Run from the repository root after the spline, EBM, and seed-ensemble scripts:
    python src/blend_candidates_v1.py
Weights are selected on an 80% split of OOF rows and reported on the held-out
20%. Writes OOF and test .npy files under artifacts/; creates no submission.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def ranked(x):
    return rankdata(x) / len(x)


def fit_weights(R, y, train_idx, step=0.05):
    best_auc, best_w = -1.0, None
    if R.shape[1] == 2:
        candidates = ([w, 1 - w] for w in np.arange(0, 1.001, step))
    else:
        candidates = ([a, b, 1 - a - b]
                      for a in np.arange(0, 1.001, step)
                      for b in np.arange(0, 1.001 - a, step))
    for weights in candidates:
        weights = np.asarray(weights)
        auc = roc_auc_score(y[train_idx], R[train_idx] @ weights)
        if auc > best_auc:
            best_auc, best_w = auc, weights
    return best_auc, best_w


def report(name, preds, y, train_idx, hold_idx):
    R = np.column_stack([ranked(p) for p in preds])
    auc_train, w = fit_weights(R, y, train_idx)
    blend = R @ w
    print(name, 'rank corr', np.round(np.corrcoef(R.T), 4).tolist(),
          'weights', np.round(w, 2).tolist(),
          'train', round(auc_train, 6),
          'holdout', round(roc_auc_score(y[hold_idx], blend[hold_idx]), 6),
          'full OOF', round(roc_auc_score(y, blend), 6), flush=True)
    return blend, w


def main():
    tr, te = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    y = tr.readmit_30d.to_numpy()
    names = ['lgb', 'seed', 'gam', 'ebm', 'pairwise']
    oof = [np.load(f'artifacts/oof_{n}.npy') for n in [
        'blend_baseline_lgb_v2', 'lgbm_seed_ensemble_v1',
        'gam_logreg_v1', 'ebm_top150_v1', 'pairwise_lgb_v1']]
    test = [pd.read_csv('submissions/v2_lgbm_reg_te.csv').set_index('encounter_id')
            .loc[te.encounter_id, 'readmit_30d'].to_numpy()]
    test += [np.load(f'artifacts/test_{n}.npy') for n in [
        'lgbm_seed_ensemble_v1', 'gam_logreg_v1', 'ebm_top150_v1',
        'pairwise_lgb_v1']]
    a, b = train_test_split(np.arange(len(y)), test_size=0.2,
                            random_state=0, stratify=y)
    for i, name in enumerate(names):
        print(name, 'OOF AUC', round(roc_auc_score(y, oof[i]), 6), flush=True)
    candidates = [
        ('lgb_gam', [0, 2]), ('lgb_ebm', [0, 3]),
        ('lgb_seed', [0, 1]), ('lgb_pairwise', [0, 4]),
        ('lgb_gam_ebm', [0, 2, 3]),
    ]
    best = None
    for name, inds in candidates:
        blend, w = report(name, [oof[i] for i in inds], y, a, b)
        if best is None or roc_auc_score(y, blend) > best[0]:
            best = (roc_auc_score(y, blend), name, inds, blend, w)
    full_auc, name, inds, blend_oof, weights = best
    np.save('artifacts/oof_best_new_blend_v1.npy', blend_oof)

    test_ranks = np.column_stack([ranked(test[i]) for i in inds])
    blend_test_rank = test_ranks @ weights
    order = rankdata(blend_test_rank, method='ordinal').astype(int) - 1
    probabilities = np.sort(test[0])[order]
    np.save('artifacts/test_best_new_blend_v1.npy', probabilities)
    print('best', name, 'OOF', round(float(full_auc), 6),
          'probability mean', round(float(probabilities.mean()), 6),
          'range', (float(probabilities.min()), float(probabilities.max())), flush=True)


if __name__ == '__main__':
    main()
