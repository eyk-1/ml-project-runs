"""Build a corrected-feature LightGBM/EBM rank blend for offline evaluation.

Run after lgbm_corrected_v2.py and ebm_top150_corrected_v1.py:
    python src/blend_corrected_lgb_ebm_v1.py
Uses fixed 0.65/0.35 ranks (selected on an 80% OOF split), reports the held-out
20%, and rescales test ranks to corrected LightGBM probabilities. No CSV is made.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main():
    y = pd.read_csv('data/train.csv').readmit_30d.to_numpy()
    oof_lgb = np.load('artifacts/oof_lgbm_corrected_v2.npy')
    oof_ebm = np.load('artifacts/oof_ebm_top150_corrected_v1.npy')
    rank_lgb, rank_ebm = rankdata(oof_lgb) / len(y), rankdata(oof_ebm) / len(y)
    oof = 0.65 * rank_lgb + 0.35 * rank_ebm
    a, b = train_test_split(np.arange(len(y)), test_size=0.2,
                            random_state=0, stratify=y)
    print('OOF AUC', round(roc_auc_score(y, oof), 6),
          'weight-fit split AUC', round(roc_auc_score(y[a], oof[a]), 6),
          'holdout AUC', round(roc_auc_score(y[b], oof[b]), 6), flush=True)

    test_lgb = np.load('artifacts/test_lgbm_corrected_v2.npy')
    test_ebm = np.load('artifacts/test_ebm_top150_corrected_v1.npy')
    blend_test = 0.65 * rankdata(test_lgb) / len(test_lgb) + 0.35 * rankdata(test_ebm) / len(test_ebm)
    order = rankdata(blend_test, method='ordinal').astype(int) - 1
    probabilities = np.sort(test_lgb)[order]
    np.save('artifacts/oof_blend_corrected_lgb_ebm_v1.npy', oof)
    np.save('artifacts/test_blend_corrected_lgb_ebm_v1.npy', probabilities)
    print('test mean', round(float(probabilities.mean()), 6),
          'range', (float(probabilities.min()), float(probabilities.max())), flush=True)


if __name__ == '__main__':
    main()
