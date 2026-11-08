"""Recheck the corrected 0.65/0.35 rank blend using fresh OOF reruns.

Run after lgbm_corrected_v3.py and ebm_top150_corrected_v2.py. Prints the
same full OOF and fixed 20% holdout AUCs and saves separate blend predictions.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main():
    y = pd.read_csv("data/train.csv").readmit_30d.to_numpy()
    lgb_oof = np.load("artifacts/oof_lgbm_corrected_v3.npy")
    ebm_oof = np.load("artifacts/oof_ebm_top150_corrected_v2.npy")
    blend = 0.65 * rankdata(lgb_oof) / len(y) + 0.35 * rankdata(ebm_oof) / len(y)
    fit, hold = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)
    print("full OOF", roc_auc_score(y, blend), "fit split", roc_auc_score(y[fit], blend[fit]),
          "holdout", roc_auc_score(y[hold], blend[hold]), flush=True)

    lgb_test = np.load("artifacts/test_lgbm_corrected_v3.npy")
    ebm_test = np.load("artifacts/test_ebm_top150_corrected_v2.npy")
    test_rank = 0.65 * rankdata(lgb_test) / len(lgb_test) + 0.35 * rankdata(ebm_test) / len(ebm_test)
    order = rankdata(test_rank, method="ordinal").astype(int) - 1
    probs = np.sort(lgb_test)[order]
    np.save("artifacts/oof_blend_corrected_lgb_ebm_v2.npy", blend)
    np.save("artifacts/test_blend_corrected_lgb_ebm_v2.npy", probs)
    print("test probability range", float(probs.min()), float(probs.max()), flush=True)


if __name__ == "__main__":
    main()
