"""Holdout-screen blends of corrected LightGBM/EBM with new candidate models.

Selects each candidate's blend weight on 80% of fixed-fold OOF rows and reports
the reserved 20% AUC. Writes OOF/test probabilities for every screened blend.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def rank(x):
    return rankdata(x, method="average") / len(x)


def main():
    y = pd.read_csv(ROOT / "data" / "train.csv").readmit_30d.to_numpy()
    fit, hold = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)
    lgb_oof = np.load(ART / "oof_lgbm_corrected_v2.npy")
    ebm_oof = np.load(ART / "oof_ebm_top150_corrected_v1.npy")
    base = 0.65 * rank(lgb_oof) + 0.35 * rank(ebm_oof)
    lgb_test = np.load(ART / "test_lgbm_corrected_v2.npy")
    ebm_test = np.load(ART / "test_ebm_top150_corrected_v1.npy")
    base_test = 0.65 * rank(lgb_test) + 0.35 * rank(ebm_test)
    candidates = {
        "adaboost": ("oof_adaboost_corrected_v1.npy", "test_adaboost_corrected_v1.npy"),
        "native_categories": ("oof_lgbm_native_categories_corrected_v1.npy", "test_lgbm_native_categories_corrected_v1.npy"),
        "monotone": ("oof_lgbm_monotone_corrected_v1.npy", "test_lgbm_monotone_corrected_v1.npy"),
    }
    print("baseline full/holdout", roc_auc_score(y, base), roc_auc_score(y[hold], base[hold]))
    for name, (oof_file, test_file) in candidates.items():
        pred = rank(np.load(ART / oof_file))
        pred_test = rank(np.load(ART / test_file))
        best_w, best_fit = 0.0, -1.0
        for w in np.arange(0, 0.501, 0.025):
            blend = (1 - w) * base + w * pred
            score = roc_auc_score(y[fit], blend[fit])
            if score > best_fit:
                best_fit, best_w = score, float(w)
        blend = (1 - best_w) * base + best_w * pred
        blend_test = (1 - best_w) * base_test + best_w * pred_test
        order = rankdata(blend_test, method="ordinal").astype(int) - 1
        probs = np.sort(lgb_test)[order]
        np.save(ART / f"oof_blend_{name}_corrected_v1.npy", blend)
        np.save(ART / f"test_blend_{name}_corrected_v1.npy", probs)
        print(name, "solo", round(roc_auc_score(y, pred), 6), "weight", best_w,
              "fit", round(best_fit, 6), "holdout", round(roc_auc_score(y[hold], blend[hold]), 6),
              "full OOF", round(roc_auc_score(y, blend), 6), flush=True)


if __name__ == "__main__":
    main()
