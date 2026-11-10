"""Holdout-screen the hospital-by-admission-month model against the v9 blend.

Selects the extra-model rank weight on 80% OOF rows, scores the reserved 20%,
and saves a test blend rescaled to the corrected LightGBM probability scale.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def ranks(p):
    return rankdata(p, method="average") / len(p)


def main():
    y = pd.read_csv(ROOT / "data" / "train.csv").readmit_30d.to_numpy()
    fit, hold = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)
    lgb_oof = np.load(ART / "oof_lgbm_corrected_v2.npy")
    ebm_oof = np.load(ART / "oof_ebm_top150_corrected_v1.npy")
    month_oof = np.load(ART / "oof_lgbm_hospital_month_te_corrected_v1.npy")
    base = 0.65 * ranks(lgb_oof) + 0.35 * ranks(ebm_oof)
    candidate = ranks(month_oof)
    baseline_hold = roc_auc_score(y[hold], base[hold])
    best_w, best_fit = 0.0, -1.0
    for w in np.arange(0, 0.501, 0.025):
        score = roc_auc_score(y[fit], ((1 - w) * base + w * candidate)[fit])
        if score > best_fit:
            best_w, best_fit = float(w), score
    blend = (1 - best_w) * base + best_w * candidate
    print("candidate solo", roc_auc_score(y, candidate), "weight", best_w,
          "full OOF", roc_auc_score(y, blend), "holdout", roc_auc_score(y[hold], blend[hold]),
          "baseline holdout", baseline_hold, flush=True)
    lgb_test = np.load(ART / "test_lgbm_corrected_v2.npy")
    ebm_test = np.load(ART / "test_ebm_top150_corrected_v1.npy")
    month_test = np.load(ART / "test_lgbm_hospital_month_te_corrected_v1.npy")
    base_test = 0.65 * ranks(lgb_test) + 0.35 * ranks(ebm_test)
    mix_test = (1 - best_w) * base_test + best_w * ranks(month_test)
    order = rankdata(mix_test, method="ordinal").astype(int) - 1
    np.save(ART / "oof_blend_hospital_month_corrected_v1.npy", blend)
    np.save(ART / "test_blend_hospital_month_corrected_v1.npy", np.sort(lgb_test)[order])


if __name__ == "__main__":
    main()
