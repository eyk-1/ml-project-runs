"""Screen last-day model variants against the corrected v9 blend.

For each candidate, selects a rank weight on 80% of OOF rows, scores the
reserved 20%, reports full OOF, and saves probability-scale test blends.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def rank(p):
    return rankdata(p, method="average") / len(p)


def main():
    y = pd.read_csv(ROOT / "data" / "train.csv").readmit_30d.to_numpy()
    fit, hold = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)
    lgb_oof = np.load(ART / "oof_lgbm_corrected_v2.npy")
    ebm_oof = np.load(ART / "oof_ebm_top150_corrected_v1.npy")
    lgb_test = np.load(ART / "test_lgbm_corrected_v2.npy")
    ebm_test = np.load(ART / "test_ebm_top150_corrected_v1.npy")
    base = 0.65 * rank(lgb_oof) + 0.35 * rank(ebm_oof)
    base_test = 0.65 * rank(lgb_test) + 0.35 * rank(ebm_test)
    candidates = {
        "hospital_month": ("oof_lgbm_hospital_month_te_corrected_v1.npy", "test_lgbm_hospital_month_te_corrected_v1.npy"),
        "clinical_interactions": ("oof_lgbm_clinical_interactions_corrected_v1.npy", "test_lgbm_clinical_interactions_corrected_v1.npy"),
        "monotone": ("oof_lgbm_monotone_corrected_v1.npy", "test_lgbm_monotone_corrected_v1.npy"),
        "class_weight_0p75": ("oof_lgbm_class_weight_0p75_corrected_v1.npy", "test_lgbm_class_weight_0p75_corrected_v1.npy"),
        "class_weight_1p25": ("oof_lgbm_class_weight_1p25_corrected_v1.npy", "test_lgbm_class_weight_1p25_corrected_v1.npy"),
        "class_weight_1p5": ("oof_lgbm_class_weight_1p5_corrected_v1.npy", "test_lgbm_class_weight_1p5_corrected_v1.npy"),
    }
    print("v9 baseline", "full OOF", roc_auc_score(y, base), "holdout", roc_auc_score(y[hold], base[hold]))
    for name, (oof_name, test_name) in candidates.items():
        oof = rank(np.load(ART / oof_name))
        pred_test = rank(np.load(ART / test_name))
        best_w, best_fit = 0.0, -1.0
        for weight in np.arange(0, 0.501, 0.025):
            blend = (1 - weight) * base + weight * oof
            score = roc_auc_score(y[fit], blend[fit])
            if score > best_fit:
                best_w, best_fit = float(weight), score
        blend = (1 - best_w) * base + best_w * oof
        blend_test = (1 - best_w) * base_test + best_w * pred_test
        order = rankdata(blend_test, method="ordinal").astype(int) - 1
        probs = np.sort(lgb_test)[order]
        np.save(ART / f"oof_lastday_{name}_v1.npy", blend)
        np.save(ART / f"test_lastday_{name}_v1.npy", probs)
        print(name, "solo", round(roc_auc_score(y, oof), 6), "weight", best_w,
              "fit", round(best_fit, 6), "holdout", round(roc_auc_score(y[hold], blend[hold]), 6),
              "full OOF", round(roc_auc_score(y, blend), 6), flush=True)


if __name__ == "__main__":
    main()
