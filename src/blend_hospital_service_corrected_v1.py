"""Screen hospital x service-line TE against the corrected v9 blend."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ART, SUB = ROOT / "artifacts", ROOT / "submissions"


def rank(values):
    return rankdata(values, method="average") / len(values)


def main():
    y = pd.read_csv(ROOT / "data" / "train.csv").readmit_30d.to_numpy()
    ids = pd.read_csv(ROOT / "data" / "test.csv").encounter_id
    fit, hold = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)
    lgb_oof = np.load(ART / "oof_lgbm_corrected_v2.npy")
    ebm_oof = np.load(ART / "oof_ebm_top150_corrected_v1.npy")
    base_oof = 0.65 * rank(lgb_oof) + 0.35 * rank(ebm_oof)
    candidate_oof = rank(np.load(ART / "oof_lgbm_hospital_service_te_corrected_v1.npy"))
    lgb_test = np.load(ART / "test_lgbm_corrected_v2.npy")
    ebm_test = np.load(ART / "test_ebm_top150_corrected_v1.npy")
    candidate_test = rank(np.load(ART / "test_lgbm_hospital_service_te_corrected_v1.npy"))
    base_test = 0.65 * rank(lgb_test) + 0.35 * rank(ebm_test)
    baseline_hold = roc_auc_score(y[hold], base_oof[hold])
    weight, fit_auc = 0.0, -1.0
    for w in np.arange(0, 0.501, 0.025):
        score = roc_auc_score(y[fit], (1 - w) * base_oof[fit] + w * candidate_oof[fit])
        if score > fit_auc:
            weight, fit_auc = float(w), score
    blend_oof = (1 - weight) * base_oof + weight * candidate_oof
    hold_auc = roc_auc_score(y[hold], blend_oof[hold])
    print(f"candidate solo OOF={roc_auc_score(y, candidate_oof):.6f}")
    print(f"v9 holdout={baseline_hold:.6f}; selected weight={weight:.3f}; "
          f"fit={fit_auc:.6f}; blend holdout={hold_auc:.6f}; "
          f"full OOF={roc_auc_score(y, blend_oof):.6f}", flush=True)
    if hold_auc > baseline_hold + 0.0002:
        blend_test = (1 - weight) * base_test + weight * candidate_test
        order = rankdata(blend_test, method="ordinal").astype(int) - 1
        probabilities = np.sort(lgb_test)[order]
        filename = "v14_hospital_service_blend.csv"
        pd.DataFrame({"encounter_id": ids, "readmit_30d": probabilities}).to_csv(SUB / filename, index=False)
        np.save(ART / "oof_v9_hospital_service_blend_corrected_v1.npy", blend_oof)
        print(f"wrote submissions/{filename}", flush=True)


if __name__ == "__main__":
    main()
