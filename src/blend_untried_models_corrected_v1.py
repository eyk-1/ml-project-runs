"""Screen the boosted-stump GAM and temporal-window model against corrected v9."""
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
    lgb_test = np.load(ART / "test_lgbm_corrected_v2.npy")
    ebm_test = np.load(ART / "test_ebm_top150_corrected_v1.npy")
    base_oof = 0.65 * rank(lgb_oof) + 0.35 * rank(ebm_oof)
    base_test = 0.65 * rank(lgb_test) + 0.35 * rank(ebm_test)
    print(f"v9: OOF={roc_auc_score(y, base_oof):.6f}; holdout={roc_auc_score(y[hold], base_oof[hold]):.6f}")
    candidates = {
        "stump_gam": ("oof_lgbm_stump_gam_corrected_v1.npy", "test_lgbm_stump_gam_corrected_v1.npy"),
        "temporal_windows": ("oof_lgbm_temporal_windows_corrected_v1.npy", "test_lgbm_temporal_windows_corrected_v1.npy"),
    }
    for name, (oof_name, test_name) in candidates.items():
        pred, pred_test = rank(np.load(ART / oof_name)), rank(np.load(ART / test_name))
        weight, best_fit = 0.0, -1.0
        for w in np.arange(0, 0.501, 0.025):
            score = roc_auc_score(y[fit], (1 - w) * base_oof[fit] + w * pred[fit])
            if score > best_fit:
                weight, best_fit = float(w), score
        blend = (1 - weight) * base_oof + weight * pred
        blend_test = (1 - weight) * base_test + weight * pred_test
        # Preserve a probability scale close to the established LightGBM submission.
        order = rankdata(blend_test, method="ordinal").astype(int) - 1
        probabilities = np.sort(lgb_test)[order]
        np.save(ART / f"oof_v9_{name}_blend_corrected_v1.npy", blend)
        np.save(ART / f"test_v9_{name}_blend_corrected_v1.npy", probabilities)
        print(f"{name}: solo={roc_auc_score(y, pred):.6f}; weight={weight:.3f}; "
              f"fit={best_fit:.6f}; holdout={roc_auc_score(y[hold], blend[hold]):.6f}; "
              f"full OOF={roc_auc_score(y, blend):.6f}", flush=True)
        if weight > 0 and roc_auc_score(y[hold], blend[hold]) > roc_auc_score(y[hold], base_oof[hold]):
            fname = f"v11_{name}_blend.csv"
            pd.DataFrame({"encounter_id": ids, "readmit_30d": probabilities}).to_csv(SUB / fname, index=False)
            print(f"wrote submissions/{fname}", flush=True)


if __name__ == "__main__":
    main()
