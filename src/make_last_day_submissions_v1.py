"""Create two last-day Kaggle CSV candidates from screened OOF blends.

Run from the repository root after blend_last_day_screen_v1.py. Refuses to
overwrite existing submission files and validates sample IDs/probability ranges.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SUB = ROOT / "submissions"
CANDIDATES = {
    "v12_class_weight_075_blend.csv": "test_lastday_class_weight_0p75_v1.npy",
}


def main():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    for filename, artifact in CANDIDATES.items():
        path = SUB / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        prob = np.load(ART / artifact)
        if len(prob) != len(test) or not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
            raise ValueError(f"Invalid predictions in {artifact}")
        submission = pd.DataFrame({"encounter_id": test.encounter_id, "readmit_30d": prob})
        if list(submission.columns) != list(sample.columns):
            raise ValueError("Submission columns do not match the sample file")
        if not submission.encounter_id.is_unique or not submission.encounter_id.equals(sample.encounter_id):
            raise ValueError("Submission IDs/order do not match sample_submission.csv")
        submission.to_csv(path, index=False)
        print(path.name, submission.shape, "mean", float(prob.mean()),
              "range", (float(prob.min()), float(prob.max())), flush=True)


if __name__ == "__main__":
    main()
