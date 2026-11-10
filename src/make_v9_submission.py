"""Write the v9 submission CSV (corrected LightGBM + EBM rank blend).

Run from the repository root after:
    python src/features_v4.py
    python src/features_v4_v2.py
    python src/features_v4_v3.py
    python src/lgbm_corrected_v2.py
    python src/ebm_top150_corrected_v1.py
    python src/blend_corrected_lgb_ebm_v1.py
Reads artifacts/test_blend_corrected_lgb_ebm_v1.npy (blend ranks rescaled to the
corrected LightGBM probability scale) and writes
submissions/v9_corrected_lgb_ebm_blend.csv in sample_submission.csv format.
"""
import numpy as np
import pandas as pd


def main():
    test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    pred = np.load('artifacts/test_blend_corrected_lgb_ebm_v1.npy')
    sub = pd.DataFrame({'encounter_id': test.encounter_id, 'readmit_30d': pred})
    assert len(sub) == len(sample) and set(sub.encounter_id) == set(sample.encounter_id)
    assert sub.encounter_id.is_unique and sub.readmit_30d.between(0, 1).all()
    sub.to_csv('submissions/v9_corrected_lgb_ebm_blend.csv', index=False)
    print('wrote', sub.shape, 'mean', round(float(sub.readmit_30d.mean()), 6))


if __name__ == '__main__':
    main()
