"""Write the best-CV rank blend as a Kaggle-format submission candidate.

Run from the repository root after src/blend_candidates_v1.py:
    python src/make_v8_submission.py
Needs data/test.csv, data/sample_submission.csv, and
artifacts/test_best_new_blend_v1.npy. Writes submissions/v8_best_cv_blend.csv.
The saved probabilities are already rescaled to the v2 LightGBM probability scale.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    output = Path('submissions/v8_best_cv_blend.csv')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    prob = np.load('artifacts/test_best_new_blend_v1.npy')
    if len(prob) != len(test):
        raise ValueError(f'{len(prob)} predictions for {len(test)} test rows')
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError('Predictions must be finite probabilities in [0, 1]')
    sub = pd.DataFrame({'encounter_id': test.encounter_id,
                        'readmit_30d': prob})
    if list(sub.columns) != list(sample.columns):
        raise ValueError('Submission columns do not match sample_submission.csv')
    if not sub.encounter_id.is_unique or not sub.encounter_id.equals(sample.encounter_id):
        raise ValueError('Submission encounter IDs do not exactly match sample order')
    sub.to_csv(output, index=False)
    print('wrote', output, sub.shape, 'mean', round(float(prob.mean()), 6),
          'range', (float(prob.min()), float(prob.max())), flush=True)


if __name__ == '__main__':
    main()
