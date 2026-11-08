"""Build a self-contained Kaggle notebook for the corrected v9 pipeline.

Run from the repository root. Embeds the corrected feature/model scripts as
notebook cells, so the Kaggle notebook needs only competition CSVs as input.
Writes submissions/kaggle_readmission_notebook_v2.ipynb.
"""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions" / "kaggle_readmission_notebook_v2.ipynb"


def main():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# 30-Day Hospital Readmission: corrected v9 model

This notebook builds the corrected encounter features from the competition CSVs, runs five-fold validation, fits corrected LightGBM and Explainable Boosting Machine (EBM) models, then rank-blends their predictions and writes a Kaggle submission.

The fixed validation split is `StratifiedKFold(5, shuffle=True, random_state=0)` on `train.csv` row order. The expected reference scores are LightGBM **0.752053**, EBM **0.750411**, and the 0.65/0.35 rank blend about **0.7526**. These are CV results, not guaranteed leaderboard scores.

Run it in a Kaggle notebook with the competition dataset attached. The full run is CPU-heavy. The EBM stage is the slowest (about 25 minutes in the measured local environment). If EBM or `interpret-core` is unavailable, the lighter fallback is the corrected LightGBM-only submission; it is valid, but it is not the v9 blend and its CV AUC is 0.752053.

Use only the competition data. The output is `/kaggle/working/v9_corrected_lgb_ebm_blend.csv`.
"""),
        nbf.v4.new_code_cell("""import glob, os, shutil, subprocess, sys, time
from pathlib import Path

# Kaggle competition data is attached as an input.
matches = glob.glob('/kaggle/input/**/train.csv', recursive=True)
assert matches, 'Attach the competition dataset to this Kaggle notebook.'
DATA = str(Path(matches[0]).parent)
assert Path(DATA, 'train.csv').exists(), 'Attach the competition dataset or set DATA_DIR.'
OUT = '/kaggle/working'
os.chdir(OUT)
os.environ['DATA_DIR'] = DATA
os.environ['ART_DIR'] = str(Path(OUT, 'artifacts'))
Path('artifacts').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
for name in ('train.csv', 'test.csv', 'sample_submission.csv'):
    shutil.copy2(Path(DATA, name), Path('data', name))
print('competition input:', DATA)
print('working output:', OUT)
print('CPU threads:', os.cpu_count())
"""),
        nbf.v4.new_markdown_cell("""## 1. Check the raw split

The source tables include both train and test encounters. Feature construction applies the same row-level cleaning and aggregation to both splits. Outcome-informed transformations (target encodings, category-level alignment, feature selection, and model fitting) are learned using training-fold rows only."""),
        nbf.v4.new_code_cell("""import pandas as pd
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')
assert train.encounter_id.is_unique and test.encounter_id.is_unique
assert set(train.encounter_id).isdisjoint(test.encounter_id)
assert len(test) == len(sample) and set(test.encounter_id) == set(sample.encounter_id)
print('train:', train.shape, '| positive rate:', round(train.readmit_30d.mean(), 5))
print('test:', test.shape, '| sample submission:', sample.shape)
"""),
    ]

    stages = [
        ("features_v4.py", "## 2a. Corrected base feature table (298 columns)"),
        ("features_v4_v2.py", "## 2b. Add training-vocabulary diagnosis and clinical score features (316 columns)"),
        ("features_v4_v3.py", "## 2c. Add discharge-state and timing features (461 stored columns)"),
    ]
    for script, heading in stages:
        target = f"/kaggle/working/{script}"
        source = (ROOT / "src" / script).read_text(encoding="utf-8")
        cells.append(nbf.v4.new_markdown_cell(heading + "\n\nThis stage cleans and aggregates the linked raw tables by `encounter_id`."))
        cells.append(nbf.v4.new_code_cell(f"%%writefile {target}\n" + source))
        cells.append(nbf.v4.new_code_cell(f"!python {target}"))

    cells += [
        nbf.v4.new_markdown_cell("""## 3. Install the EBM dependency if needed

LightGBM and scikit-learn are usually available in Kaggle. The EBM comes from `interpret-core`. If it is missing, this cell attempts to install it; internet must be enabled for that one install. If installation is unavailable, skip the EBM cells and use the corrected LightGBM-only fallback described above."""),
        nbf.v4.new_code_cell("""try:
    import interpret.glassbox
    print('interpret-core is available')
except ImportError:
    print('Installing interpret-core for the EBM stage. This requires Kaggle Internet access.')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'interpret-core==0.7.8'])
"""),
    ]

    # The feature scripts embed directly; these small helper/model modules are
    # also written from embedded source so the notebook is portable by itself.
    for script, desc in [
        ("cv_v2.py", "Fold-local target encoding helpers"),
        ("lgbm_corrected_v2.py", "Corrected LightGBM cross-validation and test fit"),
        ("ebm_top150_corrected_v1.py", "Corrected EBM cross-validation and test fit"),
        ("blend_corrected_lgb_ebm_v1.py", "OOF evaluation and probability-scale rank blend"),
    ]:
        target = f"/kaggle/working/{script}"
        source = (ROOT / "src" / script).read_text(encoding="utf-8")
        cells.append(nbf.v4.new_markdown_cell("### " + desc))
        cells.append(nbf.v4.new_code_cell(f"%%writefile {target}\n" + source))
        if script != "cv_v2.py":
            cells.append(nbf.v4.new_code_cell(f"!python {target}"))

    cells += [
        nbf.v4.new_markdown_cell("""## 4. Write and check the submission

The rank blend is converted back to a probability scale by sorting the corrected LightGBM test probabilities and assigning them in blended-rank order. This preserves a probability value for every test encounter."""),
        nbf.v4.new_code_cell("""import numpy as np
blend_path = Path('artifacts/test_blend_corrected_lgb_ebm_v1.npy')
if blend_path.exists():
    pred = np.load(blend_path)
    print('Using the corrected LightGBM + EBM v9 blend.')
else:
    pred = np.load('artifacts/test_lgbm_corrected_v2.npy')
    print('EBM was skipped; using the corrected LightGBM-only fallback, not the v9 blend.')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')
submission = pd.DataFrame({'encounter_id': test.encounter_id, 'readmit_30d': pred})
assert list(submission.columns) == ['encounter_id', 'readmit_30d']
assert len(submission) == len(sample) and submission.encounter_id.is_unique
assert set(submission.encounter_id) == set(sample.encounter_id)
assert submission.readmit_30d.notna().all() and submission.readmit_30d.between(0, 1).all()
path = '/kaggle/working/v9_corrected_lgb_ebm_blend.csv'
submission.to_csv(path, index=False)
print('saved:', path, '| shape:', submission.shape,
      '| probability range:', (submission.readmit_30d.min(), submission.readmit_30d.max()))
"""),
    ]
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
