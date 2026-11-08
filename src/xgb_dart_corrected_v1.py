"""Five-fold XGBoost DART benchmark on corrected features and fold-local encodings."""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv_v2 import add_te, prep
from lgbm_corrected_v2 import add_fold_counts, align_categories


def to_xgb(frame):
    """Turn pandas categories into train-aligned integer codes."""
    out = frame.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes.astype("int32")
        elif not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.Categorical(out[col]).codes.astype("int32")
    return out.replace([np.inf, -np.inf], np.nan)


def main():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    X = pd.read_pickle(root / "artifacts" / "X4_3.pkl")
    tr = pd.read_csv(root / "data" / "train.csv")
    te = pd.read_csv(root / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    params = dict(
        objective="binary:logistic", eval_metric="auc", booster="dart",
        tree_method="hist", eta=0.03, max_depth=3, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.35, reg_lambda=40, reg_alpha=1,
        max_bin=128, rate_drop=0.05, skip_drop=0.3, normalize_type="tree",
        nthread=-1, seed=0,
    )
    oof = np.zeros(len(y))
    its, started = [], time.time()
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)):
        A, B = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b])
        A, (B,) = add_te(A, y[a], [B], seed=fold)
        A, B = prep(A), prep(B)
        A, B = align_categories(A, B)
        A, B = to_xgb(A), to_xgb(B)
        dA, dB = xgb.DMatrix(A, label=y[a]), xgb.DMatrix(B, label=y[b])
        # DART validation is unusually expensive on this wide table; use a
        # fixed, conservative budget for a first screening run.
        model = xgb.train(params, dA, 500, verbose_eval=False)
        # DART prediction must specify a nonzero iteration range to disable dropout.
        n = model.num_boosted_rounds()
        oof[b] = model.predict(dB, iteration_range=(0, n))
        its.append(n)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={n}, elapsed={time.time()-started:.0f}s", flush=True)

    auc = roc_auc_score(y, oof)
    print("OOF AUC", round(auc, 6), "mean rounds", int(np.mean(its)), flush=True)
    np.save(root / "artifacts" / "oof_xgb_dart_corrected_v1.npy", oof)

    A, B = add_fold_counts(Xtr, Xte)
    A, (B,) = add_te(A, y, [B], seed=0)
    A, B = prep(A), prep(B)
    A, B = align_categories(A, B)
    A, B = to_xgb(A), to_xgb(B)
    final = xgb.train(params, xgb.DMatrix(A, label=y), max(1, int(np.mean(its) * 1.1)))
    test_pred = final.predict(xgb.DMatrix(B), iteration_range=(0, max(1, int(np.mean(its) * 1.1))))
    np.save(root / "artifacts" / "test_xgb_dart_corrected_v1.npy", test_pred)
    print("test mean", float(test_pred.mean()), "range", (float(test_pred.min()), float(test_pred.max())), flush=True)


if __name__ == "__main__":
    main()
