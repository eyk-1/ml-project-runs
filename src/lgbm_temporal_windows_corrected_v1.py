"""Test extra admission/discharge-window lab and vital features on corrected folds."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import cv_v2 as c
from lgbm_corrected_v2 import add_fold_counts, align_categories

ROOT = Path(__file__).resolve().parents[1]


def _wide_stats(frame, key, value, groups, prefix):
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name="encounter_id"))
    agg = frame.groupby(["encounter_id", key])[value].agg(groups)
    out = agg.unstack(key)
    out.columns = [f"{prefix}_{str(col).lower()}_{stat}" for stat, col in out.columns]
    return out


def make_temporal_features(index):
    encounters = pd.read_csv(
        ROOT / "data" / "index_encounters.csv",
        parse_dates=["admit_dt", "discharge_dt"],
    )[["encounter_id", "admit_dt", "discharge_dt"]]

    lab = pd.read_csv(ROOT / "data" / "lab_results.csv").drop_duplicates()
    lab["collected_dt"] = pd.to_datetime(lab.collected_dt, errors="coerce")
    conv = {("Albumin", "g/L"): 0.1, ("Creatinine", "umol/L"): 1 / 88.4,
            ("Glucose", "mmol/L"): 18.016, ("Hemoglobin", "g/L"): 0.1}
    lab["value_clean"] = lab.value * [conv.get((n, u), 1.0) for n, u in zip(lab.lab_name, lab.unit)]
    ranges = {"Albumin": (1, 6), "Bnp": (5, 40000), "Bun": (2, 300), "Creatinine": (0.2, 20),
              "Glucose": (30, 1000), "Hba1C": (3, 18), "Hemoglobin": (3, 22), "Inr": (0.7, 15),
              "Lactate": (0.2, 30), "Potassium": (1.5, 9), "Sodium": (100, 180),
              "Troponin_I": (0, 50), "Wbc": (0.5, 100)}
    low = lab.lab_name.map(lambda name: ranges[name])
    lab["lo"] = [pair[0] for pair in low]
    lab["hi"] = [pair[1] for pair in low]
    lab.loc[(lab.value_clean < lab.lo) | (lab.value_clean > lab.hi), "value_clean"] = np.nan
    lab = lab.merge(encounters, on="encounter_id", how="inner")
    lab["hours_from_admit"] = (lab.collected_dt - lab.admit_dt).dt.total_seconds() / 3600
    lab["hours_to_discharge"] = (lab.discharge_dt - lab.collected_dt).dt.total_seconds() / 3600
    lab = lab[lab.value_clean.notna()]
    early = lab[lab.hours_from_admit.between(0, 24, inclusive="both")]
    late24 = lab[lab.hours_to_discharge.between(0, 24, inclusive="both")]
    late12 = lab[lab.hours_to_discharge.between(0, 12, inclusive="both")]
    parts = [
        _wide_stats(early, "lab_name", "value_clean", ["mean", "max"], "tw_lab_adm24"),
        _wide_stats(late24, "lab_name", "value_clean", ["mean", "max", "min"], "tw_lab_dc24"),
        _wide_stats(late12, "lab_name", "value_clean", ["mean", "max", "min"], "tw_lab_dc12"),
    ]
    labs = pd.concat(parts, axis=1)
    for name in ranges:
        early_col, late_col = f"tw_lab_adm24_{name.lower()}_mean", f"tw_lab_dc24_{name.lower()}_mean"
        if early_col in labs and late_col in labs:
            labs[f"tw_lab_{name.lower()}_dc24_minus_adm24"] = labs[late_col] - labs[early_col]

    vit = pd.read_csv(ROOT / "data" / "vitals.csv").drop_duplicates()
    vit["charttime"] = pd.to_datetime(vit.charttime, format="%m/%d/%Y %H:%M", errors="coerce")
    vit.loc[(vit.vital_type == "TEMP") & (vit.value > 45), "value"] = (vit.value - 32) * 5 / 9
    vranges = {"HR": (20, 250), "SBP": (50, 260), "DBP": (20, 160), "RR": (4, 60),
               "SPO2": (50, 100), "TEMP": (30, 43)}
    vit["lo"] = vit.vital_type.map(lambda name: vranges[name][0])
    vit["hi"] = vit.vital_type.map(lambda name: vranges[name][1])
    vit.loc[(vit.value < vit.lo) | (vit.value > vit.hi), "value"] = np.nan
    vit = vit.merge(encounters, on="encounter_id", how="inner")
    vit["hours_from_admit"] = (vit.charttime - vit.admit_dt).dt.total_seconds() / 3600
    vit["hours_to_discharge"] = (vit.discharge_dt - vit.charttime).dt.total_seconds() / 3600
    vit = vit[vit.value.notna()]
    parts = []
    for tag, frame in [
        ("adm24", vit[vit.hours_from_admit.between(0, 24, inclusive="both")]),
        ("dc24", vit[vit.hours_to_discharge.between(0, 24, inclusive="both")]),
        ("dc12", vit[vit.hours_to_discharge.between(0, 12, inclusive="both")]),
    ]:
        parts.append(_wide_stats(frame, "vital_type", "value", ["mean", "max", "min", "std"], f"tw_vit_{tag}"))
    out = pd.concat([labs, *parts], axis=1).reindex(index)
    return out.astype("float32")


def main():
    X = pd.read_pickle(ROOT / "artifacts" / "X4_3.pkl")
    tr, te = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = tr.readmit_30d.to_numpy()
    X = X.join(make_temporal_features(X.index))
    Xtr, Xte = X.loc[tr.encounter_id], X.loc[te.encounter_id]
    c.M = 50
    params = dict(c.P, num_leaves=4, min_child_samples=400, feature_fraction=0.2,
                  lambda_l2=60, learning_rate=0.04, n_jobs=2)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y))
    oof, test_pred, rounds = np.zeros(len(y)), np.zeros(len(te)), []
    for fold, (a, b) in enumerate(folds):
        A, B, T = add_fold_counts(Xtr.iloc[a], Xtr.iloc[b], Xte)
        A, (B, T) = c.add_te(A, y[a], [B, T], seed=fold)
        A, B, T = c.prep(A), c.prep(B), c.prep(T)
        A, B, T = align_categories(A, B, T)
        model = lgb.train(
            params, lgb.Dataset(A, label=y[a]), num_boost_round=3000,
            valid_sets=[lgb.Dataset(B, label=y[b])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        rounds.append(model.best_iteration)
        oof[b] = model.predict(B, num_iteration=model.best_iteration)
        test_pred += model.predict(T, num_iteration=model.best_iteration) / len(folds)
        print(f"fold {fold}: AUC={roc_auc_score(y[b], oof[b]):.6f}, rounds={rounds[-1]}", flush=True)
    auc = roc_auc_score(y, oof)
    np.save(ROOT / "artifacts" / "oof_lgbm_temporal_windows_corrected_v1.npy", oof)
    np.save(ROOT / "artifacts" / "test_lgbm_temporal_windows_corrected_v1.npy", test_pred)
    print(f"OOF AUC={auc:.6f}; mean rounds={np.mean(rounds):.1f}; test mean={test_pred.mean():.6f}", flush=True)


if __name__ == "__main__":
    main()
