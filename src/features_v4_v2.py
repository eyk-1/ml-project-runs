"""Add v2 features to corrected X4.pkl, with diagnosis vocabulary from train only.

Run from the repository root after src/features_v4.py:
    python src/features_v4_v2.py
Produces artifacts/X4_2.pkl.
"""
import os
import numpy as np
import pandas as pd

D = os.environ.get('DATA_DIR', 'data') + '/'
ART = os.environ.get('ART_DIR', 'artifacts') + '/'
X = pd.read_pickle(ART + 'X4.pkl')
train_ids = set(pd.read_csv(D + 'train.csv').encounter_id)

# ---------- diagnoses: Charlson-style score + extra 3-char flags ----------
dx = pd.read_csv(D + 'diagnoses.csv').drop_duplicates(['encounter_id', 'seq'])
dx['icd'] = dx.icd10_code.str.strip().str.upper()
dx['c3'] = dx.icd.str[:3]

charlson = {  # prefix -> weight (simplified Charlson)
    'I21': 1, 'I22': 1, 'I50': 1, 'I70': 1, 'I73': 1, 'G45': 1, 'F00': 1, 'F01': 1, 'F02': 1, 'F03': 1,
    'G30': 1, 'J40': 1, 'J41': 1, 'J42': 1, 'J43': 1, 'J44': 1, 'J45': 1, 'J47': 1, 'M05': 1, 'M06': 1,
    'M32': 1, 'K25': 1, 'K26': 1, 'K27': 1, 'K28': 1, 'K70': 1, 'K74': 1, 'E10': 1, 'E11': 1,
    'G81': 2, 'G82': 2, 'N18': 2, 'N19': 2, 'B20': 6}
w = dx.c3.map(charlson).fillna(0)
w[dx.c3.str[0].eq('C') & dx.c3.between('C00', 'C76')] = 2
w[dx.c3.between('C77', 'C80')] = 6
dx['w'] = w
ch = dx.groupby('encounter_id').w.sum().rename('charlson')

# flags for the 150 most common 3-char codes (v1 had 60)
top = dx[dx.encounter_id.isin(train_ids)].c3.value_counts().head(150).index
fl = pd.crosstab(dx[dx.c3.isin(top)].encounter_id, dx[dx.c3.isin(top)].c3).clip(upper=1).add_prefix('dx3b_')
# only add the ones v1 did not have
fl = fl[[c for c in fl if 'dx3_' + c[5:] not in X.columns]]

# raw-code count of distinct 3-char codes and share of chronic
nd = dx.groupby('encounter_id').c3.nunique().rename('dx_distinct3')

X = X.join(ch).join(nd).join(fl)
X['charlson'] = X.charlson.fillna(0)
for c in fl.columns:
    X[c] = X[c].fillna(0)

# ---------- LACE-style score ----------
los = X.los_days
L = pd.cut(los, [-1, 1, 2, 3, 6, 13, 1000], labels=[0, 1, 2, 3, 4, 5], right=False).astype(float)
L = np.select([los < 1, los < 2, los < 3, los < 4, los < 7, los < 14], [0, 1, 2, 3, 4, 5], default=7)
A = (X.admission_type.astype(str) == 'Emergency').astype(int) * 3
C = np.select([X.charlson == 0, X.charlson == 1, X.charlson == 2, X.charlson == 3], [0, 1, 2, 3], default=5)
E = np.minimum(X.pu_ED_180d.fillna(0), 4)
X['lace'] = L + A + C + E

# ---------- ratios / interactions ----------
X['bun_creat'] = X.lab_Bun_last / X.lab_Creatinine_last
X['med_cont_frac'] = X.med_cont_n / X.med_n.replace(0, np.nan)
X['med_per_day'] = X.med_n / X.los_days
X['dx_per_day'] = X.dx_n / X.los_days
X['pu_acute_frac'] = X.pu_acute_365 / X.pu_all.replace(0, np.nan)
X['prior_ip_recent'] = (X.pu_IP_lastgap <= 30).astype(int)
X['age_x_charlson'] = X.age * X.charlson
X['los_x_charlson'] = X.los_days * X.charlson
X['icu_hours'] = X.get('tr_hours_ICU')
X['shock_index'] = X.vit_HR_last / X.vit_SBP_last

# ---------- outlier flags: how many raw lab values fell outside plausible range ----------
lab = pd.read_csv(D + 'lab_results.csv').drop_duplicates()
conv = {('Albumin', 'g/L'): .1, ('Creatinine', 'umol/L'): 1 / 88.4, ('Glucose', 'mmol/L'): 18.016, ('Hemoglobin', 'g/L'): .1}
lab['v'] = lab.value * np.array([conv.get((n, u), 1.) for n, u in zip(lab.lab_name, lab.unit)])
rng = {'Albumin': (1, 6), 'Bnp': (5, 40000), 'Bun': (2, 300), 'Creatinine': (.2, 20), 'Glucose': (30, 1000), 'Hba1C': (3, 18),
       'Hemoglobin': (3, 22), 'Inr': (.7, 15), 'Lactate': (.2, 30), 'Potassium': (1.5, 9), 'Sodium': (100, 180),
       'Troponin_I': (0, 50), 'Wbc': (.5, 100)}
lo = lab.lab_name.map(lambda x: rng[x][0]); hi = lab.lab_name.map(lambda x: rng[x][1])
lab['out'] = ((lab.v < lo) | (lab.v > hi)).astype(int)
X = X.join(lab.groupby('encounter_id').out.sum().rename('lab_outlier_n'))
X['lab_outlier_n'] = X.lab_outlier_n.fillna(0)
X = X.join(lab.groupby('encounter_id').lab_name.nunique().rename('lab_distinct_tests'))

X = X.copy()
X.to_pickle(ART + 'X4_2.pkl')
print(X.shape)
print(X[['charlson', 'lace']].describe().T)
