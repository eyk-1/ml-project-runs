"""v3 features on top of X2.pkl: the patient's state at discharge and timing information.
  l3_*  labs:        hours between last lab and discharge, last-24h counts, abnormal flag of the last value, per-test slope
  v3_*  vitals:      last-24h mean/min/max per vital, clinical-threshold abnormal counts near discharge
  m3_*  medications: drug-name flags (names normalised: 122 spellings -> 52), started/stopped timing, high-risk drugs
  p3_*  procedures:  timing relative to admission/discharge (CPT and CPT-4 treated as one system)
  t3_*  transfers:   timing of the last unit move and of the last ICU stay relative to discharge
  u3_*  prior visits: split into same hospital as the index stay vs other hospitals
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd

D = os.environ.get('DATA_DIR', 'data') + '/'
ART = os.environ.get('ART_DIR', 'artifacts') + '/'
t0 = time.time()


def log(msg):
    print(f'[{round(time.time() - t0)}s] {msg}', flush=True)


def clean(s):
    return re.sub(r'[^0-9a-zA-Z]+', '_', str(s).strip().lower()).strip('_')


X = pd.read_pickle(ART + 'X2.pkl')
ie = pd.read_csv(D + 'index_encounters.csv', parse_dates=['admit_dt', 'discharge_dt'])[
    ['encounter_id', 'patient_id', 'hospital_id', 'admit_dt', 'discharge_dt']]
NEW = {}  # group prefix -> list of DataFrames


def add(prefix, df):
    NEW.setdefault(prefix, []).append(df)


# ================================================================ labs
lab = pd.read_csv(D + 'lab_results.csv').drop_duplicates()
lab['collected_dt'] = pd.to_datetime(lab.collected_dt, errors='coerce')
conv = {('Albumin', 'g/L'): .1, ('Creatinine', 'umol/L'): 1 / 88.4, ('Glucose', 'mmol/L'): 18.016, ('Hemoglobin', 'g/L'): .1}
lab['v'] = lab.value * np.array([conv.get((n, u), 1.) for n, u in zip(lab.lab_name, lab.unit)])
rng = {'Albumin': (1, 6), 'Bnp': (5, 40000), 'Bun': (2, 300), 'Creatinine': (.2, 20), 'Glucose': (30, 1000), 'Hba1C': (3, 18),
       'Hemoglobin': (3, 22), 'Inr': (.7, 15), 'Lactate': (.2, 30), 'Potassium': (1.5, 9), 'Sodium': (100, 180),
       'Troponin_I': (0, 50), 'Wbc': (.5, 100)}
lo = lab.lab_name.map(lambda x: rng[x][0]); hi = lab.lab_name.map(lambda x: rng[x][1])
lab.loc[(lab.v < lo) | (lab.v > hi), 'v'] = np.nan
lab = lab.merge(ie[['encounter_id', 'admit_dt', 'discharge_dt']], on='encounter_id')
lab['hb'] = (lab.discharge_dt - lab.collected_dt).dt.total_seconds() / 3600       # hours before discharge
lab['ta'] = (lab.collected_dt - lab.admit_dt).dt.total_seconds() / 3600          # hours since admission
lab['l24'] = (lab.hb <= 24).astype(int)
lab['abn'] = lab.abnormal_flag.isin(['H', 'L']).astype(int)
g = lab.groupby('encounter_id')
f = pd.DataFrame({'l3_hrs_last_lab': g.hb.min(), 'l3_hrs_mean_lab': g.hb.mean(), 'l3_n_last24': g.l24.sum(),
                  'l3_frac_last24': g.l24.mean(), 'l3_abn_last24': (lab.abn * lab.l24).groupby(lab.encounter_id).sum()})
add('l3', f)
lab = lab.sort_values(['encounter_id', 'lab_name', 'collected_dt'])
gl = lab.groupby(['encounter_id', 'lab_name'])
f = gl.hb.min().unstack(); f.columns = [f'l3_{clean(c)}_hrs_since' for c in f.columns]; add('l3', f)
flag = lab.assign(fl=lab.abnormal_flag.map({'H': 1, 'L': -1, 'N': 0})).groupby(['encounter_id', 'lab_name']).fl.last().unstack()
flag.columns = [f'l3_{clean(c)}_lastflag' for c in flag.columns]; add('l3', flag)
lv = lab.dropna(subset=['v']).copy(); lv['tt'] = lv.ta ** 2; lv['tv'] = lv.ta * lv.v
s = lv.groupby(['encounter_id', 'lab_name']).agg(n=('v', 'size'), st=('ta', 'sum'), sv=('v', 'sum'), stt=('tt', 'sum'), stv=('tv', 'sum'))
den = s.n * s.stt - s.st ** 2
slope = ((s.n * s.stv - s.st * s.sv) / den).where((s.n >= 3) & (den > 1e-6))
slope = slope.unstack(); slope.columns = [f'l3_{clean(c)}_slope' for c in slope.columns]; add('l3', slope)
del lab, lv, s
log('labs done')

# ================================================================ vitals
v = pd.read_csv(D + 'vitals.csv').drop_duplicates()
v['t'] = pd.to_datetime(v.charttime, format='%m/%d/%Y %H:%M', errors='coerce')
v.loc[(v.vital_type == 'TEMP') & (v.value > 45), 'value'] = (v.value - 32) * 5 / 9
r = {'HR': (20, 250), 'SBP': (50, 260), 'DBP': (20, 160), 'RR': (4, 60), 'SPO2': (50, 100), 'TEMP': (30, 43)}
lo = v.vital_type.map(lambda x: r[x][0]); hi = v.vital_type.map(lambda x: r[x][1])
v.loc[(v.value < lo) | (v.value > hi), 'value'] = np.nan
v = v.merge(ie[['encounter_id', 'discharge_dt']], on='encounter_id')
v['hb'] = (v.discharge_dt - v.t).dt.total_seconds() / 3600
x, vt = v.value, v.vital_type
v['abn'] = (((vt == 'HR') & ((x > 100) | (x < 50))) | ((vt == 'SBP') & ((x < 90) | (x > 180))) | ((vt == 'SPO2') & (x < 92))
            | ((vt == 'TEMP') & (x > 38)) | ((vt == 'RR') & (x > 24))).astype(int)
v['l24'] = (v.hb <= 24).astype(int); v['l12'] = (v.hb <= 12).astype(int)
g = v.groupby('encounter_id')
f = pd.DataFrame({'v3_hrs_last_vital': g.hb.min(), 'v3_abn_total': g.abn.sum(), 'v3_abn_frac': g.abn.mean(), 'v3_n_last24': g.l24.sum(),
                  'v3_abn_last24': (v.abn * v.l24).groupby(v.encounter_id).sum(), 'v3_abn_last12': (v.abn * v.l12).groupby(v.encounter_id).sum()})
f['v3_abn_last24_frac'] = f.v3_abn_last24 / f.v3_n_last24.replace(0, np.nan); add('v3', f)
w = v[v.l24 == 1].groupby(['encounter_id', 'vital_type']).value.agg(['mean', 'min', 'max']).unstack()
w.columns = [f'v3_{b}_l24_{a}' for a, b in w.columns]; add('v3', w)
del v
log('vitals done')

# ================================================================ medications
m = pd.read_csv(D + 'medications.csv')
m['dn'] = m.drug_name.map(clean)
m = m.drop_duplicates(['encounter_id', 'dn', 'start_dt', 'route'])
m['s'] = pd.to_datetime(m.start_dt.astype(str), format='%Y%m%d', errors='coerce')
m['e'] = pd.to_datetime(m.stop_dt.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
m = m.merge(ie[['encounter_id', 'admit_dt', 'discharge_dt']], on='encounter_id')
dcd, adm = m.discharge_dt.dt.normalize(), m.admit_dt.dt.normalize()
m['st_last1'] = (m.s >= dcd - pd.Timedelta(days=1)).astype(int)
m['st_adm'] = (m.s <= adm).astype(int)
m['stopped'] = (m.e < dcd).astype(int)
m['dur'] = (m.e - m.s).dt.days
hr = m.dn.str.contains('enoxaparin|heparin|warfarin|insulin|morphine|oxycodone|hydromorphone|fentanyl|digoxin|tramadol', regex=True)
m['hr'] = hr.astype(int)
g = m.groupby('encounter_id')
f = pd.DataFrame({'m3_started_last1d': g.st_last1.sum(), 'm3_started_admit_day': g.st_adm.sum(), 'm3_stopped_before_dc': g.stopped.sum(),
                  'm3_mean_dur_days': g.dur.mean(), 'm3_n_highrisk': g.hr.sum(), 'm3_n_names': g.dn.nunique()}); add('m3', f)
top = m.dn.value_counts().head(50).index
fl = pd.crosstab(m[m.dn.isin(top)].encounter_id, m[m.dn.isin(top)].dn).clip(upper=1); fl.columns = [f'm3_drug_{c}' for c in fl.columns]; add('m3', fl)
cont = m[m.continued_at_discharge == 1].groupby('encounter_id').dn.nunique().rename('m3_n_cont_names'); add('m3', cont.to_frame())
del m
log('meds done')

# ================================================================ procedures
p = pd.read_csv(D + 'procedures.csv').drop_duplicates()
p['d'] = pd.to_datetime(p.proc_dt, format='%d-%b-%Y', errors='coerce')
p = p.merge(ie[['encounter_id', 'admit_dt', 'discharge_dt']], on='encounter_id')
p['da'] = (p.d - p.admit_dt.dt.normalize()).dt.days; p['db'] = (p.discharge_dt.dt.normalize() - p.d).dt.days
p['code'] = p.code.astype(str).str.strip()
g = p.groupby('encounter_id')
f = pd.DataFrame({'p3_first_day': g.da.min(), 'p3_last_day': g.da.max(), 'p3_last_days_before_dc': g.db.min(),
                  'p3_any_last48h': (p.db <= 2).groupby(p.encounter_id).max().astype(int), 'p3_distinct_codes': g.code.nunique()}); add('p3', f)
del p
log('procedures done')

# ================================================================ unit transfers
u = pd.read_csv(D + 'unit_transfers.csv', parse_dates=['in_dt', 'out_dt']).drop_duplicates()
u = u.merge(ie[['encounter_id', 'discharge_dt']], on='encounter_id').sort_values(['encounter_id', 'in_dt'])
u['in_hb'] = (u.discharge_dt - u.in_dt).dt.total_seconds() / 3600; u['out_hb'] = (u.discharge_dt - u.out_dt).dt.total_seconds() / 3600
u['dur'] = (u.out_dt - u.in_dt).dt.total_seconds() / 3600
icu = u.unit.astype(str).str.upper().str.contains('ICU')
g = u.groupby('encounter_id')
f = pd.DataFrame({'t3_last_in_hrs_before_dc': g.in_hb.min(), 'm_tmp': g.dur.last(), 'tmp_n24': (u.in_hb <= 24).groupby(u.encounter_id).sum()})
f = f.rename(columns={'m_tmp': 't3_last_unit_hours', 'tmp_n24': 't3_n_moves_last24'})
f['t3_hrs_since_icu_out'] = u[icu].groupby('encounter_id').out_hb.min()
f['t3_icu_last24'] = ((u.out_hb <= 24) & icu).groupby(u.encounter_id).max().astype(int); add('t3', f)
del u
log('transfers done')

# ================================================================ prior visits: same hospital vs elsewhere
pu = pd.read_csv(D + 'prior_utilization.csv', parse_dates=['visit_dt']).drop_duplicates()
pu = pu.merge(ie[['patient_id', 'hospital_id', 'admit_dt']].rename(columns={'hospital_id': 'idx_h'}), on='patient_id')
pu['d'] = (pu.admit_dt - pu.visit_dt).dt.days; pu = pu[pu.d > 0]
pu['same'] = (pu.hospital_id == pu.idx_h).astype(int)
cols = {}
for tag, mask in [('same', pu.same == 1), ('diff', pu.same == 0)]:
    for w in [90, 365, 10 ** 6]:
        s_ = pu[mask & (pu.d <= w)].groupby('patient_id').size(); cols[f'u3_{tag}_{"all" if w > 1000 else w}d'] = s_
    cols[f'u3_{tag}_ip_all'] = pu[mask & (pu.visit_type == 'IP')].groupby('patient_id').size()
f = pd.DataFrame(cols).fillna(0)
f['u3_same_frac'] = f.u3_same_alld / (f.u3_same_alld + f.u3_diff_alld).replace(0, np.nan)
f = f.rename(columns={'u3_same_alld': 'u3_same_all', 'u3_diff_alld': 'u3_diff_all'})
log('prior done')

# ================================================================ assemble
X3 = X.copy(); groups = {}
for pre, dfs in NEW.items():
    d = pd.concat(dfs, axis=1)
    X3 = X3.join(d); groups[pre] = list(d.columns)
X3 = X3.join(ie.set_index('encounter_id')[['patient_id']].join(f, on='patient_id').drop(columns='patient_id'))
groups['u3'] = list(f.columns)
zero = [c for c in X3.columns if c.startswith(('u3_', 'm3_drug_', 'm3_n', 'm3_started', 'm3_stopped', 'v3_n_', 'v3_abn', 'l3_n_', 'l3_abn_', 't3_n_', 't3_icu_', 'p3_any'))]
for c in zero:
    if c in X3 and c != 'u3_same_frac':
        X3[c] = X3[c].fillna(0)
X3 = X3.copy(); X3.to_pickle(ART + 'X3.pkl'); json.dump(groups, open(ART + 'v3_groups.json', 'w'))
log(f'X3 {X3.shape} (X2 had {X.shape[1]} cols); new per group: ' + str({k: len(v) for k, v in groups.items()}))
print(X3[[c for c in X3 if c[:3] in ('l3_', 'v3_', 'm3_', 'p3_', 't3_', 'u3_')]].isna().mean().round(2).describe())
