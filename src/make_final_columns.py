"""Write final_columns.csv: every column of the final modelling table (artifacts/X3.pkl) with its source table,
how it was created, the feature-set version it first appeared in, its data type and % missing.
Run from the repo root:  python src/make_final_columns.py"""
import re
import pandas as pd

X1 = pd.read_pickle('artifacts/X.pkl'); X2 = pd.read_pickle('artifacts/X2.pkl'); X3 = pd.read_pickle('artifacts/X3.pkl')
v1, v2 = set(X1.columns), set(X2.columns)

LAB_STAT = {'last': 'last value of the stay (by collection time)', 'mean': 'mean over the stay', 'min': 'minimum over the stay',
            'max': 'maximum over the stay', 'cnt': 'number of valid measurements', 'delta': 'last value minus first value'}
VIT_STAT = dict(LAB_STAT, std='standard deviation over the stay')
LABCLEAN = 'after converting to one unit and setting implausible values to missing'
VITCLEAN = 'temperature converted to Celsius, implausible readings set to missing'

EXPL = {  # (source table, how created)
    'hospital_id': ('index_encounters', 'copied (categorical)'),
    'admission_type': ('index_encounters', "copied; 'Not Available'/'Not Mapped' set to missing"),
    'admission_source': ('index_encounters', "copied; 'Not Available'/'Not Mapped' set to missing"),
    'service_line': ('index_encounters', 'copied (categorical)'),
    'discharge_disposition': ('index_encounters', "copied; 'Not Available'/'Not Mapped' set to missing"),
    'attending_id': ('index_encounters', 'copied (categorical, ~1000 levels); NOT fed to the models raw, replaced by te_attending_id and attending_n'),
    'attending_specialty': ('index_encounters', 'copied (categorical), 46% missing'),
    'admit_weight_kg': ('index_encounters', 'copied, 87% missing'),
    'sex': ('patients', "copied; 'U' set to missing"),
    'race_ethnicity': ('patients', "copied; '?' set to missing"),
    'language': ('patients', 'copied'), 'marital_status': ('patients', 'copied'),
    'insurance_type': ('patients', 'copied; case variants merged (MEDICARE -> Medicare), 10 spellings -> 5 values'),
    'zip3': ('patients', 'copied as a number'), 'deprivation_index': ('patients', 'copied'),
    'bed_size': ('hospitals', 'joined on hospital_id'), 'teaching_flag': ('hospitals', 'joined on hospital_id'),
    'urban_rural': ('hospitals', 'joined on hospital_id'), 'ownership': ('hospitals', 'joined on hospital_id'),
    'los_days': ('index_encounters', 'discharge_dt minus admit_dt in days; <=0 or >120 set to missing'),
    'age': ('patients + index_encounters', 'admission year minus birth_year (birth_year < 1910 set to missing)'),
    'admit_hour': ('index_encounters', 'hour of admit_dt'), 'admit_dow': ('index_encounters', 'weekday of admit_dt'),
    'admit_month': ('index_encounters', 'month of admit_dt'), 'admit_year': ('index_encounters', 'year of admit_dt'),
    'weight_missing': ('index_encounters', '1 if admit_weight_kg is missing'),
    'spec_missing': ('index_encounters', '1 if attending_specialty is missing'),
    'attending_n': ('index_encounters', 'number of stays of the same attending_id (all train+test stays)'),
    'hosp_n': ('index_encounters', 'number of stays of the same hospital (all train+test stays)'),
    'followup_scheduled_days': ('discharge_planning', 'copied; values > 60 set to missing; missing for stays without a row'),
    'home_health_referral': ('discharge_planning', 'Y/N mapped to 1/0'), 'education_done': ('discharge_planning', 'Yes/No mapped to 1/0'),
    'med_reconciliation_done': ('discharge_planning', 'copied (0/1)'),
    'dp_present': ('discharge_planning', '1 if the stay has a discharge-planning row'),
    'followup_missing': ('discharge_planning', '1 if followup_scheduled_days is missing in an existing row'),
    'charlson': ('diagnoses', 'simplified Charlson comorbidity score: weights per 3-character ICD-10 prefix summed over the stay'),
    'lace': ('derived (index_encounters, diagnoses, prior_utilization)', 'LACE-style score: length-of-stay points + 3 if emergency admission + Charlson points + min(ED visits in 180 d, 4)'),
    'bun_creat': ('lab_results', 'last BUN divided by last creatinine'),
    'prior_ip_recent': ('prior_utilization', '1 if the last prior inpatient visit was within 30 days of admission'),
    'age_x_charlson': ('derived (patients, diagnoses)', 'age times Charlson score'),
    'los_x_charlson': ('derived (index_encounters, diagnoses)', 'length of stay times Charlson score'),
    'icu_hours': ('unit_transfers', 'total hours in ICU (same as tr_hours_ICU)'),
    'shock_index': ('vitals', 'last heart rate divided by last systolic BP'),
    'dx_n': ('diagnoses', 'number of diagnosis rows after removing duplicates'),
    'dx_chronic_n': ('diagnoses + icd10_lookup', 'number of diagnoses flagged chronic in the lookup'),
    'dx_poa_n': ('diagnoses', "number of diagnoses not present on admission ('N')"),
    'dx_poa_missing': ('diagnoses', 'number of diagnoses with missing present_on_admission'),
    'dx_primary_chapter': ('diagnoses + icd10_lookup', 'ICD-10 chapter of the primary diagnosis'),
    'dx_primary3': ('diagnoses', 'first 3 characters of the primary ICD-10 code (after adding missing decimal points)'),
    'dx_distinct3': ('diagnoses', 'number of distinct 3-character ICD-10 codes'),
    'dx_per_day': ('diagnoses + index_encounters', 'dx_n divided by length of stay'),
    'proc_n': ('procedures', 'number of procedures (0 if the stay has none)'), 'proc_first': ('procedures', 'first procedure code; NOT fed to the models raw, replaced by te_proc_first'),
    'tr_n': ('unit_transfers', 'number of unit rows'), 'tr_last_unit': ('unit_transfers', 'last unit of the stay'),
    'tr_first_unit': ('unit_transfers', 'first unit of the stay'), 'tr_icu': ('unit_transfers', '1 if the stay included an ICU unit'),
    'pu_all': ('prior_utilization', 'number of prior visits before the index admission (duplicates removed)'),
    'pu_nhosp': ('prior_utilization', 'number of distinct hospitals visited before'),
    'pu_acute_365': ('prior_utilization', 'prior ED + inpatient visits in the last 365 days'),
    'pu_acute_90': ('prior_utilization', 'prior ED + inpatient visits in the last 90 days'),
    'pu_acute_frac': ('prior_utilization', 'acute (ED+IP) share of all prior visits'),
    'lab_abn_H': ('lab_results', "number of results flagged 'H'"), 'lab_abn_L': ('lab_results', "number of results flagged 'L'"),
    'lab_total': ('lab_results', 'total number of lab results'),
    'lab_outlier_n': ('lab_results', 'number of results outside the plausible range after unit conversion'),
    'lab_distinct_tests': ('lab_results', 'number of different lab tests done'),
    'med_n': ('medications', 'number of medication rows (duplicates removed)'), 'med_distinct': ('medications', 'number of distinct drug names'),
    'med_cont_n': ('medications', 'number of medications continued at discharge'),
    'med_cont_missing': ('medications', 'number of medications with missing continued_at_discharge'),
    'med_iv_n': ('medications', 'number of IV-route medications'),
    'med_nostop_n': ('medications', 'number of medications with no stop date'),
    'med_cont_frac': ('medications', 'med_cont_n divided by med_n'), 'med_per_day': ('medications + index_encounters', 'med_n divided by length of stay'),
    'l3_hrs_last_lab': ('lab_results + index_encounters', 'hours between the last lab collection and discharge'),
    'l3_hrs_mean_lab': ('lab_results + index_encounters', 'mean hours between each lab collection and discharge'),
    'l3_n_last24': ('lab_results + index_encounters', 'number of lab results in the last 24 h before discharge'),
    'l3_frac_last24': ('lab_results + index_encounters', 'share of lab results in the last 24 h'),
    'l3_abn_last24': ('lab_results + index_encounters', 'number of abnormal (H/L) results in the last 24 h'),
    'v3_hrs_last_vital': ('vitals + index_encounters', 'hours between the last vital reading and discharge'),
    'v3_abn_total': ('vitals', 'readings outside clinical thresholds (HR>100 or <50, SBP<90 or >180, SpO2<92, temp>38, RR>24)'),
    'v3_abn_frac': ('vitals', 'share of readings outside those thresholds'),
    'v3_n_last24': ('vitals + index_encounters', 'number of readings in the last 24 h'),
    'v3_abn_last24': ('vitals + index_encounters', 'threshold-abnormal readings in the last 24 h'),
    'v3_abn_last12': ('vitals + index_encounters', 'threshold-abnormal readings in the last 12 h'),
    'v3_abn_last24_frac': ('vitals + index_encounters', 'abnormal share of the readings in the last 24 h'),
    'm3_started_last1d': ('medications + index_encounters', 'medications started within 1 day of discharge'),
    'm3_started_admit_day': ('medications + index_encounters', 'medications started on or before the admission day'),
    'm3_stopped_before_dc': ('medications + index_encounters', 'medications with a stop date before discharge'),
    'm3_mean_dur_days': ('medications', 'mean days between start and stop date'),
    'm3_n_highrisk': ('medications', 'medications whose normalised name matches a high-risk list (anticoagulants, insulin, opioids, digoxin, tramadol)'),
    'm3_n_names': ('medications', 'distinct normalised drug names'),
    'm3_n_cont_names': ('medications', 'distinct drug names continued at discharge'),
    'p3_first_day': ('procedures + index_encounters', 'days from admission to the first procedure'),
    'p3_last_day': ('procedures + index_encounters', 'days from admission to the last procedure'),
    'p3_last_days_before_dc': ('procedures + index_encounters', 'days between the last procedure and discharge'),
    'p3_any_last48h': ('procedures + index_encounters', '1 if a procedure happened within 2 days of discharge'),
    'p3_distinct_codes': ('procedures', 'number of distinct procedure codes (CPT and CPT-4 treated as one system)'),
    't3_last_in_hrs_before_dc': ('unit_transfers + index_encounters', 'hours between the last unit entry and discharge'),
    't3_last_unit_hours': ('unit_transfers', 'hours spent in the last unit'),
    't3_n_moves_last24': ('unit_transfers + index_encounters', 'unit entries within the last 24 h'),
    't3_hrs_since_icu_out': ('unit_transfers + index_encounters', 'hours between leaving the ICU and discharge (missing if no ICU)'),
    't3_icu_last24': ('unit_transfers + index_encounters', '1 if the patient left an ICU within 24 h of discharge'),
}


def explain(c):
    if c in EXPL:
        return EXPL[c]
    m = re.match(r'^dxch_(.+)$', c)
    if m: return ('diagnoses + icd10_lookup', f'number of diagnoses in ICD-10 chapter "{m.group(1)}"')
    m = re.match(r'^(dx3|dx3b)_(.+)$', c)
    if m: return ('diagnoses', f'1 if the stay has a diagnosis starting with {m.group(2)} (one of the {"60" if m.group(1) == "dx3" else "next most common"} most frequent 3-character codes)')
    m = re.match(r'^te_(.+)$', c)
    m = re.match(r'^med_([A-Z_]+)$', c)
    if m: return ('medications', f'1 if the stay has a medication of class {m.group(1)}')
    m = re.match(r'^tr_hours_(.+)$', c)
    if m: return ('unit_transfers', f'total hours spent in unit {m.group(1)}')
    m = re.match(r'^pu_(ED|IP|OP)_(\d+)d$', c)
    if m: return ('prior_utilization', f'number of prior {m.group(1)} visits within {m.group(2)} days before admission')
    m = re.match(r'^pu_(ED|IP|OP)_all$', c)
    if m: return ('prior_utilization', f'number of all prior {m.group(1)} visits')
    m = re.match(r'^pu_(ED|IP|OP)_lastgap$', c)
    if m: return ('prior_utilization', f'days since the most recent prior {m.group(1)} visit')
    m = re.match(r'^lab_(.+)_(last|mean|min|max|cnt|delta)$', c)
    if m: return ('lab_results', f'{LAB_STAT[m.group(2)]} of {m.group(1)}, {LABCLEAN}')
    m = re.match(r'^vit_(.+)_(last|mean|min|max|std|cnt|delta)$', c)
    if m: return ('vitals', f'{VIT_STAT[m.group(2)]} of {m.group(1)}, {VITCLEAN}')
    m = re.match(r'^l3_(.+)_hrs_since$', c)
    if m: return ('lab_results + index_encounters', f'hours between the last {m.group(1)} measurement and discharge')
    m = re.match(r'^l3_(.+)_lastflag$', c)
    if m: return ('lab_results', f'abnormal flag of the last {m.group(1)} result (-1 = L, 0 = N, 1 = H)')
    m = re.match(r'^l3_(.+)_slope$', c)
    if m: return ('lab_results + index_encounters', f'least-squares slope of {m.group(1)} against hours since admission (needs at least 3 values)')
    m = re.match(r'^v3_(.+)_l24_(mean|min|max)$', c)
    if m: return ('vitals + index_encounters', f'{m.group(2)} of {m.group(1)} over the last 24 h before discharge')
    m = re.match(r'^m3_drug_(.+)$', c)
    if m: return ('medications', f'1 if the stay has drug "{m.group(1)}" (names lowercased/trimmed: 122 spellings -> 52; 50 most common kept)')
    m = re.match(r'^u3_(same|diff)_(90d|365d|all|ip_all)$', c)
    if m:
        where = 'at the same hospital as the index stay' if m.group(1) == 'same' else 'at other hospitals'
        what = {'90d': 'prior visits in the last 90 days', '365d': 'prior visits in the last 365 days', 'all': 'all prior visits', 'ip_all': 'all prior inpatient visits'}[m.group(2)]
        return ('prior_utilization + index_encounters', f'number of {what} {where}')
    if c == 'u3_same_frac': return ('prior_utilization + index_encounters', 'share of all prior visits that were at the same hospital')
    raise KeyError(c)


rows = []
for c in X3.columns:
    src, how = explain(c)
    ver = 'v1' if c in v1 else ('v2' if c in v2 else 'v3')
    rows.append(dict(column=c, source_table=src, how_created=how, added_in=ver,
                     dtype='categorical' if str(X3[c].dtype) == 'category' else 'numeric', pct_missing=round(100 * X3[c].isna().mean(), 1)))
for c, base, what in [('te_hospital_id', 'hospital_id', 'hospital'), ('te_attending_id', 'attending_id', 'attending'),
                      ('te_dx_primary3', 'dx_primary3', 'primary diagnosis'), ('te_proc_first', 'proc_first', 'first procedure code')]:
    rows.append(dict(column=c, source_table='train.csv labels + ' + base, added_in='v2 (model time)', dtype='numeric', pct_missing=0.0,
                     how_created=f'smoothed (m=50) mean readmission rate of the {what} category; computed inside each training fold (out-of-fold for the training rows), not stored in X3'))
out = pd.DataFrame(rows)
out.to_csv('final_columns.csv', index=False, lineterminator='\n')
print(len(X3.columns), 'columns in X3 +', len(rows) - len(X3.columns), 'model-time columns ->', len(out), 'rows written')
print(out.groupby('source_table').size().sort_values(ascending=False).head(12))
