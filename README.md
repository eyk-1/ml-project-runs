# 30-Day Hospital Readmission - Kaggle Challenge

Intro to Machine Learning, Fall 2026 (IBA Karachi). Predict whether a patient is readmitted within 30 days of discharge (`readmit_30d`) from a multi-table synthetic EHR dataset. Metric: ROC-AUC on predicted probabilities.

Competition: https://www.kaggle.com/t/83081d62f0604355948c9313230de90d

## Data pipeline (summary)

All tables are used. Many-row tables are aggregated to one row per `encounter_id`.

| Table | Aggregation |
|---|---|
| index_encounters, patients, hospitals | joined directly; age, length of stay, admit hour/weekday/month added |
| diagnoses (+ icd10_lookup) | counts (total, chronic, not-present-on-admission), primary diagnosis, chapter counts, flags for top 60 3-char codes |
| procedures | count, first code |
| lab_results | per test: last, mean, min, max, count, last-minus-first; abnormal H/L counts |
| vitals | per vital: last, mean, min, max, std, count, last-minus-first |
| medications | counts, distinct drugs, continued at discharge, IV count, class flags |
| unit_transfers | number of moves, first/last unit, hours per unit, ICU flag |
| discharge_planning | all fields plus a "row present" flag and missing-follow-up flag |
| prior_utilization | ED / IP / OP counts in 30/90/180/365-day windows, totals, days since last visit |

## Data quality fixes

- Lab units mixed (g/L vs g/dL, umol/L vs mg/dL, mmol/L vs mg/dL): converted to one unit per test.
- Implausible lab and vital values (unit slips, SpO2 = 0, HR = 300): set to missing.
- Temperature in both F and C: values above 45 converted from F to C.
- ICD codes missing the decimal point (`E669`): normalised to `E66.9` so they match the lookup.
- Exact duplicate rows dropped in diagnoses, labs, vitals, medications, prior utilization.
- Placeholders (`?`, `Not Available`, `Not Mapped`, sex `U`) turned into missing; insurance case normalised.
- Missing values kept as missing for tree models, with missing-indicator flags where missingness looks informative (weight, attending specialty, discharge planning).

## Experiment log

Add a row after every Kaggle submission. Full detail (including CV scores) lives in `experiments/results.csv`.

| ID | Model | Key params | Public LB |
|---|---|---|---|
| v1 | LightGBM, 3-seed avg | lr 0.03, 15 leaves, 290 rounds | **0.74326** |
| v2 | LightGBM regularised + v2 features + target encoding | lr 0.04, 4 leaves, 700 rounds | **0.74589** |


## Rules followed

Only competition data is used. All imputers/scalers/encoders are fit inside CV folds on training data only. No private sharing of code or predictions.