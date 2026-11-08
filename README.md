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
| v3 | CatBoost, 2-seed avg, native categoricals | depth 5, lr 0.06, 445 rounds | **0.74337** |
| v4 | Rank blend: 0.8 LightGBM (v2) + 0.2 Logistic Regression | LR C = 0.02 | **0.74682** |
| v5 | XGBoost, 3-seed avg, target encoding | eta 0.04, depth 3, 363 rounds | **0.74397** |
| v6 | Rank blend: 0.4 LightGBM (v2 features) + 0.4 LightGBM (v3 features) + 0.2 Logistic Regression | weights from OOF grid search | **0.74685** |
| v7 | Kaggle notebook: rank blend 0.4 LightGBM (v2 features, 3 leaves) + 0.4 LightGBM (v3 features, 3 leaves) + 0.2 Logistic Regression | settings chosen by CV inside the notebook | **0.74679** |
| v8 | Rank blend: LightGBM v2 + v3 + spline logistic regression + EBM | weights 0.3 / 0.3 / 0.2 / 0.2 | **0.74695** |
| v9 | Rank blend: corrected LightGBM + corrected EBM (clean fold-local pipeline, features v4) | weights 0.65 / 0.35; CV 0.7526 | **0.74755** |
| v10 | Rank blend: corrected LightGBM (5-seed final fit) + corrected EBM | weights 0.65 / 0.35; CV 0.7526 | **0.74752** |
| v11 | Rank blend: corrected v9 blend + boosted-stump GAM | GAM weight 0.325; CV 0.752794; holdout 0.741811 | **0.74714** |
| v12 | Rank blend: corrected v9 + LightGBM with positive-class weight 0.75 | 17.5% added rank weight; CV 0.752600; holdout 0.741773 | **0.74728** |

## Ran but not submitted (CV only)

5-fold CV ROC-AUC on the shared folds. This table contains experiments that were not submitted.

| ID | Model | Key params | CV AUC |
|---|---|---|---|
| - | LightGBM, in-fold feature selection | top 150 columns by gain (top 100 + extra_trees also 0.7514; top 100 0.7510; top 60 0.7504) | 0.7514 |
| - | HistGradientBoosting | lr 0.03, depth 3, 8 leaves, min leaf 200, L2 10, 30% of columns | 0.7491 |
| - | Logistic Regression, elastic-net | C = 0.012, l1 ratio 0.2 | 0.7466 |
| - | Linear SVM (calibrated) | C = 0.0066 | 0.7442 |
| - | Linear Discriminant Analysis | shrinkage 0.5 | 0.7385 |
| - | ExtraTrees | 400 trees, depth 14, min leaf 40, 20% of columns | 0.7359 |
| - | Random Forest | 500 trees, depth 16, min leaf 40, 20% of columns, 80% row sampling | 0.7344 |
| - | MLP | layers 128-64-32, L2 0.1, lr 0.003 | 0.7336 |
| - | Gaussian Naive Bayes (tuned) | quantile transform, top 80 columns, smoothing 1e-6 (untuned 0.6968) | 0.7212 |
| - | KNN (tuned) | standard scaler, top 80-200 columns, k = 800, distance weighted (untuned 0.6844) | 0.7211 |
| - | Decision Tree (tuned) | depth 5, min leaf 200 (untuned 0.6988) | 0.6958 |
| - | Blend: 0.7 LightGBM + 0.1 CatBoost + 0.2 Logistic Regression | rank average, weights from OOF grid | 0.7526 |
| - | Blend: greedy selection over all 11 models | mostly LightGBM, some logistic regression, CatBoost, linear SVM | 0.7527 |
| - | Stacking: logistic regression on 11 models' predictions | out-of-fold stacking | 0.7518 |
| - | LightGBM LambdaRank | random ranking groups of 1,024 rows | 0.7407 |
| - | Logistic regression with numeric interactions | top 40 numeric features, pairwise terms, C = 0.02 | 0.7337 |
| - | Target-encoding smoothing sweep | M = 10/25/50/100/200; best M = 25 (0.752085) | 0.7521 |
| - | Hospital x primary-diagnosis target encoding | nested OOF, M = 25 | 0.7514 |
| - | Corrected feature-block ablation | each table dropped once; best was dropping procedures (0.7516), all below full v4 baseline | 0.7516 |
| - | XGBoost DART (corrected features) | 500 rounds, depth 3, dropout 0.05; same folds and fold-local encodings | 0.7468 |
| - | LightGBM, corrected features + 16 row-level features | prior-utilization trends and ratios, age/LOS/LACE/Charlson interactions, follow-up-days buckets; v2 parameters; TE M = 50 | 0.7514 |
| - | LightGBM, row-level features + fold-local hospital case-mix | adds 20 columns: hospital means of 10 variables (LOS, Charlson, LACE, age, prior visits...) and each stay's difference from them, computed on the training fold only | 0.7514 |
| - | LightGBM, learning rate 0.02 | v2 parameters otherwise; single seed; 5-fold early stopping (903-1373 rounds) | 0.7516 |
| - | AdaBoost, corrected features | 60 depth-3 trees, learning rate 0.1; fold-local median imputation and one-hot encoding | 0.7378 |
| - | LightGBM with native attending/procedure categories | M=25 target encoding retained alongside raw categories; corrected features | 0.7471 |
| - | LightGBM with monotone constraints | positive constraints on age, diagnosis burden and prior utilization | 0.7520 |
| - | Blend: corrected LightGBM/EBM plus monotone LightGBM | monotone rank weight 0.05 selected on 80% OOF | 0.7526 |
| - | LightGBM, hospital-by-admission-month target encoding | nested OOF, M=50 | 0.7516 |
| - | LightGBM, four clinical interactions | ICU-by-LOS; discharge planning-by-prior use; medication-by-diagnosis burden; lab abnormalities-by-LOS | 0.7517 |
| - | LightGBM, positive-class weight 1.25 | scale_pos_weight 1.25 | 0.7517 |
| - | LightGBM, positive-class weight 1.5 | scale_pos_weight 1.5 | 0.7518 |
| - | Blend: corrected v9 blend + hospital-month model | 5% added rank weight; 80/20 OOF split | 0.7526 |
| - | Blend: corrected v9 blend + clinical-interaction model | 15% added rank weight; 80/20 OOF split | 0.7526 |
| - | Blend: corrected v9 blend + class-weight 1.25 model | 5% added rank weight; 80/20 OOF split | 0.7526 |
| - | Blend: corrected v9 blend + class-weight 1.5 model | 7.5% added rank weight; 80/20 OOF split | 0.7526 |
| - | LightGBM with extra lab/vital time windows | Early-admission and last-12/24-hour discharge summaries; corrected v4 baseline | 0.7514 |
| - | Blend: corrected v9 + extra lab/vital windows | 20% added rank weight selected on 80% OOF; holdout declined, no candidate kept | 0.7526 |
| - | LightGBM with hospital-by-service-line target encoding | Fold-local M = 50; corrected v4 features | 0.7516 |
| - | Blend: corrected v9 + hospital-by-service-line model | 7.5% added rank weight; holdout +0.000069, too small to trust | 0.7526 |



## Rules followed

Only competition data is used. All imputers/scalers/encoders are fit inside CV folds on training data only. No private sharing of code or predictions.
