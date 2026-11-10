"""Build the course report PDF in the layout of the original report."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "Readmission_Challenge_Report_revised.pdf"

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyA", fontName="Arial", fontSize=8.3, leading=11.2, spaceAfter=5, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="TitleA", fontName="Arial-Bold", fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name="SubTitleA", fontName="Arial", fontSize=12, leading=15, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name="HeadA", fontName="Arial-Bold", fontSize=12, leading=15, spaceBefore=2, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="SubHeadA", fontName="Arial-Bold", fontSize=9.5, leading=12, spaceBefore=4, spaceAfter=4, keepWithNext=True))
styles.add(ParagraphStyle(name="CellA", fontName="Arial", fontSize=6.7, leading=8.4, spaceAfter=0))
styles.add(ParagraphStyle(name="CellHeadA", fontName="Arial-Bold", fontSize=6.8, leading=8.3, textColor=colors.white, spaceAfter=0))
styles.add(ParagraphStyle(name="CaptionA", fontName="Arial-Italic", fontSize=6.7, leading=8.4, spaceBefore=3, spaceAfter=5))
styles.add(ParagraphStyle(name="BulletA", fontName="Arial", fontSize=8.2, leading=11, leftIndent=12, firstLineIndent=-8, spaceAfter=4))
styles.add(ParagraphStyle(name="CoverInfoA", fontName="Arial", fontSize=9.2, leading=15, alignment=TA_CENTER))


def P(text, style="BodyA"):
    return Paragraph(text, styles[style])


def table(rows, widths, header=True, font=6.7, padd=3.0):
    cooked = []
    for ri, row in enumerate(rows):
        sty = "CellHeadA" if header and ri == 0 else "CellA"
        cooked.append([cell if isinstance(cell, (Paragraph, Table)) else P(str(cell), sty) for cell in row])
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    ts = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#333333")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padd),
        ("RIGHTPADDING", (0, 0), (-1, -1), padd),
        ("TOPPADDING", (0, 0), (-1, -1), padd),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padd),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), colors.black), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        if len(rows) > 1:
            ts += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")])]
    t.setStyle(TableStyle(ts))
    return t


def bullet(text):
    return P("&#8226;&nbsp; " + text, "BulletA")


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Arial", 7)
    canvas.drawString(56, 22, "E")
    canvas.drawCentredString(A4[0] / 2, 22, f"Page {doc.page}")
    canvas.drawRightString(A4[0] - 56, 22, "31651")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=50, rightMargin=50,
    topMargin=43, bottomMargin=42, title="Predicting 30-Day Hospital Readmission",
    author="Introduction to Machine Learning",
)
story = []

# Page 1: cover
story += [Spacer(1, 155), P("Predicting 30-Day Hospital Readmission", "TitleA"),
          P("Kaggle Challenge Report", "SubTitleA"), Spacer(1, 46),
          P("<b>Task:</b> predict whether a patient is readmitted within 30 days of discharge", "CoverInfoA"),
          P("<b>Metric:</b> ROC-AUC on predicted probabilities", "CoverInfoA"),
          P("<b>Best public leaderboard score:</b> 0.74755 (v9)", "CoverInfoA"),
          P("<b>Highest public score reported on the leaderboard:</b> 0.75020", "CoverInfoA"),
          PageBreak()]

# Page 2: overview
story += [P("1. Overview", "HeadA"),
          P("The task is to predict the probability that a hospital stay is followed by a readmission within 30 days (readmit_30d). The training set contains 76,975 stays and the test set contains 32,812. About 10.1% of the training labels are positive. Kaggle scores submitted probabilities using ROC-AUC."),
          P("We joined and aggregated the linked encounter, patient, hospital, diagnosis, procedure, laboratory, vital-sign, medication, unit-transfer, discharge-planning and prior-utilization data. We cleaned inconsistent units and codes, retained informative missingness, and trained models on a common five-fold stratified split."),
          P("The strongest public result was the corrected LightGBM/EBM rank blend, v9, at 0.74755. Its five-fold out-of-fold (OOF) AUC was 0.752575. Other blends achieved similar OOF scores but lower public scores: v10 scored 0.74752, v12 scored 0.74728 and v11 scored 0.74714."),
          P("The top public score reported during the competition was 0.75020. The public leaderboard used about 30% of the test stays; the final leaderboard uses the remaining 70%, so the private score and final rank were not yet known when this report was prepared."),
          P("The public results support regularised gradient boosting as the most useful approach for these data. Differences of a few ten-thousandths in OOF AUC did not reliably predict the public score."),
          PageBreak()]

# Page 3: pipeline
story += [P("2. Data pipeline", "HeadA"), P("2.1 Tables used and dropped", "SubHeadA"),
          P("We used all 11 encounter and patient data tables: index_encounters, patients, hospitals, diagnoses, procedures, lab_results, vitals, medications, unit_transfers, discharge_planning and prior_utilization. We used icd10_lookup to add diagnosis chapter and chronic-condition information. data_dictionary.csv was reference material; train.csv and test.csv supplied the split and labels. We dropped no source table."),
          P("2.2 Joins", "SubHeadA"),
          P("index_encounters is the encounter spine. Patient and hospital attributes were joined to it; every many-row table was first reduced to one row per encounter_id, then left-joined. Prior-utilization visits were restricted to dates before the index admission. The label was attached only for training. Both splits used the same row-level cleaning and feature construction."),
          P("2.3 Aggregation of many-row tables", "SubHeadA")]
pipeline_rows = [
    ["Table", "Rows / coverage", "Features aggregated per stay"],
    ["diagnoses", "451,976 / 100%", "Counts; primary code and chapter; chapter counts; common-code indicators; chronic burden and Charlson-style score."],
    ["procedures", "19,547 / 17.1%", "Count, first code, distinct codes, and timing relative to admission and discharge."],
    ["lab_results", "1,668,783 / 100%", "Per-test last, mean, min, max, count, first-to-last change, discharge timing, abnormal flags and trends."],
    ["vitals", "2,267,367 / 100%", "Per-vital last, mean, min, max, standard deviation, count and change; discharge-window summaries and abnormal counts."],
    ["medications", "447,278 / 98.0%", "Counts, distinct and common drug names, continuation, route, drug class, and start/stop timing."],
    ["unit_transfers", "211,097 / 100%", "Number and type of transfers, hours per unit, ICU use and discharge-relative timing."],
    ["discharge_planning", "97,552 / 88.9%", "Fields retained, plus row-present and missing-follow-up indicators."],
    ["prior_utilization", "714,062 / 99.5% of patients", "ED, inpatient and outpatient counts by lookback window; time since prior visits; same- versus other-hospital use."],
]
story += [table(pipeline_rows, [82, 82, 331]), P("Table 1. Core multi-row tables and their encounter-level summaries.", "CaptionA"), PageBreak()]

# Page 4: final dataset
story += [P("3. Final dataset", "HeadA"),
          P("The final feature table contains 109,787 rows: 76,975 training and 32,812 test stays. It stores 461 predictors (442 numeric and 19 categorical). Four additional fold-local target encodings are generated during model fitting. Two high-cardinality raw columns, attending_id and proc_first, are then dropped, giving 463 model inputs. The target and encounter_id are not model predictors."),
          P("The complete feature inventory is attached as experiments/final_columns_v4.csv. It lists each of 465 stored or model-time columns, its source, construction, type and percentage missing."),
          P("Feature counts by first appearance", "SubHeadA")]
story += [table([["Feature block", "Columns", "Notes"],
                 ["Version 1", "297", "Core joins and encounter-level aggregates."],
                 ["Version 2", "17", "Clinical burden scores, ratios and data-quality indicators."],
                 ["Version 3", "145", "Discharge state and timing features for labs, vitals, medications, procedures, transfers and prior visits."],
                 ["Version 4", "2", "Final corrected pipeline additions."],
                 ["Fold-local target encodings", "4", "Hospital, attending, primary diagnosis and first procedure; computed within each training fold."],
                 ["Inventory total", "465", "461 stored predictors plus four model-time encodings."]], [137, 55, 303]),
          P("Table 2. Feature inventory. The two high-cardinality raw identifiers are documented but excluded from the fitted model matrix.", "CaptionA"),
          P("The model input is assembled inside each fold. Category levels, target encodings and hospital/attending counts are learned from that fold's training rows only; unseen validation or test categories receive the appropriate missing or global fallback value."),
          PageBreak()]

# Page 5: data quality
story += [P("4. Data quality", "HeadA"), P("The source tables contain inconsistent formats, duplicate records and implausible measurements. The same corrections are applied to training and test data.")]
quality_rows = [
    ["Problem", "Evidence", "Handling"],
    ["Exact duplicates", "Diagnoses 902; labs 3,331; vitals 2,265; medications 893; prior visits 1,284.", "Exact duplicates removed before aggregation."],
    ["Mixed laboratory units", "Albumin, creatinine, glucose and hemoglobin appear in mixed units.", "Convert to a common unit per test before summarising."],
    ["Implausible laboratory results", "5,194 results (0.31%) outside physiological ranges after conversion.", "Set to missing; counts and measurement summaries retain information about data quality."],
    ["Temperature units", "66,538 of 378,097 readings (17.6%) exceed 45 degrees.", "Convert Fahrenheit values to Celsius."],
    ["Impossible vital signs", "369 readings, including SpO2 of 0 and heart rate of 300.", "Set impossible readings to missing."],
    ["Malformed ICD-10 codes", "182,326 codes lack the decimal; 124,211 rows initially miss the lookup.", "Insert the decimal after the third character and join the lookup."],
    ["Drug-name variants", "122 spellings for 52 drugs.", "Normalise case and whitespace before building drug features."],
    ["Procedure systems", "CPT and CPT-4 both appear.", "Treat them as one code system."],
    ["Placeholder categories", "?, U, Not Available and Not Mapped occur in categorical fields.", "Convert placeholders to missing; standardise insurance-type case."],
    ["Implausible dates and values", "Very early birth years, zero/very long stays, remote follow-up dates.", "Set invalid values to missing; parse each table's date format explicitly."],
]
story += [table(quality_rows, [112, 175, 208]), P("Table 3. Data-quality issues and corrections.", "CaptionA"), PageBreak()]

# Page 6: missingness
story += [P("5. Missing values", "HeadA"),
          P("Missingness often reflects clinical decisions—for example, specialised tests are ordered for patients who need them. The explanations below are interpretations of the dataset, not observed causes."),
          table([["Where", "Missingness / pattern", "Interpretation and response"],
                 ["Admission weight", "87.0% missing; readmission 12.6% when recorded vs 9.7% otherwise.", "Recorded for a subset of stays; preserve missingness and an explicit flag."],
                 ["Attending specialty", "46.4% missing; readmission rates are similar.", "Likely source-system gaps; retain missing category/flag."],
                 ["Procedures", "No row for 82.9% of stays.", "Usually no procedure; convert absent procedure counts to zero, leave timing missing."],
                 ["Discharge planning", "No row for 11.1%; follow-up days missing in 41.8% of existing rows.", "May mean planning was not done or not recorded; add row-present and missing-follow-up flags."],
                 ["Selective labs", "Albumin ordered in 37.2%; HbA1c 16.0%; troponin 14.4%; BNP 12.3%; lactate 9.7%.", "Test ordering is informative; keep counts and missing values."],
                 ["Registration fields", "Insurance 17.2%; marital status 9.0%; language 7.1%; deprivation 4.7%; ZIP 4.0%.", "Retain missing categories; tree models can use missingness directly."],
                 ["Prior visits", "No prior-utilization rows for 0.5% of patients.", "Represents first-time patients; counts become zero and timing remains missing."]], [98, 165, 232]),
          P("Table 4. Main missing-data patterns. Rates are from the training data.", "CaptionA"),
          bullet("LightGBM and other boosted-tree models handle numeric missing values natively; explicit missing indicators are added when useful."),
          bullet("Imputation for models that require complete matrices is fitted inside each training fold, with missing indicators and a separate missing category for text."),
          bullet("Target encodings use an inner five-fold scheme on the outer training partition, so a training row does not receive an encoding derived from its own label."),
          PageBreak()]

# Page 7: methodology and top individual models
story += [P("6. Models", "HeadA"), P("6.1 How we tuned and evaluated", "SubHeadA"),
          bullet("Models use StratifiedKFold with five folds, shuffle=True and random_state=0 on the 76,975 training stays."),
          bullet("Preprocessing, target encodings, category alignment and provider counts are fitted from each outer training fold. LightGBM iteration counts use early stopping on that fold's validation portion."),
          bullet("The main LightGBM setting is strongly regularised: four leaves, at least 400 rows per leaf, 20% feature sampling, L2=60 and learning rate 0.04. Target-encoding smoothing was screened across M=10, 25, 50, 100 and 200."),
          bullet("Blends combine OOF predictions by rank, then map ranks back to a LightGBM probability scale. Their weights were screened on 80% of OOF rows. This is a useful diagnostic, but because the underlying OOF folds differ from that 80/20 split, it is not a fully nested estimate."),
          P("6.2 Three strongest individual components", "SubHeadA")]
single_rows = [
    ["Model", "Key hyperparameters", "Five-fold OOF AUC"],
    ["Corrected LightGBM", "4 leaves; min leaf 400; feature fraction 0.2; L2=60; learning rate 0.04; M=50 target encoding; early stopping.", "0.752053"],
    ["LightGBM boosted-stump GAM", "2 leaves; depth 1; learning rate 0.1; min child 100; feature fraction 0.5; L2=20; max_bin 63.", "0.751929"],
    ["Corrected EBM", "150 LightGBM gain-selected features per fold; 10 pairwise interactions; two outer bags.", "0.750382"],
]
story += [table(single_rows, [112, 330, 53]), P("Table 5. Best standalone components. None of these three component models was submitted alone; public scores below are for their corresponding ensemble submissions.", "CaptionA"),
          P("6.3 Other model and feature experiments", "SubHeadA"), PageBreak()]

# Page 8: other experiments
other_rows = [
    ["Experiment", "OOF AUC", "Outcome"],
    ["LightGBM with version 3 discharge-state features", "0.7518", "Below the corrected version 2 setting."],
    ["LightGBM with monotone constraints", "0.752043", "Effectively tied with the baseline; blend gain was negligible."],
    ["Positive-class weight 0.75", "0.751981", "Below the unweighted LightGBM; a small blend gain did not transfer to the public score."],
    ["Clinical interaction features", "0.751675", "Lower than baseline."],
    ["Hospital x primary-diagnosis target encoding", "0.7514", "Did not improve the baseline."],
    ["Hospital x admission-month target encoding", "0.751576", "Did not improve the baseline."],
    ["Hospital x service-line target encoding", "0.751616", "Did not improve the baseline."],
    ["Extra early/late lab and vital windows", "0.751442", "Existing features already included lab trends and discharge summaries; added windows did not help."],
    ["XGBoost DART", "0.746786", "Weaker than the standard LightGBM."],
    ["AdaBoost", "0.7378", "Weaker than the main boosted-tree models."],
    ["Logistic regression with pairwise numeric interactions", "0.7337", "No useful gain."],
]
story += [table(other_rows, [181, 58, 256]), P("Table 6. Selected corrected-pipeline model and feature tests. Scores use the same five OOF folds unless otherwise stated.", "CaptionA"),
          P("We also evaluated CatBoost, XGBoost, spline logistic regression, random forests, ExtraTrees, SVM, Naive Bayes, KNN, HistGradientBoosting and LambdaRank. These did not produce a stronger public submission. Diagnosis and medication SVD features and a large set of explicit pairwise features were also neutral or harmful."),
          P("A patient-ID target encoding was not pursued: each encounter has a unique patient_id, and there is no patient overlap between train and test."),
          PageBreak()]

# Page 9: submissions/ensembles
story += [P("6.4 Ensembles and scored submissions", "HeadA"),
          P("The four best public submissions are compared below. v9 remains the strongest public result. v11 obtained the highest OOF AUC, but its leaderboard score was lower; this illustrates why small CV differences should not be treated as reliable leaderboard gains."),
          table([["ID", "Ensemble", "OOF AUC", "Public AUC"],
                 ["v9", "Corrected LightGBM + EBM; rank weights 0.65 / 0.35.", "0.752575", "0.74755"],
                 ["v10", "Corrected LightGBM with five-seed final fit + EBM; rank weights 0.65 / 0.35.", "0.7526", "0.74752"],
                 ["v11", "v9 blend + boosted-stump GAM; 32.5% GAM rank weight.", "0.752794", "0.74714"],
                 ["v12", "v9 blend + class-weight-0.75 LightGBM; 17.5% added rank weight.", "0.752600", "0.74728"]], [35, 333, 62, 65]),
          P("Table 7. Best four public submissions by leaderboard score, with their OOF AUC.", "CaptionA"),
          P("The v9 and v10 public scores differ by only 0.00003. v11 and v12 had slightly higher OOF scores than v9 but lower public scores. The remaining experimental blends changed the reserved OOF holdout by less than 0.0001; these differences were too small to trust."),
          P("Rank averaging suits ROC-AUC because it evaluates ordering rather than probability calibration. Each submission still contains probabilities, obtained by mapping the blend ranks back to the corrected LightGBM scale."),
          PageBreak()]

# Page 10: results and discussion
story += [P("7. Results and discussion", "HeadA"), P("7.1 Leaderboard result", "SubHeadA"),
          P("Our best public leaderboard score is 0.74755 from v9. The top score reported on the public leaderboard was 0.75020. v10 scored 0.74752, v12 scored 0.74728 and v11 scored 0.74714. The final 70% private leaderboard score was not available when this report was prepared."),
          P("The public score is about 0.005 below the v9 OOF AUC. The public leaderboard uses about 30% of test encounters; the final ranking uses the other 70%. The public ordering therefore gives a useful, but noisy, comparison."),
          P("7.2 Which approach performed best and why", "SubHeadA"),
          P("The corrected LightGBM/EBM rank blend v9 is our best submission. LightGBM provides the strongest standalone ranking, handling mixed numeric and categorical features and missing values with shallow, heavily regularised trees. EBM adds a different additive structure. The 0.65 / 0.35 blend reached OOF AUC 0.752575 and public AUC 0.74755."),
          P("The result did not improve simply by adding more models. The stump-GAM candidate had the highest OOF score (0.752794) but scored 0.74714 publicly. The class-weighted blend reached OOF 0.752600 but scored 0.74728. Both were below v9 on the leaderboard."),
          P("7.3 Why the score may be difficult to raise", "SubHeadA"),
          bullet("Several different model families and feature sets land near the same OOF range, around 0.75."),
          bullet("The strongest model and blend differences are only a few ten-thousandths of AUC; the latest public scores did not follow those tiny CV gains."),
          bullet("Hospital, diagnosis, utilization and clinical features are all represented, and the main data-quality corrections and temporal summaries are already included."),
          P("These results suggest that the remaining performance gap is difficult to capture reliably with the available data. They do not prove that a better model is impossible."),
          PageBreak()]

# Page 11: limitations and future
story += [P("7.4 Limitations", "SubHeadA"),
          bullet("Early stopping uses each validation fold to select the number of boosting rounds, giving a small optimism to the reported OOF scores."),
          bullet("Blend weights were screened on a subset of OOF predictions, but the base OOF folds were not nested around that subset. This is not a fully independent estimate of the whole blend-selection process."),
          bullet("The public leaderboard covers only 30% of test encounters. The final private score may reorder submissions."),
          bullet("Scores vary by split and run. Differences under 0.001 should not be over-interpreted."),
          P("7.5 What we would try next", "SubHeadA"),
          bullet("Evaluate a logistic or non-negative meta-learner using fully nested stacking: for each untouched outer fold, generate fresh inner-fold predictions using only the outer training data, train the meta-learner on those predictions, then score the outer fold."),
          bullet("Repeat the main comparison over several outer split seeds to estimate how stable the small score differences are."),
          bullet("Inspect private leaderboard results when released; the v9 blend is the best-supported submission based on the evidence currently available."),
          P("A future stack should be judged on genuinely outer-fold predictions. The previously reported stacked score from another project is not used here because its base OOF predictions and stack evaluation folds did not match."),
          PageBreak()]

# Page 12: submission history
story += [P("Appendix A. Submission history", "HeadA"),
          P("The table lists all twelve submissions and their recorded five-fold OOF and public leaderboard scores. v9 is the highest-scoring public submission."),
          table([["ID", "Model / blend", "OOF AUC", "Public AUC"],
                 ["v1", "LightGBM, 3 seeds, 15 leaves.", "0.7494", "0.74326"],
                 ["v2", "Regularised LightGBM, version 2 features, target encoding.", "0.7521", "0.74589"],
                 ["v3", "CatBoost, 2 seeds.", "0.7476", "0.74337"],
                 ["v4", "0.8 LightGBM + 0.2 logistic regression.", "0.7525", "0.74682"],
                 ["v5", "XGBoost, 3 seeds, target encoding.", "0.7502", "0.74397"],
                 ["v6", "0.4 LightGBM v2 + 0.4 LightGBM v3 + 0.2 logistic regression.", "0.7527", "0.74685"],
                 ["v7", "Kaggle notebook blend: two LightGBM feature sets + logistic regression.", "0.7528", "0.74679"],
                 ["v8", "LightGBM v2/v3 + spline logistic regression + EBM.", "0.7528", "0.74695"],
                 ["v9", "Corrected LightGBM + EBM rank blend.", "0.752575", "0.74755"],
                 ["v10", "Five-seed corrected LightGBM + EBM rank blend.", "0.7526", "0.74752"],
                 ["v11", "v9 + boosted-stump GAM rank blend.", "0.752794", "0.74714"],
                 ["v12", "v9 + class-weight-0.75 LightGBM rank blend.", "0.752600", "0.74728"]], [34, 338, 61, 62]),
          P("Table 8. Kaggle submission history.", "CaptionA"),
          P("The top three public scores were v9 (0.74755), v10 (0.74752) and v12 (0.74728)."),
          PageBreak()]

# Page 13: appendix and deliverables
story += [P("Appendix B. Attached files and reproducibility", "HeadA"),
          bullet("experiments/final_columns_v4.csv: 465-row inventory of the corrected feature table, including source, construction method, feature version, dtype and missingness."),
          bullet("submissions/kaggle_readmission_notebook_v2.ipynb: notebook for the corrected v9 feature/model pipeline. The earlier kaggle_readmission_notebook.ipynb records the previous end-to-end version."),
          bullet("src/: feature construction, cross-validation and model scripts. README.md describes the project layout; experiments/results.csv records model and submission scores."),
          bullet("Challenge.pdf: assignment brief."),
          P("All experiments use only the competition data. Submission files follow sample_submission.csv and contain predicted probabilities for every test encounter_id."),
          P("The report and code do not include the raw competition data. The data files remain local and are excluded from Git."),
          P("The final public score is known, but the private leaderboard score is pending. Any final ranking statement should be updated once Kaggle releases the private leaderboard."),
          Spacer(1, 14), P("End of report", "CaptionA")]

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
