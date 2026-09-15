## Context

The repository is an initial Python/uv package with no analysis implementation. The source workbook `datas/580-analysis.xlsx` contains 580 patients and 24 fields: group1 has 476 patients and 58 events; group2 has 104 patients and 9 events. Group labels agree with admission-date ranges, but patient hashes, encounter identifiers, event dates, and follow-up dates are not present. The analysis must therefore support a fixed one-year binary endpoint and retrospective temporal validation, while refusing to infer survival times.

## Goals / Non-Goals

**Goals:**

- Build a read-only, reproducible pipeline from the workbook to audited features and reports.
- Define leakage-safe M0, M1, and M2 prediction-time feature sets.
- Develop models only on group1 and evaluate the frozen specification once on group2.
- Evaluate a broad but pre-registered model pool, while retaining penalized Logistic regression as the interpretability reference and selecting by proper scoring rather than an AUC leaderboard.
- Report calibration, Brier/log-loss, discrimination, DCA, confidence intervals, subgroup checks, and sensitivity analyses.
- Preserve raw-data provenance through configuration, hashes, model metadata, and tests.

**Non-Goals:**

- No clinical API, automated deployment, retraining, or monitoring service.
- No Cox, competing-risk, or time-dependent survival analysis without event/follow-up dates.
- No use of group2 for model selection, tuning, threshold selection, or recalibration.
- No automatic all-pairs interaction mining, post-hoc model cherry-picking, or group2-based algorithm leaderboard.
- No upload or public publication of patient-level data.

## Decisions

### 1. Use explicit admission-date group locking

Parse Excel serial dates, verify group1 is 2020-03-08–2022-06-25 and group2 is 2022-09-26–2023-08-07, and record the empty July–August 2022 washout. Preserve the source group label for audit but derive the locked split from the date rule. This is safer than trusting a manually entered group column alone.

### 2. Treat LSR encodings as categorical and avoid redundant primary predictors

The primary M1 representation uses one-hot `zyg_lsr` and `man_lsr`; a separate sensitivity model uses the three-state `lsr_combo`. This avoids placing the two original LSR variables and their deterministic combination into one unrestricted model. Invalid LSR values block the run.

### 3. Bind preprocessing to every fitted model

Use scikit-learn `Pipeline`/`ColumnTransformer` so imputation, scaling, and one-hot encoding are fitted inside each development fold. Group2 is transformed only with frozen group1-fitted objects.

### 4. Use a broad candidate pool with staged nested selection

The pre-registered pool SHALL include: prevalence baseline; Logistic regression with L1, L2, and elastic-net penalties; LDA/QDA; Gaussian Naive Bayes; KNN; linear and RBF SVM with probability calibration; Decision Tree; Random Forest; ExtraTrees; AdaBoost; Gradient Boosting; HistGradientBoosting; and MLP. XGBoost, LightGBM, and CatBoost are optional adapters enabled only when explicitly installed and version-locked. All candidates use the same leakage-safe preprocessing and nested group1 cross-validation. Selection is primarily by Brier/log-loss, with AUROC/AUPRC, calibration, runtime, convergence, and stability reported as secondary evidence. If differences are uncertain, prefer the simpler calibrated model.

The pool is broad for scientific comparison, not a license to select a winner from group2. The final model is frozen once, and group2 is evaluated only after that freeze.

### 5. Separate confirmatory prediction from exploratory interaction analysis

The primary result is M1 temporal validation. Only LSR-residual × duration and LSR-residual × age are pre-specified exploratory interactions. Report effect sizes and confidence intervals without requiring group2 significance; group2's 9 events cannot support a confirmation gate.

### 6. Keep group2 evaluation immutable

Create a freeze manifest containing feature definitions, preprocessing, model search space, calibration, threshold, package lock, Git commit, and hashes before evaluation. The evaluator must write an immutable run record and reject a second evaluation under the same validation lock.

## Risks / Trade-offs

- **[Small validation event count]** → Report absolute event counts, wide intervals, and label results as initial temporal validation; do not claim definitive external validation.
- **[No patient hash]** → Require a documented limitation and check available hospital numbers for duplicates; block claims about cross-encounter deduplication.
- **[Group/date ambiguity]** → Fail the audit if non-washout group labels disagree with date-derived groups.
- **[Calibration instability]** → Prefer intercept/slope with intervals and calibration plots; avoid rigid pass/fail cutoffs.
- **[Interaction overinterpretation]** → Keep interactions exploratory and distinguish generation from confirmation in reports.
- **[Raw data exposure]** → Keep workbook outside generated artifacts and ignore raw/local data paths in version control.
- **[Large candidate pool]** → Use fixed search spaces, nested CV, convergence/runtime logs, and a pre-specified primary scoring rule; do not rank models by group2 results.
