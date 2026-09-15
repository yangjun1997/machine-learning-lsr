## 1. Repository and configuration setup

- [x] 1.1 Add the analysis package layout under `src/mechine_learning_lsr/` for audit, features, modeling, evaluation, and artifact metadata.
- [x] 1.2 Add `configs/data_dictionary.yaml`, `configs/split.yaml`, `configs/features_m0.yaml`, `configs/features_m1.yaml`, and `configs/evaluation.yaml` with the frozen field names, date windows, seeds, metrics, and threshold range.
- [x] 1.3 Add and version-lock the core modeling dependencies through `uv`; keep XGBoost/LightGBM/CatBoost as explicitly optional adapters and record availability.
- [x] 1.4 Extend ignore rules so raw/local patient data and generated artifacts are not committed accidentally.

## 2. Data audit and contract

- [x] 2.1 Implement read-only workbook loading for `datas/580-analysis.xlsx` with Excel serial-date parsing.
- [x] 2.2 Implement schema, row/column, missingness, duplicate hospital-number, binary-outcome, and LSR-code checks.
- [x] 2.3 Implement date-derived group1, washout, and group2 checks against the source group labels.
- [x] 2.4 Emit `reports/data_audit.md` and aggregate audit tables without raw patient rows or direct identifiers.
- [x] 2.5 Add data-contract tests covering valid input, invalid LSR values, conflicting date/group labels, duplicate identifiers, and schema drift.

## 3. Feature construction

- [x] 3.1 Implement explicit M0 preoperative, M1 intraoperative, and M2 post-day-7 feature manifests.
- [x] 3.2 Implement categorical one-hot encoding for `zyg_lsr` and `man_lsr` and a separate `lsr_combo` representation with documented mapping.
- [x] 3.3 Enforce exclusion of post-day-7 spasm and length of stay from M0/M1.
- [x] 3.4 Add bounded exploratory interaction features for LSR residual × duration and LSR residual × age only.
- [x] 3.5 Add leakage tests proving that preprocessing is fitted only inside group1 training folds.

## 4. Group1 model development

- [x] 4.1 Implement a common estimator interface for prevalence baseline, L1/L2/elastic-net Logistic, LDA/QDA, GaussianNB, KNN, calibrated linear/RBF SVM, DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoosting, HistGradientBoosting, and MLP.
- [x] 4.2 Add optional version-checked adapters for XGBoost, LightGBM, and CatBoost without making them required for a valid run.
- [x] 4.3 Implement nested group1 cross-validation with fixed seeds and proper-scoring primary selection by Brier score/log-loss across the full available pool.
- [x] 4.4 Record convergence, warnings, runtime, calibration status, and stability for every candidate; unavailable optional models must be explicit.
- [x] 4.5 Fit calibration methods and choose a threshold policy using group1 outputs only.
- [x] 4.6 Produce group1 cross-validation metrics, uncertainty summaries, model-selection rationale, and feature representation sensitivity results.
- [x] 4.7 Fit the selected specification on all group1 data and write a freeze manifest before any group2 evaluation.

## 5. Locked temporal validation

- [x] 5.1 Implement a one-time group2 evaluator that consumes only the frozen manifest and group2 features.
- [x] 5.2 Prevent group2 tuning, feature selection, threshold selection, recalibration, or overwrite of an existing locked run.
- [x] 5.3 Record dataset/model hashes, code revision, configuration, timestamp, and run identifier for the validation.
- [x] 5.4 Generate the locked group2 prediction table in protected aggregate/pseudonymous form.

## 6. Statistical evaluation and figures

- [x] 6.1 Implement AUROC, AUPRC, Brier, log-loss, sensitivity, specificity, PPV, NPV, F1, and bootstrap confidence intervals where estimable.
- [x] 6.2 Implement calibration intercept/slope estimates, calibration plots, patient-level frozen-model probability intervals, and explicit sparse-event caveats.
- [x] 6.3 Implement threshold-specific DCA against treat-all and treat-none using a configured clinically relevant range.
- [ ] 6.4 Implement bounded subgroup and sensitivity analyses without redefining the frozen primary model.
- [ ] 6.5 Implement exploratory interaction effect estimates with event-count tables, confidence intervals, and exploratory labeling.
- [x] 6.6 Generate `reports/temporal_validation.md`, publication tables, and vector/raster figures without exporting patient-level data.

## 7. Reproducibility and quality gates

- [x] 7.1 Add dataset, configuration, model, and report SHA-256 metadata plus software-version capture.
- [x] 7.2 Add privacy checks for generated reports and artifacts.
- [x] 7.3 Add an end-to-end smoke test from workbook audit through group1 freeze using a small synthetic fixture; keep raw patient data out of fixtures.
- [x] 7.4 Run the full test suite and verify the documented commands reproduce the reports from a clean `uv` environment.
- [x] 7.5 Document that survival analysis, clinical API deployment, automatic retraining, and prospective validation are out of scope until their required data and approvals exist.
