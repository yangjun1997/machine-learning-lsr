# clinical-risk-modeling Specification

## Purpose
TBD - created by archiving change sci-lsr-temporal-validation. Update Purpose after archive.
## Requirements
### Requirement: Prediction-time feature sets are explicit
The modeling pipeline SHALL expose M0 preoperative, M1 intraoperative, and M2 post-day-7 feature sets, and SHALL exclude post-day-7 spasm and length of stay from M0 and M1.

#### Scenario: M1 feature build
- **WHEN** M1 features are requested
- **THEN** the result SHALL contain preoperative fields plus LSR fields and SHALL not contain post-day-7 spasm or length of stay

#### Scenario: M2 feature build
- **WHEN** M2 features are requested
- **THEN** the result MAY include post-day-7 spasm and length of stay and SHALL label the prediction time as post-day-7/discharge

### Requirement: Primary LSR representation is leakage-safe
The primary M1 model SHALL use categorical one-hot LSR features, and the combo representation SHALL be evaluated as a separate sensitivity representation rather than being silently treated as an ordered numeric variable.

#### Scenario: LSR encoding
- **WHEN** a model matrix is generated
- **THEN** LSR codes SHALL be one-hot encoded and no numeric 1<2<3 ordering SHALL be assumed

### Requirement: Model pool is broad and reproducible
The development pipeline SHALL evaluate a pre-registered broad model pool using nested group1 cross-validation and fixed configuration seeds. The pool SHALL include prevalence baseline, L1/L2/elastic-net Logistic regression, LDA/QDA, Gaussian Naive Bayes, KNN, calibrated linear/RBF SVM, Decision Tree, Random Forest, ExtraTrees, AdaBoost, Gradient Boosting, HistGradientBoosting, and MLP. XGBoost, LightGBM, and CatBoost MAY be enabled only as explicitly version-locked optional adapters.

#### Scenario: Broad model selection
- **WHEN** nested validation completes
- **THEN** the pipeline SHALL rank candidates primarily by Brier score/log-loss, retain AUROC/AUPRC and calibration as secondary metrics, and record convergence, runtime, and stability for every candidate

#### Scenario: Uncertain winner
- **WHEN** candidate uncertainty overlaps materially
- **THEN** the pipeline SHALL prefer the simpler calibrated model and record the decision without consulting group2 results

#### Scenario: Optional external adapter unavailable
- **WHEN** XGBoost, LightGBM, or CatBoost is not installed or fails version checks
- **THEN** the pipeline SHALL mark that candidate unavailable and continue with the core pool without silently changing the primary selection rule

### Requirement: Exploratory interactions are bounded
The pipeline SHALL support only pre-specified LSR-residual × duration and LSR-residual × age analyses as exploratory outputs and SHALL not perform automatic all-pairs interaction search.

#### Scenario: Interaction output
- **WHEN** an exploratory interaction is fitted
- **THEN** the report SHALL include effect size, confidence interval, direction, event counts, and an exploratory label

