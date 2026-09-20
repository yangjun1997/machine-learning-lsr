## MODIFIED Requirements

### Requirement: Reduced model uses exactly the approved five predictors
The reduced prediction model SHALL require exactly `age`, `duration`, `acupuncture`, `zyg_lsr`, and `man_lsr`, corresponding to the five highest-ranked source variables in the Group 1 SHAP analysis, with no sex, botulinum-toxin history, side, hypertension, diabetes, BMI, KPS, outcome, identifier, or post-outcome field in the inference contract.

#### Scenario: Five-variable inference frame
- **WHEN** a valid prediction request is converted to a model frame
- **THEN** the frame SHALL contain the five canonical fields in the frozen schema order and SHALL contain no outcome or patient-identifier column

### Requirement: Reduced model is selected and frozen without Group 2 leakage
The reduced model SHALL use the existing date-derived Group 1/Group 2 split and the same fixed `GradientBoostingClassifier` specification as the 12-predictor index model. It SHALL report Group 1 cross-validated performance for that fixed specification but SHALL NOT rerun candidate-algorithm selection for the reduced model. Group 2 SHALL remain unavailable to fitting, tuning, feature selection, parameter selection, threshold selection, and recalibration.

#### Scenario: Model freeze
- **WHEN** reduced-model training completes
- **THEN** the fixed Gradient Boosting estimator SHALL be refit on Group 1, frozen as a separate artifact, and accompanied by the model name, feature schema, dataset hash, model hash, and training metadata

### Requirement: Reduced model receives locked temporal evaluation
The corrected reduced model SHALL be evaluated on Group 2 only after freezing, and its report SHALL include sample size, event count, AUROC, AUPRC, Brier score, log-loss, calibration estimates where estimable, and uncertainty or an explicit non-estimable status. Because Group 2 was previously used to evaluate the superseded prototype, the corrected-model report SHALL describe the result as internal temporal validation and SHALL NOT describe it as first-use untouched validation.

#### Scenario: Temporal validation report
- **WHEN** the frozen corrected reduced model is evaluated on Group 2
- **THEN** the report SHALL state that no Group 2 fitting, tuning, or recalibration occurred and SHALL disclose that Group 2 had already been evaluated with the superseded prototype
