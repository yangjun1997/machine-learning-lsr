## Purpose
Define the approved five-variable research model, its leakage-free training and frozen evaluation contract, and its bounded-probability inference behavior.

## Requirements

### Requirement: Reduced model uses exactly the approved five predictors
The reduced prediction model SHALL require exactly `duration`, `botox`, `acupuncture`, `zyg_lsr`, and `man_lsr`, with no sex, age, side, hypertension, diabetes, BMI, KPS, outcome, identifier, or post-outcome field in the inference contract.

#### Scenario: Five-variable inference frame
- **WHEN** a valid prediction request is converted to a model frame
- **THEN** the frame SHALL contain the five canonical fields in the frozen schema order and SHALL contain no outcome or patient-identifier column

### Requirement: Reduced model preserves categorical LSR semantics
The reduced model SHALL treat `zyg_lsr` and `man_lsr` as nominal categorical variables with source codes 1, 2, and 3, and SHALL not interpret those codes as ordered numeric distances.

#### Scenario: LSR encoding
- **WHEN** a training or inference frame contains an LSR field
- **THEN** the preprocessing path SHALL encode its categories independently and SHALL preserve the three source states without ordinal arithmetic

### Requirement: Reduced model is selected and frozen without Group 2 leakage
The reduced model SHALL use the existing date-derived Group 1/Group 2 split, stratified nested cross-validation within Group 1, Brier score as the primary selection metric, and log-loss as the tie-breaker; Group 2 SHALL remain unavailable to model selection, tuning, feature selection, and recalibration.

#### Scenario: Model freeze
- **WHEN** reduced-model training completes
- **THEN** exactly one selected estimator SHALL be refit on Group 1, frozen as a separate artifact, and accompanied by the selected model name, feature schema, dataset hash, model hash, and training metadata

### Requirement: Reduced model receives locked temporal evaluation
The reduced model SHALL be evaluated once on Group 2 after freezing, and its report SHALL include sample size, event count, AUROC, AUPRC, Brier score, log-loss, calibration estimates where estimable, and uncertainty or an explicit non-estimable status.

#### Scenario: Temporal validation report
- **WHEN** the frozen reduced model is evaluated on Group 2
- **THEN** the report SHALL identify the evaluation as internal temporal validation and SHALL state that no Group 2 tuning or recalibration occurred

### Requirement: Reduced inference returns a bounded probability
The reduced inference adapter SHALL return one finite probability in the closed interval [0, 1] for every valid five-variable request and SHALL fail visibly rather than returning a result when artifact loading or schema validation fails.

#### Scenario: Valid prediction
- **WHEN** all five inputs pass validation and the frozen artifact loads successfully
- **THEN** the adapter SHALL return a finite predicted probability between 0 and 1

#### Scenario: Invalid prediction
- **WHEN** a required field is missing, non-finite, outside its allowed categorical set, or the artifact cannot load
- **THEN** the adapter SHALL return a user-visible validation or availability error and SHALL not fabricate a probability
