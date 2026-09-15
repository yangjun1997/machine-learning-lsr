## ADDED Requirements

### Requirement: Evaluation reports discrimination and proper scoring
The evaluation pipeline SHALL report AUROC, AUPRC, Brier score, and log-loss with uncertainty for development and locked temporal validation where estimable.

#### Scenario: Group2 metrics
- **WHEN** frozen predictions exist for all group2 patients
- **THEN** the report SHALL include absolute event counts and uncertainty for each estimable metric

### Requirement: Evaluation reports calibration
The pipeline SHALL report calibration-in-the-large/intercept, calibration slope when estimable, and a calibration plot, without applying a universal rigid pass/fail cutoff.

#### Scenario: Sparse-event calibration
- **WHEN** calibration estimates are unstable because group2 has few events
- **THEN** the report SHALL show wide intervals or an explicit non-estimable status rather than hiding the result

### Requirement: Clinical utility is threshold-specific
The pipeline SHALL produce decision-curve net benefit over a pre-specified clinically relevant threshold range and compare model, treat-all, and treat-none strategies.

#### Scenario: Unspecified threshold range
- **WHEN** no clinically approved threshold range is configured
- **THEN** DCA SHALL be marked exploratory and SHALL NOT be presented as proof of clinical benefit

### Requirement: Sensitivity and subgroup analyses are bounded
The evaluation SHALL support pre-specified LSR representation, calibration, temporal-fold, and limited clinical subgroup sensitivity analyses without using group2 to redefine the primary model.

#### Scenario: Sensitivity disagreement
- **WHEN** a sensitivity analysis materially differs from the primary result
- **THEN** the report SHALL display both results and explain the difference without silently replacing the primary analysis

