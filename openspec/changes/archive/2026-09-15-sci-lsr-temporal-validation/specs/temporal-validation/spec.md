## ADDED Requirements

### Requirement: Development and temporal validation are isolated
The system SHALL fit preprocessing, feature selection, model parameters, calibration, and thresholds using group1 only, then evaluate the frozen result once on group2.

#### Scenario: First locked evaluation
- **WHEN** the freeze manifest exists and group1 development completes
- **THEN** the evaluator SHALL generate group2 predictions and metrics without fitting on group2

#### Scenario: Attempted group2 tuning
- **WHEN** a run requests tuning, feature selection, threshold selection, or recalibration using group2
- **THEN** the system SHALL reject the run and explain that group2 is locked

### Requirement: Validation is labeled honestly
Reports SHALL label the current group2 analysis as retrospective initial temporal validation and SHALL state that nine validation events limit precision.

#### Scenario: Report generation
- **WHEN** validation metrics are written
- **THEN** the report SHALL include group sizes, event counts, uncertainty, and the retrospective temporal-validation label

### Requirement: Evaluation is auditable and immutable
The validation run SHALL record the freeze manifest, dataset/model hashes, code revision, configuration, and evaluation timestamp, and SHALL prevent a second result from overwriting the first locked run.

#### Scenario: Repeated evaluation
- **WHEN** an existing locked validation record is present
- **THEN** a new evaluation SHALL write a separately identified run or stop without replacing the original record

