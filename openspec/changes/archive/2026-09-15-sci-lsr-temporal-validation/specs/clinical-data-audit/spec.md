## ADDED Requirements

### Requirement: Workbook audit is reproducible
The audit pipeline SHALL read `datas/580-analysis.xlsx` without modifying it, record its SHA-256 hash, and emit row count, column count, field inventory, missingness, duplicate hospital numbers, outcome counts, LSR counts, date ranges, and group/date consistency.

#### Scenario: Valid 580-patient workbook
- **WHEN** the workbook contains 580 patient rows and the expected 24 fields
- **THEN** the audit SHALL produce a passing report with the observed counts and hash

#### Scenario: Structural mismatch
- **WHEN** required fields are missing, duplicated, or the row/column schema changes unexpectedly
- **THEN** the audit SHALL fail before feature generation and identify the offending fields

### Requirement: Clinical codes are validated
The audit SHALL require the outcome to be binary 0/1 and both LSR variables to contain only categorical codes 1, 2, or 3; invalid values SHALL block downstream modeling.

#### Scenario: Invalid LSR value
- **WHEN** any LSR value is outside 1, 2, or 3
- **THEN** the audit SHALL fail and report the row identifier and offending value

### Requirement: Date-derived temporal split is checked
The audit SHALL parse the admission date, derive group1, washout, and group2 windows, and reject non-washout rows whose source group label disagrees with the date-derived group.

#### Scenario: Consistent temporal labels
- **WHEN** all group1 and group2 labels agree with their date windows
- **THEN** the audit SHALL record the split counts and the empty or populated washout interval

#### Scenario: Conflicting label
- **WHEN** a non-washout patient has a source label inconsistent with the locked date rule
- **THEN** the audit SHALL fail before model fitting

