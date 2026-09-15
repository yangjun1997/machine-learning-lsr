# reproducible-research-artifacts Specification

## Purpose
TBD - created by archiving change sci-lsr-temporal-validation. Update Purpose after archive.
## Requirements
### Requirement: Research artifacts are versioned
The pipeline SHALL write configuration, package versions, code revision, dataset hash, model hash, freeze manifest, report metadata, and test results for each analysis lock.

#### Scenario: Successful freeze
- **WHEN** the group1 model and evaluation specification are frozen
- **THEN** all required metadata SHALL be present before group2 evaluation is allowed

### Requirement: Raw patient data is protected
Generated reports and artifacts SHALL not contain direct patient identifiers or raw row-level patient data, and repository ignore rules SHALL exclude local raw-data paths and generated artifacts as configured.

#### Scenario: Artifact inspection
- **WHEN** a report or artifact is created
- **THEN** it SHALL contain aggregate results or pseudonymous metadata only and SHALL pass the privacy check

