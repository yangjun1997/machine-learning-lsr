## ADDED Requirements

### Requirement: Completed TRIPOD+AI checklist is supplied as supplementary material
The publication package SHALL generate a standalone completed TRIPOD+AI checklist using authoritative checklist item wording. Each applicable item SHALL identify its manuscript location, and absent or non-applicable information SHALL be marked explicitly rather than inferred or fabricated.

#### Scenario: Supplementary checklist is generated
- **WHEN** the checklist generation command completes
- **THEN** a readable DOCX checklist SHALL exist under `results/` and SHALL identify the manuscript title, TRIPOD+AI framework, checklist items, and reporting locations or explicit status

#### Scenario: Manuscript lacks required information
- **WHEN** a checklist item cannot be mapped to reported manuscript content
- **THEN** the checklist SHALL state `Not reported`, `Author input required`, or `Not applicable` with a concise reason

### Requirement: Corrected web-model reporting discloses validation reuse
Documentation of the corrected five-variable calculator SHALL identify Group 1 as the training cohort, the fixed Gradient Boosting algorithm, and the five SHAP-ranked predictors. It SHALL disclose that Group 2 had previously been used to evaluate the superseded prototype and SHALL not call the corrected-model Group 2 analysis first-use untouched or external validation.

#### Scenario: Corrected calculator is described
- **WHEN** a reader inspects the README, model report, manuscript implementation text, or supplementary checklist
- **THEN** the corrected predictors, algorithm, development cohort, temporal-validation role, and Group 2 reuse limitation SHALL be internally consistent
