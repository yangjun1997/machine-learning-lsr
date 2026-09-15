## ADDED Requirements

### Requirement: Publication artifacts are reproducible and traceable
The system SHALL generate the canonical publication package from repository data and structured aggregate analysis outputs through a documented `uv run` command. It SHALL record, for every selected figure and table, the canonical filename, manuscript role, cohort, source artifact, and generating module or function.

#### Scenario: Publication package is generated
- **WHEN** the publication generation command completes successfully
- **THEN** five main figures, three main Markdown tables, the SCI Methods and Results document, and the publication manifest SHALL exist at their documented paths

#### Scenario: Source provenance is inspected
- **WHEN** a reader checks the publication manifest
- **THEN** each main figure and table SHALL identify an aggregate source and reproducible generation path rather than relying only on an unexplained copied image

### Requirement: Curation is non-destructive
The system SHALL preserve raw data, structured reports, model artifacts, and historical generated outputs while staging canonical publication files under `results/图/` and `results/表/`.

#### Scenario: Package is regenerated
- **WHEN** canonical publication files are regenerated
- **THEN** the generator SHALL replace only its documented generated targets and SHALL NOT delete or modify the source workbook, frozen model artifacts, or unrelated historical outputs

### Requirement: The main figure set is complete and scientifically labeled
The system SHALL provide five sequentially numbered main figures covering study flow, Group 1 model comparison, Group 1 SHAP interpretation, Group 2 validation calibration and exploratory decision curves, and pooled exploratory residual-LSR effect modification. Every figure SHALL have a detailed self-contained legend and SHALL avoid unsupported external, prospective, causal, or clinical-utility claims.

#### Scenario: Main figures are reviewed
- **WHEN** Figures 1 through 5 and their legends are inspected
- **THEN** the cohort, analysis role, displayed estimates, uncertainty convention, abbreviations, and exploratory status SHALL be identifiable without relying on the Results prose

#### Scenario: Temporal validation terminology is rendered
- **WHEN** Figure 1 or Figure 4 describes Group 2
- **THEN** it SHALL call Group 2 an internal temporal validation cohort and SHALL NOT call it an external validation cohort

### Requirement: Study-flow and model-selection claims match the implementation
Figure 1 and Figure 2 SHALL report the observed cohort date ranges, the fixed candidate-specification comparison used in Group 1, the prespecified freeze rule, and the one-time Group 2 evaluation without implying a non-empty hyperparameter search or Group 2 model selection.

#### Scenario: Development workflow is described
- **WHEN** a reader follows Figure 1 and the Figure 2 legend
- **THEN** the reader SHALL be able to distinguish fixed candidate comparison, model freezing, and locked temporal validation, and Group 2 SHALL not influence candidate ranking or recalibration

#### Scenario: Leading retained candidate models are reported
- **WHEN** Gradient Boosting and elastic-net logistic regression are compared
- **THEN** the package SHALL identify Gradient Boosting as the index model selected by the predefined rule while reporting both models' Brier score, log-loss, AUROC, and AUPRC without claiming broad superiority

### Requirement: Figure 5 remains explicitly exploratory and statistically coherent
Figure 5 SHALL describe a pooled 580-patient association analysis that is separate from model development and temporal validation. It SHALL use human-readable age-stratum labels, report the stratum denominators or provide them in the legend, and distinguish descriptive event rates from interaction inference.

#### Scenario: Age-stratified rates are displayed
- **WHEN** Figure 5 panel B is rendered
- **THEN** it SHALL show <60 years and >=60 years in publication language, remove programmatic labels, and associate the rates with 9/369, 43/99, 1/92, and 14/20 as applicable

#### Scenario: Interaction uncertainty is displayed
- **WHEN** Figure 5 panel A presents interaction odds ratios
- **THEN** the displayed confidence intervals and any P values SHALL have clearly compatible or explicitly separated inferential definitions, and unadjusted multiplicity and bootstrap details SHALL be disclosed

#### Scenario: Sparse interaction is not estimable
- **WHEN** a modifier stratum contains no informative outcome events for an interaction estimate
- **THEN** the figure and text SHALL mark the estimate as not estimable rather than displaying false precision such as OR 1.0 with 95% CI 1.0-1.0

### Requirement: The main table set is complete and editable
The system SHALL generate three canonical Markdown tables: baseline characteristics by cohort, Group 1 cross-validated candidate-model performance, and Group 2 frozen-model validation performance with uncertainty.

#### Scenario: Tables are generated
- **WHEN** the publication package is built
- **THEN** Table 1, Table 2, and Table 3 SHALL be available as Markdown source files and SHALL be linked or embedded in the SCI document

#### Scenario: Table notes are read independently
- **WHEN** a reader reviews any main table without the Methods section
- **THEN** its title and notes SHALL define the cohort, sample size, summary format, units, abbreviations, uncertainty, tests where applicable, missing-data handling, and descriptive or exploratory status

### Requirement: Methods accurately describe the executed analysis
The SCI Methods SHALL describe a single-center retrospective cohort study with internal temporal validation, the date-locked cohorts, outcome and predictor handling, independent one-hot encoding of the two branch-specific LSR variables, estimator-specific preprocessing, 14 fixed candidate specifications, Group 1 selection and freezing, Group 2 evaluation, bootstrap uncertainty, calibration, exploratory DCA, and exploratory interaction analysis.

#### Scenario: Feature and preprocessing text is checked
- **WHEN** the Methods text is compared with the model pipeline
- **THEN** it SHALL identify the 12 direct predictors, state that the two branch-specific LSR variables were independently one-hot encoded, omit derived LSR-combination variables, and SHALL not claim numeric standardization for Gradient Boosting

#### Scenario: Candidate comparison text is checked
- **WHEN** the Methods text is compared with the development configuration
- **THEN** it SHALL describe fixed hyperparameter candidate specifications evaluated by stratified Group 1 cross-validation and SHALL not claim substantive hyperparameter optimization when the search grid was empty

### Requirement: Results are bounded by the available evidence
The SCI Results SHALL report cohort flow, baseline characteristics, Group 1 candidate comparison and index-model selection, Group 2 validation, model interpretation, calibration and exploratory clinical utility, and exploratory interaction findings using values traceable to repository outputs.

#### Scenario: Group 2 validation is reported
- **WHEN** the primary validation paragraph is rendered
- **THEN** it SHALL report 104 patients and 9 events together with AUROC 0.899, AUPRC 0.361, Brier score 0.080, and log-loss 0.247 and their available bootstrap 95% confidence intervals

#### Scenario: Sparse validation evidence is interpreted
- **WHEN** calibration or clinical utility is discussed
- **THEN** the text SHALL report uncertainty and SHALL not claim definitive calibration, clinical benefit, transportability, or external validation from nine events

#### Scenario: Pooled interaction findings are reported
- **WHEN** the Figure 5 results are described
- **THEN** the text SHALL identify the analysis as pooled and hypothesis-generating and SHALL not use it to alter the frozen prediction model or claim confirmed effect modification

### Requirement: Unknown study facts are not fabricated
The SCI document SHALL use explicit `[[AUTHOR INPUT REQUIRED: ...]]` placeholders for unavailable center, ethics, consent, eligibility, outcome-adjudication, follow-up, and target-journal information.

#### Scenario: Required metadata is unavailable
- **WHEN** a required study fact cannot be verified from repository evidence
- **THEN** the document SHALL retain a specific author-input placeholder and SHALL list it in a completion checklist rather than inferring or inventing content

### Requirement: Publication outputs protect privacy
The publication package SHALL contain only aggregate results and non-identifying metadata and SHALL not expose direct identifiers or patient-level prediction rows.

#### Scenario: Publication artifacts are scanned
- **WHEN** privacy checks run against generated Markdown, tables, legends, and the manifest
- **THEN** no hospital number, direct patient identifier, or patient-level prediction record SHALL be present

### Requirement: Focused publication checks detect drift
The repository SHALL include a focused automated check for canonical file presence, resolvable Markdown links, key numeric consistency, privacy, and prohibited overclaims.

#### Scenario: Publication values or links drift
- **WHEN** a canonical file is missing, a link is broken, a primary value differs from its structured source, or prohibited validation language appears
- **THEN** the publication check SHALL fail with a message identifying the affected artifact or claim
