## Purpose
Define the public Streamlit research predictor's approved inputs, validation, non-clinical output, privacy boundaries, disclaimer, and deployment safeguards.

## Requirements

### Requirement: Public page exposes only the five approved inputs
The Streamlit page SHALL expose controls for disease duration, prior botulinum toxin treatment, acupuncture, zygomatic-branch LSR, and mandibular-branch LSR, and SHALL not expose identifiers, free-text clinical notes, file uploads, or the seven omitted predictors.

#### Scenario: Form rendering
- **WHEN** a user opens the prediction page
- **THEN** the page SHALL show exactly the five approved clinical inputs with human-readable labels and explicit categorical choices

### Requirement: User input is validated before prediction
The application SHALL reject missing, non-finite, negative, or otherwise invalid numeric values and SHALL reject categorical values outside the documented source states before calling the model.

#### Scenario: Invalid form submission
- **WHEN** a user submits an invalid or incomplete form
- **THEN** the page SHALL identify the validation problem and SHALL not display a prediction probability

### Requirement: Prediction output is probability-only and non-clinical
The application SHALL display the reduced model's predicted probability, model identity/version metadata, and validation context, but SHALL not display a clinical risk category, treatment recommendation, diagnosis, or approved decision threshold.

#### Scenario: Valid form submission
- **WHEN** a user submits five valid inputs and the artifact is available
- **THEN** the page SHALL display one bounded predicted probability and state that it is a research-model output

### Requirement: Research disclaimer is prominent and repeated
The application SHALL display a prominent disclaimer on the landing page and alongside every prediction stating that the output is for scientific research only, is not diagnosis, treatment advice, medical advice, clinical evidence, or a basis for clinical decision-making, and has not undergone external validation.

#### Scenario: Disclaimer visibility
- **WHEN** a user opens the page or receives a result
- **THEN** the research disclaimer SHALL be visible without requiring navigation to a separate help page

### Requirement: Public prototype does not retain submitted clinical values
The application SHALL not persist submitted inputs or prediction history, SHALL not send entered values to third-party analytics, and SHALL warn users not to enter identifiable patient information.

#### Scenario: Anonymous research use
- **WHEN** a user submits a prediction
- **THEN** the app SHALL compute the result in memory, SHALL avoid writing the submitted values to application-managed storage or logs, and SHALL show the no-identifiers warning

### Requirement: Public deployment uses secure transport and fail-closed availability
The deployment instructions SHALL require public HTTPS, and the application SHALL show an explicit unavailable state rather than returning a result if the reduced artifact or runtime dependencies cannot be loaded.

#### Scenario: Artifact unavailable
- **WHEN** the app cannot load the frozen reduced model
- **THEN** the page SHALL show a clear service-unavailable message and SHALL not display a stale or fabricated prediction
