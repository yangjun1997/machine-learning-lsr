## MODIFIED Requirements

### Requirement: Public page exposes only the five approved inputs
The Streamlit page SHALL expose controls for age, disease duration, acupuncture, zygomatic-branch LSR, and mandibular-branch LSR, and SHALL not expose botulinum-toxin history, identifiers, free-text clinical notes, file uploads, or the other omitted predictors.

#### Scenario: Form rendering
- **WHEN** a user opens the prediction page
- **THEN** the page SHALL show exactly the five Group 1 SHAP-ranked clinical inputs with human-readable labels and explicit categorical choices
