## Context

The repository currently freezes a Gradient Boosting pipeline trained on 12 M1 variables and evaluates it with a locked Group 2 temporal cohort. The requested public prototype has a narrower input contract: disease duration, prior botulinum toxin treatment, acupuncture, zygomatic-branch LSR, and mandibular-branch LSR. The existing 12-variable artifact cannot be used with only five submitted fields because its fitted column transformer expects the original feature schema.

The prototype will be a Streamlit application backed by a separately trained and frozen five-variable model. It is a research interface, not a clinical decision-support product. The public deployment must not solicit identifiers or retain submitted clinical values.

## Goals / Non-Goals

**Goals:**

- Train and freeze a reproducible five-variable model using the existing Group 1 selection and Group 2 locked temporal-validation conventions.
- Preserve categorical treatment of both LSR variables; numeric codes must not become ordinal model inputs.
- Provide a single-page Streamlit form with Chinese labels, explicit choices, and probability-only output.
- Make the app deployable over public HTTPS while avoiding application-level persistence, uploads, third-party analytics, and patient identifiers.
- Show research scope, model version, validation context, and disclaimer text before and after prediction.

**Non-Goals:**

- Reusing the 12-variable model by filling omitted inputs with arbitrary defaults.
- Replacing the existing 12-variable publication artifact or changing its reported results.
- Displaying a clinical risk category, treatment threshold, diagnosis, treatment recommendation, or claimed clinical utility.
- Accepting names, hospital numbers, free-text clinical notes, files, or longitudinal patient records.
- Performing external validation, prospective validation, recalibration, or automated model updating.

## Decisions

### 1. Create a separate `reduced5` model variant

Add a dedicated feature schema with one numeric field (`duration`) and four categorical fields (`botox`, `acupuncture`, `zyg_lsr`, `man_lsr`). Train the reduced model from the source workbook rather than adapting the 12-variable artifact. Keep artifacts separate, for example `artifacts/reduced5_model.joblib` and `artifacts/reduced5_freeze_manifest.json`, so the two input contracts cannot be confused.

**Alternative considered:** retain the 12-variable model and impute the seven hidden variables. Rejected because the resulting prediction path would not match the validated model specification and would silently add unsupported assumptions.

### 2. Reuse the existing model-selection and temporal split rules

Use the same date-derived Group 1/Group 2 split, stratified nested cross-validation, core fixed candidate configurations, Brier score as the primary selection metric, log-loss as the tie-breaker, and locked Group 2 evaluation. Optional XGBoost/LightGBM/CatBoost adapters are excluded from the public reduced artifact unless their runtime dependencies are explicitly pinned. Group 2 must not influence selection, tuning, feature selection, or recalibration. Reports must include cohort counts, event counts, selected model, dataset/model hashes, and uncertainty where estimable.

**Alternative considered:** select the five-variable model using all 580 records or Group 2 performance. Rejected because it would invalidate the existing temporal-validation design.

### 3. Separate training features from online inference input construction

Add an inference adapter that constructs a one-row DataFrame from the five validated canonical fields and passes it to the frozen pipeline. Do not use the current evaluation helper unchanged because it is coupled to labeled outcome data. The adapter will share field names and categorical semantics with the training schema.

### 4. Use Streamlit for the first web surface

Place the app in a small Streamlit entry point that loads the reduced artifact once per process, renders five typed controls, validates values before prediction, and displays one probability plus model metadata. Streamlit is preferred over a separate frontend/API stack for the first prototype because it minimizes code and dependency surface while preserving a deployable HTTPS application.

**Alternative considered:** FastAPI plus a separate JavaScript frontend. Keep it as a later option if authentication, multi-user governance, API consumers, or structured audit logging become requirements.

### 5. Make privacy and disclaimer behavior part of the product contract

The app will contain no identifier fields, file upload, database, prediction history, or analytics integration. It will not log submitted values. Public hosting documentation will require HTTPS and will warn users not to submit identifiable patient information. The same research disclaimer will appear on the landing page and result view:

> Research prototype only. This output is for scientific research and does not constitute diagnosis, treatment advice, medical advice, clinical evidence, or a basis for clinical decision-making. The model was developed from a single-center retrospective cohort with internal temporal validation and has not undergone external validation. Do not enter identifiable patient information.

### 6. Keep the output deliberately narrow

The result will show the predicted probability for the documented one-year outcome, the selected model name/hash, and a short validation note. It will not label a patient as high/low risk or recommend an action because no clinically approved operating threshold exists.

### 7. Treat public hosting as a deployable prototype, not a clinical release

The repository will include a reproducible run command and deployment notes suitable for a public Streamlit host. Deployment credentials, domain configuration, and provider-specific operations remain outside the code change. If real patient data must be entered, the public app is the wrong deployment boundary and an authenticated institutional deployment is required.

## Risks / Trade-offs

- **Reduced predictive information** → Report the five-variable model's Group 1 and Group 2 metrics before presenting the URL; do not imply equivalence to the 12-variable model.
- **Small temporal-validation event count** → Show event counts and uncertainty, retain the internal-validation wording, and avoid clinical thresholds.
- **Public users may submit sensitive data** → Exclude identifiers and uploads, show a repeated warning, avoid persistence and value logging, require HTTPS, and state that identifiable data must not be entered.
- **Streamlit hosting/provider logs may exist outside the application** → Document the provider limitation and keep the prototype for anonymous research inputs only; use institutional hosting for real patient data.
- **Training/inference schema drift** → Store a feature-schema manifest and test that the app's one-row frame exactly matches the frozen pipeline contract.
- **Artifact load or dependency failure** → Fail closed with a visible non-prediction error; never show a fabricated or stale result.

## Migration Plan

1. Add the `reduced5` feature schema and inference adapter without modifying the current 12-variable path.
2. Train the reduced candidate pool, freeze the selected estimator, write hashes/metadata, and generate locked Group 2 evaluation reports.
3. Add focused tests for feature order, categorical encoding, artifact loading, probability bounds, and privacy/disclaimer behavior.
4. Add the Streamlit entry point and local run/deployment instructions.
5. Validate the app locally with synthetic or non-identifying inputs before public deployment.
6. Roll back by removing the reduced-model app entry point and using the unchanged 12-variable artifacts; no existing model artifact is overwritten.

## Open Questions

- Which public hosting provider should receive the prototype (Streamlit Community Cloud or another HTTPS host)?
- Should the interface be Chinese-only, or include an English toggle for international research users?
- What domain-specific numeric bounds should be enforced for disease duration, beyond rejecting non-finite or negative values?
- Is the intended outcome label wording “1-year postoperative spasm,” or should the author provide a more formal clinical definition before deployment?
