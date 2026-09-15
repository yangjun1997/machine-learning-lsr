## Why

The current frozen predictor requires 12 inputs, but the intended online research prototype must use only five variables: disease duration, prior botulinum toxin treatment, acupuncture, zygomatic-branch LSR, and mandibular-branch LSR. A five-variable model must therefore be retrained and re-evaluated; removing fields from the web form while reusing the 12-variable artifact would produce an unsupported prediction path.

The project also needs a public, low-friction Streamlit interface for research exploration, with explicit non-clinical disclaimers and no collection of patient identifiers or persistent clinical records.

## What Changes

- Add a reduced five-variable feature specification and train a new frozen model using the existing Group 1 nested cross-validation, Brier/log-loss selection rule, and locked Group 2 temporal validation workflow.
- Store the reduced-model artifact, freeze metadata, feature schema, and evaluation report separately from the existing 12-variable model.
- Add a Streamlit prototype that accepts only the five approved variables and returns the reduced model's predicted probability.
- Use labeled categorical choices for botulinum toxin, acupuncture, and both LSR variables; do not expose raw encoded values as the primary user interface.
- Display the study scope, model version, validation summary, and a prominent research-only disclaimer on the public page and prediction result.
- Do not collect names, hospital numbers, contact information, uploads, or persistent prediction histories; do not add analytics that transmit entered clinical values to third parties.
- Document public-deployment limitations and keep the prototype explicitly outside clinical decision support, diagnosis, treatment recommendation, and clinical-evidence claims.

## Capabilities

### New Capabilities

- `reduced5-model`: Defines the five-variable feature schema, reduced-model training/freeze process, locked temporal evaluation, and reproducible model artifacts.
- `research-prediction-web`: Defines the public Streamlit prediction page, input validation, probability-only output, privacy behavior, and research disclaimer.

### Modified Capabilities

None.

## Impact

- Affects feature/model training code under `src/mechine_learning_lsr/`, model artifacts and aggregate reports under `artifacts/` and `reports/`, and tests under `test/`.
- Adds a Streamlit runtime dependency and a web entry point for local and hosted execution.
- Adds a public-facing application surface, so input validation, non-persistence, HTTPS deployment, and disclaimer text become release requirements.
- Leaves the existing 12-variable `artifacts/model.joblib` and its publication outputs unchanged.
- Does not use Group 2 to select, tune, recalibrate, or modify the reduced model.
