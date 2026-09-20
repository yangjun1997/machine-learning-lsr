## Why

The deployed five-variable prototype does not implement the stated reduction rule: it uses prior botulinum toxin treatment instead of age and reselects a histogram-based boosting algorithm rather than retaining the study's frozen Gradient Boosting algorithm. The manuscript also cites TRIPOD+AI without supplying a completed reporting checklist, leaving the model implementation and reporting package internally inconsistent.

## What Changes

- **BREAKING**: Replace `botox` with `age` in the reduced-model feature schema, inference contract, Streamlit form, tests, and frozen artifact.
- Fix the reduced model to the same `GradientBoostingClassifier` specification used by the 12-predictor index model; do not rerun candidate-algorithm selection for the web model.
- Derive and record the reduced feature set from the existing Group 1 source-variable SHAP ranking: MAN-LSR, ZYG-LSR, disease duration, prior acupuncture, and age.
- Retrain the corrected reduced model on Group 1, regenerate its artifact and development report, and evaluate the frozen artifact on Group 2 without using Group 2 to choose features, algorithm, parameters, or recalibration.
- Disclose that Group 2 had already been used to evaluate the superseded web prototype, so the corrected web-model evaluation is temporal internal validation but not a first-use untouched validation exercise.
- Generate a completed TRIPOD+AI checklist as standalone supplementary material, mapping each applicable item to the manuscript section and marking genuinely unreported or non-applicable items explicitly.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `reduced5-model`: Change the required predictors to the Group 1 SHAP top five and fix the algorithm to the index-model Gradient Boosting specification without reduced-model candidate reselection.
- `research-prediction-web`: Replace the botulinum-toxin input with age and serve predictions only from the corrected frozen artifact.
- `publication-results-package`: Add a reproducible, completed TRIPOD+AI checklist to the supplementary-material package.

## Impact

- Affects reduced feature construction, training, evaluation, artifact validation, Streamlit controls, tests, and model documentation under `src/mechine_learning_lsr/`, `test/`, `artifacts/`, `reports/`, and `README.md`.
- Replaces the deployed model artifact and changes the public calculator's input contract.
- Adds a supplementary DOCX checklist under `results/` and a small reproducible generator/test path.
- Does not alter the 12-predictor index model, its SHAP analysis, or its reported primary validation results.
