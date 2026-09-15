## 1. Reduced feature contract

- [x] 1.1 Add the `reduced5` feature schema with `duration` as numeric and `botox`, `acupuncture`, `zyg_lsr`, and `man_lsr` as categorical fields.
- [x] 1.2 Add shared categorical validation and labeled source-state mappings for binary treatments and the three LSR states.
- [x] 1.3 Add a prediction-only frame builder that does not require an `outcome` column or any patient identifier.

## 2. Reduced model training and evaluation

- [x] 2.1 Add a reduced-model training entry point that reuses the existing Group 1 nested cross-validation and fixed candidate configurations.
- [x] 2.2 Apply the existing Brier-score primary selection and log-loss tie-breaker, then refit exactly one selected estimator on Group 1.
- [x] 2.3 Write separate reduced-model artifacts, freeze metadata, feature schema, dataset/model hashes, and training report without overwriting the 12-variable artifacts.
- [x] 2.4 Evaluate the frozen reduced model once on locked Group 2 and write metrics, event counts, calibration estimates, and uncertainty/non-estimable statuses.

## 3. Streamlit research prototype

- [x] 3.1 Add the Streamlit dependency and a runnable app entry point for local and public HTTPS hosting.
- [x] 3.2 Build a Chinese-language form exposing only the five approved variables with numeric and categorical controls.
- [x] 3.3 Load the reduced artifact once per process, validate submitted values, and display only a bounded predicted probability plus model/version metadata.
- [x] 3.4 Add fail-closed handling for missing artifacts, invalid schemas, invalid values, and prediction exceptions.
- [x] 3.5 Add the research-only disclaimer to the landing page and result view, including the single-center/internal-validation and no-clinical-evidence statements.

## 4. Privacy and deployment safeguards

- [x] 4.1 Ensure the app has no identifier fields, file uploads, database writes, prediction history, or third-party analytics integration.
- [x] 4.2 Ensure submitted clinical values are not written to application-managed logs and display a warning against entering identifiable information.
- [x] 4.3 Add public HTTPS deployment instructions and document that real patient data require an authenticated institutional deployment.

## 5. Verification and documentation

- [x] 5.1 Add tests for exact five-field order, categorical LSR encoding, probability bounds, artifact hash/schema metadata, and Group 2 isolation.
- [x] 5.2 Add Streamlit-focused tests or a deterministic prediction smoke check covering valid input, invalid input, and unavailable-artifact behavior.
- [x] 5.3 Document the run command, model limitations, validation summary, disclaimer text, and public-deployment privacy boundary.
- [x] 5.4 Run the full test suite and record the reduced-model training/evaluation commands and outputs before publishing a public URL.
