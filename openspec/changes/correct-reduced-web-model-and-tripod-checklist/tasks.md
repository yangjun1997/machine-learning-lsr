## 1. Correct the reduced model contract

- [x] 1.1 Replace botulinum-toxin history with age in the shared reduced feature schema, validation, and inference-frame builder.
- [x] 1.2 Update the Streamlit form and focused tests to expose the five Group 1 SHAP-ranked variables.

## 2. Fix training and validation

- [x] 2.1 Train only the existing fixed Gradient Boosting specification for the reduced model and record descriptive Group 1 cross-validated metrics without candidate reselection.
- [x] 2.2 Retrain the Group 1 artifact and regenerate its manifest and development report.
- [x] 2.3 Evaluate the corrected frozen artifact on Group 2 and add the prior-use limitation to validation metadata.

## 3. Add TRIPOD+AI supplementary material

- [x] 3.1 Add a reproducible generator using authoritative TRIPOD+AI item wording and explicit manuscript-location/status mappings.
- [x] 3.2 Generate `results/TRIPOD_AI_Checklist.docx` and add a focused automated check.
- [x] 3.3 Update `results/Manuscripts.docx` to cite the completed checklist and accurately describe the web calculator where relevant.

## 4. Documentation and verification

- [x] 4.1 Update README and model metadata to describe the corrected predictors, fixed algorithm, regenerated results, and Group 2 reuse limitation.
- [x] 4.2 Run focused tests, the full pytest suite, the package build, and OpenSpec validation.
