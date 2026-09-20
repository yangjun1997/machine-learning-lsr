## Context

The public five-variable calculator was intended to operationalize the five highest-ranked source variables from the Group 1 SHAP analysis while retaining the study's frozen Gradient Boosting algorithm. The current implementation instead substitutes prior botulinum toxin treatment for age and reruns candidate-model comparison, which selected HistGradientBoosting. The manuscript also states that reporting followed TRIPOD+AI but does not provide the completed checklist expected as supplementary material.

The correction spans the reduced feature contract, training and validation metadata, the Streamlit form, generated artifacts, tests, and the manuscript supplement. Group 1 remains the only cohort used to define predictors and fit the corrected model. Group 2 was already inspected by the superseded prototype, so a corrected-model evaluation on Group 2 remains an internal temporal validation analysis but cannot be described as first-use untouched validation.

## Goals / Non-Goals

**Goals:**

- Make the web model use exactly MAN-LSR, ZYG-LSR, disease duration, prior acupuncture, and age.
- Fit the corrected web model on Group 1 with the same fixed `GradientBoostingClassifier` specification as the 12-predictor index model.
- Regenerate and verify the frozen reduced artifact and its Group 2 temporal-validation report with transparent reuse disclosure.
- Replace the public form's botulinum-toxin control with age.
- Generate a completed TRIPOD+AI checklist as a standalone supplementary DOCX and point readers to it from the manuscript.

**Non-Goals:**

- Rerun the 12-predictor model comparison, SHAP analysis, or primary validation.
- Compare candidate algorithms for the corrected five-variable web model.
- Claim that Group 2 is external validation or a newly untouched test set.
- Fill unavailable study facts or checklist locations by inference.

## Decisions

### Use the existing reduced-model contract as the single source of truth

`REDUCED5_NUMERIC`, `REDUCED5_CATEGORICAL`, and their validation/frame helpers will be corrected centrally. Training, inference, and the Streamlit page will continue to consume that shared contract. This is smaller and less error-prone than maintaining a separate web-only schema.

### Reuse the existing Gradient Boosting candidate specification directly

The corrected trainer will obtain the existing `gradient_boosting` pipeline from `models.py`, evaluate that fixed specification by Group 1 out-of-fold cross-validation for descriptive development metrics, and refit it on all Group 1 records. It will not invoke candidate ranking. Reusing the existing candidate definition prevents parameter drift between the index and web models.

### Preserve temporal evaluation while disclosing prior Group 2 use

After the corrected Group 1 artifact is frozen, Group 2 will be evaluated without fitting, tuning, threshold selection, or recalibration. Reports and documentation will state that Group 2 had already been used to evaluate the superseded prototype. This preserves an honest temporal-validation estimate without misrepresenting the cohort as newly untouched.

### Generate the checklist from a small structured source

The checklist generator will use the authoritative TRIPOD+AI item wording and a repository-owned mapping of manuscript locations/status. It will mark unavailable or non-applicable items explicitly and produce a standalone DOCX under `results/`. A focused test will check document existence and core identifying content. No new dependency is needed because `python-docx` is already installed.

## Risks / Trade-offs

- [Risk] Group 2 performance may change after correcting the predictors and algorithm. → Regenerate all reduced-model metrics and do not preserve superseded values in current documentation.
- [Risk] Reuse of Group 2 can be mistaken for pristine validation. → Put the reuse limitation in the validation report, manifest, README, manuscript, and checklist mapping.
- [Risk] Checklist mappings can overstate reporting completeness. → Use explicit `Not reported`, `Not applicable`, and author-input statuses where the manuscript lacks evidence.
- [Risk] Streamlit deployment can temporarily load an old artifact. → Commit the corrected code, manifest, report, and model artifact together; inference fails closed on schema/hash mismatch.

## Migration Plan

1. Correct the shared five-variable schema and tests.
2. Change reduced training to the fixed Gradient Boosting specification and retrain on Group 1.
3. Evaluate the frozen corrected artifact on Group 2 and regenerate validation metadata.
4. Update the Streamlit form and repository documentation.
5. Generate the TRIPOD+AI checklist and update the manuscript supplementary-material statement.
6. Run focused tests, the full test suite, and the package build before deployment.

Rollback consists of reverting the correction commit and redeploying the preceding artifact/code pair. The superseded model will not be presented as satisfying the SHAP-top-five requirement.

## Open Questions

- The final journal-specific supplement numbering can be assigned at submission; the generated file uses a descriptive title in the meantime.
