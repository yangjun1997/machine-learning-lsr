## 1. Source and publication contract

- [x] 1.1 Inventory the existing figure, table, JSON, CSV, workbook, and model-report sources and record the authoritative source for every manuscript value.
- [x] 1.2 Define canonical filenames and manuscript roles for Figures 1-5, Tables 1-3, and supplementary figures without deleting historical outputs.
- [x] 1.3 Reconcile this package with the unfinished subgroup/sensitivity and interaction tasks in `sci-lsr-temporal-validation`, documenting which available outputs are complete enough to publish and which remain exploratory.

## 2. Reproducible publication generator

- [x] 2.1 Add one minimal publication-generation entry point that creates `results/图/`, `results/表/`, `results/SCI_Methods_Results.md`, and the publication manifest from repository sources.
- [x] 2.2 Make regeneration replace only documented canonical targets and preserve the raw workbook, frozen artifacts, structured reports, and unrelated historical outputs.
- [x] 2.3 Document the exact `uv run` command and source-to-output mapping needed to reproduce the package.

## 3. Main and supplementary figures

- [x] 3.1 Regenerate Figure 1 with the observed Group 1 and Group 2 date ranges, fixed-candidate stratified development comparison, model freeze, and locked internal temporal validation wording.
- [x] 3.2 Regenerate Figure 2 with 14 primary models plus supplementary LightGBM, identifying Gradient Boosting by the predefined selection rule while retaining elastic-net logistic regression.
- [x] 3.3 Regenerate Figure 3 from the structured SHAP outputs and label the cohort, index model, aggregation method, and feature directions accurately.
- [x] 3.4 Regenerate Figure 4 with Group 2 calibration and explicitly exploratory decision-curve analysis, removing redundant or weak panels unless they add independent evidence.
- [x] 3.5 Regenerate Figure 5 with human-readable age labels, descriptive denominators, coherent interaction uncertainty, multiplicity disclosure, and not-estimable handling for sparse modifiers.
- [x] 3.6 Stage the learning curve, patient prediction intervals, overall performance confidence intervals, and other retained diagnostics as clearly numbered supplementary figures.
- [x] 3.7 Visually inspect every PNG and SVG at publication and reduced review sizes for blank panels, clipping, font substitution, overlap, label readability, and consistency with its source data.

## 4. Canonical tables

- [x] 4.1 Generate Table 1 baseline characteristics for Group 1 and Group 2 from the audited workbook, including appropriate summaries, event counts, tests, missingness, and a complete table note.
- [x] 4.2 Generate Table 2 from Group 1 cross-validation outputs with candidate availability, fixed specification, discrimination, proper-scoring metrics, and the index-model selection note.
- [x] 4.3 Generate Table 3 from the locked Group 2 outputs with AUROC, AUPRC, Brier score, log-loss, calibration intercept and slope, bootstrap confidence intervals, and sparse-event caveats.
- [x] 4.4 Replace the placeholder table index with links to the three canonical Markdown tables and identify any rendered table images as convenience copies rather than numeric sources.

## 5. SCI Methods and Results document

- [x] 5.1 Draft the English Methods sections for study design, cohort construction, predictors and outcome, temporal split, model development and freezing, validation metrics, calibration, SHAP, exploratory DCA, exploratory interactions, software, and reproducibility.
- [x] 5.2 Ensure Methods states the 12 direct predictors, independent one-hot encoding of the two branch-specific LSR variables, estimator-specific preprocessing, 14 fixed candidate specifications, empty hyperparameter grid, and prohibition on Group 2 tuning or recalibration.
- [x] 5.3 Draft the English Results sections for cohort flow and baseline data, Group 1 model comparison, frozen-model Group 2 validation, interpretation, calibration and exploratory utility, and pooled exploratory effect modification.
- [x] 5.4 Report the primary Group 2 values and confidence intervals directly from structured sources and describe the Gradient Boosting versus elastic-net logistic regression comparison without superiority overclaiming.
- [x] 5.5 Add self-contained legends for Figures 1-5 and titles and notes for Tables 1-3, defining all panels, cohorts, encodings, units, abbreviations, confidence intervals, resampling, and exploratory analyses.
- [x] 5.6 Insert consistent `[[AUTHOR INPUT REQUIRED: ...]]` placeholders and a completion checklist for center, ethics, consent, eligibility, endpoint adjudication, follow-up, and target-journal details.

## 6. Verification and handoff

- [x] 6.1 Add a focused publication test that checks canonical file presence, resolvable Markdown links, source-consistent cohort counts and Group 2 metrics, privacy, and prohibited overclaim language.
- [x] 6.2 Run the publication generator, the focused test, and the full `uv run python -m pytest` suite and record the commands and outcomes in the manifest.
- [x] 6.3 Perform a final evidence-to-text audit across Methods, Results, figures, and tables and leave the package explicitly incomplete until every author-input placeholder is resolved.
