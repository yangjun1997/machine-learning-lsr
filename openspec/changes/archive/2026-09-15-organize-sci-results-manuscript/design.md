## Context

The repository already contains a completed modeling pipeline and aggregate outputs for 580 patients, including 476 Group 1 patients with 58 events and 104 Group 2 patients with 9 events. The current `results/` directory mixes source outputs, rendered tables, draft composites, duplicates, and a placeholder `results/图/表.md`. Figures 1-4 do not have a traceable publication-composition entry point, and Figure 5 is present but combines approximate Wald P values with bootstrap confidence intervals and displays programmatic age labels.

The publication package must reflect the implementation rather than earlier narrative descriptions. The study is a single-center retrospective cohort study with internal temporal validation, the final model independently one-hot encodes `zyg_lsr` and `man_lsr`, the Gradient Boosting numeric path is not standardized, and the reported comparison contains 14 fixed candidate specifications rather than a non-empty hyperparameter search. Group 2 remains a locked temporal validation cohort and must not be used to select, tune, or recalibrate the model.

Several manuscript inputs are not available in the repository: center name, ethics committee and approval number, consent handling, eligibility criteria, formal 1-year outcome definition, follow-up method, and target journal. These facts cannot be inferred from the workbook or code.

## Goals / Non-Goals

**Goals:**

- Produce a reproducible publication package containing five main figures, three main tables, supplementary-item inventory, and an English SCI-style Methods and Results document.
- Make every reported number and visual traceable to an aggregate report, configured analysis, or auditable baseline-table calculation.
- Correct misleading labels and descriptions while preserving the frozen model and temporal validation results.
- Provide detailed, self-contained figure legends and table notes that define cohorts, variables, estimates, uncertainty, abbreviations, and exploratory analyses.
- Retain unknown study details as conspicuous placeholders for author completion.

**Non-Goals:**

- Refit candidates based on Group 2 performance, change the frozen index model, or recalibrate on Group 2.
- Add new clinical thresholds, claim that exploratory decision curves establish utility, or promote pooled interaction analyses to confirmatory findings.
- Perform survival analysis, prospective validation, multicenter external validation, or new literature review.
- Draft the Introduction, Discussion, abstract, references, or submission statements outside the requested Methods and Results package.
- Delete or rewrite raw data and historical generated outputs.

## Decisions

### 1. Generate one non-destructive publication package

Add one repository entry point that stages canonical files under `results/图/` and `results/表/`, writes `results/SCI_Methods_Results.md`, and writes a compact publication manifest. The generator will reuse installed packages and existing analysis helpers. Historical outputs in `results/` and `reports/` remain source evidence and are not deleted.

This is preferred to manual renaming because a single command can reproduce links, captions, and numeric tables. A separate publication framework or templating dependency is unnecessary for one Markdown deliverable.

### 2. Use structured outputs as the numeric source of truth

Model performance, calibration, SHAP, interaction, and supplementary values will come from the existing JSON/CSV reports or the functions that generate them. Baseline summaries may be recomputed read-only from `datas/580-analysis.xlsx` using the existing loader. Existing prose reports and rasterized table images are display artifacts, not numeric authorities.

The publication manifest will map each canonical item to source data, generation code, cohort, and manuscript role. Focused snapshot checks will cover the primary cohort counts and Group 2 performance values so accidental narrative drift is visible.

### 3. Fix the main figure set in advance

The main manuscript set is:

1. Study design and analysis flow.
2. Group 1 candidate-model comparison and index-model selection.
3. Group 1 SHAP-based index-model interpretation.
4. Group 2 calibration and exploratory decision-curve analysis.
5. Exploratory pooled-cohort residual-LSR effect modification.

Figure 1 will state the observed date ranges and describe a stratified fivefold comparison of fixed candidate specifications, with any inner resampling described only where it was actually used. Figure 2 will not imply that Gradient Boosting dominated all metrics: it was frozen by the predefined selection rule, while the lower Brier score and stronger discrimination of elastic-net logistic regression remain visible. Figure 4 will label decision-curve analysis as exploratory because no clinically approved threshold range is available.

Figure 5 will remain in the main set at the user's request but will be visually and textually separated from temporal validation. Panel A will identify weak ridge stabilization, the interaction-OR units, bootstrap resampling, sparse estimates, and multiplicity status. Displayed inference will use a coherent convention; approximate Wald P values must not be visually presented as if they are the tests corresponding to percentile bootstrap intervals. A modifier with no informative events, including `other_vessel`, will be marked not estimable instead of OR 1.0 with CI 1.0-1.0. Panel B will use human-readable age labels, remove the generic `label` axis title, show denominators or define them in the legend, and remain descriptive rather than a confirmation panel.

### 4. Use three Markdown tables as the canonical tabular set

Table 1 will report baseline characteristics for Group 1 and Group 2. Table 2 will report Group 1 cross-validated candidate performance and identify the fixed index-model selection rule. Table 3 will report Group 2 discrimination, proper scoring, and calibration with bootstrap confidence intervals where available. Rendered table images may be retained as convenience copies, but the Markdown tables are authoritative and editable.

Table notes will state summary formats, denominators, missing-data handling, statistical tests where used, units, abbreviations, resampling details, and whether P values are descriptive. Unavailable information will be shown as `[[AUTHOR INPUT REQUIRED: ...]]`, not as an em dash that could be mistaken for a measured absence.

### 5. Separate evidence, inference, and placeholders in the manuscript

`results/SCI_Methods_Results.md` will contain Methods, Results, Figure Legends, and Table Titles and Notes. Methods will use past tense and Results will report estimates without causal language. The document will use the terms "single-center retrospective cohort study with internal temporal validation," "development cohort," and "temporal validation cohort."

The text will explicitly describe the 12 direct predictors, independent encoding of the two branch-specific LSR variables, 14 fixed candidate specifications, preprocessing by estimator, freeze rule, bootstrap uncertainty, sparse Group 2 event count, and exploratory status of DCA and interaction analyses. Missing center, ethics, consent, eligibility, endpoint-adjudication, follow-up, and journal details will use consistent author-input placeholders and will be listed in a completion checklist.

### 6. Validate the package with narrow automated checks

One focused test module will verify that all canonical files exist, Markdown image/table links resolve, key cohort counts and Group 2 metrics agree with structured reports, direct identifiers are absent, and prohibited overclaims do not appear. Existing unit tests remain the source for modeling behavior; the publication tests will not duplicate the model suite.

## Risks / Trade-offs

- **[Missing clinical metadata]** -> Keep explicit author-input placeholders and do not label the document submission-ready until they are resolved.
- **[Only nine Group 2 events]** -> Report event counts and confidence intervals prominently and use "internal temporal validation" rather than external or definitive validation.
- **[Near-tied model scores]** -> State the prespecified freeze rule and show competing metrics rather than asserting broad superiority.
- **[Sparse interaction strata]** -> Mark non-estimable effects, use one coherent inferential display, and characterize all pooled interaction findings as hypothesis-generating.
- **[Composite figures lack current provenance]** -> Recreate them through the publication entry point and record their sources; do not treat the existing raster files as the only masters.
- **[Upstream reports may be regenerated]** -> Fail focused snapshot/link checks when primary values change, prompting manuscript regeneration and review.
- **[Journal-specific style is unknown]** -> Produce neutral medical-journal English and defer exact size, reference, and submission formatting until the target journal is supplied.

## Migration Plan

1. Add the publication generator and focused checks without removing existing outputs.
2. Regenerate the canonical figures, tables, manifest, and Markdown manuscript into the requested directories.
3. Compare primary counts and metrics against current structured reports and visually inspect all five figures at publication and mobile review sizes.
4. Replace the placeholder table file only after the generated Table 1-3 files are verified.
5. If generation fails, remove only the newly generated canonical files; existing analysis outputs remain the rollback source.

## Open Questions

- Which hospital or research center supplied the cohort?
- What ethics committee, approval number, and consent or waiver statement apply?
- What were the exact inclusion and exclusion criteria?
- How was 1-year postoperative spasm defined, adjudicated, and followed up?
- Which journal and house style will govern final wording, dimensions, and file formats?
