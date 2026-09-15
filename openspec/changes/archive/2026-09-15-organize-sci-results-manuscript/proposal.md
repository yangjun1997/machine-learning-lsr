## Why

The current analysis outputs contain duplicate files, placeholder documentation, inconsistent figure labels, and manuscript statements that do not always match the implemented pipeline. A curated, traceable results package is needed so the five main figures, three main tables, and SCI-style Methods and Results can be used without overstating the single-center temporal validation or exploratory analyses.

## What Changes

- Create a publication-oriented figure and table index that maps each selected item to its source data, generation code, manuscript placement, and canonical filename.
- Curate five main figures and three main tables, retain useful secondary analyses as supplementary material, and avoid deleting the original generated outputs.
- Correct known presentation and provenance issues, including the Figure 1 dates and validation wording, the Figure 5 age labels and exploratory framing, and inconsistent statistical terminology.
- Generate an English SCI-style Markdown document containing Methods, Results, detailed figure legends, and detailed table titles and notes.
- Ensure manuscript statements match the implemented feature encoding, fixed candidate-model comparison, frozen index-model selection, temporal validation metrics, calibration analysis, and exploratory decision-curve and interaction analyses.
- Preserve explicit placeholders for unavailable center, ethics, consent, eligibility, outcome-adjudication, follow-up, and target-journal information rather than fabricating study details.
- Add focused checks for figure/table completeness, Markdown links, reported values, privacy, and reproducibility.

## Capabilities

### New Capabilities

- `publication-results-package`: Curate traceable publication figures and tables and produce an evidence-bounded SCI Methods and Results manuscript package.

### Modified Capabilities

None. The related modeling and evaluation capabilities remain in the in-progress `sci-lsr-temporal-validation` change; this change consumes their outputs without redefining the frozen model or validation analysis.

## Impact

- Affects publication files under `results/`, aggregate source reports under `reports/`, figure-generation code under `src/mechine_learning_lsr/`, and focused tests under `test/`.
- Reuses the existing Python and `uv` environment and introduces no new dependency unless an existing output format cannot be reproduced with installed packages.
- Does not rerun model selection against Group 2, recalibrate the frozen model, alter patient-level data, or claim external/prospective validation.
- Keeps raw patient data local and limits publication artifacts to aggregate, non-identifying content.
