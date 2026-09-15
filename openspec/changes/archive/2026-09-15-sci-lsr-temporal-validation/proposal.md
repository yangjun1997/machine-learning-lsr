## Why

The repository contains a 580-patient single-center dataset and a study plan, but no reproducible analysis pipeline. The study needs a defensible SCI-oriented workflow that separates development from locked temporal validation, prevents leakage, and reports calibration and clinical utility rather than relying on AUC alone.

## What Changes

- Add an auditable data-contract and quality-gate workflow for `datas/580-analysis.xlsx`.
- Add M0 preoperative, M1 intraoperative LSR, and M2 sensitivity feature definitions.
- Add reproducible group1 development and group2 one-time temporal validation.
- Compare a broad pre-registered model pool spanning statistical baselines, linear models, probabilistic models, distance/kernel methods, tree ensembles, boosting, and neural networks; select using group1 nested cross-validation rather than group2 performance.
- Report discrimination, calibration, Brier/log-loss, decision-curve net benefit, confidence intervals, and predefined sensitivity analyses.
- Treat LSR-by-duration and LSR-by-age interactions as exploratory effect-modification analyses, not confirmation gates.
- Produce versioned reports, figures, model artifacts, hashes, and leakage/data-contract tests.
- Keep survival analysis and deployment outside this change until event-time and follow-up fields are available.

## Capabilities

### New Capabilities

- `clinical-data-audit`: Validate the 580-row workbook, normalize fields, enforce value ranges, and emit an auditable data-quality report.
- `temporal-validation`: Lock group1/group2 by admission date, train only on group1, and evaluate the frozen model once on group2.
- `clinical-risk-modeling`: Build M0/M1/M2 feature sets and fit the broad model pool with leakage-safe preprocessing.
- `clinical-model-evaluation`: Compute discrimination, calibration, uncertainty, DCA, subgroup, and sensitivity outputs for publication reports.
- `reproducible-research-artifacts`: Record dataset/model hashes, configuration, software versions, reports, and test results.

### Modified Capabilities

None. The repository has no existing analysis specifications whose requirements are being changed.

## Impact

- Adds analysis modules under `src/mechine_learning_lsr/`, tests under `test/`, configuration under `configs/`, and generated reports/artifacts under `reports/` and `artifacts/`.
- Uses the existing Python/`uv` project; dependencies may include pandas, openpyxl, scikit-learn, scipy, statsmodels, matplotlib, and a calibration/DCA implementation only when needed.
- Raw patient data remains local and is not uploaded or committed to a public repository.
- No clinical API, automatic deployment, survival model, or automatic retraining is introduced.
