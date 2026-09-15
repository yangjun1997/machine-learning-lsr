# mechine-learning-LSR

The repository contains the frozen temporal-validation analysis and its publication package.

## Rebuild the publication package

Run from the repository root:

```powershell
uv run python -m mechine_learning_lsr.publication
```

The generator reads the audited workbook in `datas/`, aggregate JSON/CSV reports in
`reports/`, and the frozen pipeline in `artifacts/model.joblib`. It writes the five main
figures to `results/图/`, the three editable main tables to `results/表/`, and regenerates
`results/SCI_Methods_Results.md` plus `results/publication_manifest.md`. The manifest is
the authoritative source-to-output map. Historical outputs and source artifacts are not
deleted.

Run the checks with:

```powershell
uv run python -m pytest
```

Generate the Word manuscript with embedded main figures and editable tables:

```powershell
uv run python -m mechine_learning_lsr.word_export
```

## Five-variable research prediction prototype

The public prototype uses a separately trained `reduced5` model with only
disease duration, prior botulinum toxin treatment, acupuncture, zygomatic-branch
LSR, and mandibular-branch LSR. It does not reuse the 12-variable model.

Regenerate the reduced model and its locked Group 2 evaluation:

```powershell
uv run python -m mechine_learning_lsr.reduced5_train
uv run python -m mechine_learning_lsr.reduced5_evaluate
```

The frozen reduced model is `hist_gradient_boosting` and uses exactly five
inputs: duration, prior botulinum toxin, acupuncture, zygomatic-branch LSR,
and mandibular-branch LSR. Locked internal temporal validation on Group 2 had
104 records and 9 events: AUROC 0.839, AUPRC 0.304, Brier score 0.090, and
log-loss 0.282 (95% bootstrap intervals are in
`reports/reduced5_temporal_validation.json`). Group 2 was evaluated once after
freezing; it was not used for selection, tuning, or recalibration. These
results are internal validation only and do not establish clinical utility or
equivalence to the existing 12-variable model.

Run the Streamlit app locally:

```powershell
uv run streamlit run src/mechine_learning_lsr/streamlit_app.py
```

Research prototype only. This output is for scientific research and does not
constitute diagnosis, treatment advice, medical advice, clinical evidence, or a
basis for clinical decision-making. The model was developed from a single-center
retrospective cohort with internal temporal validation and has not undergone
external validation. Do not enter identifiable patient information.

For public hosting, deploy the repository with an HTTPS Streamlit host (for
example, Streamlit Community Cloud): select
`src/mechine_learning_lsr/streamlit_app.py` as the entry point, use Python 3.12
and the committed `uv.lock`/`pyproject.toml`, and share only the provider HTTPS
URL. Keep application-level persistence, uploads, and analytics disabled.
Provider-level access logs may still exist outside this repository; the public
prototype is therefore for anonymous research inputs only. Real patient data
require an authenticated institutional deployment.

Verification record (2026-09-15):

```text
uv run python -m mechine_learning_lsr.reduced5_train
selected_model: hist_gradient_boosting
development_rows: 476; development_events: 58; validation_locked: false
model_sha256: 897f2d98f49eceddaf664ea599055dda0b7279d10b7182ae5021c5651c469e61

uv run python -m mechine_learning_lsr.reduced5_evaluate
rows: 104; events: 9
AUROC: 0.839181; AUPRC: 0.303718; Brier: 0.089543; log-loss: 0.282463
validation_locked: true

uv run python -m pytest
19 passed, 2 warnings in 8.96s
```
