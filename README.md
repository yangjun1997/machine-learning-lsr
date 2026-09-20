# machine-learning-LSR

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
age, disease duration, prior acupuncture, zygomatic-branch LSR, and
mandibular-branch LSR: the five highest-ranked original variables in the Group 1
SHAP analysis. It does not reuse the fitted 12-variable model.

Regenerate the reduced model and its locked Group 2 evaluation:

```powershell
uv run python -m mechine_learning_lsr.reduced5_train
uv run python -m mechine_learning_lsr.reduced5_evaluate
```

The reduced model uses the same fixed `gradient_boosting` specification as the
12-variable index model; candidate algorithms were not re-compared after the
feature reduction. Internal temporal validation of the corrected model on Group
2 had 104 records and 9 events: AUROC 0.895, AUPRC 0.342, Brier score 0.081,
and log-loss 0.251 (95% bootstrap intervals are in
`reports/reduced5_temporal_validation.json`). Group 2 was not used for fitting,
tuning, feature selection, threshold selection, or recalibration. However, it
had already been used to evaluate the superseded web prototype, so this result
must not be described as first-use untouched or external validation. It does
not establish clinical utility or equivalence to the 12-variable model.

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

The repository is public at
<https://github.com/yangjun1997/machine-learning-lsr>. The web-based research
calculator is publicly accessible at
<https://hfs-mvd-risk-calculator.streamlit.app/>.

Generate the completed TRIPOD+AI supplementary checklist and update the
manuscript's implementation and availability statements with:

```powershell
uv run python -m mechine_learning_lsr.tripod_checklist
```

Verification record (2026-09-20):

```text
uv run python -m mechine_learning_lsr.reduced5_train
selected_model: gradient_boosting
development_rows: 476; development_events: 58; validation_locked: false
model_sha256: 902f0d7e20451b356d6fe57dde79f8bf17d210223ca9fd1af15523c4d8521ee4

uv run python -m mechine_learning_lsr.reduced5_evaluate
rows: 104; events: 9
AUROC: 0.894737; AUPRC: 0.342095; Brier: 0.080657; log-loss: 0.251242
validation_locked: true
```
