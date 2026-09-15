import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mechine_learning_lsr.data import load_workbook, locked_split, sha256
from mechine_learning_lsr.features import (
    REDUCED5_COLUMNS,
    build_reduced5_frame,
    validate_reduced5,
)
from mechine_learning_lsr.reduced5 import load_reduced5_model, predict_reduced5


MANIFEST = Path("artifacts/reduced5_freeze_manifest.json")
REPORT = Path("reports/reduced5_temporal_validation.json")


def test_reduced5_schema_validation_and_probability():
    values = {"duration": 3.0, "botox": 0, "acupuncture": 1, "zyg_lsr": 1, "man_lsr": 3}
    assert validate_reduced5(values) == []
    assert validate_reduced5({**values, "man_lsr": 4})
    assert list(build_reduced5_frame(values).columns) == REDUCED5_COLUMNS
    assert build_reduced5_frame({**values, "duration": "3"}).loc[0, "duration"] == 3.0
    probability = predict_reduced5(values)
    assert 0 <= probability <= 1


def test_reduced5_artifact_schema_categories_and_hash():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model, _ = load_reduced5_model()
    encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    assert REDUCED5_COLUMNS == ["duration", "botox", "acupuncture", "zyg_lsr", "man_lsr"]
    assert [values.tolist() for values in encoder.categories_] == [[0, 1], [0, 1], [1, 2, 3], [1, 2, 3]]
    assert manifest["model_sha256"] == sha256("artifacts/reduced5_model.joblib")
    assert manifest["feature_schema"]["columns"] == REDUCED5_COLUMNS


def test_reduced5_artifacts_are_validation_locked_and_separate():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert manifest["validation_locked"] is True
    assert manifest["feature_schema"]["columns"] == REDUCED5_COLUMNS
    assert manifest["validation_dataset_sha256"] == sha256("datas/580-analysis.xlsx")
    assert report["rows"] == 104
    assert report["events"] == 9
    assert report["note"].startswith("Group 2 was evaluated once")


def test_reduced5_group2_is_temporally_isolated():
    df = load_workbook("datas/580-analysis.xlsx")
    development, washout, validation = locked_split(df)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(development) == manifest["development_rows"] == 476
    assert len(validation) == manifest["validation_rows"] == 104
    assert len(development) + len(washout) + len(validation) == len(df) == 580
    assert development.admission_date.max() <= pd.Timestamp("2022-06-30")
    assert validation.admission_date.min() > pd.Timestamp("2022-08-31")
    assert manifest["validation_dataset_sha256"] == sha256("datas/580-analysis.xlsx")


def test_reduced5_invalid_input_and_unavailable_artifact(monkeypatch, tmp_path):
    values = {"duration": 3.0, "botox": 0, "acupuncture": 1, "zyg_lsr": 1, "man_lsr": 3}
    with pytest.raises(ValueError, match="man_lsr"):
        predict_reduced5({**values, "man_lsr": 9})

    import mechine_learning_lsr.reduced5 as reduced5

    monkeypatch.setattr(reduced5, "ARTIFACT_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(reduced5, "MANIFEST_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        reduced5.load_reduced5_model()


def test_streamlit_research_page_has_five_inputs_and_returns_probability():
    app = AppTest.from_file(Path(__file__).parents[1] / "src/mechine_learning_lsr/streamlit_app.py").run()
    assert not app.exception
    assert app.title[0].value == "One-Year Postoperative Spasm Probability"
    assert app.caption[0].value == "Five-variable Streamlit research prototype"
    assert app.warning[0].value.startswith("Research prototype only.")
    assert len(app.number_input) == 1
    assert len(app.selectbox) == 4
    assert len(app.warning) == 1
    assert not app.text_input
    assert not app.file_uploader

    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 1
    assert len(app.info) == 1
    assert app.metric[0].label == "Predicted probability"
    assert app.caption[-1].value.startswith("Model: hist_gradient_boosting")
