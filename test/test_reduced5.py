import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
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
    values = {"age": 52, "duration": 3.0, "acupuncture": 1, "zyg_lsr": 1, "man_lsr": 3}
    assert validate_reduced5(values) == []
    assert validate_reduced5({**values, "age": 17})
    assert validate_reduced5({**values, "age": 121})
    assert validate_reduced5({**values, "man_lsr": 4})
    assert list(build_reduced5_frame(values).columns) == REDUCED5_COLUMNS
    assert build_reduced5_frame({**values, "duration": "3"}).loc[0, "duration"] == 3.0
    probability = predict_reduced5(values)
    assert 0 <= probability <= 1


def test_reduced5_artifact_schema_categories_and_hash():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model, _ = load_reduced5_model()
    encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    assert REDUCED5_COLUMNS == ["age", "duration", "acupuncture", "zyg_lsr", "man_lsr"]
    assert [values.tolist() for values in encoder.categories_] == [[0, 1], [1, 2, 3], [1, 2, 3]]
    estimator = model.named_steps["model"]
    assert isinstance(estimator, GradientBoostingClassifier)
    expected_parameters = {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "random_state": 20260904,
    }
    parameters = estimator.get_params()
    assert {name: parameters[name] for name in expected_parameters} == expected_parameters
    assert manifest["selected_model"] == "gradient_boosting"
    assert manifest["model_sha256"] == sha256("artifacts/reduced5_model.joblib")
    assert manifest["feature_schema"]["columns"] == REDUCED5_COLUMNS


def test_reduced5_features_match_group1_shap_top_five():
    ranking = pd.read_csv("reports/shap_variable_importance.csv").sort_values("rank")
    assert set(REDUCED5_COLUMNS) == set(ranking.head(5)["variable"])
    assert "botox" not in REDUCED5_COLUMNS


def test_reduced5_artifacts_are_validation_locked_and_separate():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert manifest["validation_locked"] is True
    assert manifest["feature_schema"]["columns"] == REDUCED5_COLUMNS
    assert manifest["validation_dataset_sha256"] == sha256("datas/580-analysis.xlsx")
    assert report["rows"] == 104
    assert report["events"] == 9
    assert "superseded web prototype" in report["note"]


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
    values = {"age": 52, "duration": 3.0, "acupuncture": 1, "zyg_lsr": 1, "man_lsr": 3}
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
    assert len(app.number_input) == 2
    assert len(app.selectbox) == 3
    assert app.number_input[0].label == "Age (years)"
    assert all(widget.label != "Prior botulinum toxin treatment" for widget in app.selectbox)
    assert len(app.warning) == 1
    assert not app.text_input
    assert not app.file_uploader

    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 1
    assert len(app.info) == 1
    assert app.metric[0].label == "Predicted probability"
    assert len(app.caption) == 1
