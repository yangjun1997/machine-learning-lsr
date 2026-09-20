from __future__ import annotations

import json
from pathlib import Path

import joblib

from .audit import audit
from .data import locked_split, sha256
from .features import REDUCED5_CATEGORICAL, REDUCED5_COLUMNS, REDUCED5_NUMERIC, get_features
from .models import available_candidates
from .train import OPTIONAL_MODEL_NAMES, nested_development


SEED = 20260904
SOURCE = Path("datas/580-analysis.xlsx")
ARTIFACT = Path("artifacts/reduced5_model.joblib")
MANIFEST = Path("artifacts/reduced5_freeze_manifest.json")
REPORT = Path("reports/reduced5_development_models.json")


def train_reduced5() -> dict:
    df, audit_result = audit(SOURCE)
    development, _, _ = locked_split(df)
    X, y, feature_set = get_features(development, "reduced5", "independent")
    candidates_report, selected_name, _ = nested_development(
        X,
        y,
        feature_set,
        seed=SEED,
        include_optional=False,
        candidate_names={"gradient_boosting"},
    )
    selected = next(
        candidate
        for candidate in available_candidates(feature_set.numeric, feature_set.categorical, seed=SEED)
        if candidate.name == selected_name and candidate.name not in OPTIONAL_MODEL_NAMES
    )
    selected.estimator.fit(X, y)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected.estimator, ARTIFACT)
    model_hash = sha256(ARTIFACT)
    feature_schema = {
        "name": "reduced5",
        "columns": REDUCED5_COLUMNS,
        "numeric": REDUCED5_NUMERIC,
        "categorical": REDUCED5_CATEGORICAL,
        "prediction_time": "intraoperative",
    }
    report = {
        "dataset_sha256": audit_result.sha256,
        "model": selected_name,
        "feature_schema": feature_schema,
        "development_rows": len(development),
        "development_events": int(y.sum()),
        "seed": SEED,
        "candidate_pool": "fixed gradient_boosting specification used by the 12-predictor index model",
        "selection_rule": "No reduced-model candidate reselection; gradient_boosting was fixed in advance.",
        "feature_selection": "Five highest-ranked source variables from the Group 1 SHAP analysis.",
        "candidates": candidates_report,
    }
    manifest = {
        "dataset_sha256": audit_result.sha256,
        "model_sha256": model_hash,
        "selected_model": selected_name,
        "feature_schema": feature_schema,
        "development_rows": len(development),
        "development_events": int(y.sum()),
        "validation_locked": False,
        "candidate_pool": "fixed gradient_boosting specification used by the 12-predictor index model",
        "training_note": "The five SHAP-ranked predictors and Gradient Boosting specification were fixed from Group 1, then the estimator was refit on Group 1. Group 2 was not used for fitting, tuning, feature selection, threshold selection, or recalibration.",
        "group2_reuse_disclosure": "Group 2 had already been used to evaluate the superseded web prototype before this corrected model was frozen.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(train_reduced5(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
