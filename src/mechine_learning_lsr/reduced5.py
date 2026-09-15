from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib

from .data import sha256
from .features import REDUCED5_COLUMNS, build_reduced5_frame


ARTIFACT_PATH = Path("artifacts/reduced5_model.joblib")
MANIFEST_PATH = Path("artifacts/reduced5_freeze_manifest.json")
VALIDATION_REPORT_PATH = Path("reports/reduced5_temporal_validation.json")


def load_reduced5_model() -> tuple[Any, dict]:
    """Load only a validation-locked reduced model with the expected schema."""
    if not ARTIFACT_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError("reduced5 model artifacts are unavailable")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("feature_schema", {}).get("columns") != REDUCED5_COLUMNS:
        raise RuntimeError("reduced5 feature schema does not match the application contract")
    if manifest.get("model_sha256") != sha256(ARTIFACT_PATH):
        raise RuntimeError("reduced5 model hash does not match the freeze manifest")
    if not manifest.get("validation_locked") or not VALIDATION_REPORT_PATH.is_file():
        raise RuntimeError("reduced5 model has not completed locked temporal validation")
    return joblib.load(ARTIFACT_PATH), manifest


def predict_reduced5(values: dict[str, object], model: Any | None = None) -> float:
    """Return one bounded probability for a validated five-variable input."""
    frame = build_reduced5_frame(values)
    if model is None:
        model, _ = load_reduced5_model()
    probability = float(model.predict_proba(frame)[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("model returned an invalid probability")
    return probability
