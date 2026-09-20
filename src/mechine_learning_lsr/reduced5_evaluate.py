from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .audit import audit
from .data import locked_split, sha256
from .features import get_features
from .metrics import calibration_metrics, classification_metrics
from .reduced5 import ARTIFACT_PATH, MANIFEST_PATH, VALIDATION_REPORT_PATH


N_BOOT = 2000
SEED = 20260908
SOURCE = Path("datas/580-analysis.xlsx")


def _interval(values: list[float | None]) -> list[float] | None:
    clean = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(clean):
        return None
    return [float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))]


def _metric_intervals(y: np.ndarray, probability: np.ndarray) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(SEED)
    names = ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope")
    samples = {name: [] for name in names}
    for _ in range(N_BOOT):
        indices = rng.integers(0, len(y), len(y))
        metrics = {**classification_metrics(y[indices], probability[indices]), **calibration_metrics(y[indices], probability[indices])}
        for name in names:
            samples[name].append(metrics.get(name))
    return {name: _interval(values) for name, values in samples.items()}


def evaluate_reduced5() -> dict:
    if not ARTIFACT_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise SystemExit("missing reduced5 model artifacts; run reduced5_train first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("validation_locked"):
        raise SystemExit("reduced5 validation is already locked; refusing to overwrite")
    df, audit_result = audit(SOURCE)
    if audit_result.sha256 != manifest["dataset_sha256"]:
        raise SystemExit("dataset hash differs from reduced5 freeze manifest")
    _, _, validation = locked_split(df)
    X, y, _ = get_features(validation, "reduced5", "independent")
    probability = joblib.load(ARTIFACT_PATH).predict_proba(X)[:, 1]
    y_array = np.asarray(y, dtype=int)
    point_estimates = {**classification_metrics(y_array, probability), **calibration_metrics(y_array, probability)}
    result = {
        "analysis_type": "internal temporal validation of the corrected model; not first-use untouched validation",
        "model": manifest["selected_model"],
        "feature_schema": manifest["feature_schema"],
        "rows": len(y_array),
        "events": int(y_array.sum()),
        "bootstrap_replicates": N_BOOT,
        "point_estimates": point_estimates,
        "point_estimate_ci": _metric_intervals(y_array, probability),
        "validation_dataset_sha256": sha256(SOURCE),
        "note": "The corrected model was frozen before this evaluation, and no Group 2 fitting, tuning, feature selection, threshold selection, or recalibration was performed. However, Group 2 had already been used to evaluate the superseded web prototype, so this is not a first-use untouched validation exercise.",
    }
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.update({
        "validation_locked": True,
        "validation_dataset_sha256": sha256(SOURCE),
        "validation_rows": len(y_array),
        "validation_events": int(y_array.sum()),
        "group2_reuse_disclosure": "Group 2 had already been used to evaluate the superseded web prototype before this corrected model was frozen.",
    })
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(evaluate_reduced5(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
