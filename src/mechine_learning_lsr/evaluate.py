from __future__ import annotations

from pathlib import Path
import json

import joblib

from .audit import audit
from .data import locked_split, sha256
from .features import get_features
from .metrics import calibration_metrics, classification_metrics, decision_curve


def main():
    manifest_path = Path("artifacts/freeze_manifest.json")
    if not manifest_path.exists():
        raise SystemExit("missing artifacts/freeze_manifest.json; run train first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("validation_locked"):
        raise SystemExit("group2 validation is already locked; refusing to overwrite")
    source = Path("datas/580-analysis.xlsx")
    df, audit_result = audit(source)
    if audit_result.sha256 != manifest["dataset_sha256"]:
        raise SystemExit("dataset hash differs from freeze manifest")
    _, _, validation = locked_split(df)
    X, y, _ = get_features(validation, manifest["feature_set"], "independent")
    model = joblib.load("artifacts/model.joblib")
    probability = model.predict_proba(X)[:, 1]
    result = {"model": manifest["selected_model"], "rows": len(y), "events": int(y.sum()), **classification_metrics(y, probability), **calibration_metrics(y, probability), "decision_curve": decision_curve(y, probability, [0.05, 0.10, 0.15, 0.20, 0.30, 0.40])}
    Path("reports/temporal_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["validation_locked"] = True
    manifest["validation_dataset_sha256"] = sha256(source)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
