from __future__ import annotations

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .audit import audit
from .data import locked_split, sha256
from .features import get_features
from .models import available_candidates, candidates
from .metrics import classification_metrics
from .metrics import bootstrap_interval


OPTIONAL_MODEL_NAMES = {"xgboost", "lightgbm", "catboost"}


def nested_development(X, y, feature_set, seed=20260904, include_optional=True):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    oof_by_model = {}
    candidate_pool = candidates(feature_set.numeric, feature_set.categorical, seed)
    if not include_optional:
        candidate_pool = [candidate for candidate in candidate_pool if candidate.name not in OPTIONAL_MODEL_NAMES]
    for candidate in candidate_pool:
        if candidate.status != "available":
            results.append({"model": candidate.name, "status": candidate.status, "note": candidate.note})
            continue
        oof = np.full(len(y), np.nan)
        started = time.perf_counter()
        warnings_seen = []
        for train_idx, test_idx in outer.split(X, y):
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            search = GridSearchCV(
                candidate.estimator,
                param_grid={},
                scoring="neg_brier_score",
                cv=inner,
                refit=True,
                error_score="raise",
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search.fit(X.iloc[train_idx], y.iloc[train_idx])
                    warnings_seen.extend(str(w.message) for w in caught)
                oof[test_idx] = search.predict_proba(X.iloc[test_idx])[:, 1]
            except Exception as exc:
                results.append({"model": candidate.name, "status": "failed", "note": repr(exc)})
                break
        else:
            metrics = classification_metrics(y, oof)
            oof_by_model[candidate.name] = oof
            results.append({
                "model": candidate.name,
                "status": "complete",
                **metrics,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "warning_count": len(warnings_seen),
                "warnings": sorted(set(warnings_seen))[:10],
                "auroc_ci": bootstrap_interval(y, oof, "auroc"),
                "auprc_ci": bootstrap_interval(y, oof, "auprc"),
            })
    complete = [r for r in results if r.get("status") == "complete"]
    if not complete:
        raise RuntimeError("no candidate completed nested development")
    selected = min(complete, key=lambda r: (r["brier"], r["log_loss"]))
    return results, selected["model"], oof_by_model[selected["model"]]


def main():
    source = Path("datas/580-analysis.xlsx")
    df, audit_result = audit(source)
    dev, _, _ = locked_split(df)
    X, y, feature_set = get_features(dev, "m1", "independent")
    results, selected_name, selected_oof = nested_development(X, y, feature_set)
    selected = next(c for c in available_candidates(feature_set.numeric, feature_set.categorical) if c.name == selected_name)
    selected.estimator.fit(X, y)
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(selected.estimator, "artifacts/model.joblib")
    report = {
        "dataset_sha256": audit_result.sha256,
        "model": selected_name,
        "feature_set": feature_set.name,
        "prediction_time": feature_set.prediction_time,
        "development_rows": len(dev),
        "development_events": int(y.sum()),
        "candidates": results,
    }
    Path("reports/development_models.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    threshold = min((0.05, 0.10, 0.15, 0.20, 0.30, 0.40), key=lambda t: abs(((selected_oof >= t) == y.to_numpy()).mean() - 0.8))
    manifest = {"dataset_sha256": audit_result.sha256, "model_sha256": sha256("artifacts/model.joblib"), "selected_model": selected_name, "feature_set": "m1", "threshold": threshold, "validation_locked": False}
    Path("artifacts/freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "candidates": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
