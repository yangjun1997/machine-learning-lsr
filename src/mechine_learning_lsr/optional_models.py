from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .audit import audit
from .data import locked_split
from .features import get_features
from .metrics import bootstrap_interval, classification_metrics
from .models import available_candidates


def main() -> None:
    df, audit_result = audit("datas/580-analysis.xlsx")
    dev, _, _ = locked_split(df)
    X, y, feature_set = get_features(dev, "m1", "independent")
    candidates = {candidate.name: candidate for candidate in available_candidates(feature_set.numeric, feature_set.categorical)}
    names = [name for name in ("xgboost", "lightgbm", "catboost") if name in candidates]
    unavailable = [name for name in ("xgboost", "lightgbm", "catboost") if name not in candidates]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260904)
    rows = []
    for name in names:
        oof = np.full(len(y), np.nan)
        candidate = candidates[name]
        for train_idx, test_idx in cv.split(X, y):
            model = candidate.estimator
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append({
            "model": name,
            "status": "complete",
            **classification_metrics(y, oof),
            "auroc_ci": bootstrap_interval(y, oof, "auroc"),
            "auprc_ci": bootstrap_interval(y, oof, "auprc"),
            "note": "Supplementary Group 1 five-fold OOF comparison; not used to alter the frozen Gradient Boosting model or Group 2 validation.",
        })
    rows.extend({"model": name, "status": "unavailable", "note": "Package could not be imported in this environment."} for name in unavailable)
    payload = {"dataset_sha256": audit_result.sha256, "development_rows": len(dev), "development_events": int(y.sum()), "models": rows}
    Path("reports/optional_boosting_models.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# XGBoost、LightGBM、CatBoost补充比较", "", "该表仅为 Group 1 的五折 OOF 补充比较，不改变已经冻结的 Gradient Boosting，也没有使用 Group 2 进行选模。", "", "| 模型 | 状态 | Brier | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI |", "|---|---|---:|---:|---|---:|---|"]
    for row in rows:
        if row["status"] == "complete":
            lines.append(f"| {row['model']} | complete | {row['brier']:.4f} | {row['auroc']:.4f} | {row['auroc_ci'][0]:.4f}–{row['auroc_ci'][1]:.4f} | {row['auprc']:.4f} | {row['auprc_ci'][0]:.4f}–{row['auprc_ci'][1]:.4f} |")
        else:
            lines.append(f"| {row['model']} | unavailable | NA | NA | NA | NA | NA |")
    Path("reports/optional_boosting_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
