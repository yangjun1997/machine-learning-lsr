from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .audit import audit
from .data import locked_split
from .features import get_features


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _source_variable(feature_name: str, columns: list[str]) -> str:
    raw = feature_name.split("__", 1)[-1]
    matches = [column for column in columns if raw == column or raw.startswith(f"{column}_")]
    return max(matches, key=len) if matches else raw


def _save(fig, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(Path("results") / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    source = Path("datas/580-analysis.xlsx")
    df, audit_result = audit(source)
    dev, _, _ = locked_split(df)
    X, y, feature_set = get_features(dev, "m1", "independent")
    pipeline = joblib.load("artifacts/model.joblib")
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    shap_values = shap.TreeExplainer(estimator).shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    columns = feature_set.columns
    source_names = [_source_variable(name, columns) for name in feature_names]
    encoded = pd.DataFrame({
        "encoded_feature": feature_names,
        "source_variable": source_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    encoded["rank"] = np.arange(1, len(encoded) + 1)

    variable_rows = []
    for variable in columns:
        indices = [i for i, name in enumerate(source_names) if name == variable]
        contribution = shap_values[:, indices].sum(axis=1)
        variable_rows.append({
            "variable": variable,
            "mean_abs_shap": float(np.abs(contribution).mean()),
            "mean_shap": float(contribution.mean()),
            "encoded_columns": ", ".join(feature_names[i] for i in indices),
        })
    variables = pd.DataFrame(variable_rows).sort_values("mean_abs_shap", ascending=False)
    variables["rank"] = np.arange(1, len(variables) + 1)

    Path("reports").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    encoded.to_csv("reports/shap_encoded_feature_importance.csv", index=False)
    variables.to_csv("reports/shap_variable_importance.csv", index=False)
    top_variables = variables.head(5).copy()
    top_variables.to_csv("reports/shap_variable_importance_top5.csv", index=False)

    params = {
        "dataset_sha256": audit_result.sha256,
        "development_rows": len(dev),
        "development_events": int(y.sum()),
        "selected_model": "gradient_boosting",
        "feature_set": feature_set.name,
        "prediction_time": feature_set.prediction_time,
        "training_note": "The frozen estimator was refit on all 476 Group 1 patients after nested-CV model selection. Group 2 was not used for fitting or tuning.",
        "numeric_features": list(feature_set.numeric),
        "categorical_features": list(feature_set.categorical),
        "preprocessing": _jsonable(preprocessor.get_params(deep=True)),
        "estimator": _jsonable(estimator.get_params(deep=True)),
        "shap": {
            "explainer": "shap.TreeExplainer",
            "output": "raw Gradient Boosting margin (log-odds contribution)",
            "rows_explained": len(dev),
            "encoded_features": len(feature_names),
        },
    }
    Path("artifacts/model_parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    key_params = [
        ("n_estimators", estimator.n_estimators),
        ("learning_rate", estimator.learning_rate),
        ("max_depth", estimator.max_depth),
        ("min_samples_leaf", estimator.min_samples_leaf),
        ("subsample", estimator.subsample),
        ("random_state", estimator.random_state),
        ("loss", estimator.loss),
    ]
    lines = [
        "# 476例开发队列：最终模型参数与SHAP解释", "",
        "## 模型如何得到", "",
        "先在 Group 1 的476例患者中进行嵌套交叉验证比较候选模型，再按 Brier score 选择 Gradient Boosting，最后用全部476例 Group 1 重新拟合并保存为 `artifacts/model.joblib`。Group 2 没有参与拟合、调参或特征选择。", "",
        "## 最终模型关键参数", "",
        "| 参数 | 值 |", "|---|---:|",
    ]
    lines += [f"| `{name}` | `{value}` |" for name, value in key_params]
    lines += [
        "", "## 预处理规则", "",
        "- 数值变量：中位数填补，并保留缺失指示变量；随后标准化。",
        "- 分类变量：众数填补，再进行 one-hot 编码；未知类别忽略。",
        f"- 数值变量：`{', '.join(feature_set.numeric)}`。",
        f"- 分类变量：`{', '.join(feature_set.categorical)}`。", "",
        "完整的 sklearn pipeline 参数见 [`../artifacts/model_parameters.json`](../artifacts/model_parameters.json)。", "",
        "## SHAP如何解释", "",
        "SHAP 使用 `TreeExplainer` 解释冻结的 Gradient Boosting。SHAP 值表示某个变量相对于模型基线对预测风险的贡献；本报告使用模型原始输出尺度（log-odds），正值表示推高预测风险，负值表示降低预测风险。分类变量的多个 one-hot 水平已重新合并为原始变量后计算变量级贡献。", "",
        "| 文件 | 含义 |", "|---|---|",
        "| `reports/shap_variable_importance.csv` | 原始变量级 SHAP 排名和方向 |",
        "| `reports/shap_encoded_feature_importance.csv` | one-hot 展开后的详细 SHAP 排名 |",
        "| `results/06_shap_variable_importance.png` | 原始变量平均绝对 SHAP 贡献 |",
        "| `results/07_shap_summary_beeswarm.png` | 编码后特征的 SHAP 分布 |", "",
        "平均绝对 SHAP 越大，表示该变量对模型预测的平均影响越大；它不是因果效应，也不是 p 值。", "",
        "## 注意事项", "",
        "SHAP 图解释的是已经拟合的开发队列模型，不是独立验证集上的因果分析。Group 2 只有9个事件，因此不应使用 SHAP 排名证明变量具有临床因果作用。"
    ]
    Path("reports/development_model_parameters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    top = variables.head(5).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["variable"], top["mean_abs_shap"], color="#4c78a8")
    ax.set_xlabel("Mean absolute SHAP value (raw model output)")
    ax.set_title("Development cohort: top 5 SHAP variables")
    _save(fig, "06_shap_variable_importance")

    top_source = set(top_variables["variable"])
    encoded_indices = [i for i, name in enumerate(source_names) if name in top_source]
    display = min(15, len(encoded_indices))
    shap.summary_plot(
        shap_values[:, encoded_indices],
        transformed[:, encoded_indices],
        feature_names=[feature_names[i] for i in encoded_indices],
        max_display=display,
        show=False,
        plot_size=(10, 7),
    )
    plt.title("Development cohort: SHAP summary for top 5 variables")
    plt.tight_layout()
    plt.savefig("results/07_shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/07_shap_summary_beeswarm.svg", bbox_inches="tight")
    plt.close()
    print(json.dumps({"rows": len(dev), "variables": len(variables), "encoded_features": len(encoded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
