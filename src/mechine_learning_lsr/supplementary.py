from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve

from .audit import audit
from .data import locked_split
from .features import get_features
from .metrics import calibration_metrics, classification_metrics, threshold_metrics


THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
SEED = 20260908
N_BOOT = 2000
N_MODEL_BOOT = 300


def _ci(values: list[float | None]) -> list[float] | None:
    clean = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(clean):
        return None
    return [float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))]


def _bootstrap_metric_rows(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> dict:
    idx = rng.integers(0, len(y), len(y))
    metrics = {**classification_metrics(y[idx], p[idx]), **calibration_metrics(y[idx], p[idx])}
    return metrics


def _metric_ci(y: np.ndarray, p: np.ndarray) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(SEED)
    samples = [_bootstrap_metric_rows(y, p, rng) for _ in range(N_BOOT)]
    names = ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope")
    return {name: _ci([row.get(name) for row in samples]) for name in names}


def _threshold_ci(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for threshold in THRESHOLDS:
        point = threshold_metrics(y, p, threshold)
        samples = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            samples.append(threshold_metrics(y[idx], p[idx], threshold))
        row = {"threshold": threshold}
        for name in ("sensitivity", "specificity", "ppv", "npv", "f1"):
            row[name] = point.get(name)
            row[f"{name}_ci"] = _ci([sample.get(name) for sample in samples])
        rows.append(row)
    return rows


def _patient_probability_intervals(model, X_dev, y_dev, X_val) -> np.ndarray:
    rng = np.random.default_rng(SEED + 2)
    predictions = np.empty((N_MODEL_BOOT, len(X_val)), dtype=float)
    y_array = np.asarray(y_dev, dtype=int)
    for repeat in range(N_MODEL_BOOT):
        indices = rng.integers(0, len(X_dev), len(X_dev))
        fitted = clone(model).fit(X_dev.iloc[indices], y_array[indices])
        predictions[repeat] = fitted.predict_proba(X_val)[:, 1]
    return np.quantile(predictions, [0.025, 0.975], axis=0).T


def _calibration_plot(y: np.ndarray, p: np.ndarray, output: str) -> None:
    bins = np.quantile(p, np.linspace(0, 1, 6))
    bins[0], bins[-1] = 0.0, 1.0
    groups = np.digitize(p, np.unique(bins)[1:-1], right=True)
    rows = []
    rng = np.random.default_rng(SEED + 3)
    for group in sorted(np.unique(groups)):
        mask = groups == group
        if not np.any(mask):
            continue
        boot_obs = []
        for _ in range(N_BOOT):
            sample = rng.choice(np.flatnonzero(mask), size=int(mask.sum()), replace=True)
            boot_obs.append(float(y[sample].mean()))
        rows.append({
            "bin": int(group + 1),
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "observed": float(y[mask].mean()),
            "observed_ci": _ci(boot_obs),
        })
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.array([row["mean_predicted"] for row in rows])
    observed = np.array([row["observed"] for row in rows])
    lower = np.array([row["observed_ci"][0] for row in rows])
    upper = np.array([row["observed_ci"][1] for row in rows])
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.errorbar(x, observed, yerr=[observed - lower, upper - observed], fmt="o-", capsize=3, label="Frozen model")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed event rate", xlim=(0, 1), ylim=(0, 1), title="Group 2 calibration (bootstrap 95% CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output).with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    Path("reports/group2_calibration_bins.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _performance_plots(metric_point: dict, metric_ci: dict, threshold_rows: list[dict], predictions: pd.DataFrame) -> None:
    names = ["AUROC", "AUPRC", "1-Brier"]
    values = [metric_point["auroc"], metric_point["auprc"], 1 - metric_point["brier"]]
    intervals = [metric_ci["auroc"], metric_ci["auprc"], [1 - metric_ci["brier"][1], 1 - metric_ci["brier"][0]]]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    lower = [values[i] - intervals[i][0] for i in range(len(values))]
    upper = [intervals[i][1] - values[i] for i in range(len(values))]
    ax.errorbar(x, values, yerr=[lower, upper], fmt="o", capsize=5, lw=2, color="#4c78a8")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Group 2 overall performance (bootstrap 95% CI)")
    for index, value in enumerate(values):
        ax.text(index, min(value + 0.06, 0.98), f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig("results/09_group2_overall_performance_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/09_group2_overall_performance_ci.svg", bbox_inches="tight")
    plt.close(fig)


def _learning_curve_plot(model, X_dev: pd.DataFrame, y_dev: pd.Series) -> None:
    sizes, train_scores, validation_scores = learning_curve(
        clone(model), X_dev, y_dev,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        n_jobs=1,
    )
    train_mean = train_scores.mean(axis=1)
    train_sd = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_sd = validation_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_mean, "o-", label="Training AUROC", color="#4c78a8")
    ax.fill_between(sizes, train_mean - train_sd, train_mean + train_sd, color="#4c78a8", alpha=0.15)
    ax.plot(sizes, validation_mean, "o-", label="5-fold CV AUROC", color="#e45756")
    ax.fill_between(sizes, validation_mean - validation_sd, validation_mean + validation_sd, color="#e45756", alpha=0.15)
    ax.set(xlabel="Number of Group 1 patients used", ylabel="AUROC", ylim=(0.4, 1.02), title="Gradient Boosting learning curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/13_group1_learning_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/13_group1_learning_curve.svg", bbox_inches="tight")
    plt.close(fig)
    Path("reports/group1_learning_curve.json").write_text(json.dumps({"train_sizes": sizes.tolist(), "training_auroc_mean": train_mean.tolist(), "training_auroc_sd": train_sd.tolist(), "cv_auroc_mean": validation_mean.tolist(), "cv_auroc_sd": validation_sd.tolist()}, indent=2), encoding="utf-8")


def _ks_curve_plot(y: np.ndarray, p: np.ndarray) -> None:
    order = np.argsort(-p)
    y_sorted = y[order]
    total_events = max(int(y.sum()), 1)
    total_nonevents = max(int((1 - y).sum()), 1)
    cumulative_events = np.cumsum(y_sorted) / total_events
    cumulative_nonevents = np.cumsum(1 - y_sorted) / total_nonevents
    ks = cumulative_events - cumulative_nonevents
    max_index = int(np.argmax(ks))
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, len(y) + 1) / len(y)
    ax.plot(x, cumulative_events, label="Cumulative events", color="#e45756")
    ax.plot(x, cumulative_nonevents, label="Cumulative non-events", color="#4c78a8")
    ax.axvline(x[max_index], color="#54a24b", linestyle="--", label=f"Max KS={ks[max_index]:.3f}")
    ax.set(xlabel="Fraction of Group 2 ranked by predicted risk", ylabel="Cumulative proportion", ylim=(0, 1.02), title="Group 2 KS curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/14_group2_ks_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/14_group2_ks_curve.svg", bbox_inches="tight")
    plt.close(fig)
    Path("reports/group2_ks.json").write_text(json.dumps({"ks": float(ks[max_index]), "rank_fraction": float(x[max_index]), "threshold_at_max_ks": float(p[order[max_index]]), "events": int(y.sum()), "non_events": int((1 - y).sum())}, indent=2), encoding="utf-8")


def _confusion_matrix_plot(y: np.ndarray, p: np.ndarray, threshold: float = 0.10) -> None:
    predicted = (p >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Patients")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted no event", "Predicted event"], yticklabels=["Observed no event", "Observed event"], title=f"Group 2 confusion matrix (threshold={threshold:.2f})")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max() * 0.5 else "black", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/15_group2_confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/15_group2_confusion_matrix.svg", bbox_inches="tight")
    plt.close(fig)
    Path("reports/group2_confusion_matrix.json").write_text(json.dumps({"threshold": threshold, "labels": ["no_event", "event"], "matrix": matrix.tolist(), "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])}, indent=2), encoding="utf-8")


def _performance_table(metric_point: dict, metric_ci: dict, threshold_rows: list[dict], predictions: pd.DataFrame) -> None:
    def interval(value, ci):
        return f"{value:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"

    overall_rows = [
        ["AUROC", interval(metric_point["auroc"], metric_ci["auroc"])],
        ["AUPRC", interval(metric_point["auprc"], metric_ci["auprc"])],
        ["Brier", interval(metric_point["brier"], metric_ci["brier"])],
        ["Log-loss", interval(metric_point["log_loss"], metric_ci["log_loss"])],
        ["Calibration intercept", interval(metric_point["calibration_intercept"], metric_ci["calibration_intercept"])],
        ["Calibration slope", interval(metric_point["calibration_slope"], metric_ci["calibration_slope"])],
    ]
    threshold_rows_display = []
    for row in threshold_rows:
        threshold_rows_display.append([
            f"{row['threshold']:.2f}",
            interval(row["sensitivity"], row["sensitivity_ci"]),
            interval(row["specificity"], row["specificity_ci"]),
            interval(row["ppv"], row["ppv_ci"]),
            interval(row["npv"], row["npv_ci"]),
            interval(row["f1"], row["f1_ci"]),
        ])
    lines = [
        "# Group 2完整性能表", "", "冻结模型在Group 2（104例，9个事件）中的点估计与bootstrap 95% CI。", "",
        "## 总体性能", "", "| 指标 | 点估计（95% CI） |", "|---|---|",
    ]
    lines += [f"| {row[0]} | {row[1]} |" for row in overall_rows]
    lines += ["", "## 预设阈值下分类性能", "", "| 阈值 | 敏感度 | 特异度 | PPV | NPV | F1 |", "|---:|---|---|---|---|---|"]
    lines += ["| " + " | ".join(row) + " |" for row in threshold_rows_display]
    Path("results/group2_performance_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1.35]})
    axes[0].axis("off")
    axes[1].axis("off")
    table1 = axes[0].table(cellText=overall_rows, colLabels=["Overall metric", "Estimate (95% CI)"], loc="center", cellLoc="center", colWidths=[0.32, 0.48])
    table2 = axes[1].table(cellText=threshold_rows_display, colLabels=["Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "F1"], loc="center", cellLoc="center", colWidths=[0.11, 0.18, 0.18, 0.18, 0.18, 0.18])
    for table in (table1, table2):
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4c78a8")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#eef3f8")
    axes[0].set_title("Group 2 overall performance", fontweight="bold", pad=12)
    axes[1].set_title("Threshold performance", fontweight="bold", pad=12)
    fig.suptitle("Frozen Gradient Boosting model: performance table", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/12_group2_performance_table.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/12_group2_performance_table.svg", bbox_inches="tight")
    plt.close(fig)

    metric_names = ["sensitivity", "specificity", "ppv", "npv", "f1"]
    labels = ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for row, color in zip(threshold_rows, colors):
        values = [float(row[name]) for name in metric_names]
        values += values[:1]
        axis.plot(angles, values, "o-", lw=2, color=color, label=f"Threshold {row['threshold']:.2f}")
        axis.fill(angles, values, color=color, alpha=0.06)
    axis.set_xticks(angles[:-1], labels)
    axis.set_ylim(0, 1)
    axis.set_yticks([0.25, 0.50, 0.75, 1.00])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    axis.set_title("Group 2 threshold performance", pad=25)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)
    fig.text(0.5, 0.02, "Point estimates shown; bootstrap 95% CI are reported in the accompanying table.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/10_group2_threshold_performance_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/10_group2_threshold_performance_ci.svg", bbox_inches="tight")
    plt.close(fig)

    ordered = predictions.sort_values("predicted_probability").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ordered))
    colors = np.where(ordered["outcome"].to_numpy() == 1, "#e45756", "#4c78a8")
    ax.errorbar(x, ordered["predicted_probability"], yerr=[ordered["predicted_probability"] - ordered["probability_ci_lower"], ordered["probability_ci_upper"] - ordered["predicted_probability"]], fmt="none", ecolor="#9aa0a6", alpha=0.5, capsize=1)
    ax.scatter(x, ordered["predicted_probability"], c=colors, s=18)
    ax.set(xlabel="Group 2 patient order (sorted by predicted risk)", ylabel="Predicted probability", title="Group 2 patient-level probability intervals")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="w", markerfacecolor="#e45756", label="Event", markersize=7), plt.Line2D([], [], marker="o", color="w", markerfacecolor="#4c78a8", label="No event", markersize=7)], frameon=False)
    fig.tight_layout()
    fig.savefig("results/11_group2_probability_intervals.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/11_group2_probability_intervals.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    source = Path("datas/580-analysis.xlsx")
    df, _ = audit(source)
    dev, _, validation = locked_split(df)
    X_dev, y_dev, feature_set = get_features(dev, "m1", "independent")
    X_val, y_val, _ = get_features(validation, "m1", "independent")
    model = joblib.load("artifacts/model.joblib")
    probability = model.predict_proba(X_val)[:, 1]
    y = np.asarray(y_val, dtype=int)
    metric_point = {**classification_metrics(y, probability), **calibration_metrics(y, probability)}
    metric_ci = _metric_ci(y, probability)
    threshold_rows = _threshold_ci(y, probability)
    intervals = _patient_probability_intervals(model, X_dev, y_dev, X_val)

    Path("reports").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    predictions = pd.DataFrame({
        "validation_row": np.arange(1, len(y) + 1),
        "outcome": y,
        "predicted_probability": probability,
        "probability_ci_lower": intervals[:, 0],
        "probability_ci_upper": intervals[:, 1],
    })
    predictions.to_csv("reports/group2_patient_predictions.csv", index=False)
    payload = {
        "analysis_type": "supplementary frozen-model analysis",
        "note": "Predictions use the already frozen model; no Group 2 tuning or refitting was performed. Patient-level probability intervals are bootstrap parameter-uncertainty intervals from 300 refits on resampled Group 1 data.",
        "model": "gradient_boosting",
        "group2_rows": len(y),
        "group2_events": int(y.sum()),
        "bootstrap_replicates_metrics": N_BOOT,
        "bootstrap_replicates_probability_intervals": N_MODEL_BOOT,
        "point_estimates": metric_point,
        "point_estimate_ci": metric_ci,
        "threshold_metrics": threshold_rows,
    }
    Path("reports/group2_supplementary_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_lines = [
        "# Group 2完整性能与校准结果", "",
        "> 这是冻结模型的补充分析，不改变已锁定的时间验证。Group 2含104例、9个事件；所有置信区间均为bootstrap 95% CI。", "",
        "## 总体性能", "",
        "| 指标 | 点估计 | 95% CI |", "|---|---:|---:|",
    ]
    for name, label in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier"), ("log_loss", "Log-loss"), ("calibration_intercept", "校准截距"), ("calibration_slope", "校准斜率")):
        point = metric_point[name]
        ci = metric_ci[name]
        metric_lines.append(f"| {label} | {point:.4f} | {ci[0]:.4f}–{ci[1]:.4f} |" if point is not None and ci else f"| {label} | {point} | {ci} |")
    metric_lines += ["", "## 预设阈值下的分类指标", "", "| 阈值 | 敏感度 | 特异度 | PPV | NPV | F1 |", "|---:|---|---|---|---|---|"]
    for row in threshold_rows:
        def fmt(name):
            value, ci = row[name], row[f"{name}_ci"]
            return f"{value:.3f} ({ci[0]:.3f}–{ci[1]:.3f})" if value is not None and ci else "NA"
        metric_lines.append(f"| {row['threshold']:.2f} | {fmt('sensitivity')} | {fmt('specificity')} | {fmt('ppv')} | {fmt('npv')} | {fmt('f1')} |")
    metric_lines += ["", "## 校准结果", "", "![校准曲线](../results/08_group2_calibration.png)", "", "校准曲线按预测概率五分位分组，误差线为组内观察事件率的bootstrap 95% CI。每位 Group 2 患者的预测概率及参数不确定性区间见 `reports/group2_patient_predictions.csv`。这些区间不是患者结局的确定性区间，而是对开发队列重抽样并重新拟合固定超参数模型后得到的预测概率不确定性区间。", "", "## 文件", "", "- `reports/group2_supplementary_metrics.json`：机器可读的点估计、CI和阈值指标。", "- `reports/group2_patient_predictions.csv`：104例验证患者的匿名行号、实际结局、预测概率及95%参数不确定性区间。", "- `reports/group2_calibration_bins.json`：校准分箱数据。"]
    Path("reports/group2_complete_performance.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")
    _calibration_plot(y, probability, "results/08_group2_calibration.png")
    _performance_plots(metric_point, metric_ci, threshold_rows, predictions)
    _performance_table(metric_point, metric_ci, threshold_rows, predictions)
    _learning_curve_plot(model, X_dev, y_dev)
    _ks_curve_plot(y, probability)
    _confusion_matrix_plot(y, probability, threshold=0.10)
    print(json.dumps({"rows": len(y), "events": int(y.sum()), "bootstrap_metrics": N_BOOT, "bootstrap_probability_intervals": N_MODEL_BOOT}, ensure_ascii=False))


if __name__ == "__main__":
    main()
