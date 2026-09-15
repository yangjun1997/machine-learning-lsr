from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

from .data import load_workbook, locked_split, sha256
from .features import get_features


RESULTS = Path("results")
FIGURES = RESULTS / "图"
TABLES = RESULTS / "表"
REPORTS = Path("reports")
ARTIFACTS = Path("artifacts")
OPTIONAL_MODELS = {"xgboost", "lightgbm", "catboost"}
EXCLUDED_PUBLICATION_MODELS = {"logistic_l1", "logistic_l2"}
COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "grey": "#BBBBBB",
    "dark_grey": "#555555",
}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
}


MODEL_LABELS = {
    "logistic_l1": "Logistic regression (L1)",
    "logistic_l2": "Logistic regression (L2)",
    "logistic_elasticnet": "Logistic regression (elastic net)",
    "lda": "Linear discriminant analysis",
    "qda": "Quadratic discriminant analysis",
    "gaussian_nb": "Gaussian naive Bayes",
    "knn": "k-nearest neighbors",
    "svm_linear": "Linear support vector machine",
    "svm_rbf": "RBF support vector machine",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "mlp": "Multilayer perceptron",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, text: str) -> None:
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=14, fontweight="bold")


def _flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        linespacing=1.25,
    )


def _flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#555555",
        )
    )


def figure_1(df: pd.DataFrame, development: dict) -> None:
    dev, washout, val = locked_split(df)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Study design and analysis workflow", fontsize=14, fontweight="bold", pad=10)

    _flow_box(
        ax,
        (0.25, 0.82),
        0.50,
        0.11,
        f"Retrospective single-center cohort\nN={len(df)}; events={int(df.outcome.sum())}\n"
        f"Observed: {df.admission_date.min():%d %b %Y} to {df.admission_date.max():%d %b %Y}",
        COLORS["blue"],
    )
    _flow_box(
        ax,
        (0.04, 0.62),
        0.41,
        0.13,
        f"Group 1: development cohort\nn={len(dev)}; events={int(dev.outcome.sum())} "
        f"({100 * dev.outcome.mean():.1f}%)\n"
        f"{dev.admission_date.min():%d %b %Y} to {dev.admission_date.max():%d %b %Y}",
        COLORS["teal"],
    )
    _flow_box(
        ax,
        (0.55, 0.62),
        0.41,
        0.13,
        f"Group 2: internal temporal validation\nn={len(val)}; events={int(val.outcome.sum())} "
        f"({100 * val.outcome.mean():.1f}%)\n"
        f"{val.admission_date.min():%d %b %Y} to {val.admission_date.max():%d %b %Y}",
        COLORS["orange"],
    )
    _flow_box(
        ax,
        (0.04, 0.37),
        0.41,
        0.16,
        "Final model: 12 source predictors\n"
        "Age, duration, BMI, KPS + 6 binary covariates\n"
        "Two branch LSR variables independently one-hot encoded\n"
        f"{len(_development_rows(development))} fixed candidates; stratified five-fold comparison",
        COLORS["blue"],
    )
    _flow_box(
        ax,
        (0.55, 0.37),
        0.41,
        0.16,
        "Held out until model freeze\n\nOne-time evaluation of the frozen model\n"
        "No model selection, tuning, feature selection, or recalibration",
        COLORS["red"],
    )
    _flow_box(
        ax,
        (0.04, 0.17),
        0.41,
        0.11,
        "Primary selection by lowest out-of-fold Brier score\n"
        "Gradient Boosting refitted on all Group 1 patients and frozen",
        COLORS["teal"],
    )
    _flow_box(
        ax,
        (0.55, 0.17),
        0.41,
        0.11,
        "Discrimination, proper scoring, and calibration\n"
        "Exploratory decision-curve and threshold analyses",
        COLORS["orange"],
    )
    _flow_box(
        ax,
        (0.16, 0.025),
        0.68,
        0.07,
        "Five main figures, three main tables, and supplementary diagnostics",
        COLORS["dark_grey"],
    )

    _flow_arrow(ax, (0.42, 0.82), (0.25, 0.75))
    _flow_arrow(ax, (0.58, 0.82), (0.75, 0.75))
    _flow_arrow(ax, (0.245, 0.62), (0.245, 0.53))
    _flow_arrow(ax, (0.755, 0.62), (0.755, 0.53))
    _flow_arrow(ax, (0.245, 0.37), (0.245, 0.28))
    _flow_arrow(ax, (0.755, 0.37), (0.755, 0.28))
    _flow_arrow(ax, (0.245, 0.17), (0.37, 0.095))
    _flow_arrow(ax, (0.755, 0.17), (0.63, 0.095))
    ax.text(
        0.50,
        0.675,
        f"Washout\nJul-Aug 2022\nn={len(washout)}",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )
    _save_figure(fig, "Figure_1_study_flow")


def _development_rows(development: dict) -> list[dict]:
    return [
        row
        for row in development["candidates"]
        if row.get("status") == "complete"
        and row["model"] not in OPTIONAL_MODELS | EXCLUDED_PUBLICATION_MODELS
    ]


def figure_2(development: dict, optional: dict) -> None:
    rows = [
        dict(row, comparison_role="Primary fixed candidate")
        for row in _development_rows(development)
    ]
    primary_count = len(rows)
    rows.extend(
        dict(row, comparison_role="Post-freeze supplementary")
        for row in optional.get("models", [])
        if row.get("status") == "complete"
    )
    rows.sort(key=lambda row: row["brier"])
    names = [MODEL_LABELS[row["model"]] for row in rows]
    y = np.arange(len(rows))
    colors = [
        COLORS["red"]
        if row["model"] == "gradient_boosting"
        else COLORS["teal"]
        if row["model"] == "logistic_elasticnet"
        else COLORS["orange"]
        if row["comparison_role"] == "Post-freeze supplementary"
        else COLORS["blue"]
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    axes[0].barh(y, [row["brier"] for row in rows], color=colors)
    axes[1].barh(y, [row["auroc"] for row in rows], color=colors)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(row["brier"] for row in rows) * 1.17)
    axes[1].set_xlim(0, 1)
    axes[0].set_xlabel("Out-of-fold Brier score (lower is better)")
    axes[1].set_xlabel("Out-of-fold AUROC (higher is better)")
    axes[0].set_title("A  Proper-scoring comparison")
    axes[1].set_title("B  Discrimination comparison")
    for ax, key in zip(axes, ("brier", "auroc")):
        for pos, row in enumerate(rows):
            ax.text(row[key] + ax.get_xlim()[1] * 0.01, pos, f"{row[key]:.3f}", va="center", fontsize=7)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Group 1 performance of available candidate models",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.01,
        f"Displayed: {primary_count + 1} models total ({primary_count} fixed primary models plus "
        "1 post-freeze supplementary LightGBM). "
        "Gradient Boosting remains the frozen index model.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.29, bottom=0.09, top=0.90, wspace=0.18)
    _save_figure(fig, "Figure_2_model_comparison")


def _source_variable(feature_name: str, columns: list[str]) -> str:
    raw = feature_name.split("__", 1)[-1]
    matches = [column for column in columns if raw == column or raw.startswith(f"{column}_")]
    return max(matches, key=len) if matches else raw


def _feature_label(name: str) -> str:
    raw = name.split("__", 1)[-1]
    labels = {
        "duration": "Disease duration",
        "age": "Age",
        "bmi": "BMI",
        "kps": "KPS",
        "acupuncture": "Prior acupuncture",
        "acupuncture_0": "Prior acupuncture: no",
        "acupuncture_1": "Prior acupuncture: yes",
        "zyg_lsr": "Zygomatic-branch LSR",
        "zyg_lsr_1": "Zygomatic LSR: disappeared",
        "zyg_lsr_2": "Zygomatic LSR: not elicited",
        "zyg_lsr_3": "Zygomatic LSR: persisted",
        "man_lsr": "Mandibular-branch LSR",
        "man_lsr_1": "Mandibular LSR: disappeared",
        "man_lsr_2": "Mandibular LSR: not elicited",
        "man_lsr_3": "Mandibular LSR: persisted",
    }
    return labels.get(raw, raw.replace("_", " ").capitalize())


def figure_3(df: pd.DataFrame) -> None:
    dev, _, _ = locked_split(df)
    X, y, feature_set = get_features(dev, "m1", "independent")
    pipeline = joblib.load(ARTIFACTS / "model.joblib")
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(X), dtype=float)
    feature_names = list(preprocessor.get_feature_names_out())
    shap_values = shap.TreeExplainer(estimator).shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values, dtype=float)
    source_names = [_source_variable(name, feature_set.columns) for name in feature_names]

    variable_rows = []
    for variable in feature_set.columns:
        indices = [i for i, source in enumerate(source_names) if source == variable]
        contribution = shap_values[:, indices].sum(axis=1)
        variable_rows.append((variable, float(np.abs(contribution).mean())))
    variables = sorted(variable_rows, key=lambda item: item[1], reverse=True)[:5]
    top_sources = {name for name, _ in variables}
    encoded_indices = [i for i, source in enumerate(source_names) if source in top_sources]
    encoded_indices = sorted(
        encoded_indices,
        key=lambda i: float(np.abs(shap_values[:, i]).mean()),
        reverse=True,
    )[:10]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [0.9, 1.35]},
    )
    ordered = list(reversed(variables))
    ax_a.barh(
        [_feature_label(name) for name, _ in ordered],
        [value for _, value in ordered],
        color=COLORS["blue"],
    )
    ax_a.set_xlabel("Mean absolute SHAP value (raw model output)")
    ax_a.set_title("Original-variable importance")
    ax_a.grid(axis="x", alpha=0.2)

    rng = np.random.default_rng(20260914)
    y_positions = np.arange(len(encoded_indices))[::-1]
    scatter = None
    for y_pos, index in zip(y_positions, encoded_indices):
        feature_value = transformed[:, index]
        low, high = np.nanpercentile(feature_value, [5, 95])
        normalized = np.zeros_like(feature_value) if high <= low else np.clip((feature_value - low) / (high - low), 0, 1)
        jitter = rng.normal(0, 0.075, len(feature_value))
        scatter = ax_b.scatter(
            shap_values[:, index],
            y_pos + jitter,
            c=normalized,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=10,
            alpha=0.75,
            linewidths=0,
        )
    ax_b.axvline(0, color="#777777", linewidth=0.8)
    ax_b.set_yticks(y_positions, [_feature_label(feature_names[i]) for i in encoded_indices])
    ax_b.set_xlabel("SHAP value (impact on raw model output)")
    ax_b.set_title("Encoded-feature SHAP distribution")
    ax_b.grid(axis="x", alpha=0.15)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax_b, fraction=0.035, pad=0.03)
        colorbar.set_label("Encoded feature value\nlow to high", fontsize=8)
        colorbar.set_ticks([0, 1], labels=["Low", "High"])
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    fig.suptitle(
        f"Gradient Boosting interpretation in Group 1 (n={len(y)})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.16, bottom=0.11, top=0.86, right=0.95, wspace=0.48)
    _save_figure(fig, "Figure_3_shap_interpretation")


def figure_4(
    validation: dict,
    calibration_bins: list[dict],
    supplementary: dict,
    confusion: dict,
) -> None:
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0], projection="polar")
    ax_d = fig.add_subplot(grid[1, 1])
    x = np.array([row["mean_predicted"] for row in calibration_bins])
    observed = np.array([row["observed"] for row in calibration_bins])
    lower = np.array([row["observed_ci"][0] for row in calibration_bins])
    upper = np.array([row["observed_ci"][1] for row in calibration_bins])
    ax_a.plot([0, 1], [0, 1], "--", color="#777777", label="Ideal")
    ax_a.errorbar(
        x,
        observed,
        yerr=[observed - lower, upper - observed],
        fmt="o-",
        capsize=3,
        color=COLORS["blue"],
        label="Frozen model",
    )
    ax_a.set(
        xlabel="Mean predicted probability",
        ylabel="Observed 1-year event rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Calibration by predicted-risk group",
    )
    ax_a.legend(frameon=False)

    curve = pd.DataFrame(validation["decision_curve"])
    ax_b.plot(curve.threshold, curve.net_benefit, "o-", color=COLORS["blue"], label="Frozen model")
    ax_b.plot(curve.threshold, curve.treat_all, "s--", color=COLORS["orange"], label="Treat all")
    ax_b.plot(curve.threshold, curve.treat_none, "^--", color=COLORS["dark_grey"], label="Treat none")
    ax_b.axhline(0, color="#777777", linewidth=0.7)
    ax_b.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title="Exploratory decision-curve analysis",
    )
    ax_b.legend(frameon=False)
    ax_b.grid(alpha=0.15)

    metric_names = ["sensitivity", "specificity", "ppv", "npv", "f1"]
    metric_labels = ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]
    threshold_colors = [COLORS["blue"], COLORS["orange"], COLORS["teal"], COLORS["red"]]
    for row, color in zip(supplementary["threshold_metrics"], threshold_colors):
        values = [float(row[name]) for name in metric_names]
        values += values[:1]
        ax_c.plot(angles, values, "o-", linewidth=1.5, color=color, label=f"Threshold {row['threshold']:.2f}")
        ax_c.fill(angles, values, color=color, alpha=0.05)
    ax_c.set_xticks(angles[:-1], metric_labels)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_c.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax_c.set_title("Exploratory threshold performance", pad=18)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=7)

    matrix = np.asarray(confusion["matrix"], dtype=int)
    image = ax_d.imshow(matrix, cmap="Blues", vmin=0)
    ax_d.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Predicted\nno event", "Predicted\nevent"],
        yticklabels=["Observed no event", "Observed event"],
        title=f"Confusion matrix (threshold={confusion['threshold']:.2f})",
    )
    for row in range(2):
        for column in range(2):
            ax_d.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() * 0.5 else "black",
                fontsize=14,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.04, label="Patients")
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    _panel_label(ax_c, "C")
    _panel_label(ax_d, "D")
    fig.suptitle(
        "Internal temporal validation in Group 2 (n=104; 9 events)",
        fontsize=14,
        fontweight="bold",
    )
    _save_figure(fig, "Figure_4_temporal_validation")


def figure_5(interactions: pd.DataFrame, strata: pd.DataFrame) -> None:
    order = ["duration", "age", "va", "pica", "aica", "other_vessel"]
    rows = interactions.set_index("modifier").loc[order].reset_index()
    labels = {
        "duration": f"Disease duration (per {rows.loc[rows.modifier == 'duration', 'scale'].iloc[0]:.2f}-year SD)",
        "age": f"Age (per {rows.loc[rows.modifier == 'age', 'scale'].iloc[0]:.2f}-year SD)",
        "va": "Vertebral artery (present vs absent)",
        "pica": "Posterior inferior cerebellar artery (present vs absent)",
        "aica": "Anterior inferior cerebellar artery (present vs absent)",
        "other_vessel": "Other vessel (not estimable)",
    }
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    y = np.arange(len(rows))[::-1]
    for pos, row in zip(y, rows.itertuples()):
        if row.modifier == "other_vessel":
            ax_a.text(0.0013, pos, "NE", va="center", fontsize=8, color=COLORS["dark_grey"])
            continue
        color = COLORS["blue"] if row.modifier in {"duration", "age"} else COLORS["dark_grey"]
        marker = "o" if row.modifier in {"duration", "age"} else "s"
        ax_a.errorbar(
            row.interaction_or,
            pos,
            xerr=[[row.interaction_or - row.ci_low], [row.ci_high - row.interaction_or]],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
        )
    ax_a.axvline(1, color="#777777", linestyle="--", linewidth=0.9)
    ax_a.set_xscale("log")
    ax_a.set_xlim(1e-3, 1e4)
    ax_a.set_ylim(-0.5, 5.5)
    ax_a.set_yticks(y, [labels[modifier] for modifier in order])
    ax_a.set_xlabel("Interaction odds ratio (log scale), bootstrap 95% CI")
    ax_a.set_title("Pooled exploratory interaction estimates")
    ax_a.plot([], [], "o", color=COLORS["blue"], label="Continuous modifier")
    ax_a.plot([], [], "s", color=COLORS["dark_grey"], label="Additional vessel analysis")
    ax_a.legend(frameon=False, loc="upper left")

    age = (
        strata.loc[strata.modifier == "age_ge_60"]
        .groupby(["modifier_group", "lsr_residual"], as_index=False)
        .agg(n=("n", "sum"), events=("events", "sum"))
    )
    combinations = [("<60", 0), ("<60", 1), (">=60", 0), (">=60", 1)]
    values = {}
    for group, residual in combinations:
        row = age[(age.modifier_group == group) & (age.lsr_residual == residual)].iloc[0]
        values[(group, residual)] = (int(row.n), int(row.events), 100 * row.events / row.n)
    x = np.arange(2)
    width = 0.34
    no_residual = [values[(group, 0)][2] for group in ("<60", ">=60")]
    residual = [values[(group, 1)][2] for group in ("<60", ">=60")]
    bars_a = ax_b.bar(
        x - width / 2,
        no_residual,
        width,
        color=COLORS["blue"],
        edgecolor="#333333",
        linewidth=0.5,
        label="No residual LSR",
    )
    bars_b = ax_b.bar(
        x + width / 2,
        residual,
        width,
        color=COLORS["orange"],
        edgecolor="#333333",
        linewidth=0.5,
        label="Residual LSR",
    )
    ax_b.set_ylim(0, 80)
    for bar in bars_b:
        _add_explicit_hatch(ax_b, bar.get_x(), bar.get_width(), bar.get_height())
    for bars, residual_value in ((bars_a, 0), (bars_b, 1)):
        for bar, group in zip(bars, ("<60", ">=60")):
            n, events, rate = values[(group, residual_value)]
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{events}/{n}\n{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_b.set_xticks(x, ["<60 years\n(n=468)", ">=60 years\n(n=112)"])
    ax_b.set_ylabel("Observed 1-year event rate (%)")
    ax_b.set_title("Observed rates by age and residual LSR")
    _add_bar_legend(ax_b)
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    fig.suptitle(
        "Residual-LSR effect modification in the pooled cohort (N=580)",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.14, top=0.86, wspace=0.27)
    _save_figure(fig, "Figure_5_exploratory_interaction")


def _add_explicit_hatch(
    ax: plt.Axes,
    x: float,
    width: float,
    height: float,
    *,
    y: float = 0,
    transform=None,
    spacing: float = 8.0,
    zorder: float = 3,
) -> None:
    """Draw slash hatch as independent line segments for Illustrator-safe SVG."""
    trans = transform or ax.transData
    inverse = trans.inverted()
    left, bottom = trans.transform((x, y))
    right, top = trans.transform((x + width, y + height))
    # For y = x - offset, this interval is exactly where the line intersects
    # the rectangle. Starting at left - top includes the upper-left corner.
    for offset in np.arange(left - top, right - bottom + spacing * 0.5, spacing):
        points = []
        for px in (left, right):
            py = px - offset
            if bottom <= py <= top:
                points.append((px, py))
        for py in (bottom, top):
            px = py + offset
            if left <= px <= right:
                points.append((px, py))
        unique = []
        for point in points:
            if not any(np.allclose(point, previous) for previous in unique):
                unique.append(point)
        if len(unique) >= 2:
            unique.sort(key=lambda point: point[0])
            data_points = inverse.transform(np.asarray([unique[0], unique[-1]]))
            ax.plot(
                data_points[:, 0],
                data_points[:, 1],
                color="#333333",
                linewidth=0.5,
                solid_capstyle="butt",
                zorder=zorder,
                clip_on=False,
                transform=trans,
            )


def _add_bar_legend(ax: plt.Axes) -> None:
    """Add a fully editable two-entry legend without SVG pattern fills."""
    left, width, height = 0.08, 0.10, 0.055
    for index, (label, color, hatched) in enumerate(
        (("No residual LSR", COLORS["blue"], False), ("Residual LSR", COLORS["orange"], True))
    ):
        bottom = 0.875 - index * 0.075
        patch = plt.Rectangle(
            (left, bottom),
            width,
            height,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.5,
            zorder=4,
        )
        ax.add_patch(patch)
        if hatched:
            _add_explicit_hatch(
                ax,
                left,
                width,
                height,
                y=bottom,
                transform=ax.transAxes,
                spacing=4.0,
                zorder=5,
            )
        ax.text(
            left + width + 0.035,
            bottom + height / 2,
            label,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
            zorder=4,
        )


def _p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.mean():.2f} ({values.std(ddof=1):.2f})"


def _median_iqr(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.2f} [{q1:.2f}, {q3:.2f}]"


def _n_percent(series: pd.Series, value: int) -> str:
    valid = series.dropna()
    count = int((valid == value).sum())
    return f"{count} ({100 * count / len(valid):.1f}%)"


def _continuous_p(dev: pd.DataFrame, val: pd.DataFrame, column: str) -> float:
    return float(
        mannwhitneyu(
            pd.to_numeric(dev[column], errors="coerce").dropna(),
            pd.to_numeric(val[column], errors="coerce").dropna(),
            alternative="two-sided",
            method="auto",
        ).pvalue
    )


def _categorical_p(dev: pd.DataFrame, val: pd.DataFrame, column: str) -> tuple[float, str]:
    levels = sorted(pd.concat([dev[column], val[column]]).dropna().unique())
    observed = np.array(
        [
            [int((dev[column] == level).sum()), int((val[column] == level).sum())]
            for level in levels
        ]
    )
    expected = chi2_contingency(observed, correction=False).expected_freq
    if observed.shape == (2, 2) and np.any(expected < 5):
        return float(fisher_exact(observed, alternative="two-sided").pvalue), "Fisher exact test (two-sided, 2 x 2)"
    shape = f"{observed.shape[0]} x {observed.shape[1]}"
    return float(chi2_contingency(observed, correction=False).pvalue), f"Pearson chi-square test ({shape})"


def table_1(df: pd.DataFrame) -> tuple[str, dict]:
    dev, _, val = locked_split(df)
    rows: list[list[str]] = []
    tests: set[str] = set()

    for label, column, summary in (
        ("Age, years", "age", _mean_sd),
        ("Disease duration, years", "duration", _median_iqr),
        ("Body mass index, kg/m2", "bmi", _mean_sd),
        ("Admission KPS score", "kps", _median_iqr),
        ("Intraoperative blood loss, mL", "blood_loss", _median_iqr),
    ):
        rows.append(
            [
                label,
                summary(dev[column]),
                summary(val[column]),
                _p_text(_continuous_p(dev, val, column)),
                "Mann-Whitney U test (two-sided)",
            ]
        )
        tests.add("Mann-Whitney U")

    binary = (
        ("Female sex", "sex", 0),
        ("Left-sided disease", "side", 1),
        ("Hypertension", "hypertension", 1),
        ("Diabetes mellitus", "diabetes", 1),
        ("Type 2 incision", "incision", 2),
        ("Prior botulinum toxin treatment", "botox", 1),
        ("Prior acupuncture", "acupuncture", 1),
        ("Vertebral artery involved", "va", 1),
        ("Posterior inferior cerebellar artery involved", "pica", 1),
        ("Anterior inferior cerebellar artery involved", "aica", 1),
        ("Other vessel involved", "other_vessel", 1),
    )
    for label, column, value in binary:
        p_value, test = _categorical_p(dev, val, column)
        tests.add(test)
        rows.append([label, _n_percent(dev[column], value), _n_percent(val[column], value), _p_text(p_value), test])

    lsr_levels = {
        1: "elicited and disappeared",
        2: "not elicited",
        3: "elicited and persisted",
    }
    for column, branch in (("zyg_lsr", "Zygomatic-branch LSR"), ("man_lsr", "Mandibular-branch LSR")):
        p_value, test = _categorical_p(dev, val, column)
        tests.add(test)
        for index, (value, level) in enumerate(lsr_levels.items()):
            rows.append(
                [
                    f"{branch}: {level}",
                    _n_percent(dev[column], value),
                    _n_percent(val[column], value),
                    _p_text(p_value) if index == 0 else "",
                    test if index == 0 else "Included in preceding omnibus test",
                ]
            )

    outcome_p, outcome_test = _categorical_p(dev, val, "outcome")
    tests.add(outcome_test)
    rows.append(
        [
            "1-year postoperative spasm",
            _n_percent(dev.outcome, 1),
            _n_percent(val.outcome, 1),
            _p_text(outcome_p),
            outcome_test,
        ]
    )
    lines = [
        "# Table 1. Cohort characteristics by temporal group",
        "",
        "| Characteristic | Group 1 development (n=476) | Group 2 internal temporal validation (n=104) | P value | Statistical method |",
        "|---|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines += [
        "",
        "**Note.** Values are mean (SD), median [IQR], or n (%). Percentages use the nonmissing value-specific denominator; the audited workbook contained no missing values. The final column gives the method used for each comparison. All continuous variables were compared using the two-sided Mann-Whitney U test. Binary or multinomial categorical variables were compared using the Pearson chi-square test without continuity correction; a two-sided Fisher exact test replaced it for a 2 x 2 table when any expected cell count was below 5. The branch-specific LSR P value is one omnibus 3 x 2 comparison across all three response categories and is shown on the first category row only. P values are descriptive, were not adjusted for multiple comparisons, and were not used for model selection. KPS, Karnofsky Performance Status; LSR, lateral spread response; SD, standard deviation; IQR, interquartile range.",
    ]
    return "\n".join(lines), {
        "tests": sorted(tests),
        "age_dev": _mean_sd(dev.age),
        "age_val": _mean_sd(val.age),
        "age_p": _continuous_p(dev, val, "age"),
        "duration_dev": _median_iqr(dev.duration),
        "duration_val": _median_iqr(val.duration),
        "duration_p": _continuous_p(dev, val, "duration"),
    }


def _ci_text(interval: list[float] | None, digits: int = 3) -> str:
    if not interval:
        return "NA"
    return f"{interval[0]:.{digits}f}-{interval[1]:.{digits}f}"


def table_2(development: dict, optional: dict) -> str:
    rows = []
    for row in sorted(_development_rows(development), key=lambda item: item["brier"]):
        rows.append(
            [
                MODEL_LABELS[row["model"]],
                "Primary fixed comparison",
                "Complete",
                f"{row['brier']:.4f}",
                f"{row['log_loss']:.4f}",
                f"{row['auroc']:.3f} ({_ci_text(row.get('auroc_ci'))})",
                f"{row['auprc']:.3f} ({_ci_text(row.get('auprc_ci'))})",
            ]
        )
    for row in optional.get("models", []):
        if row["status"] == "complete":
            rows.append(
                [
                    MODEL_LABELS[row["model"]],
                    "Post-freeze supplementary",
                    "Complete",
                    f"{row['brier']:.4f}",
                    f"{row['log_loss']:.4f}",
                    f"{row['auroc']:.3f} ({_ci_text(row.get('auroc_ci'))})",
                    f"{row['auprc']:.3f} ({_ci_text(row.get('auprc_ci'))})",
                ]
            )
        else:
            rows.append([MODEL_LABELS[row["model"]], "Post-freeze supplementary", "Unavailable", "NA", "NA", "NA", "NA"])
    lines = [
        "# Table 2. Group 1 cross-validated performance of fixed candidate specifications",
        "",
        "| Model | Comparison role | Status | Brier score | Log-loss | AUROC (95% CI) | AUPRC (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines += [
        "",
        "**Note.** Primary values are out-of-fold predictions from stratified five-fold cross-validation in Group 1 (n=476; 58 events). The publication comparison contains 14 fixed primary specifications. The implementation passed each outer-training fold through a three-fold GridSearchCV wrapper with an empty parameter grid; no substantive hyperparameter optimization occurred. Gradient Boosting was frozen because it had the lowest predefined Brier score, with log-loss used as the tie-breaker. AUROC and AUPRC were secondary. The stored primary-candidate report does not contain confidence intervals, so they are shown as NA. LightGBM was evaluated after model freeze as a supplementary comparison with bootstrap AUROC and AUPRC intervals and did not alter selection; XGBoost and CatBoost were unavailable in that environment. AUROC, area under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve; CI, confidence interval; RBF, radial basis function.",
    ]
    return "\n".join(lines)


def table_3(supplementary: dict) -> str:
    point = supplementary["point_estimates"]
    intervals = supplementary["point_estimate_ci"]
    labels = (
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Brier score", "brier"),
        ("Log-loss", "log_loss"),
        ("Calibration intercept", "calibration_intercept"),
        ("Calibration slope", "calibration_slope"),
    )
    lines = [
        "# Table 3. Performance of the frozen Gradient Boosting model in Group 2",
        "",
        "| Metric | Estimate | Bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for label, key in labels:
        lines.append(f"| {label} | {point[key]:.3f} | {_ci_text(intervals[key])} |")
    lines += [
        "",
        f"**Note.** Group 2 was the locked internal temporal validation cohort (n={supplementary['group2_rows']}; {supplementary['group2_events']} events). The 12-predictor model and its preprocessing pipeline were frozen before Group 2 evaluation; no Group 2 tuning, feature selection, or recalibration was performed. Confidence intervals are percentile intervals from {supplementary['bootstrap_replicates_metrics']:,} patient-level bootstrap resamples of Group 2 and are wide because only nine events occurred. Calibration intercept and slope were estimated together by fitting a logistic recalibration model to the frozen predicted log odds. AUROC, area under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve; CI, confidence interval.",
    ]
    return "\n".join(lines)


def supplementary_threshold_table(supplementary: dict) -> str:
    def metric(row: dict, name: str) -> str:
        return f"{row[name]:.3f} ({_ci_text(row[f'{name}_ci'])})"

    lines = [
        "# Table S1. Exploratory classification performance at configured thresholds in Group 2",
        "",
        "| Threshold | Sensitivity | Specificity | PPV | NPV | F1 score |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in supplementary["threshold_metrics"]:
        lines.append(
            f"| {row['threshold']:.2f} | {metric(row, 'sensitivity')} | "
            f"{metric(row, 'specificity')} | {metric(row, 'ppv')} | "
            f"{metric(row, 'npv')} | {metric(row, 'f1')} |"
        )
    lines += [
        "",
        "**Note.** Values are estimates (bootstrap 95% CI). Thresholds were configured before reporting, but no threshold range was confirmed as clinically actionable. These results are exploratory. PPV, positive predictive value; NPV, negative predictive value; CI, confidence interval.",
    ]
    return "\n".join(lines)


def _stage_supplementary_figures() -> list[tuple[str, str]]:
    mapping = [
        ("13_group1_learning_curve", "Figure_S1_learning_curve"),
        ("11_group2_probability_intervals", "Figure_S2_patient_predictions"),
        ("09_group2_overall_performance_ci", "Figure_S3_overall_performance_ci"),
        ("14_group2_ks_curve", "Figure_S4_ks_curve"),
        ("15_group2_confusion_matrix", "Figure_S5_confusion_matrix"),
    ]
    for source, target in mapping:
        for suffix in ("png", "svg"):
            source_path = RESULTS / f"{source}.{suffix}"
            if source_path.exists():
                shutil.copy2(source_path, FIGURES / f"{target}.{suffix}")
    return mapping


def _software_versions() -> str:
    packages = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "shap"]
    values = [f"Python {platform.python_version()}"]
    for package in packages:
        values.append(f"{package} {importlib.metadata.version(package)}")
    return ", ".join(values)


def manuscript(
    df: pd.DataFrame,
    development: dict,
    validation: dict,
    supplementary: dict,
    baseline: dict,
    interactions: pd.DataFrame,
    confusion: dict,
) -> str:
    dev, washout, val = locked_split(df)
    candidates = {row["model"]: row for row in _development_rows(development)}
    gb = candidates["gradient_boosting"]
    elastic_net = candidates["logistic_elasticnet"]
    point = supplementary["point_estimates"]
    ci = supplementary["point_estimate_ci"]
    shap_rows = pd.read_csv(REPORTS / "shap_variable_importance.csv").head(5)
    shap_summary = ", ".join(
        f"{_feature_label(row.variable)} ({row.mean_abs_shap:.3f})"
        for row in shap_rows.itertuples()
    )
    interaction = interactions.set_index("modifier")
    age = interaction.loc["age"]
    duration = interaction.loc["duration"]

    return f"""# SCI Methods and Results Draft

> **Document status:** Draft with unresolved author inputs. This file is evidence-bounded to the repository outputs and is not submission-ready until every `[[AUTHOR INPUT REQUIRED: ...]]` field is resolved.

## Methods

### Study design and participants

This single-center retrospective cohort study with internal temporal validation analyzed 580 patient records from [[AUTHOR INPUT REQUIRED: hospital/research center and department]]. Patients were treated for [[AUTHOR INPUT REQUIRED: formal diagnosis and operative procedure]]. Eligibility was determined according to [[AUTHOR INPUT REQUIRED: complete inclusion criteria]] and [[AUTHOR INPUT REQUIRED: complete exclusion criteria]]. The observed admission dates extended from {df.admission_date.min():%d %B %Y} through {df.admission_date.max():%d %B %Y}.

The study protocol was reviewed by [[AUTHOR INPUT REQUIRED: ethics committee name]] and assigned approval number [[AUTHOR INPUT REQUIRED: ethics approval number]]. [[AUTHOR INPUT REQUIRED: state whether written informed consent was obtained or waived and identify the authority for the waiver]]. Only aggregate, non-identifying results were included in the publication package.

### Outcome and predictors

The binary outcome was postoperative spasm recorded at 1 year. The formal clinical definition, assessor, source record, permissible visit window, handling of deaths or loss to follow-up, and adjudication procedure require author confirmation: [[AUTHOR INPUT REQUIRED: 1-year outcome definition and adjudication]] and [[AUTHOR INPUT REQUIRED: follow-up method, timing, and completeness]]. No event dates, censoring times, or last-follow-up dates were available, so time-to-event analyses were not performed.

The prediction model used 12 source predictors. Four were numerical: age, disease duration, body mass index, and admission Karnofsky Performance Status score. Eight were categorical: sex, affected side, hypertension, diabetes, prior botulinum toxin treatment, prior acupuncture, zygomatic-branch lateral spread response (LSR), and mandibular-branch LSR. Each branch-specific LSR variable had three nominal categories: elicited and disappeared, not elicited, and elicited but persisted. The two LSR variables were independently one-hot encoded.

### Temporal split and data quality

Records dated on or before 30 June 2022 formed Group 1, the development cohort. The observed Group 1 range was {dev.admission_date.min():%d %B %Y} to {dev.admission_date.max():%d %B %Y}. Records after 31 August 2022 formed Group 2, the locked internal temporal validation cohort, with an observed range of {val.admission_date.min():%d %B %Y} to {val.admission_date.max():%d %B %Y}. The prespecified washout from 1 July through 31 August 2022 contained {len(washout)} patients. The source labels agreed with the date-derived split. The audited workbook contained 580 rows, no missing fields, no duplicate hospital numbers, and no LSR values outside the prespecified codes 1, 2, and 3.

### Model development and freezing

The reported Group 1 comparison retained {len(candidates)} fixed candidate specifications: elastic-net logistic regression; linear and quadratic discriminant analysis; Gaussian naive Bayes; k-nearest neighbors; linear and radial-basis-function support vector machines; Decision Tree; Random Forest; Extra Trees; AdaBoost; Gradient Boosting; Histogram Gradient Boosting; and a multilayer perceptron. Numerical variables were median-imputed with missingness indicators. They were standardized only for candidates configured with scaling; the Gradient Boosting numerical pipeline was not standardized. Categorical variables were mode-imputed and one-hot encoded within each training fold.

Stratified five-fold cross-validation generated out-of-fold predictions for every fixed specification. Within each outer training fold, the implementation used a three-fold `GridSearchCV` wrapper with an empty parameter grid. The wrapper therefore refitted the same fixed specification and did not perform substantive hyperparameter optimization. Cross-validation shuffling and stochastic estimators used random seed 20260904. Candidate selection minimized the out-of-fold Brier score, with log-loss as the tie-breaker; AUROC and AUPRC were secondary. The selected Gradient Boosting model used 100 estimators, a learning rate of 0.03, maximum tree depth of 2, and a minimum of 10 samples per leaf. It was refitted on all {len(dev)} Group 1 records and frozen before Group 2 was accessed. A later LightGBM run was supplementary and did not change the frozen model.

### Internal temporal validation and statistical analysis

The frozen pipeline generated one probability for each Group 2 patient. No Group 2 data were used for model selection, feature selection, parameter tuning, or recalibration. Discrimination was summarized by AUROC and AUPRC. Overall predictive accuracy was assessed with the Brier score and log-loss. Calibration was summarized by the calibration intercept and slope from a logistic recalibration model and by grouped observed-versus-predicted event rates. Percentile 95% confidence intervals were estimated from {supplementary['bootstrap_replicates_metrics']:,} patient-level bootstrap resamples of Group 2. Classification measures at thresholds 0.05, 0.10, 0.15, and 0.20 were supplementary because no clinically approved operating threshold was available.

Decision-curve net benefit was calculated at thresholds from 0.05 to 0.40 and compared with treat-all and treat-none strategies. This analysis was exploratory because the threshold range had not received clinical confirmation. SHAP values were calculated with `TreeExplainer` for the frozen Gradient Boosting model refitted on all Group 1 patients. SHAP values were expressed on the raw model-output scale and summarized both for encoded features and after aggregation to original variables; they describe model behavior rather than causal effects.

### Exploratory effect-modification analysis

A separate pooled-cohort analysis examined whether the association between residual LSR and the 1-year outcome varied with age, disease duration, or responsible vessel. Residual LSR was defined as code 3 for either the zygomatic- or mandibular-branch response. Separate weakly ridge-stabilized logistic models (`C=1000`) included residual LSR, one modifier, and their product term. Age and disease duration were standardized by their pooled standard deviations ({age['scale']:.2f} and {duration['scale']:.2f} years, respectively); vessel indicators compared presence with absence. Interaction-OR intervals were percentile intervals from 500 bootstrap resamples. All modifier analyses were exploratory, and vessel estimates were susceptible to sparse-data bias. No multiplicity adjustment was applied. Approximate Wald P values were not displayed with the bootstrap intervals because those inferential procedures produced discordant results for sparse vessel strata. The other-vessel interaction was treated as not estimable because the 12 exposed records contained no outcome events.

All analyses were run using {_software_versions()}. The dataset and frozen model were identified by SHA-256 hashes recorded in the publication manifest.

## Results

### Cohort flow and characteristics

The cohort comprised {len(dev)} Group 1 patients and {len(val)} Group 2 patients (Figure 1). The 1-year event occurred in {int(dev.outcome.sum())}/{len(dev)} ({100 * dev.outcome.mean():.1f}%) and {int(val.outcome.sum())}/{len(val)} ({100 * val.outcome.mean():.1f}%), respectively. Mean age was {baseline['age_dev']} years in Group 1 and {baseline['age_val']} years in Group 2 (descriptive P={baseline['age_p']:.3f}). Median disease duration was {baseline['duration_dev']} years and {baseline['duration_val']} years, respectively (descriptive P={baseline['duration_p']:.3f}). The complete cohort comparison is reported in Table 1.

![Figure 1](图/Figure_1_study_flow.png)

### Candidate-model comparison and index-model selection

Gradient Boosting had the lowest Group 1 out-of-fold Brier score among the {len(candidates)} fixed candidate models ({gb['brier']:.5f}), followed by elastic-net logistic regression ({elastic_net['brier']:.5f}; absolute difference {abs(gb['brier'] - elastic_net['brier']):.5f}) (Table 2). Their log-loss values were {gb['log_loss']:.4f} and {elastic_net['log_loss']:.4f}, respectively. Elastic-net logistic regression had higher AUROC ({elastic_net['auroc']:.3f} vs {gb['auroc']:.3f}) and AUPRC ({elastic_net['auprc']:.3f} vs {gb['auprc']:.3f}). Gradient Boosting was retained as the index model under the predefined primary selection rule; these results do not establish broad performance superiority over elastic-net logistic regression. Figure 2 displays 15 models in total: the 14 fixed candidate models plus supplementary LightGBM.

![Figure 2](图/Figure_2_model_comparison.png)

### Model interpretation

The five largest original-variable mean absolute SHAP values in Group 1 were {shap_summary} (Figure 3). The two branch-specific LSR variables accounted for the largest average changes in the model's raw output. These values describe how the fitted model used the predictors and do not estimate independent or causal clinical effects.

![Figure 3](图/Figure_3_shap_interpretation.png)

### Locked internal temporal validation

In Group 2 (n={len(val)}; {int(val.outcome.sum())} events), the frozen model yielded an AUROC of {point['auroc']:.3f} (95% CI, {_ci_text(ci['auroc'])}), an AUPRC of {point['auprc']:.3f} (95% CI, {_ci_text(ci['auprc'])}), a Brier score of {point['brier']:.3f} (95% CI, {_ci_text(ci['brier'])}), and a log-loss of {point['log_loss']:.3f} (95% CI, {_ci_text(ci['log_loss'])}) (Table 3). The calibration intercept was {point['calibration_intercept']:.3f} (95% CI, {_ci_text(ci['calibration_intercept'])}) and the calibration slope was {point['calibration_slope']:.3f} (95% CI, {_ci_text(ci['calibration_slope'])}). Both intervals were wide and included the ideal values of 0 and 1, respectively.

Grouped calibration showed no observed events in the three lowest predicted-risk groups; the highest group had a mean predicted probability of 0.503 and an observed event rate of 0.350 (Figure 4A). In exploratory decision-curve analysis, model net benefit exceeded both reference strategies at thresholds 0.05 through 0.30, but fell below treat-none at 0.40 (Figure 4B). Threshold-specific sensitivity, specificity, positive predictive value, negative predictive value, and F1 score are summarized descriptively in Figure 4C. At threshold 0.10, the confusion matrix contained {confusion['tn']} true negatives, {confusion['fp']} false positives, {confusion['fn']} false negative, and {confusion['tp']} true positives (Figure 4D). These estimates are uncertain because Group 2 contained nine events, and they do not establish a clinically actionable threshold.

![Figure 4](图/Figure_4_temporal_validation.png)

### Exploratory pooled-cohort effect modification

In the pooled 580-patient analysis, the interaction OR for age was {age['interaction_or']:.3f} per {age['scale']:.2f}-year increase (bootstrap 95% CI, {age['ci_low']:.3f}-{age['ci_high']:.3f}); the corresponding estimate for disease duration was {duration['interaction_or']:.3f} per {duration['scale']:.2f}-year increase (bootstrap 95% CI, {duration['ci_low']:.3f}-{duration['ci_high']:.3f}) (Figure 5A). Vessel interaction estimates were highly unstable, and the other-vessel interaction was not estimable.

Among patients younger than 60 years, the observed event rate was 2.4% (9/369) without residual LSR and 43.4% (43/99) with residual LSR. Among patients aged 60 years or older, the corresponding rates were 1.1% (1/92) and 70.0% (14/20) (Figure 5B). Because Group 1 and Group 2 were pooled, the analysis was not a validation analysis. The sparse strata, additional vessel comparisons, and lack of multiplicity adjustment make these findings hypothesis-generating; they were not used to modify the frozen prediction model.

![Figure 5](图/Figure_5_exploratory_interaction.png)

## Figure Legends

### Figure 1. Study design and analysis workflow

The 580 records were separated by admission date into Group 1 for model development and Group 2 for locked internal temporal validation. The displayed dates are the observed ranges, and the predefined July-August 2022 washout contained no patients. Group 1 was used for fixed-specification comparison, model selection, refitting, and SHAP analysis. Group 2 remained held out until the Gradient Boosting pipeline had been frozen and was then evaluated once without tuning or recalibration. BMI, body mass index; KPS, Karnofsky Performance Status; LSR, lateral spread response.

### Figure 2. Group 1 performance of available candidate models

Panel A shows out-of-fold Brier scores and panel B shows out-of-fold AUROCs for 15 displayed models: 14 fixed primary candidates plus the post-freeze supplementary LightGBM analysis. Elastic-net logistic regression is included among the primary models. Predictions for the primary candidates were produced by stratified five-fold cross-validation in Group 1 (n=476; 58 events). Red identifies the frozen Gradient Boosting index model, teal identifies elastic-net logistic regression, and orange identifies supplementary LightGBM. The stored primary-candidate report does not contain confidence intervals; the supplementary LightGBM report contains bootstrap AUROC and AUPRC intervals. AUROC, area under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve.

### Figure 3. SHAP interpretation of the frozen Gradient Boosting model in Group 1

Panel A ranks the five original variables with the largest mean absolute SHAP values after contributions from one-hot levels were aggregated to their source variable. Panel B shows the distribution of encoded-feature SHAP values for those variables; color represents the encoded feature value. SHAP values are on the raw Gradient Boosting output scale. Positive values increase the model output relative to its expected value and negative values decrease it. The analysis describes the model refitted on all 476 Group 1 patients and does not estimate causality or validate predictor effects in Group 2. LSR, lateral spread response; SHAP, Shapley additive explanations.

### Figure 4. Internal temporal validation and exploratory threshold performance in Group 2

Panel A plots observed 1-year event rates against mean frozen-model predictions in five quantile-based risk groups. Error bars are percentile 95% confidence intervals from 2,000 within-group bootstrap resamples; tied predictions produced group sizes of 22, 22, 19, 21, and 20. The diagonal denotes ideal calibration. Panel B shows exploratory net benefit for the frozen model, treat-all, and treat-none strategies at configured thresholds from 0.05 to 0.40. Panel C is a radar plot of sensitivity, specificity, positive predictive value, negative predictive value, and F1 score at thresholds 0.05, 0.10, 0.15, and 0.20; point estimates are shown and bootstrap intervals are reported in Table S1. Panel D is the confusion matrix at threshold 0.10, with rows denoting observed outcomes and columns denoting predicted outcomes. Group 2 contained 104 patients and nine events. No threshold was clinically confirmed, and none of these panels was used to recalibrate the model or select an operating threshold. F1, harmonic mean of sensitivity and positive predictive value; NPV, negative predictive value; PPV, positive predictive value.

### Figure 5. Exploratory residual-LSR effect modification in the pooled cohort

Panel A shows interaction odds ratios from separate weakly ridge-stabilized logistic models fitted to all 580 patients. Each model contained residual LSR, one modifier, and their product term. Points are interaction odds ratios and bars are percentile 95% confidence intervals from 500 bootstrap resamples. Age and disease duration are expressed per pooled standard deviation; vessel variables compare presence with absence. The dashed line denotes an interaction odds ratio of 1. All modifier analyses were exploratory, and no multiplicity adjustment was applied. The other-vessel estimate was marked not estimable because the 12 exposed patients had no events. Panel B displays unadjusted observed event rates by age and residual-LSR status; labels provide events/total and percentages. Residual LSR was defined as persistence (code 3) in either branch-specific LSR record. This pooled association analysis was separate from model development and temporal validation. CI, confidence interval; LSR, lateral spread response; NE, not estimable; OR, odds ratio; SD, standard deviation.

## Table Titles and Notes

### Table 1. Cohort characteristics by temporal group

[Open Table 1](表/Table_1_baseline_characteristics.md). Values are mean (SD), median [IQR], or n (%). The final column reports the exact two-sided statistical method for each row: Mann-Whitney U for continuous variables, Pearson chi-square for categorical variables, or Fisher exact for sparse 2 x 2 tables. Branch-specific LSR P values are omnibus 3 x 2 tests. P values are descriptive, were not adjusted for multiplicity, and did not affect model selection.

### Table 2. Group 1 cross-validated performance of fixed candidate specifications

[Open Table 2](表/Table_2_model_comparison.md). Primary values are based on out-of-fold predictions from the 476-patient development cohort. The table reports candidate status, Brier score, log-loss, AUROC, AUPRC, and available bootstrap intervals. It distinguishes the primary fixed comparison from post-freeze optional-model analyses.

### Table 3. Performance of the frozen Gradient Boosting model in Group 2

[Open Table 3](表/Table_3_temporal_validation.md). Estimates are from the 104-patient internal temporal validation cohort, which contained nine events. Percentile 95% confidence intervals used 2,000 patient-level bootstrap resamples. No Group 2 tuning, feature selection, or recalibration was performed.

## Supplementary Material Index

- [Figure S1: Group 1 learning curve](图/Figure_S1_learning_curve.png)
- [Figure S2: Group 2 patient prediction intervals](图/Figure_S2_patient_predictions.png)
- [Figure S3: Group 2 overall performance intervals](图/Figure_S3_overall_performance_ci.png)
- [Figure S4: Group 2 Kolmogorov-Smirnov curve](图/Figure_S4_ks_curve.png)
- [Figure S5: Group 2 confusion matrix at threshold 0.10](图/Figure_S5_confusion_matrix.png)
- [Table S1: exploratory threshold performance](表/Table_S1_threshold_performance.md)

## Author Completion Checklist

- [ ] [[AUTHOR INPUT REQUIRED: hospital/research center and department]]
- [ ] [[AUTHOR INPUT REQUIRED: formal diagnosis and operative procedure]]
- [ ] [[AUTHOR INPUT REQUIRED: complete inclusion criteria]]
- [ ] [[AUTHOR INPUT REQUIRED: complete exclusion criteria]]
- [ ] [[AUTHOR INPUT REQUIRED: ethics committee name]]
- [ ] [[AUTHOR INPUT REQUIRED: ethics approval number]]
- [ ] [[AUTHOR INPUT REQUIRED: informed consent or waiver statement]]
- [ ] [[AUTHOR INPUT REQUIRED: 1-year outcome definition and adjudication]]
- [ ] [[AUTHOR INPUT REQUIRED: follow-up method, timing, and completeness]]
- [ ] [[AUTHOR INPUT REQUIRED: target journal and house style]]
"""


def publication_manifest(supplementary_mapping: list[tuple[str, str]]) -> str:
    verification_path = REPORTS / "publication_verification.json"
    verification = _read_json(verification_path) if verification_path.exists() else {
        "status": "PENDING",
        "publication_command": "not recorded",
        "focused_test": "not recorded",
        "full_test_suite": "not recorded",
    }
    supplement_rows = "\n".join(
        f"| `{target}.png` / `.svg` | `results/{source}.png` / `.svg` | Supplementary diagnostic |"
        for source, target in supplementary_mapping
    )
    return f"""# Publication Results Manifest

## Status

- Manuscript status: **DRAFT, AUTHOR INPUT REQUIRED**.
- Dataset SHA-256: `{sha256('datas/580-analysis.xlsx')}`.
- Frozen model SHA-256: `{sha256(ARTIFACTS / 'model.joblib')}`.
- Canonical generation command: `uv run python -m mechine_learning_lsr.publication`.
- Generator: `src/mechine_learning_lsr/publication.py`.
- Generation is non-destructive: only the canonical targets listed below are replaced.

## Source hierarchy

1. `datas/580-analysis.xlsx` is authoritative for audited cohort and baseline summaries.
2. Structured JSON and CSV files under `reports/` are authoritative for model, validation, calibration, SHAP, and interaction values.
3. `artifacts/model.joblib` is authoritative for the frozen pipeline and Figure 3 SHAP recomputation.
4. Existing prose reports and rendered table images are explanatory copies, not numeric sources.

## Main figures

| Canonical file | Cohort and role | Authoritative sources | Transformation | Limitation |
|---|---|---|---|---|
| `图/Figure_1_study_flow.png` / `.svg` | All records; design and temporal split | Workbook, `configs/split.yaml`, `reports/development_models.json` | `publication.figure_1` | Center, eligibility, and follow-up details require author input |
| `图/Figure_2_model_comparison.png` / `.svg` | Group 1; fixed primary comparison | `reports/development_models.json`, `reports/optional_boosting_models.json` | `publication.figure_2` | No non-empty hyperparameter search; differences were not formally compared |
| `图/Figure_3_shap_interpretation.png` / `.svg` | Group 1; fitted-model interpretation | Workbook, frozen model, SHAP CSV outputs | `publication.figure_3` | Describes model behavior and is not causal evidence |
| `图/Figure_4_temporal_validation.png` / `.svg` | Group 2; calibration, exploratory DCA, threshold metrics, and confusion matrix | `reports/group2_calibration_bins.json`, `reports/temporal_validation.json`, `reports/group2_supplementary_metrics.json`, `reports/group2_confusion_matrix.json` | `publication.figure_4` | Only nine events; thresholds lack clinical confirmation |
| `图/Figure_5_exploratory_interaction.png` / `.svg` | Pooled cohort; exploratory association | `reports/interaction_overall_sensitivity.csv`, `reports/interaction_stratified_rates.csv` | `publication.figure_5` | Pooled, sparse, unadjusted for multiplicity, and not a validation analysis |

## Main tables

| Canonical file | Cohort and role | Authoritative sources | Transformation | Limitation |
|---|---|---|---|---|
| `表/Table_1_baseline_characteristics.md` | Group 1 vs Group 2 characteristics | Workbook | `publication.table_1` | P values are descriptive and unadjusted |
| `表/Table_2_model_comparison.md` | Group 1 candidate comparison | Development and optional-model JSON | `publication.table_2` | Optional models evaluated post-freeze cannot change selection |
| `表/Table_3_temporal_validation.md` | Group 2 frozen-model performance | Group 2 supplementary JSON | `publication.table_3` | Bootstrap intervals are imprecise with nine events |

## Supplementary figures

| Canonical file | Source | Role |
|---|---|---|
{supplement_rows}

`表/Table_S1_threshold_performance.md` reports supplementary classification metrics and bootstrap intervals. No operating threshold has been clinically approved.

## Reconciliation with `sci-lsr-temporal-validation`

- Tasks 6.4 and 6.5 in the earlier change remain unchecked.
- Aggregate subgroup counts and interaction outputs exist, but this package treats them only as descriptive or exploratory.
- Figure 5 pools Group 1 and Group 2 for a separate association analysis. It does not confirm subgroup transportability and does not modify the frozen model.
- Primary temporal validation results remain the locked Group 2 metrics from the frozen 12-predictor Gradient Boosting pipeline.

## Verification record

- Status: **{verification['status']}**
- Publication generation: {verification['publication_command']}
- Focused publication test: {verification['focused_test']}
- Full test suite: {verification['full_test_suite']}

## Author-input gate

The package intentionally retains placeholders for the center, diagnosis and procedure, eligibility criteria, ethics review, consent, outcome adjudication, follow-up, and target journal. It must not be described as submission-ready until those placeholders have been resolved and checked by the authors.
"""


def table_index() -> str:
    return """# 主文表格索引

以下 Markdown 文件是数值主文件；PNG/SVG 表格仅作为查看副本，不作为数值来源。

| 编号 | 文件 | 内容 |
|---|---|---|
| Table 1 | [Table_1_baseline_characteristics.md](../表/Table_1_baseline_characteristics.md) | Group 1 与 Group 2 队列特征 |
| Table 2 | [Table_2_model_comparison.md](../表/Table_2_model_comparison.md) | Group 1 固定候选模型比较 |
| Table 3 | [Table_3_temporal_validation.md](../表/Table_3_temporal_validation.md) | Group 2 冻结模型验证性能 |
| Table S1 | [Table_S1_threshold_performance.md](../表/Table_S1_threshold_performance.md) | 探索性阈值分类指标 |

完整 Methods、Results、图注和表注见 [SCI_Methods_Results.md](../SCI_Methods_Results.md)。
"""


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    df = load_workbook("datas/580-analysis.xlsx")
    development = _read_json(REPORTS / "development_models.json")
    validation = _read_json(REPORTS / "temporal_validation.json")
    supplementary = _read_json(REPORTS / "group2_supplementary_metrics.json")
    calibration_bins = _read_json(REPORTS / "group2_calibration_bins.json")
    confusion = _read_json(REPORTS / "group2_confusion_matrix.json")
    optional = _read_json(REPORTS / "optional_boosting_models.json")
    interactions = pd.read_csv(REPORTS / "interaction_overall_sensitivity.csv")
    strata = pd.read_csv(REPORTS / "interaction_stratified_rates.csv")

    figure_1(df, development)
    figure_2(development, optional)
    figure_3(df)
    figure_4(validation, calibration_bins, supplementary, confusion)
    figure_5(interactions, strata)
    supplementary_mapping = _stage_supplementary_figures()

    table1_text, baseline = table_1(df)
    _write(TABLES / "Table_1_baseline_characteristics.md", table1_text)
    _write(TABLES / "Table_2_model_comparison.md", table_2(development, optional))
    _write(TABLES / "Table_3_temporal_validation.md", table_3(supplementary))
    _write(TABLES / "Table_S1_threshold_performance.md", supplementary_threshold_table(supplementary))
    _write(TABLES / "README.md", table_index())
    _write(FIGURES / "表.md", table_index())

    _write(
        RESULTS / "SCI_Methods_Results.md",
        manuscript(df, development, validation, supplementary, baseline, interactions, confusion),
    )
    _write(RESULTS / "publication_manifest.md", publication_manifest(supplementary_mapping))
    print("Generated five main figures, three main tables, supplementary index, and SCI Methods/Results.")


if __name__ == "__main__":
    main()
