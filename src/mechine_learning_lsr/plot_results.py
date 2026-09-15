from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_workbook


RESULTS = Path("results")
STYLE = {"figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", bbox_inches="tight")
    fig.savefig(RESULTS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def model_comparison() -> None:
    report = json.loads(Path("reports/development_models.json").read_text(encoding="utf-8"))
    rows = [dict(r, source="core") for r in report["candidates"] if r.get("status") == "complete"]
    optional_path = Path("reports/optional_boosting_models.json")
    optional = json.loads(optional_path.read_text(encoding="utf-8")) if optional_path.exists() else {"models": []}
    rows.extend(dict(r, source="optional") for r in optional.get("models", []) if r.get("status") == "complete")
    df = pd.DataFrame(rows).sort_values("brier")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#d62728" if x == report["model"] else "#4c78a8" for x in df.model]
    axes[0].barh(df.model, df.brier, color=colors)
    axes[0].invert_yaxis()
    axes[0].set(title="Group 1 nested CV: Brier score", xlabel="Lower is better")
    axes[1].barh(df.model, df.auroc, color=colors)
    axes[1].invert_yaxis()
    axes[1].set(title="Group 1 nested CV: AUROC", xlabel="Higher is better", xlim=(0, 1))
    save(fig, "01_model_comparison")

    unavailable = [r for r in report["candidates"] if r.get("status") == "unavailable"]
    unavailable.extend(r for r in optional.get("models", []) if r.get("status") == "unavailable")
    by_model = {row["model"]: row for row in rows}
    for row in unavailable:
        by_model.setdefault(row["model"], dict(row, source="optional"))
    all_rows = sorted(by_model.values(), key=lambda r: (r.get("status") != "complete", r["model"]))
    table_lines = [
        "# Group 1统一模型比较表", "", "可运行模型使用Group 1五折/嵌套交叉验证结果；XGBoost和CatBoost当前环境未成功导入，因此保留为 unavailable，不填入虚构性能。", "",
        "| 模型 | 来源 | 状态 | Brier | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI |", "|---|---|---|---:|---:|---|---:|---|",
    ]
    for row in all_rows:
        if row.get("status") == "complete":
            auroc_ci = row.get("auroc_ci")
            auprc_ci = row.get("auprc_ci")
            auroc_text = f"{auroc_ci[0]:.3f}–{auroc_ci[1]:.3f}" if auroc_ci else "NA"
            auprc_text = f"{auprc_ci[0]:.3f}–{auprc_ci[1]:.3f}" if auprc_ci else "NA"
            table_lines.append(f"| {row['model']} | {row.get('source', 'core')} | complete | {row['brier']:.4f} | {row['auroc']:.4f} | {auroc_text} | {row['auprc']:.4f} | {auprc_text} |")
        else:
            table_lines.append(f"| {row['model']} | {row.get('source', 'optional')} | unavailable | NA | NA | NA | NA | NA |")
    (RESULTS / "01_model_comparison_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    display_rows = [row for row in all_rows if row.get("status") == "complete"]
    fig, ax = plt.subplots(figsize=(14, max(5, 0.42 * len(display_rows) + 1.5)))
    ax.axis("off")
    cell_rows = [[row["model"], row.get("source", "core"), f"{row['brier']:.4f}", f"{row['auroc']:.4f}", f"{row['auprc']:.4f}"] for row in display_rows]
    table = ax.table(cellText=cell_rows, colLabels=["Model", "Source", "Brier", "AUROC", "AUPRC"], loc="center", cellLoc="center", colWidths=[0.28, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4c78a8")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef3f8")
        if row > 0 and display_rows[row - 1]["model"] == report["model"]:
            cell.set_facecolor("#f8cccc")
    ax.set_title("Group 1 unified model comparison", fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(RESULTS / "01_model_comparison_table.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS / "01_model_comparison_table.svg", bbox_inches="tight")
    plt.close(fig)


def dca() -> None:
    report = json.loads(Path("reports/temporal_validation.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(report["decision_curve"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df.threshold, df.net_benefit, "o-", lw=2, label="Frozen model")
    ax.plot(df.threshold, df.treat_all, "o--", label="Treat all")
    ax.plot(df.threshold, df.treat_none, "--", label="Treat none")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Group 2 decision-curve analysis", xlabel="Threshold probability", ylabel="Net benefit")
    ax.legend(frameon=False)
    save(fig, "02_decision_curve")


def cohort_outcomes() -> None:
    df = load_workbook("datas/580-analysis.xlsx")
    summary = df.groupby("group").outcome.agg(n="size", events="sum").reindex(["组1", "组2"])
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary.n - summary.events, 0.36, label="No 1-year spasm", color="#72b7b2")
    ax.bar(x + 0.18, summary.events, 0.36, label="1-year spasm", color="#e45756")
    ax.set_xticks(x, ["Group 1\nDevelopment", "Group 2\nTemporal validation"])
    ax.set_ylabel("Patients")
    ax.set_title("Cohort and outcome composition")
    ax.legend(frameon=False)
    for i, row in enumerate(summary.itertuples()):
        ax.text(i, row.n + 5, f"n={row.n}\nevents={row.events}", ha="center")
    save(fig, "03_cohort_outcomes")


def lsr_outcomes() -> None:
    df = load_workbook("datas/580-analysis.xlsx")
    df["lsr_combo"] = np.select(
        [(df.zyg_lsr == 1) & (df.man_lsr == 1), (df.zyg_lsr == 3) & (df.man_lsr == 3)],
        ["both_gone", "both_persist"],
        default="partial",
    )
    summary = df.groupby("lsr_combo").outcome.agg(n="size", events="sum").reindex(["both_gone", "partial", "both_persist"])
    summary["rate"] = summary.events / summary.n
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary.index, summary.rate * 100, color=["#59a14f", "#f28e2b", "#e15759"])
    ax.set_ylabel("1-year spasm rate (%)")
    ax.set_title("Observed outcome by LSR combination")
    ax.set_ylim(0, max(65, summary.rate.max() * 100 + 10))
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{row.events}/{row.n}", ha="center")
    save(fig, "04_lsr_combination_outcomes")


def validation_summary() -> None:
    report = json.loads(Path("reports/temporal_validation.json").read_text(encoding="utf-8"))
    names = ["AUROC", "AUPRC", "1-Brier"]
    values = [report["auroc"], report["auprc"], 1 - report["brier"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Locked group 2 validation summary (9 events)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    save(fig, "05_group2_validation_summary")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    model_comparison()
    dca()
    cohort_outcomes()
    lsr_outcomes()
    validation_summary()
    index = """# Analysis Results\n\nGenerated from the frozen analysis artifacts and the unchanged `datas/580-analysis.xlsx`.\n\n| File | Description |\n|---|---|\n| `01_model_comparison.png` / `.svg` | Group 1 unified model comparison; available optional models included. |\n| `01_model_comparison_table.md` | Complete model table, including unavailable optional-model status. |\n| `01_model_comparison_table.png` / `.svg` | Publication-style runnable-model comparison table. |\n| `02_decision_curve.png` / `.svg` | Group 2 locked-validation decision curve. |\n| `03_cohort_outcomes.png` / `.svg` | Group sizes and one-year outcome composition. |\n| `04_lsr_combination_outcomes.png` / `.svg` | Observed outcome rate by LSR combination. |\n| `05_group2_validation_summary.png` / `.svg` | Locked group2 summary metrics. |\n\nSelected model: `gradient_boosting`. Group 2 contains 104 patients and 9 events; results are initial retrospective temporal validation, not prospective validation.\n"""
    index += "\n## Model comparison\n\n- `01_model_comparison_table.md`: unified model table including optional model status.\n- `01_model_comparison_table.png` / `.svg`: publication-style model comparison table.\n\n## SHAP explanations\n\n- `06_shap_variable_importance.png` / `.svg`: Group 1 frozen-model SHAP importance by original variable.\n- `07_shap_summary_beeswarm.png` / `.svg`: Group 1 frozen-model SHAP distribution for encoded features.\n"
    (RESULTS / "README.md").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()
