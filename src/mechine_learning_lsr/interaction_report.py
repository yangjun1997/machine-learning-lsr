from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_workbook, locked_split
from .interactions import formal_interactions, interaction_table, vessel_table, lsr_stratum_associations


def _stratified(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["lsr_residual"] = ((x.zyg_lsr == 3) | (x.man_lsr == 3)).astype(int)
    rows = []
    for modifier, cutoff, title in [("duration", 3, "duration_ge_3y"), ("age", 60, "age_ge_60")]:
        x["modifier_group"] = np.where(x[modifier] >= cutoff, f">={cutoff}", f"<{cutoff}")
        t = x.groupby(["cohort", "lsr_residual", "modifier_group"], dropna=False).outcome.agg(n="size", events="sum").reset_index()
        t["event_rate"] = t.events / t.n
        t["modifier"] = title
        rows.append(t)
    for vessel in ["va", "pica", "aica", "other_vessel"]:
        t = x.groupby(["cohort", "lsr_residual", vessel], dropna=False).outcome.agg(n="size", events="sum").reset_index()
        t["modifier_group"] = t[vessel].astype(int).map({0: "absent", 1: "present"})
        t["event_rate"] = t.events / t.n
        t["modifier"] = vessel
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _forest_axis(ax, rows: pd.DataFrame, title: str) -> None:
    y = np.arange(len(rows))[::-1]
    x = rows.interaction_or.to_numpy()
    lo = np.maximum(x - rows.ci_low.to_numpy(), 1e-6)
    hi = np.maximum(rows.ci_high.to_numpy() - x, 1e-6)
    ax.errorbar(x, y, xerr=[lo, hi], fmt="o", color="#2f5597", ecolor="#2f5597", capsize=3)
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_yticks(y, rows.label)
    ax.set_xscale("log")
    ax.set_xlabel("Interaction OR (log scale), 95% bootstrap CI")
    ax.set_title(title)


def _plot_forest(group1: pd.DataFrame, overall: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    _forest_axis(axes[0], group1, "Group 1 formal exploratory model (n=476)")
    _forest_axis(axes[1], overall, "All 580 patients sensitivity analysis")
    fig.suptitle("LSR residual effect modification: development-only vs overall sensitivity", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "16_interaction_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "16_interaction_forest.svg", bbox_inches="tight")
    plt.close(fig)


def _draw_strata(ax, strata: pd.DataFrame, interactions: pd.DataFrame, title: str) -> None:
    significant_modifiers = interactions.loc[interactions.wald_p < 0.05, "modifier"].replace(
        {"age": "age_ge_60", "duration": "duration_ge_3y"}
    )
    s = strata.groupby(["lsr_residual", "modifier", "modifier_group"], as_index=False).agg(n=("n", "sum"), events=("events", "sum"))
    s = s[s.modifier.isin(significant_modifiers)]
    s["event_rate"] = s.events / s.n
    s["group"] = s.lsr_residual.map({0: "No residual", 1: "Residual"})
    s["label"] = s.modifier + ": " + s.modifier_group.astype(str)
    piv = s.pivot_table(index="label", columns="group", values="event_rate", aggfunc="first")
    if piv.empty:
        ax.text(0.5, 0.5, "No interaction has Wald p < 0.05", ha="center", transform=ax.transAxes)
        ax.set_xticks([])
    else:
        piv.mul(100).plot.bar(ax=ax, color=["#72b7b2", "#e45756"])
        ax.legend(title="LSR status", frameon=False)
    ax.set_ylabel("Observed 1-year spasm rate (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)


def _plot_strata(strata: pd.DataFrame, out: Path, interactions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_strata(
        ax,
        strata,
        interactions,
        "All 580 patients: stratified event rates\nModifiers with interaction Wald p < 0.05 in Fig. 19",
    )
    fig.tight_layout()
    fig.savefig(out / "17_interaction_stratified_rates.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "17_interaction_stratified_rates.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_strata(strata: pd.DataFrame, out: Path) -> None:
    s = strata.copy()
    s["group"] = s.lsr_residual.map({0: "No residual", 1: "Residual"})
    s["label"] = s.modifier + ": " + s.modifier_group.astype(str)
    piv = s.pivot_table(index="label", columns=["cohort", "group"], values="event_rate", aggfunc="first").fillna(0)
    ax = piv.mul(100).plot.bar(figsize=(12, 6), color=["#72b7b2", "#e45756", "#9ecae1", "#d62728"])
    ax.set_ylabel("Observed 1-year spasm rate (%)")
    ax.set_title("Stratified event rates by cohort (descriptive; sparse Group 2)")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(frameon=False, ncol=2)
    fig = ax.get_figure(); fig.tight_layout()
    fig.savefig(out / "18_interaction_cohort_rates.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "18_interaction_cohort_rates.svg", bbox_inches="tight")
    plt.close(fig)


def _draw_all580_forest(ax, rows: pd.DataFrame, title: str) -> None:
    y = np.arange(len(rows))[::-1]
    x = rows.interaction_or.to_numpy()
    ax.errorbar(x, y, xerr=[np.maximum(x - rows.ci_low, 1e-6), np.maximum(rows.ci_high - x, 1e-6)], fmt="o", color="#2f5597", ecolor="#2f5597", capsize=3)
    ax.axvline(1, color="#888", ls="--")
    labels = [
        f"{label} (Wald p={p:.4f})" if pd.notna(p) else f"{label} (Wald p=NA)"
        for label, p in zip(rows.label, rows.wald_p)
    ]
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("Interaction OR (log scale), 95% bootstrap CI")
    ax.set_title(title)


def _plot_all580_forest(rows: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_all580_forest(ax, rows, "All 580 patients: LSR residual effect modification")
    fig.tight_layout()
    fig.savefig(out / "19_all580_lsr_interaction_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "19_all580_lsr_interaction_forest.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_combined(strata: pd.DataFrame, interactions: pd.DataFrame, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(20, 7), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    _draw_all580_forest(ax_a, interactions, "Interaction test")
    _draw_strata(ax_b, strata, interactions, "Age-stratified observed event rates")
    for ax, label in [(ax_a, "A"), (ax_b, "B")]:
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=18, fontweight="bold")
    fig.suptitle("LSR residual effect modification in all 580 patients", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "19_17_combined_lsr_interaction.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "19_17_combined_lsr_interaction.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_all580_stratum_or(rows: pd.DataFrame, out: Path) -> None:
    x = pd.DataFrame(rows)
    x["label"] = x.modifier + ": " + x.stratum
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(x))[::-1]
    point = x.lsr_or.to_numpy()
    ax.errorbar(point, y, xerr=[point - x.ci_low, x.ci_high - point], fmt="o", capsize=3, color="#e45756", ecolor="#e45756")
    ax.axvline(1, color="#888", ls="--")
    ax.set_yticks(y, x.label)
    ax.set_xscale("log")
    ax.set_xlabel("LSR residual vs no residual OR (log scale), Haldane-corrected 95% CI")
    ax.set_title("All 580 patients: LSR association within modifier strata")
    fig.tight_layout()
    fig.savefig(out / "20_all580_lsr_stratum_or.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "20_all580_lsr_stratum_or.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = Path("results"); out.mkdir(exist_ok=True)
    df = load_workbook("datas/580-analysis.xlsx")
    dev, _, val = locked_split(df)
    formal = formal_interactions(dev, bootstrap=500)
    overall_formal = formal_interactions(df, bootstrap=500)
    all580_strata = lsr_stratum_associations(df)
    pd.DataFrame(formal).to_csv("reports/interaction_formal_results.csv", index=False, encoding="utf-8-sig")
    Path("reports/interaction_formal_results.json").write_text(json.dumps(formal, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(overall_formal).to_csv("reports/interaction_overall_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all580_strata).to_csv("reports/all580_lsr_stratum_associations.csv", index=False, encoding="utf-8-sig")
    strata = _stratified(pd.concat([dev.assign(cohort="Group 1"), val.assign(cohort="Group 2")], ignore_index=True))
    strata.to_csv("reports/interaction_stratified_rates.csv", index=False, encoding="utf-8-sig")
    overall_frame = pd.DataFrame(overall_formal)
    _plot_strata(strata, out, overall_frame)
    _plot_all580_forest(overall_frame, out)
    _plot_combined(strata, overall_frame, out)
    _plot_all580_stratum_or(all580_strata, out)
    standalone = [
        "# 全部580例：LSR残留与术后痉挛的效应修饰分析", "", 
        "本分析是独立于机器学习模型之外的全队列关联分析。使用全部580例患者，结局为术后1年痉挛，暴露为LSR残留（任一LSR持续存在）。Group 1/Group 2在这里不再承担开发集或验证集角色；不使用预测模型概率，也不进行模型调参。", "",
        "## 1. 交互作用模型", "", "模型形式：`outcome ~ LSR residual + modifier + LSR residual × modifier`。病程和年龄按每1个标准差增加；责任血管按出现 vs 未出现。交互OR<1提示LSR残留关联可能减弱，交互OR>1提示可能增强。", "", "| 交互项 | n | 事件 | 交互OR | 95% CI | Wald p |", "|---|---:|---:|---:|---|---:|"
    ]
    for r in overall_formal:
        standalone.append(f"| {r['label']} | {r['n']} | {r['events']} | {r['interaction_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} | {r['wald_p']:.4f} |")
    standalone += ["", "![全体580例交互作用森林图](../results/19_all580_lsr_interaction_forest.png)", "", "## 2. 各分层内LSR残留关联", "", "下表/图显示在每个病程、年龄或责任血管层级内，LSR残留与术后痉挛的粗OR。由于部分层级事件数少，使用0.5连续性校正；这些OR用于解释交互方向，不替代交互项检验。", "", "| 修饰变量 | 分层 | n | 事件 | LSR残留OR | 95% CI |", "|---|---|---:|---:|---:|---|"]
    for r in all580_strata:
        standalone.append(f"| {r['modifier']} | {r['stratum']} | {r['n']} | {r['events']} | {r['lsr_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} |")
    standalone += ["", "![全体580例分层LSR关联](../results/20_all580_lsr_stratum_or.png)", "", "### 分层LSR关联的具体数值（图20）", "", "| 修饰变量 | 分层 | LSR OR | 95% CI | p值 |", "|---|---|---:|---|---:|"]
    for r in all580_strata:
        standalone.append(f"| {r['modifier']} | {r['stratum']} | {r['lsr_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} | {r['wald_p']:.4f} |")
    standalone += ["", "## 3. 结论边界", "", "这是全体580例的独立关联/效应修饰分析，不是机器学习模型性能分析。交互OR的置信区间跨越1时，不能确认效应随该变量改变。`other_vessel=1`样本极少且无事件，相关结果只能视为不稳定信号。年龄、病程和责任血管的结果均不能单独证明LSR预测价值在某一亚组中衰减或增强，需要更大样本的预先指定验证。", ""]
    Path("reports/all580_lsr_effect_modification.md").write_text("\n".join(standalone), encoding="utf-8")
    rows = ["# 交互作用/效应修饰探索性分析", "", "正式交互项在Group 1（476例，58个事件）拟合；Group 2（104例，9个事件）仅作描述性分层。所有结果均为探索性，不用于改动冻结模型或重新校准。", "", "## 正式交互项（Group 1）", "", "| 交互项 | 单位 | n | 事件 | 交互OR | 95% CI | Wald p |", "|---|---|---:|---:|---:|---|---:|"]
    for r in formal:
        rows.append(f"| {r['label']} | {r['unit']} | {r['n']} | {r['events']} | {r['interaction_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} | {r['wald_p']:.4f} |" if r['wald_p'] is not None else f"| {r['label']} | {r['unit']} | {r['n']} | {r['events']} | {r['interaction_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} | NA |")
    rows += ["", "OR<1表示随着修饰变量增加，LSR残留与结局的关联可能减弱；OR>1表示可能增强。置信区间跨越1时，不能确认效应修饰。年龄和病程使用标准化连续变量（每1个标准差）；责任血管为出现 vs 未出现。", "", "## 图形", "", "![交互作用森林图](../results/16_interaction_forest.png)", "", "图17仅显示图19中交互检验p<0.05的修饰变量。", "", "![分层事件率](../results/17_interaction_stratified_rates.png)", "", "## 稀疏性与解释边界", "", "`other_vessel=1`仅12例且无事件，相关OR和区间受完全/近完全分离影响，应视为信号筛查而非可靠效应估计。Group 2的分层事件数过少，不能据此声称LSR预测价值在老年、长病程或特定责任血管中已经衰减；需要更大样本和预先指定的交互验证。", ""]
    rows += ["", "## Overall 580-patient analysis", "", "The following analysis uses all 580 patients as one cohort and is independent of model development or temporal validation.", "", "![All-580 interaction forest](../results/19_all580_lsr_interaction_forest.png)", "", "![All-580 stratified LSR associations](../results/20_all580_lsr_stratum_or.png)", ""]
    rows += [f"| Overall {r['label']} | {r['interaction_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} |" for r in overall_formal]
    Path("reports/interaction_analysis.md").write_text("\n".join(rows), encoding="utf-8")
    report_path = Path("results/完整结果解读报告.md")
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# 完整结果解读报告\n"
    marker = "## 12. 交互作用/效应修饰分析（新增）"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    report += "\n" + marker + "\n\n"
    report += "本节回答：LSR残留与术后痉挛的关联，是否会随病程、年龄或责任血管而改变。正式交互项只在Group 1（476例、58个事件）中估计；Group 2（104例、9个事件）仅用于描述性分层，未用于重新训练或修改冻结模型。\n\n"
    report += "![交互作用森林图](16_interaction_forest.png)\n\n图17仅显示图19中交互检验p<0.05的修饰变量。\n\n![分层事件率](17_interaction_stratified_rates.png)\n\n"
    report += "| 交互项 | 交互OR | 95% CI | p值 |\n|---|---:|---|---:|\n"
    for r in formal:
        p = "NA" if r["wald_p"] is None else f"{r['wald_p']:.4f}"
        report += f"| {r['label']}（{r['unit']}） | {r['interaction_or']:.3f} | {r['ci_low']:.3f}–{r['ci_high']:.3f} | {p} |\n"
    report += "\n解释：交互OR<1提示修饰变量升高时LSR残留关联可能减弱，交互OR>1提示可能增强。病程交互OR为 {:.3f}（95% CI {:.3f}–{:.3f}），未显示明确衰减；年龄交互OR为 {:.3f}（95% CI {:.3f}–{:.3f}），点估计反而提示年龄增加时LSR残留关联可能增强，但这是探索性结果，不能视为已证实的临床效应。责任血管交互项的区间较宽，尤其`va`、`aica`和`other_vessel`存在稀疏/分离问题，不能据此断言不同责任血管中LSR预测价值已经衰减。\n\n".format(formal[0]["interaction_or"], formal[0]["ci_low"], formal[0]["ci_high"], formal[1]["interaction_or"], formal[1]["ci_low"], formal[1]["ci_high"])
    report += "Group 2分层仅作方向性核对：长病程、老年及责任血管亚组的事件数很少，不能提供可靠的交互检验。后续若要形成可发表的效应修饰结论，应在更大外部队列中预先指定交互项，并报告分层校准、区分度及bootstrap CI。\n"
    report += "\n### 12.1 统计口径说明\n\n- `16_interaction_forest.png` 左侧：Group 1的476例正式探索性交互模型，用于主要推断。右侧：全部580例合并敏感性分析，仅用于描述性核对。\n- `17_interaction_stratified_rates.png`：仅显示图19中Wald交互检验p<0.05（未作多重比较校正）的修饰变量，展示全部580例合并后的分层事件率，不是模型性能，也不是Group 2外部验证结果。\n- `18_interaction_cohort_rates.png`：Group 1与Group 2分开显示的分层事件率；Group 2只有9个事件，需谨慎解读。\n\n将Group 1和Group 2合并可以增加样本量，但会混合模型开发队列和时间验证队列，因此总体结果不能替代Group 1正式交互结果，也不能用于重新训练或调参。\n"
    report += "\n## 13. 独立的全体580例关联分析\n\n你所要求的‘Group 1+Group 2全部患者’分析已单独整理，不涉及之前的机器学习模型、模型概率、开发集或验证集。完整结果见 [all580_lsr_effect_modification.md](../reports/all580_lsr_effect_modification.md)。\n\n![全体580例交互作用森林图](19_all580_lsr_interaction_forest.png)\n\n![全体580例分层LSR关联](20_all580_lsr_stratum_or.png)\n"
    report = report.replace("16_interaction_forest.png", "19_all580_lsr_interaction_forest.png")
    report = report.replace("18_interaction_cohort_rates.png", "20_all580_lsr_stratum_or.png")
    report += "\n\n说明：本报告后续采用整体580例分析口径。19图为全部580例的交互项检验；20图为各分层内LSR残留OR。两图均不涉及Group 1/Group 2分组，也不涉及机器学习模型性能。\n"
    report_path.write_text(report, encoding="utf-8")
    readme = Path("results/README.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Analysis Results\n"
    if "16_interaction_forest" not in text:
        text += "\n## Interaction/effect-modification analysis\n\n- `16_interaction_forest.png` / `.svg`: Group 1 formal exploratory estimates (left) and all-580 sensitivity estimates (right).\n- `17_interaction_stratified_rates.png` / `.svg`: overall descriptive event rates using Group 1 + Group 2.\n- `18_interaction_cohort_rates.png` / `.svg`: Group 1 and Group 2 event rates shown separately.\n- `../reports/interaction_formal_results.csv`: Group 1 interaction estimates.\n- `../reports/interaction_overall_sensitivity.csv`: all-580 sensitivity estimates.\n- `../reports/interaction_analysis.md`: methods, tables, and limitations.\n"
        readme.write_text(text, encoding="utf-8")
    print("wrote interaction results")


if __name__ == "__main__":
    main()
