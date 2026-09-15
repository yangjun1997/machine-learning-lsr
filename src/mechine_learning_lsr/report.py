from __future__ import annotations

from pathlib import Path
import json

from .audit import audit
from .data import locked_split
from .interactions import exploratory_interactions
from .sensitivity import subgroup_counts


def main():
    source = Path("datas/580-analysis.xlsx")
    df, audit_result = audit(source)
    validation = json.loads(Path("reports/temporal_validation.json").read_text(encoding="utf-8"))
    development = json.loads(Path("reports/development_models.json").read_text(encoding="utf-8"))
    formal_path = Path("reports/interaction_formal_results.json")
    formal = json.loads(formal_path.read_text(encoding="utf-8")) if formal_path.exists() else []
    dev, _, val = locked_split(df)
    lines = [
        "# Temporal Validation Report", "",
        "## Study status", "",
        "This is a retrospective initial temporal validation. Group2 was evaluated once after group1 model development and freeze.", "",
        f"- Dataset: 580 patients; group1={len(dev)}, group2={len(val)}.",
        f"- Events: group1={int(dev.outcome.sum())}, group2={int(val.outcome.sum())}.",
        f"- Dataset SHA-256: `{audit_result.sha256}`.",
        f"- Selected model: `{development['model']}`.", "",
        "## Locked group2 results", "",
    ]
    for key in ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope"):
        lines.append(f"- {key}: {validation.get(key)}")
    lines += ["", "## Decision curve", "", "| Threshold | Model net benefit | Treat-all | Treat-none |", "|---:|---:|---:|---:|"]
    for row in validation["decision_curve"]:
        lines.append(f"| {row['threshold']:.2f} | {row['net_benefit']:.4f} | {row['treat_all']:.4f} | {row['treat_none']:.4f} |")
    lines += ["", "## Exploratory interactions", "", "These tables are hypothesis-generating only; group2 has nine events and cannot confirm interaction effects.", ""]
    for name, rows in exploratory_interactions(df).items():
        lines += [f"### {name}", "", "| LSR residual | Modifier group | n | events |", "|---:|---|---:|---:|"]
        lines += [f"| {r['lsr_residual']} | {r['modifier_group']} | {r['n']} | {r['events']} |" for r in rows]
        lines.append("")
    if formal:
        lines += ["## Formal interaction estimates (exploratory)", "", "| Interaction | OR | 95% CI | Wald p |", "|---|---:|---|---:|"]
        for row in formal:
            p = "NA" if row.get("wald_p") is None else f"{row['wald_p']:.4f}"
            lines.append(f"| {row['label']} | {row['interaction_or']:.3f} | {row['ci_low']:.3f}–{row['ci_high']:.3f} | {p} |")
        lines += ["", "These crude interaction estimates are fitted in Group 1 only; Group 2 subgroup results are descriptive because it contains nine events. See `reports/interaction_analysis.md` and `results/16_interaction_forest.png`.", ""]
    lines += ["## Aggregate subgroup counts", "", "```json", json.dumps(subgroup_counts(df), ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Limitations", "", "The workbook has no patient hash, event date, last-follow-up date, or censoring fields. Survival analysis and claims of prospective validation are out of scope."]
    Path("reports/temporal_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote reports/temporal_validation.md")


if __name__ == "__main__":
    main()
