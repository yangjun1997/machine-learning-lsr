from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import scipy
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, shapiro, ttest_ind

from .data import load_workbook

CONTINUOUS = ["age", "duration", "bmi", "kps", "blood_loss", "length_of_stay"]
CATEGORICAL = ["sex", "side", "hypertension", "diabetes", "incision", "day7_spasm", "zyg_lsr", "man_lsr", "botox", "acupuncture", "va", "pica", "aica", "other_vessel"]
LABELS = {"age": "年龄（岁）", "duration": "病程（年）", "bmi": "BMI", "kps": "入院KPS", "blood_loss": "术中出血量（mL）", "length_of_stay": "住院时间（天）", "sex": "性别", "side": "患侧", "hypertension": "高血压", "diabetes": "糖尿病", "incision": "切口类型", "day7_spasm": "术后第7天痉挛", "zyg_lsr": "颧支LSR", "man_lsr": "下颌支LSR", "botox": "肉毒毒素", "acupuncture": "针灸", "va": "VA责任血管", "pica": "PICA责任血管", "aica": "AICA责任血管", "other_vessel": "其他责任血管"}

def _description(x: pd.Series) -> str:
    q1, median, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}; {median:.2f} [{q1:.2f}, {q3:.2f}]"

def _continuous(df: pd.DataFrame, col: str):
    g1 = pd.to_numeric(df.loc[df.group == "G1", col], errors="coerce").dropna(); g2 = pd.to_numeric(df.loc[df.group == "G2", col], errors="coerce").dropna(); allv = pd.to_numeric(df[col], errors="coerce").dropna()
    p1 = float(shapiro(g1).pvalue); p2 = float(shapiro(g2).pvalue)
    if p1 >= .05 and p2 >= .05:
        p = float(ttest_ind(g1, g2, equal_var=False, alternative="two-sided").pvalue); method = "Welch独立样本t检验"; func = "scipy.stats.ttest_ind(equal_var=False, alternative='two-sided')"; diagnostic = "两组均近似正态"
    else:
        p = float(mannwhitneyu(g1, g2, alternative="two-sided", method="auto").pvalue); method = "双侧Mann–Whitney U检验"; func = "scipy.stats.mannwhitneyu(alternative='two-sided', method='auto')"; diagnostic = "至少一组偏离正态"
    row = (LABELS[col], _description(g1), _description(g2), _description(allv), p, method)
    audit = (LABELS[col], "连续", len(g1), len(g2), p1, p2, diagnostic, method, func, p, "Shapiro–Wilk两组均p≥0.05→Welch t；否则→Mann–Whitney U")
    return row, audit

def _categorical(df: pd.DataFrame, col: str):
    levels = sorted(df[col].dropna().unique().tolist()); cross = pd.crosstab(df[col], df.group).reindex(index=levels, columns=["G1", "G2"], fill_value=0); observed = cross.to_numpy(); expected = chi2_contingency(observed, correction=False).expected_freq
    if observed.shape == (2, 2) and (expected < 5).any():
        p = float(fisher_exact(observed, alternative="two-sided").pvalue); method = "双侧Fisher精确检验"; func = "scipy.stats.fisher_exact(alternative='two-sided')"
    else:
        p = float(chi2_contingency(observed, correction=False).pvalue); method = "Pearson χ²检验"; func = "scipy.stats.chi2_contingency(correction=False)"
    rows = []
    for i, level in enumerate(levels):
        n1, n2 = int(cross.loc[level, "G1"]), int(cross.loc[level, "G2"]); rows.append((f"{LABELS[col]} = {level}", f"{n1} ({100*n1/476:.1f}%)", f"{n2} ({100*n2/104:.1f}%)", f"{n1+n2} ({100*(n1+n2)/580:.1f}%)", p if i == 0 else None, method if i == 0 else ""))
    audit = (LABELS[col], "分类", int(observed[:, 0].sum()), int(observed[:, 1].sum()), None, None, f"{observed.shape[0]}×{observed.shape[1]}列联表；最小期望频数={expected.min():.4f}；期望频数<5单元格={(expected < 5).sum()}", method, func, p, "仅2×2且任一期望频数<5→Fisher；其他→Pearson χ²")
    return rows, audit

def _header(ws):
    for cell in ws[1]: cell.fill = PatternFill("solid", fgColor="1F4E78"); cell.font = Font(bold=True, color="FFFFFF"); cell.alignment = Alignment(horizontal="center", wrap_text=True)

def main():
    df = load_workbook("datas/580-analysis.xlsx").copy(); df["group"] = np.where(df.admission_date <= pd.Timestamp("2022-06-30"), "G1", "G2")
    table = [("变量", "Group 1（n=476）", "Group 2（n=104）", "总体（n=580）", "P值", "实际采用的检验")]; audit = [("变量", "类型", "Group 1有效n", "Group 2有效n", "G1 Shapiro–Wilk p", "G2 Shapiro–Wilk p", "适用性诊断", "实际检验", "Python函数", "最终p值", "选择规则")]
    for col in CONTINUOUS:
        row, a = _continuous(df, col); table.append(row); audit.append(a)
    for col in CATEGORICAL:
        rows, a = _categorical(df, col); table.extend(rows); audit.append(a)
    wb = Workbook(); ws = wb.active; ws.title = "Baseline table"
    for row in table: ws.append(row)
    _header(ws); ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    for c, w in {"A": 28, "B": 29, "C": 29, "D": 29, "E": 14, "F": 28}.items(): ws.column_dimensions[c].width = w
    for row in ws.iter_rows():
        for cell in row: cell.alignment = Alignment(vertical="top", wrap_text=True)
    wa = wb.create_sheet("Test audit")
    for row in audit: wa.append(row)
    _header(wa); wa.freeze_panes = "A2"; wa.auto_filter.ref = wa.dimensions
    for c, w in {"A": 25, "B": 12, "C": 16, "D": 16, "E": 21, "F": 21, "G": 48, "H": 28, "I": 64, "J": 16, "K": 55}.items(): wa.column_dimensions[c].width = w
    for row in wa.iter_rows():
        for cell in row: cell.alignment = Alignment(vertical="top", wrap_text=True)
    wm = wb.create_sheet("Software & rules")
    metadata = [("项目", "实际信息"), ("Python", platform.python_version()), ("SciPy", scipy.__version__), ("pandas", pd.__version__), ("NumPy", np.__version__), ("openpyxl", openpyxl.__version__), ("正态性函数", "scipy.stats.shapiro；分别对Group 1和Group 2全部非缺失观测执行Shapiro–Wilk"), ("连续变量函数", "两组Shapiro–Wilk均p≥0.05时scipy.stats.ttest_ind(equal_var=False)；否则scipy.stats.mannwhitneyu(alternative='two-sided')"), ("分类变量函数", "2×2且任一期望频数<5时scipy.stats.fisher_exact(alternative='two-sided')；否则scipy.stats.chi2_contingency(correction=False)"), ("P值", "双侧Group 1 vs Group 2比较；总体列不参与检验；α=0.05；未作多重比较校正"), ("缺失值", "按变量进行complete-case分析"), ("复现命令", "uv run python -m mechine_learning_lsr.baseline_table")]
    for row in metadata: wm.append(row)
    _header(wm); wm.column_dimensions["A"].width = 28; wm.column_dimensions["B"].width = 110
    for row in wm.iter_rows():
        for cell in row: cell.alignment = Alignment(vertical="top", wrap_text=True)
    output = Path("results/临床基线表.xlsx"); wb.save(output); print(output)
    Path("results/临床基线表_说明.md").write_text("# 临床基线表\n\nExcel：`临床基线表.xlsx`。\n\n- `Baseline table`：Group 1、Group 2、总体，以及Group 1 vs Group 2的实际p值和实际检验。\n- `Test audit`：逐变量记录Shapiro–Wilk p值、期望频数、实际调用的scipy函数、最终p值。\n- `Software & rules`：Python及pandas、NumPy、SciPy、openpyxl版本和复现规则。\n\n连续变量并非预先统一使用某一种检验，而是根据本次数据实际Shapiro–Wilk结果选择：两组均p≥0.05才使用Welch t检验，否则使用双侧Mann–Whitney U。分类变量按实际列联表和期望频数选择Fisher精确检验或Pearson χ²。\n", encoding="utf-8")

if __name__ == "__main__": main()
