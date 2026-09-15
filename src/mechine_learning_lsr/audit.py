from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import load_workbook, locked_split, make_lsr_combo, sha256


@dataclass
class AuditResult:
    path: str
    sha256: str
    rows: int
    columns: int
    missing: dict[str, int]
    duplicate_hospital_numbers: int
    invalid_outcomes: int
    invalid_lsr: dict[str, int]
    date_range: tuple[str, str]
    group_counts: dict[str, int]
    event_counts: dict[str, int]
    washout_rows: int

    @property
    def passed(self) -> bool:
        return not (
            self.rows != 580
            or self.columns != 24
            or any(self.missing.values())
            or self.duplicate_hospital_numbers
            or self.invalid_outcomes
            or any(self.invalid_lsr.values())
        )


def audit(path: str | Path) -> tuple[pd.DataFrame, AuditResult]:
    df = load_workbook(path)
    dev, washout, validation = locked_split(df)
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    invalid_lsr = {c: int((~df[c].isin([1, 2, 3])).sum()) for c in ("zyg_lsr", "man_lsr")}
    result = AuditResult(
        path=str(path), sha256=sha256(path), rows=len(df), columns=len(df.columns),
        missing={k: v for k, v in missing.items() if v},
        duplicate_hospital_numbers=int(df.hospital_number.duplicated().sum()),
        invalid_outcomes=int((~df.outcome.isin([0, 1])).sum()),
        invalid_lsr=invalid_lsr,
        date_range=(df.admission_date.min().date().isoformat(), df.admission_date.max().date().isoformat()),
        group_counts=df.group.value_counts().to_dict(),
        event_counts=df.groupby("group").outcome.sum().astype(int).to_dict(),
        washout_rows=len(washout),
    )
    if not result.passed:
        raise ValueError(f"data contract failed: {result}")
    if set(dev.index) & set(validation.index):
        raise ValueError("development and validation rows overlap")
    expected = {"组1": len(dev), "组2": len(validation)}
    if result.group_counts != expected:
        raise ValueError(f"source group labels disagree with date split: {result.group_counts} != {expected}")
    df["lsr_combo"] = make_lsr_combo(df)
    return df, result


def render_report(result: AuditResult, output: str | Path) -> None:
    lines = [
        "# Data Audit Report", "", f"- Source: `{result.path}`",
        f"- SHA-256: `{result.sha256}`", f"- Shape: {result.rows} rows × {result.columns} columns",
        f"- Date range: {result.date_range[0]} to {result.date_range[1]}",
        f"- Group counts: {result.group_counts}", f"- Event counts: {result.event_counts}",
        f"- Washout rows (2022-07-01 to 2022-08-31): {result.washout_rows}",
        f"- Duplicate hospital numbers: {result.duplicate_hospital_numbers}",
        f"- Missing fields: {result.missing or 'none'}", f"- Invalid LSR values: {result.invalid_lsr}",
        f"- Contract status: {'PASS' if result.passed else 'FAIL'}", "",
        "This report contains aggregate metadata only; patient-level rows are not exported.",
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
