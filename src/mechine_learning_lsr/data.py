from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd

SOURCE_COLUMNS = [
    "group", "admission_date", "hospital_number", "outcome", "sex", "age", "side",
    "duration", "hypertension", "diabetes", "incision", "blood_loss", "day7_spasm",
    "zyg_lsr", "man_lsr", "botox", "acupuncture", "bmi", "kps", "length_of_stay",
    "va", "pica", "aica", "other_vessel",
]
CHINESE_COLUMNS = [
    "来源", "入院时间", "住院号", "一年预后痉挛1不痉挛0", "性别（男=1，女=0）", "年龄（岁）",
    "患侧（左=1，右=2）", "病程（年）", "高血压（有=1，无=0）", "糖尿病（有=1，无=0）",
    "切口类型（1类或者2类）", "术中出血量（ml）", "术后七天（痉挛=1，不痉挛=0）",
    "3颧支（引出且消失=1，引出未消失=3，未引出=2）", "3下颌是否消失（引出且消失=1，引出未消失=3，未引出=2）",
    "肉毒素（有=1，无=0）", "针灸（有=1，无=0）", "BMI", "入院时KPS评分", "住院时间",
    "椎动脉", "小脑后下动脉", "小脑前下动脉", "其他动脉",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_workbook(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if list(df.columns) != CHINESE_COLUMNS:
        raise ValueError(f"unexpected workbook columns: {list(df.columns)!r}")
    df = df.copy()
    df.columns = SOURCE_COLUMNS
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    return df


def make_lsr_combo(df: pd.DataFrame) -> pd.Series:
    combo = pd.Series(pd.NA, index=df.index, dtype="string")
    combo = combo.mask((df.zyg_lsr == 1) & (df.man_lsr == 1), "both_gone")
    combo = combo.mask((df.zyg_lsr == 3) & (df.man_lsr == 3), "both_persist")
    return combo.fillna("partial")


def locked_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev_end = pd.Timestamp("2022-06-30")
    washout_end = pd.Timestamp("2022-08-31")
    dev = df[df.admission_date <= dev_end].copy()
    washout = df[(df.admission_date > dev_end) & (df.admission_date <= washout_end)].copy()
    validation = df[df.admission_date > washout_end].copy()
    return dev, washout, validation
