from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import pandas as pd

from .data import make_lsr_combo


M0_NUMERIC = ["age", "duration", "bmi", "kps"]
M0_CATEGORICAL = ["sex", "side", "hypertension", "diabetes", "botox", "acupuncture"]
M1_NUMERIC = M0_NUMERIC
M1_CATEGORICAL = M0_CATEGORICAL + ["zyg_lsr", "man_lsr"]
M2_CATEGORICAL = M1_CATEGORICAL + ["day7_spasm"]
REDUCED5_NUMERIC = ["age", "duration"]
REDUCED5_CATEGORICAL = ["acupuncture", "zyg_lsr", "man_lsr"]
REDUCED5_COLUMNS = REDUCED5_NUMERIC + REDUCED5_CATEGORICAL
REDUCED5_ALLOWED = {
    "acupuncture": (0, 1),
    "zyg_lsr": (1, 2, 3),
    "man_lsr": (1, 2, 3),
}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    prediction_time: str

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]


FEATURE_SETS = {
    "m0": FeatureSet("m0", tuple(M0_NUMERIC), tuple(M0_CATEGORICAL), "preoperative"),
    "m1": FeatureSet("m1", tuple(M1_NUMERIC), tuple(M1_CATEGORICAL), "intraoperative"),
    "m2": FeatureSet("m2", tuple(M1_NUMERIC), tuple(M2_CATEGORICAL), "post-day-7"),
    "reduced5": FeatureSet("reduced5", tuple(REDUCED5_NUMERIC), tuple(REDUCED5_CATEGORICAL), "intraoperative"),
}


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lsr_combo"] = make_lsr_combo(out)
    return out


def get_features(df: pd.DataFrame, name: str = "m1", representation: str = "independent") -> tuple[pd.DataFrame, pd.Series, FeatureSet]:
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {name}")
    feature_set = FEATURE_SETS[name]
    data = prepare_frame(df)
    columns = feature_set.columns.copy()
    categorical = list(feature_set.categorical)
    if representation == "combo" and name == "m1":
        columns = [*M0_NUMERIC, *M0_CATEGORICAL, "lsr_combo"]
        categorical = [*M0_CATEGORICAL, "lsr_combo"]
    elif representation != "independent":
        raise ValueError("representation must be independent or combo")
    return data[columns].copy(), data["outcome"].astype(int), FeatureSet(name, tuple(c for c in columns if c not in categorical), tuple(categorical), feature_set.prediction_time)


def validate_reduced5(values: Mapping[str, object]) -> list[str]:
    """Return validation errors for the public five-variable prediction contract."""
    errors: list[str] = []
    for name in REDUCED5_NUMERIC:
        try:
            value = float(values[name])
            valid = math.isfinite(value) and (18 <= value <= 120 if name == "age" else value >= 0)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            message = "a finite number between 18 and 120" if name == "age" else "a finite non-negative number"
            errors.append(f"{name} must be {message}")
    for name, allowed in REDUCED5_ALLOWED.items():
        if values.get(name) not in allowed:
            errors.append(f"{name} must be one of {allowed}")
    return errors


def build_reduced5_frame(values: Mapping[str, object]) -> pd.DataFrame:
    """Build the exact one-row frame expected by the frozen reduced model."""
    errors = validate_reduced5(values)
    if errors:
        raise ValueError("; ".join(errors))
    numeric = {column: float(values[column]) for column in REDUCED5_NUMERIC}
    categorical = {column: values[column] for column in REDUCED5_CATEGORICAL}
    return pd.DataFrame(
        [{**numeric, **categorical}],
        columns=REDUCED5_COLUMNS,
    )
