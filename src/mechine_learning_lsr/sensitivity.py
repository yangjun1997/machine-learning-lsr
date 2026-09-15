from __future__ import annotations

import pandas as pd


def subgroup_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "sex": df.groupby("sex").outcome.agg(n="size", events="sum").astype(int).to_dict("index"),
        "age": df.assign(age_group=(df.age >= 60).map({True: "ge60", False: "lt60"})).groupby("age_group").outcome.agg(n="size", events="sum").astype(int).to_dict("index"),
        "lsr_residual": df.assign(lsr_group=((df.zyg_lsr == 3) | (df.man_lsr == 3)).map({True: "residual", False: "gone_or_unelicited"})).groupby("lsr_group").outcome.agg(n="size", events="sum").astype(int).to_dict("index"),
    }


def representation_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "independent_lsr": ["zyg_lsr", "man_lsr"],
        "combo_lsr": ["lsr_combo"],
    }
