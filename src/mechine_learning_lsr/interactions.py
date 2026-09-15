from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def interaction_table(df: pd.DataFrame, modifier: str, cutoff: float) -> pd.DataFrame:
    x = df.copy()
    x["lsr_residual"] = ((x.zyg_lsr == 3) | (x.man_lsr == 3)).astype(int)
    x["modifier_group"] = (x[modifier] >= cutoff).map({True: "high", False: "low"})
    return x.groupby(["lsr_residual", "modifier_group"], dropna=False).outcome.agg(n="size", events="sum").reset_index()


def exploratory_interactions(df: pd.DataFrame) -> dict[str, list[dict]]:
    return {
        "lsr_residual_x_duration": interaction_table(df, "duration", 3).to_dict("records"),
        "lsr_residual_x_age": interaction_table(df, "age", 60).to_dict("records"),
        **{f"lsr_residual_x_{v}": vessel_table(df, v).to_dict("records") for v in ["va", "pica", "aica", "other_vessel"]},
    }


def _lsr(df: pd.DataFrame) -> pd.Series:
    return ((df.zyg_lsr == 3) | (df.man_lsr == 3)).astype(int)


def vessel_table(df: pd.DataFrame, vessel: str) -> pd.DataFrame:
    x = df.copy()
    x["lsr_residual"] = _lsr(x)
    x["modifier_group"] = x[vessel].astype(int).map({0: "absent", 1: "present"})
    return x.groupby(["lsr_residual", "modifier_group"], dropna=False).outcome.agg(n="size", events="sum").reset_index()


def _fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-stabilized logistic fit; avoids infinite estimates under sparse vessels."""
    model = LogisticRegression(C=1e3, solver="lbfgs", max_iter=2000)
    model.fit(X[:, 1:], y.astype(int))
    beta = np.r_[float(model.intercept_[0]), model.coef_[0].astype(float)]
    p = model.predict_proba(X[:, 1:])[:, 1]
    cov = np.linalg.pinv(X.T @ ((p * (1 - p))[:, None] * X) + np.eye(X.shape[1]) * 1e-6)
    return beta, cov


def fit_interaction(df: pd.DataFrame, modifier: str, label: str, continuous: bool = False, bootstrap: int = 500) -> dict:
    x = df[["outcome", "zyg_lsr", "man_lsr", modifier]].dropna().copy()
    x["lsr_residual"] = _lsr(x)
    raw = x[modifier].astype(float).to_numpy()
    center, scale = (float(raw.mean()), float(raw.std(ddof=0))) if continuous else (0.0, 1.0)
    if continuous:
        scale = scale or 1.0
        mod = (raw - center) / scale
        unit = "per SD increase"
    else:
        mod = raw
        unit = "present vs absent"
    design = np.column_stack([np.ones(len(x)), x.lsr_residual.to_numpy(), mod, x.lsr_residual.to_numpy() * mod])
    y = x.outcome.to_numpy(dtype=float)
    beta, cov = _fit_logistic(design, y)
    se = float(np.sqrt(max(cov[3, 3], 0)))
    z = float(beta[3] / se) if se else float("nan")
    rng = np.random.default_rng(20260904)
    boots = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), len(y))
        try:
            b, _ = _fit_logistic(design[idx], y[idx])
            if np.isfinite(b[3]) and abs(b[3]) < 30:
                boots.append(float(b[3]))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) >= 50 else [float(beta[3] - 1.96 * se), float(beta[3] + 1.96 * se)]
    return {
        "label": label, "modifier": modifier, "n": int(len(x)), "events": int(y.sum()),
        "unit": unit, "center": center, "scale": scale, "interaction_log_or": float(beta[3]),
        "interaction_or": float(np.exp(np.clip(beta[3], -30, 30))),
        "ci_low": float(np.exp(np.clip(ci[0], -30, 30))), "ci_high": float(np.exp(np.clip(ci[1], -30, 30))),
        "wald_p": float(2 * norm.sf(abs(z))) if np.isfinite(z) else None,
        "bootstrap_replicates": int(len(boots)),
        "interpretation": "OR<1 suggests weaker LSR association as the modifier increases; OR>1 suggests stronger association.",
    }


def formal_interactions(df: pd.DataFrame, bootstrap: int = 500) -> list[dict]:
    rows = [
        fit_interaction(df, "duration", "LSR residual × disease duration", continuous=True, bootstrap=bootstrap),
        fit_interaction(df, "age", "LSR residual × age", continuous=True, bootstrap=bootstrap),
    ]
    rows.extend(fit_interaction(df, v, f"LSR residual × {v}", bootstrap=bootstrap) for v in ["va", "pica", "aica", "other_vessel"])
    return rows


def lsr_stratum_associations(df: pd.DataFrame) -> list[dict]:
    """Unadjusted LSR-residual OR within each modifier stratum for all-patient analysis."""
    x = df.copy()
    x["lsr_residual"] = _lsr(x)
    specifications = [("duration", 3, "病程<3年", "病程≥3年"), ("age", 60, "年龄<60岁", "年龄≥60岁")]
    rows: list[dict] = []
    for modifier, cutoff, low_label, high_label in specifications:
        x["stratum"] = np.where(x[modifier] >= cutoff, high_label, low_label)
        groups = [(low_label, x[x.stratum == low_label]), (high_label, x[x.stratum == high_label])]
        for label, g in groups:
            tab = g.groupby("lsr_residual").outcome.agg(n="size", events="sum")
            a = float(tab.loc[1, "events"] + 0.5) if 1 in tab.index else 0.5
            b = float(tab.loc[1, "n"] - tab.loc[1, "events"] + 0.5) if 1 in tab.index else 0.5
            c = float(tab.loc[0, "events"] + 0.5) if 0 in tab.index else 0.5
            d = float(tab.loc[0, "n"] - tab.loc[0, "events"] + 0.5) if 0 in tab.index else 0.5
            log_or = float(np.log((a * d) / (b * c)))
            se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
            rows.append({"modifier": modifier, "stratum": label, "n": int(len(g)), "events": int(g.outcome.sum()), "lsr_or": float(np.exp(log_or)), "ci_low": float(np.exp(log_or - 1.96 * se)), "ci_high": float(np.exp(log_or + 1.96 * se)), "wald_p": float(2 * norm.sf(abs(log_or / se)))})
    for vessel in ["va", "pica", "aica", "other_vessel"]:
        for value, label in [(0, "未出现"), (1, "出现")]:
            g = x[x[vessel] == value]
            tab = g.groupby("lsr_residual").outcome.agg(n="size", events="sum")
            a = float(tab.loc[1, "events"] + 0.5) if 1 in tab.index else 0.5
            b = float(tab.loc[1, "n"] - tab.loc[1, "events"] + 0.5) if 1 in tab.index else 0.5
            c = float(tab.loc[0, "events"] + 0.5) if 0 in tab.index else 0.5
            d = float(tab.loc[0, "n"] - tab.loc[0, "events"] + 0.5) if 0 in tab.index else 0.5
            log_or = float(np.log((a * d) / (b * c)))
            se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
            rows.append({"modifier": vessel, "stratum": label, "n": int(len(g)), "events": int(g.outcome.sum()), "lsr_or": float(np.exp(log_or)), "ci_low": float(np.exp(log_or - 1.96 * se)), "ci_high": float(np.exp(log_or + 1.96 * se)), "wald_p": float(2 * norm.sf(abs(log_or / se)))})
    return rows


def log_odds_ratio(table: pd.DataFrame) -> dict[str, float | None]:
    cells = {(int(r.lsr_residual), r.modifier_group): int(r.events) for r in table.itertuples()}
    counts = {(int(r.lsr_residual), r.modifier_group): int(r.n) for r in table.itertuples()}
    try:
        a = cells[(1, "high")] + 0.5
        b = counts[(1, "high")] - cells[(1, "high")] + 0.5
        c = cells[(0, "high")] + 0.5
        d = counts[(0, "high")] - cells[(0, "high")] + 0.5
    except KeyError:
        return {"odds_ratio": None, "ci_low": None, "ci_high": None}
    log_or = float(np.log((a * d) / (b * c)))
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    return {"odds_ratio": float(np.exp(log_or)), "ci_low": float(np.exp(log_or - 1.96 * se)), "ci_high": float(np.exp(log_or + 1.96 * se))}
