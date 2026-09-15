from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression


def classification_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    result = {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    result["auprc"] = float(average_precision_score(y, p)) if np.any(y) else None
    return result


def threshold_metrics(y_true, probability, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    predicted = np.asarray(probability, dtype=float) >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    div = lambda a, b: float(a / b) if b else None
    return {"threshold": float(threshold), "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "f1": float(f1_score(y, predicted, zero_division=0))}


def calibration_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    result = {"calibration_intercept": None, "calibration_slope": None}
    if len(np.unique(y)) < 2:
        return result
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        slope = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
        result["calibration_intercept"] = float(slope.intercept_[0])
        result["calibration_slope"] = float(slope.coef_[0, 0])
    except ValueError:
        pass
    return result


def decision_curve(y_true, probability, thresholds):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    n = len(y)
    prevalence = float(y.mean())
    rows = []
    for threshold in thresholds:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        nb = tp / n - fp / n * threshold / (1 - threshold)
        rows.append({"threshold": float(threshold), "net_benefit": float(nb), "treat_all": prevalence - (1 - prevalence) * threshold / (1 - threshold), "treat_none": 0.0})
    return rows


def bootstrap_interval(y_true, probability, metric, n_boot=500, seed=20260905):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(probability)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2 and metric in {"auroc", "auprc"}:
            continue
        values.append(classification_metrics(y[idx], p[idx]).get(metric))
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) else None


def plot_calibration(y_true, probability, output):
    import matplotlib.pyplot as plt
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    bins = np.linspace(0, 1, 11)
    centers, observed = [], []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if np.any(mask):
            centers.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(centers, observed, "o-")
    ax.set(xlabel="Predicted risk", ylabel="Observed risk", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
