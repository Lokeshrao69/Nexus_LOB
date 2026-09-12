"""Linear IC benchmark — the only model class allowed before Phase 3.

Rule (plan_2.md Phase 1): no trees / boosters / neural nets until the *linear*
baseline stands. These are univariate/multivariate OLS on z-scored features;
the predictor of a z-scored feature against the label is exactly the linear
signal whose rank IC the experiment layer reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def zscore(x: Sequence[float]) -> np.ndarray:
    """Standardize to mean 0, std 1 (epsilon-guarded)."""
    a = np.asarray(x, dtype=np.float64)
    s = a.std(ddof=1)
    if s <= 1e-12:
        return np.zeros_like(a)
    return (a - a.mean()) / s


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = zscore(x)
    b = zscore(y)
    if a.size != b.size or a.size < 2:
        return float("nan")
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def fit_ols_ic(
    features: Mapping[str, Sequence[float]],
    y: Sequence[float],
    *,
    standardize: bool = True,
) -> dict:
    """Fit a multi-feature linear model on z-scored inputs.

    Returns the OLS coefficients, in-sample R², the linear predictor, and its
    Pearson / rank IC against the label — the honest IC of the *linear*
    predictor (i.e. whether combining features adds anything over E1–E4's
    single-feature ICs).

    ``standardize`` z-scores every column AND the label, so coefficients are
    comparable (a per-tick basis-point interpretation is possible).
    """
    names = list(features.keys())
    if not names:
        raise ValueError("at least one feature required")
    X = np.column_stack([np.asarray(features[n], dtype=np.float64) for n in names])
    yv = np.asarray(y, dtype=np.float64)
    if X.shape[0] != yv.size or X.shape[0] < 2:
        raise ValueError("feature rows must match y and be >= 2")

    if standardize:
        Xs = np.column_stack([zscore(X[:, j]) for j in range(X.shape[1])])
        ys = zscore(yv)
    else:
        Xs, ys = X, yv

    design = np.column_stack([np.ones(Xs.shape[0]), Xs])
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    pred = design @ beta
    resid = ys - pred
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) or 1e-30
    r2 = float(1.0 - np.sum(resid**2) / ss_tot)

    from .experiments import rank_ic

    return {
        "coefs": dict(zip(["intercept"] + names, beta.tolist())),
        "r2": r2,
        "pearson_ic": pearson(pred, yv),
        "rank_ic": rank_ic(yv, pred),
        "pred": pred.tolist(),
    }