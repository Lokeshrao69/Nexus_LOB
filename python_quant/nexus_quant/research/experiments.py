"""Experiment + statistical-rigor toolbox (plan_2.md §5).

Everything is NumPy-only and deterministic (seeded RNG). The three comparison
primitives used across E1–E7:

  * rank_ic / icir — primary signal metrics;
  * bootstrap_ci (block)  — CIs that respect autocorrelation (never i.i.d.
    bootstrap on a tape);
  * diebold_mariano      — paired forecast comparison with HAC variance when
    horizons overlap.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from math import erf, isinf, isnan, sqrt

import numpy as np


# ---------------------------------------------------------------------------
# signal metrics
# ---------------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (standard Spearman) — ties are common on integer-tick labels.

    Vectorized: a tie block occupying sorted positions ``[i, j)`` gets rank
    ``(i + j − 1) / 2`` for every member (identical to the scalar definition).
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new_block = np.concatenate(([True], xs[1:] != xs[:-1]))
    starts = np.flatnonzero(new_block)
    ends = np.concatenate((starts[1:], [n]))
    block_rank = (starts + ends - 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = block_rank[np.cumsum(new_block) - 1]
    return ranks


def rank_ic(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation between prediction and realized outcome."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    denom = np.sqrt(((ra - ra.mean()) ** 2).sum() * ((rb - rb.mean()) ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).sum() / denom)


def icir(ic_series: Sequence[float], annualize: float = 1.0) -> float:
    """Information ratio of an IC series: mean / std * sqrt(n)."""
    x = np.asarray(ic_series, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]):
        return 0.0
    s = float(x.std(ddof=1))
    if s <= 1e-12 or np.isnan(s):
        return 0.0
    return float(x.mean() / s * np.sqrt(x.size) * annualize)


def hit_rate(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Fraction of predictions whose sign matches the realized sign."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return float("nan")
    return float(np.mean(np.sign(b) == np.sign(a)))


def decile_spread(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n: int | None = None,
    q: float = 0.10,
) -> float:
    """mean(label of top pred decile) − mean(label of bottom pred decile).

    If ``n`` is None, the bucket size is computed from quantile ``q``
    (default 0.10 for deciles: ``max(1, int(len(a) * q))``). If an explicit
    count ``n`` is provided, that fixed count is used instead.
    """
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size:
        return float("nan")
    k = n if n is not None else max(1, int(a.size * q))
    if a.size < 2 * k or k < 1:
        return float("nan")
    order = np.argsort(b, kind="mergesort")
    a_sorted = a[order]
    top = a_sorted[-k:].mean()
    bot = a_sorted[:k].mean()
    return float(top - bot)


# ---------------------------------------------------------------------------
# bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    kind: str = "block",
    block: int = 0,
    alpha: float = 0.05,
    seed: int = 0x51ED,
    stat_fn: object | None = None,
) -> dict[str, float]:
    """Bootstrap (1−alpha) CI on the mean of ``values``.

    ``kind="block"`` (default) runs the moving-block bootstrap over
    non-overlapping blocks of length ``block`` (default ``int(sqrt(n))``) to
    preserve autocorrelation. ``kind="iid"`` samples single observations.
    ``stat_fn`` overrides the statistic (default: the sample mean).

    Returns {"lo", "hi", "mean", "se"}.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("bootstrap_ci needs at least 2 observations")
    rng = np.random.default_rng(seed)
    fn = stat_fn if stat_fn is not None else _mean
    blen = block if block > 0 else max(1, int(np.sqrt(n)))
    if kind == "block":
        n_blocks = max(1, (n + blen - 1) // blen)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if kind == "block":
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate(
                [(s + np.arange(blen, dtype=np.int64)) % n for s in starts]
            )[:n]
            sample = x[idx]
        else:  # iid
            sample = x[rng.integers(0, n, size=n)]
        stats[b] = fn(sample)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(stats)), "se": float(np.std(stats))}


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------
def diebold_mariano(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    *,
    h: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold–Mariano test that the loss of A is not equal to the loss of B.

    Loss here is the *squared error* of the forecast. The variance of the loss
    differential is HAC-adjusted for overlap (Newey–West with ``lag = h``), so
    the statistic is valid when horizons overlap.

    Returns {"dm", "p_value", "mean_diff"} where ``mean_diff = mean(loss_a −
    loss_b)`` — a negative value means A has lower loss on average.
    """
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        raise ValueError("diebold_mariano needs equal-length series of >= 2")
    d = a**2 - b**2
    d = d - d.mean()
    n = d.size
    lag = int(max(1, min(h, n - 1)))
    gamma = np.array([np.mean(d[i:] * d[: n - i]) for i in range(lag + 1)])
    var = gamma[0] + 2 * np.sum(gamma[1:] * (1 - np.arange(1, lag + 1) / (lag + 1)))
    if var <= 0:
        var = np.var(d, ddof=1)
    se = sqrt(var / n)
    mean_diff = float(np.mean(a**2 - b**2))
    dm = float(mean_diff / se) if se > 0 else float("nan")
    if alternative in ("two-sided", "two_sided"):
        p = 2.0 * (1.0 - normal_cdf(abs(dm)))
    elif alternative == "less":
        p = normal_cdf(dm)
    else:  # greater
        p = 1.0 - normal_cdf(dm)
    return {"dm": dm, "p_value": float(p), "mean_diff": mean_diff}


def normal_cdf(z: float) -> float:
    """Standard-normal CDF (used for asymptotically-normal test statistics)."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# ---------------------------------------------------------------------------
# probability calibration (E5)
# ---------------------------------------------------------------------------
def brier_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean squared error of a probability forecast: mean((p − y)^2), 0 = perfect."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        raise ValueError("brier_score needs equal-length non-empty series")
    return float(np.mean((b - a) ** 2))


def calibration_curve(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict:
    """Binned calibration of predicted probabilities against realized outcomes.

    Bin *by predicted value* (equal-frequency, n-weighted); each bin reports the
    mean prediction and the observed outcome rate plus its binomial s.e. The
    return includes an n-weighted OLS ``slope`` of obs-rate on predicted — the
    honest calibration statistic: slope ≈ 1 = calibrated, slope ≈ 0 = no signal
    beyond the base rate. Returns:
    ``{"bins":[{lo,hi,pred_mean,obs_rate,se,n}...], "slope", "intercept"}``.
    """
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        raise ValueError("calibration_curve needs equal-length series of >= 2")
    n_bins = max(1, int(min(n_bins, a.size)))
    order = np.argsort(b, kind="mergesort")
    a_s, b_s = a[order], b[order]
    edges = np.linspace(0, a.size, n_bins + 1).astype(np.int64)
    bins: list[dict] = []
    for lo, hi in itertools.pairwise(edges):
        bs, aa = b_s[lo:hi], a_s[lo:hi]
        n = int(bs.size)
        p = float(bs.mean()) if n else float("nan")
        o = float(aa.mean()) if n else float("nan")
        se = sqrt(o * (1.0 - o) / n) if n else float("nan")
        bins.append({
            "lo": float(bs.min()) if n else float("nan"),
            "hi": float(bs.max()) if n else float("nan"),
            "pred_mean": p, "obs_rate": o, "se": se, "n": n,
        })
    ws = np.asarray([bb["n"] for bb in bins], dtype=np.float64)
    ps = np.asarray([bb["pred_mean"] for bb in bins], dtype=np.float64)
    oo = np.asarray([bb["obs_rate"] for bb in bins], dtype=np.float64)
    wsum = float(ws.sum())
    pm = float((ps * ws).sum() / wsum)
    om = float((oo * ws).sum() / wsum)
    sp = float((ws * (ps - pm) ** 2).sum())
    if sp > 0:
        slope = float((ws * (ps - pm) * (oo - om)).sum() / sp)
    else:
        slope = float("nan")
    return {"bins": bins, "slope": slope, "intercept": float(om - slope * pm)}


def hac_se(values: Sequence[float], *, lag: int = 0) -> float:
    """Newey–West (Bartlett) HAC standard error of the mean of ``values``.

    ``lag`` handles overlap (e.g. drift over ``h`` events uses ``lag=h``).
    Falls back to the raw sample s.e. when the HAC variance is non-positive.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    d = x - x.mean()
    L = int(min(int(lag), n - 1)) if lag > 0 else 0
    gamma = np.array([np.mean(d[i:] * d[: n - i]) for i in range(L + 1)])
    var = gamma[0] + 2.0 * np.sum(gamma[1:] * (1.0 - np.arange(1, L + 1) / (L + 1)))
    if var <= 0:
        var = np.var(d, ddof=1)
    return float(sqrt(var / n))


# ---------------------------------------------------------------------------
# experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    split_tags: Sequence[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0x51ED,
) -> dict:
    """E1–E4 result bundle for one feature on one label: JSON-serializable.

    Returns per-split (or overall) rank IC, ICIR, hit rate, decile spread, and
    a block-bootstrap CI of the rank IC.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must be equal length")

    result: dict = {"n": int(yt.size), "overall": _one_result(yt, yp, n_boot, seed)}
    if split_tags is not None:
        tags = np.asarray([str(t) for t in split_tags])
        per_split: dict[str, dict] = {}
        for tag in np.unique(tags):
            m = tags == tag
            tag_seed = seed ^ (sum(int(c) for c in tag.encode()) & 0xFFFF)  # deterministic
            per_split[str(tag)] = _one_result(yt[m], yp[m], n_boot, tag_seed)
        result["per_split"] = per_split
    return result


def _one_result(yt: np.ndarray, yp: np.ndarray, n_boot: int, seed: int) -> dict:
    n = int(yt.size)
    if n < 4:
        return {"n": n, "rank_ic": None, "icir": None, "hit_rate": None,
                "decile_spread": None, "ci95": None}
    ic = rank_ic(yt, yp)
    ci = _rank_ic_bootstrap(yt, yp, n_boot=n_boot, seed=seed)
    ic_clean = None if (isnan(ic) or isinf(ic)) else float(ic)
    val_icir = icir([ic]) if ic_clean is not None else float("nan")
    icir_clean = None if (ic_clean is None or np.isnan(val_icir)) else float(val_icir)
    return {
        "n": n,
        "rank_ic": ic_clean,
        "icir": icir_clean,
        "hit_rate": hit_rate(yt, yp),
        "decile_spread": decile_spread(yt, yp),
        "ci95": {"lo": ci["lo"], "hi": ci["hi"], "mean": ci["mean"]},
    }


def _rank_ic_bootstrap(
    yt: np.ndarray,
    yp: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    block: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Block-bootstrap CI of the rank IC, resampling (y, pred) pairs jointly.

    This is the CI that E1–E4 report: it drives the "is IC ≠ 0" decision. The
    bootstrap preserves the joint pair distribution AND, via moving blocks,
    the autocorrelation of the tape.
    """
    n = yt.size
    rng = np.random.default_rng(seed)
    blen = block if block > 0 else max(1, int(np.sqrt(n)))
    n_blocks = max(1, (n + blen - 1) // blen)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate(
            [(s + np.arange(blen, dtype=np.int64)) % n for s in starts]
        )[:n]
        stats[b] = rank_ic(yt[idx], yp[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(stats))}