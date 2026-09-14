"""E6 — adverse selection after passive fills (research, Phase 3).

A passive order that gets filled was picked off by a market order — the classic
microstructure question is *how much does the mid price move against you* right
after the fill, and whether that adverse drift is predictable from information
knowable *at* the fill:

* the side of the rest (ask = resting seller, picked off by a buy take);
* the **OFI of the take** that filled us (``FillRecord.ofi`` — the book's order
  flow in the very event that consumed the resting order);
* the queue position (``ahead_at_fill / level_size_at_fill``) — how far behind
  the touch the fill sat.

Everything here is a post-hoc measurement over recorded fills; nothing is a
feature for a live strategy. No lookahead by construction: drift is defined
from the mid *at* the fill (``mid_history[event_index]``) forward, and rows
whose horizon runs off the end of the tape are dropped, not extrapolated.

Honesty contract (plan_2.md §0): the null tape (``FLOW_PRESETS["rw"]``) should
show no systematic adverse drift (CI straddling 0); a structural-drift tape
(``FLOW_PRESETS["drift"]``) should — that sign test is what the tests assert.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..book_state import Side
from .experiments import bootstrap_ci, hac_se, normal_cdf
from .queue_dynamics import FillRecord, QueueTracker

_View = Any


def side_str(side: Side) -> str:
    """JSON-safe side label: ``"ask"`` for a resting seller, ``"bid"`` else."""
    return "ask" if side == Side.Ask else "bid"


def drift_ticks(fill: FillRecord, book: QueueTracker, h: int) -> float | None:
    """Mid move ``h`` events after the fill: ``mid[t+h] - mid[t]`` (post-fill).

    Returns ``None`` when ``t + h`` runs past the tape (dropped, never
    extrapolated). ``fill.event_index`` is the tracker's ``seq`` at the fill, so
    ``mid_history[event_index]`` is the mid *after* the fill event itself.
    """
    idx = fill.event_index
    mh = book.mid_history
    if idx + h >= len(mh):
        return None
    return float(mh[idx + h] - mh[idx])


def _is_adverse(fill: FillRecord, d: float) -> bool:
    """Mid moved against the passive side: an ask (resting seller) is adverse on
    an up-move, a bid (resting buyer) on a down-move."""
    if fill.side == Side.Ask:
        return d > 0
    return d < 0


def post_fill_drift(
    fills: Sequence[FillRecord],
    book: QueueTracker,
    h: int,
    *,
    n_boot: int = 2000,
    seed: int = 0x51ED,
) -> dict[str, Any]:
    """Mean signed drift in the ``h`` events after a passive fill.

    ``drift_ticks`` rows whose horizon runs off the tape are dropped. The
    mean-is-zero test uses a Newey–West HAC s.e. with ``lag = h - 1`` (the CIs
    and ``t`` over overlapping horizons), with a two-sided normal p-value. The
    CI is a *block* bootstrap over block ``h``.

    Returns ``{"h", "n", "mean_ticks", "mean_bps", "ci95", "t_stat",
    "p_value", "hac_se", "nw_lag"}`` — JSON-able.
    """
    ticks: list[float] = []
    bps: list[float] = []
    for f in fills:
        d = drift_ticks(f, book, h)
        if d is None:
            continue
        ticks.append(d)
        bps.append(d / max(1.0, float(book.mid_history[f.event_index])) * 1e4)
    nw_lag = int(max(0, h - 1))
    if len(ticks) == 0:
        return {
            "h": int(h), "n": 0, "mean_ticks": None, "mean_bps": None,
            "ci95": {"lo": None, "hi": None, "mean": None, "se": None},
            "t_stat": None, "p_value": None, "hac_se": None, "nw_lag": nw_lag,
        }
    if len(ticks) == 1:
        return {
            "h": int(h), "n": 1, "mean_ticks": float(ticks[0]),
            "mean_bps": float(bps[0]),
            "ci95": {"lo": None, "hi": None, "mean": None, "se": None},
            "t_stat": None, "p_value": None, "hac_se": None, "nw_lag": nw_lag,
        }
    x = np.asarray(ticks, dtype=np.float64)
    mean_ticks = float(np.mean(x))
    mean_bps = float(np.mean(bps))
    hse = hac_se(x, lag=nw_lag)
    t = mean_ticks / hse if hse and hse > 0 and np.isfinite(hse) else float("nan")
    p = 2.0 * (1.0 - normal_cdf(abs(t))) if np.isfinite(t) else float("nan")
    return {
        "h": int(h), "n": int(x.size), "mean_ticks": mean_ticks, "mean_bps": mean_bps,
        "ci95": bootstrap_ci(x, kind="block", block=max(1, h), n_boot=n_boot, seed=seed),
        "t_stat": float(t), "p_value": float(p),
        "hac_se": float(hse) if np.isfinite(hse) else float("nan"),
        "nw_lag": nw_lag,
    }


def P_adverse(
    fill: FillRecord,
    book: QueueTracker,
    ofi: float,
    queue_pos: float,
    *,
    h: int = 5,
) -> dict[str, Any]:
    """Whether one fill was adverse over the next ``h`` events.

    ``ofi`` / ``queue_pos`` are the fill-time conditionals (caller-supplied;
    this is a measurement helper, not a strategy feature). Fields are ``None``
    when the horizon runs off the tape. Returns
    ``{"side", "ofi", "queue_pos", "h", "drift_ticks", "drift_bps",
    "adverse"}`` — ``adverse`` is ``True`` iff the mid moved against the
    passive side.
    """
    d = drift_ticks(fill, book, h)
    base = {"side": side_str(fill.side), "ofi": float(ofi),
            "queue_pos": float(queue_pos), "h": int(h)}
    if d is None:
        return {**base, "drift_ticks": None, "drift_bps": None, "adverse": None}
    return {
        **base, "drift_ticks": d,
        "drift_bps": d / max(1.0, float(book.mid_history[fill.event_index])) * 1e4,
        "adverse": _is_adverse(fill, d),
    }


def adverse_groups(
    fills: Sequence[FillRecord],
    book: QueueTracker,
    *,
    horizons: Sequence[int] = (1, 5, 25),
    n_queue_terciles: int = 3,
    n_boot: int = 2000,
    seed: int = 0x51ED,
) -> dict[str, Any]:
    """Adverse-drift statistics conditioned on (side × OFI sign × queue tercile).

    Every fill is bucketed exactly once: ``queue_pos = ahead_at_fill /
    max(1, level_size_at_fill)`` with tercile edges from ``np.quantile``
    (deterministic, no RNG); ``ofi_sign = sign(fill.ofi)``. Per bin per horizon:
    mean signed drift (bps and ticks), the adverse probability
    ``P(adverse | side, ofi, queue)``, a block-bootstrap CI, and a
    Newey–West ``t``/``p`` (``lag = h - 1``). Rows whose horizon runs off the
    tape are dropped per horizon (bin ``n_fills`` still counts every fill).

    Return is fully JSON-able: ``{"horizons", "n_queue_terciles",
    "queue_tercile_edges", "bins": [{"side", "ofi_sign", "queue_tercile",
    "n_fills", "horizons": {h: {...}}}]}``.
    """
    fills = list(fills)
    if not fills:
        raise ValueError("adverse_groups needs at least one fill")
    nq = int(n_queue_terciles)
    qpos = np.asarray(
        [f.ahead_at_fill / max(1, f.level_size_at_fill) for f in fills],
        dtype=np.float64,
    )
    edges = np.quantile(qpos, np.linspace(0.0, 1.0, nq + 1)[1:-1])
    terc = np.digitize(qpos, edges)
    side_arr = np.asarray([f.side for f in fills])
    ofi_sign_arr = np.asarray([float(np.sign(f.ofi)) for f in fills])
    out_bins: list[dict[str, Any]] = []
    for side in (Side.Ask, Side.Bid):
        for og in (-1, 0, 1):
            for qt in range(nq):
                sel = np.where(
                    (side_arr == side) & (ofi_sign_arr == og) & (terc == qt)
                )[0]
                h_entries: dict[str, dict[str, Any]] = {}
                for h in horizons:
                    h_entries[str(int(h))] = _horizon_stats(
                        fills, sel, book, int(h), n_boot=n_boot, seed=seed
                    )
                out_bins.append({
                    "side": side_str(side),
                    "ofi_sign": int(og),
                    "queue_tercile": int(qt),
                    "n_fills": int(sel.size),
                    "horizons": h_entries,
                })
    return {
        "horizons": [int(h) for h in horizons],
        "n_queue_terciles": nq,
        "queue_tercile_edges": [float(e) for e in edges],
        "bins": out_bins,
    }


def _horizon_stats(
    fills: list[FillRecord],
    sel: np.ndarray,
    book: QueueTracker,
    h: int,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    ticks: list[float] = []
    bps: list[float] = []
    adverse: list[float] = []
    for i in sel:
        f = fills[i]
        d = drift_ticks(f, book, h)
        if d is None:
            continue
        ticks.append(d)
        bps.append(d / max(1.0, float(book.mid_history[f.event_index])) * 1e4)
        adverse.append(1.0 if _is_adverse(f, d) else 0.0)
    nw_lag = int(max(0, h - 1))
    if len(ticks) == 0:
        return {
            "n": 0, "mean_drift_ticks": None, "mean_drift_bps": None,
            "p_adverse": None,
            "ci95": {"lo": None, "hi": None, "mean": None, "se": None},
            "t_stat": None, "p_value": None, "hac_se": None, "nw_lag": nw_lag,
        }
    if len(ticks) == 1:
        return {
            "n": 1, "mean_drift_ticks": float(ticks[0]),
            "mean_drift_bps": float(bps[0]), "p_adverse": float(adverse[0]),
            "ci95": {"lo": None, "hi": None, "mean": None, "se": None},
            "t_stat": None, "p_value": None, "hac_se": None, "nw_lag": nw_lag,
        }
    x = np.asarray(ticks, dtype=np.float64)
    hse = hac_se(x, lag=nw_lag)
    t = float(np.mean(x)) / hse if hse and hse > 0 and np.isfinite(hse) else float("nan")
    p = 2.0 * (1.0 - normal_cdf(abs(t))) if np.isfinite(t) else float("nan")
    return {
        "n": int(x.size),
        "mean_drift_ticks": float(np.mean(x)),
        "mean_drift_bps": float(np.mean(bps)),
        "p_adverse": float(np.mean(adverse)),
        "ci95": bootstrap_ci(x, kind="block", block=max(1, h), n_boot=n_boot, seed=seed),
        "t_stat": float(t), "p_value": float(p),
        "hac_se": float(hse) if np.isfinite(hse) else float("nan"),
        "nw_lag": nw_lag,
    }