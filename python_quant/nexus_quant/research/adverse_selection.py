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
from dataclasses import dataclass
from math import erf, sqrt
from typing import Any

import numpy as np

from ..book_state import Side
from ..itch_parser import EventType, NormalizedEvent
from .experiments import bootstrap_ci, hac_se, normal_cdf
from .queue_dynamics import FillRecord, QueueTracker

_View = Any


def is_book_affecting(ev: NormalizedEvent) -> bool:
    """True if event mutates visible order book depth or quotes.

    Off-market prints, cross trades, and hidden executions (e.g. ITCH 'P' TRADE
    messages) do not alter displayed depth or quotes and must not advance the
    microstructure event clock.
    """
    return ev.kind in (
        EventType.ADD,
        EventType.ADD_MPID,
        EventType.EXECUTE,
        EventType.EXECUTE_PX,
        EventType.CANCEL,
        EventType.DELETE,
        EventType.REPLACE,
    )


def filter_book_events(events: Sequence[NormalizedEvent]) -> list[NormalizedEvent]:
    """Filter event stream to retain only depth- or quote-affecting book events."""
    return [ev for ev in events if is_book_affecting(ev)]


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


def _post_fill_drift_tracker(
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
    "p_value", "hac_se", "nw_lag"}`` — JSON-able. Note the units: ``mean_ticks``
    and ``ci95`` are in ticks; ``mean_bps`` is ``mean_ticks / base_mid * 1e4``
    (the CI is on the ticks, so compare it with ``mean_ticks``, not
    ``mean_bps``).
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


def _P_adverse_single(
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

# ---------------------------------------------------------------------------
# Person B PassiveFill and Adverse Selection Report Tools
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PassiveFill:
    """One passive execution: tape index of the fill, side of the *resting* order."""

    idx: int
    side: Side
    price: int = 0
    size: int = 0
    ofi: float = 0.0  # order-flow imbalance observed at/just before the fill
    queue_frac: float = float("nan")  # queue position at placement (0 = front)


def _mids(mids: Sequence[float]) -> np.ndarray:
    return np.asarray(mids, dtype=np.float64).reshape(-1)



def _post_fill_drift_mids(
    fills: Sequence[PassiveFill],
    mids: Sequence[float],
    h: int,
) -> np.ndarray:
    """Signed post-fill drift ``s·(mid[t+h] − mid[t])`` per fill (NaN if no future).

    ``mids`` is the mid series indexed by tape event (one entry per event the
    book was replayed through). A fill at ``idx`` beyond ``len(mids) − h − 1``
    has no ``t+h`` state and yields NaN — it is dropped by the aggregators.
    """
    m = _mids(mids)
    n = m.size
    out = np.full(len(fills), np.nan, dtype=np.float64)
    for i, f in enumerate(fills):
        t = int(f.idx)
        if t < 0 or t + h >= n or m[t] <= 0 or m[t + h] <= 0:
            continue
        s = 1.0 if f.side == Side.Bid else -1.0
        out[i] = s * (m[t + h] - m[t])
    return out




def post_fill_drift(
    fills: Sequence[Any],
    book_or_mids: Any,
    h: int = 1,
    *,
    n_boot: int = 2000,
    seed: int = 0x51ED,
) -> dict[str, Any] | np.ndarray:
    """Mean signed drift after a passive fill.

    Supports:
    - QueueTracker + FillRecord (Lokesh PR #20): returns summary dict with HAC s.e., CI, p-value.
    - mid series + PassiveFill (Person B): returns np.ndarray of per-fill signed drifts.
    """
    if hasattr(book_or_mids, "mid_history"):
        return _post_fill_drift_tracker(
            fills, book_or_mids, h, n_boot=n_boot, seed=seed
        )
    return _post_fill_drift_mids(fills, book_or_mids, h)


def P_adverse(
    fill_or_drifts: Any,
    book: Any = None,
    ofi: float = 0.0,
    queue_pos: float = 0.0,
    *,
    h: int = 5,
) -> Any:
    """Single fill adverse indicator (Lokesh PR #20) or array adverse fraction."""
    if book is not None or hasattr(fill_or_drifts, "side"):
        return _P_adverse_single(fill_or_drifts, book, ofi=ofi, queue_pos=queue_pos, h=h)
    return p_adverse(fill_or_drifts)

def nw_tstat(x: Sequence[float], *, lag: int) -> dict[str, float]:
    """Mean, Newey–West (Bartlett) standard error, t-stat and two-sided p-value."""
    a = np.asarray(x, dtype=np.float64)
    a = a[~np.isnan(a)]
    n = a.size
    if n < 3:
        return {"n": float(n), "mean": float(a.mean()) if n else float("nan"),
                "se": float("nan"), "t": float("nan"), "p": float("nan")}
    d = a - a.mean()
    L = int(max(0, min(lag, n - 1)))
    gamma0 = float(np.dot(d, d) / n)
    var = gamma0
    for j in range(1, L + 1):
        gj = float(np.dot(d[j:], d[:-j]) / n)
        var += 2.0 * (1.0 - j / (L + 1)) * gj
    var = max(var, 0.0)
    se = sqrt(var / n) if var > 0 else float("nan")
    mean = float(a.mean())
    t = mean / se if se and se > 0 else float("nan")
    p = 2.0 * (1.0 - _ncdf(abs(t))) if not np.isnan(t) else float("nan")
    return {"n": float(n), "mean": mean, "se": float(se), "t": float(t), "p": float(p)}


def _ncdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def p_adverse(drifts: Sequence[float], *, conditional: bool = True) -> float:
    """P(drift < 0).

    Args:
        drifts: Sequence of signed post-fill mid moves (negative = adverse).
        conditional: If True (default), computes P(drift < 0 | drift != 0), conditioning
            only on intervals with a non-zero mid move. If False, computes unconditional
            P(drift < 0) = N(drift < 0) / N(valid drifts), treating zero moves as non-adverse.
    """
    a = np.asarray(drifts, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    if conditional:
        nz = a[a != 0.0]
        if nz.size == 0:
            return float("nan")
        return float(np.mean(nz < 0.0))
    return float(np.mean(a < 0.0))


def p_adverse_unconditional(drifts: Sequence[float]) -> float:
    """Unconditional P(drift < 0) = N(drift < 0) / N(valid fills), treating zero moves as non-adverse."""
    return p_adverse(drifts, conditional=False)


def pre_fill_drift(
    fills: Sequence[PassiveFill],
    mids: Sequence[float],
    h: int,
) -> np.ndarray:
    """Pre-fill benchmark: the same signed drift over the ``h`` events *before* the fill.

    ``s·(mid[t−1] − mid[t−1−h])`` — the window ends strictly before the fill
    event, so it excludes the fill's own mechanical BBO move (a fully consumed
    level shifts the mid at ``t`` itself). ``post − pre`` is the fill-conditional
    excess drift: it nets out a tape that was already trending into the fill,
    so E6 reports selection, not momentum.

    Methodological Note (M03):
    This control serves as a pre-fill trend benchmark. Because it evaluates the
    same order's pre-execution trajectory rather than a counterfactual unexecuted
    resting order, it benchmarks prevailing market momentum immediately prior to
    trade arrival. For comparing executed passive orders directly against matched
    unexecuted resting orders over the forward window [t, t+h], use
    ``matched_unexecuted_control_drift``.
    """
    m = _mids(mids)
    out = np.full(len(fills), np.nan, dtype=np.float64)
    for i, f in enumerate(fills):
        t = int(f.idx)
        a, b = t - 1 - h, t - 1
        if a < 0 or b >= m.size or m[a] <= 0 or m[b] <= 0:
            continue
        s = 1.0 if f.side == Side.Bid else -1.0
        out[i] = s * (m[b] - m[a])
    return out


def matched_unexecuted_control_drift(
    controls: Sequence[PassiveFill],
    mids: Sequence[float],
    h: int,
) -> np.ndarray:
    """Matched unexecuted control: signed drift over ``[t, t+h]`` for unexecuted orders.

    Measures forward price drift over ``[t, t+h]`` for unexecuted control orders
    (e.g., cancelled or resting orders matched by side, price, queue position, and
    decision time), providing a clean counterfactual comparison against filled orders.
    """
    return _post_fill_drift_mids(controls, mids, h)


def _bucket_ofi(v: float) -> str:
    if np.isnan(v) or v == 0.0:
        return "ofi=0"
    return "ofi>0" if v > 0 else "ofi<0"


def _bucket_queue(q: float) -> str:
    if np.isnan(q):
        return "queue=n/a"
    if q < 1.0 / 3.0:
        return "queue=front"
    if q < 2.0 / 3.0:
        return "queue=mid"
    return "queue=back"


def adverse_selection_report(
    fills: Sequence[PassiveFill],
    mids: Sequence[float],
    *,
    horizons: Sequence[int] = (1, 5, 25),
    min_group: int = 20,
) -> dict[str, Any]:
    """E6 bundle: per horizon, overall + grouped drift stats and P(adverse).

    Returns ``{"n_fills": int, "horizons": {h: {"overall": {...}, "pre_fill":
    {...}, "post_minus_pre": {...}, "groups": {label: {...}}}}}`` where each
    stats dict carries ``n, mean_drift, se_nw, t_nw, p_value, p_adverse``.
    Groups with fewer than ``min_group`` fills are omitted (no claims on 5 fills).
    """
    fills = sorted(fills, key=lambda f: f.idx)
    result: dict[str, Any] = {"n_fills": len(fills), "horizons": {}}
    for h in horizons:
        d = post_fill_drift(fills, mids, h)
        c = pre_fill_drift(fills, mids, h)
        excess = d - c
        block: dict[str, Any] = {
            "overall": _stats(d, h),
            "pre_fill": _stats(c, h),
            "post_minus_pre": _stats(excess, h),
            "groups": {},
        }
        labels = {
            "side=bid": np.array([f.side == Side.Bid for f in fills]),
            "side=ask": np.array([f.side == Side.Ask for f in fills]),
        }
        for name in ("ofi<0", "ofi=0", "ofi>0"):
            labels[name] = np.array([_bucket_ofi(f.ofi) == name for f in fills])
        for name in ("queue=front", "queue=mid", "queue=back"):
            labels[name] = np.array([_bucket_queue(f.queue_frac) == name for f in fills])
        for name, mask in labels.items():
            if mask.size and int(np.sum(mask & ~np.isnan(d))) >= min_group:
                block["groups"][name] = _stats(d[mask], h)
        result["horizons"][int(h)] = block
    return result


def _stats(d: np.ndarray, h: int) -> dict[str, float]:
    t = nw_tstat(d, lag=int(h))
    return {
        "n": int(t["n"]),
        "mean_drift": t["mean"],
        "se_nw": t["se"],
        "t_nw": t["t"],
        "p_value": t["p"],
        "p_adverse": p_adverse(d, conditional=True),
        "p_adverse_conditional": p_adverse(d, conditional=True),
        "p_adverse_unconditional": p_adverse(d, conditional=False),
    }


def fills_from_tracker(completed: Sequence[Any], *, ofi_at: Sequence[float] | None = None) -> list[PassiveFill]:
    """Turn ``OrderLevelTracker.completed`` records with outcome ``"filled"`` into fills.

    The fill index is the tape event of the order's *first* execution
    (``idx_first_fill``); the queue fraction is ``ahead_at_add / (ahead_at_add +
    size0)`` at placement (0 = front of the queue). ``ofi_at[i]`` (optional) supplies the OFI observed at event
    ``i`` so fills can be bucketed by flow sign.
    """
    out: list[PassiveFill] = []
    for o in completed:
        if getattr(o, "outcome", "") != "filled":
            continue
        idx = o.idx_first_fill if getattr(o, "idx_first_fill", None) is not None else o.idx_end
        if idx is None:
            continue
        denom = float(o.ahead_at_add + max(1, o.size0))
        q = float(o.ahead_at_add) / denom if denom > 0 else float("nan")
        ofi_v = float(ofi_at[idx]) if ofi_at is not None and idx < len(ofi_at) else 0.0
        out.append(PassiveFill(idx=int(idx), side=Side(int(o.side)), price=int(o.price),
                               size=int(o.filled), ofi=ofi_v, queue_frac=q))
    out.sort(key=lambda f: f.idx)
    return out
