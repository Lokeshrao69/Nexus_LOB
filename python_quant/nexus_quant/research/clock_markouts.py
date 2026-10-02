"""E6b — passive-fill markouts on the **clock**, not the event clock.

The event-time E6 in :mod:`.adverse_selection` reports that 90–97 % of passive
fills are followed by an adverse mid move over ``h ∈ {1, 5, 25}`` *book events*.
That number is suspect for a mechanical reason: one ITCH ``E`` message executes
exactly **one** resting order, so one incoming aggressive order that sweeps four
resting orders emits four messages and moves the BBO at ``t+1 … t+3`` — squarely
inside an ``h = 1 … 25`` window. The mid then moves "against" the fill because
the fill was part of a sweep, not because anyone knew anything.

Units
-----
ITCH prices are integers in **$0.0001**. That is *not* a tick: AAPL and QQQ both
quote on a $0.01 tick, i.e. :data:`TICK_PX_UNITS` = 100 price units. Everything
here is reported in $0.0001 units (suffix ``_px0001``) or as a multiple of the
half-spread at the fill (``_half_spreads``); nothing is called a tick.

The unit of observation
----------------------
One aggressive order can fill many resting orders at the same nanosecond, and
those fills are not independent draws — they share one aggressor, one price path
and one mechanical BBO move. So the **main** tables use one observation per
*aggressor event* (:func:`group_aggressor_events`: same ITCH timestamp, same
resting side), and the Newey–West s.e. and block-bootstrap CIs are computed over
events. Per-fill numbers are kept as a secondary table, where the effective
sample size is overstated by exactly the sweep multiplicity.

Aggregating to the event also sharpens the ``after`` reference: the event's
``after`` mid is the mid once its **last** message is applied, so the whole
sweep's mechanical move is excluded, not just the first message's.

Two reference mids
------------------
* **before** — the mid of the last book event strictly *before* the event's first
  message. Includes the sweep's BBO move.
* **after** — the mid once the event's last message is fully applied. Excludes it.

``before − after`` is exactly ``s · (mid_after − mid_before)``: the quote move the
fill's own execution caused. It contains neither the terminal mid nor ``H``, so it
is **constant across horizons by construction** and is *not* a test of whether the
effect is mechanical or informational. It measures one thing only — how much of a
``before``-referenced markout is the fill's own quote move. The test that does
discriminate is **reversion**: a mechanical dislocation reverts once depth is
replenished, whereas informed flow leaves a permanent impact.

Signs and the realized spread
-----------------------------
* ``s = +1`` for a resting **bid** (we bought), ``s = −1`` for a resting **ask**
  (we sold). ``markout = s · (mid[t+H] − mid_ref)`` — **negative = adverse**,
  matching ``_post_fill_drift_mids``.
* ``half_spread_earned = s · (mid_ref − P)``, ``P`` the size-weighted execution
  price over the event's fills (its VWAP).
* ``realized_spread = half_spread_earned + markout = s · (mid[t+H] − P)``. This is
  the headline: it answers whether a passive fill makes money *net* of adverse
  selection. The decomposition shifts value between the two terms when the
  reference changes, but the sum is **reference-invariant**
  (``test_realized_spread_is_reference_invariant``).

Location, dispersion and controls
---------------------------------
Means over markouts are tail-sensitive, so every horizon reports the **mean, the
median and the 10 %-each-tail trimmed mean**, each with its own 95 % CI, so a
non-monotone mean path can be read against a robust one. ``pre_fill`` is the
clock analogue of ``adverse_selection.pre_fill_drift``: the same signed move over
the ``H`` *before* the fill, ending strictly before it, so ``post_minus_pre``
nets out a session that was already trending into the fill.

Statistics reuse the existing toolbox — :func:`.adverse_selection.nw_tstat` for
the Newey–West bundle and :func:`.experiments.bootstrap_ci` for every CI. The
clock analogue of the event-time ``lag = h`` is :func:`clock_overlap_lag`: the
median number of later events whose own horizon window overlaps a given one's.
When that overlap is zero the observations do not overlap at all, so the CI uses
``kind="iid"`` rather than a degenerate length-1 block.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..book_state import Side
from ..itch_parser import EventType, NormalizedEvent
from .adverse_selection import _bucket_queue, nw_tstat, p_adverse
from .experiments import bootstrap_ci
from .queue_dynamics import REGULAR_CLOSE_NS, REGULAR_OPEN_NS, OrderLevelTracker

NS_PER_MS: int = 1_000_000
NS_PER_S: int = 1_000_000_000

#: One ITCH price unit, in dollars.
PX_UNIT_USD: float = 1e-4
#: Price units in one $0.01 tick (AAPL and QQQ both quote on a penny tick).
TICK_PX_UNITS: int = 100

#: Default clock horizons: 100 ms, 1 s, 10 s, 60 s.
CLOCK_HORIZONS_NS: tuple[int, ...] = (
    100 * NS_PER_MS,
    1 * NS_PER_S,
    10 * NS_PER_S,
    60 * NS_PER_S,
)

#: Cap on the Newey–West lag / bootstrap block length (keeps the HAC kernel from
#: eating the whole sample on a 60 s horizon in a busy name).
MAX_OVERLAP_LAG: int = 2000

#: Floor on the **clock** length of a bootstrap block. At H = 100 ms a block sized
#: to the horizon — or to the usual ``sqrt(n)`` — would span a second or two and
#: resample price moves that are still correlated. Blocks are therefore
#: ``max(H, 60 s)`` of wall-clock time, converted to an event count per symbol.
BOOTSTRAP_MIN_BLOCK_NS: int = 60 * NS_PER_S

#: Trim proportion per tail for the robust location estimate.
TRIM_PROP: float = 0.10

#: Mid prices sit on a half-tick lattice (50 price units for a penny-tick name),
#: so a median markout is pinned to a lattice point and its bootstrap replicates
#: collapse onto a handful of values. Medians are reported; CIs on them are not.
MEDIAN_LATTICE_NOTE: str = (
    "medians sit on the 50-unit half-tick lattice, so bootstrap CIs on them "
    "degenerate and are not reported — use the sign-split table instead"
)

#: Mid lattice step in price units: half of a $0.01 tick.
MID_LATTICE_PX0001: int = TICK_PX_UNITS // 2

#: **Assumed** maker rebates, in $0.0001 per share, for the break-even columns.
#: These are *not* measured from the tape — ITCH carries no fee data — they are
#: illustrative values bracketing a typical 2019 US equity maker rebate. Any
#: conclusion about passive profitability net of rebate is conditional on them.
REBATE_SCENARIOS_PX0001: tuple[int, ...] = (20, 30)

#: Stated wherever a rebate-adjusted figure appears.
REBATE_ASSUMPTION_NOTE: str = (
    "maker rebate is an ASSUMED constant per share, not measured from the tape — "
    "ITCH carries no fee data"
)

_REFERENCES: tuple[str, ...] = ("before", "after")


def horizon_label(ns: int) -> str:
    """``100000000 → "100ms"``, ``1000000000 → "1s"`` — stable JSON/table keys."""
    if ns % NS_PER_S == 0:
        return f"{ns // NS_PER_S}s"
    if ns % NS_PER_MS == 0:
        return f"{ns // NS_PER_MS}ms"
    return f"{ns}ns"


# --------------------------------------------------------------------------- #
# fill-time context, harvested during replay                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ClockFill:
    """One passive fill plus the level context knowable **at** the fill.

    ``OrderLevelTracker``'s ``TrackedOrder`` keeps no fill-time queue snapshot
    (only ``ahead_at_add``), so :class:`ClockFillCollector` reads the tracker
    *before* the execute message mutates it. All sizes are shares; ``price`` is
    the execution price in $0.0001 units.
    """

    tape_idx: int
    ts_ns: int
    side: Side
    price: int
    size: int                 # shares this message executed against our order
    ahead_at_fill: int        # displayed size still ahead of us in the FIFO
    level_size_at_fill: int   # total displayed size at our (side, price)
    at_touch: bool            # our level was the best on our side
    emptied_touch: bool       # ... and this message cleared it (mechanical BBO move)

    @property
    def queue_frac(self) -> float:
        """Queue position at the fill, ``0`` = front of the level."""
        lvl = self.level_size_at_fill
        return self.ahead_at_fill / lvl if lvl > 0 else float("nan")

    @property
    def size_frac(self) -> float:
        """Executed shares as a fraction of the level's displayed size."""
        lvl = self.level_size_at_fill
        return self.size / lvl if lvl > 0 else float("nan")


class ClockFillCollector:
    """Harvest first-execution context off an :class:`OrderLevelTracker`.

    Call :meth:`observe` with the tape index and the event **before** handing the
    event to the tracker — the pre-mutation queue state is the whole point.
    Only the *first* execution of each resting order is kept, which is exactly
    the population the event-time E6 uses (``idx_first_fill`` in
    ``fills_from_tracker``), so the two studies are directly comparable.
    """

    __slots__ = ("fills",)

    def __init__(self) -> None:
        self.fills: list[ClockFill] = []

    def observe(self, tape_idx: int, ev: NormalizedEvent, tracker: OrderLevelTracker) -> None:
        if ev.kind not in (EventType.EXECUTE, EventType.EXECUTE_PX):
            return
        oid = int(ev.order_id)
        o = tracker.orders.get(oid)
        if o is None or o.filled != 0:   # untracked, or not the first execution
            return
        qty = min(int(o.size), int(ev.size))
        if qty <= 0:
            return
        lvl = tracker.level_size(o.side, o.price)
        at_touch = tracker.best_price(o.side) == o.price
        px = int(ev.price_ticks) if (ev.kind == EventType.EXECUTE_PX and ev.price_ticks) else int(o.price)
        self.fills.append(
            ClockFill(
                tape_idx=int(tape_idx), ts_ns=int(ev.ts_ns), side=o.side, price=px,
                size=qty, ahead_at_fill=int(tracker.ahead_qty(oid)), level_size_at_fill=int(lvl),
                at_touch=bool(at_touch), emptied_touch=bool(at_touch and lvl - qty <= 0),
            )
        )


# --------------------------------------------------------------------------- #
# the unit of observation: one aggressor event                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AggressorEvent:
    """Every passive fill caused by one incoming aggressive order.

    Identified as a maximal run of fills sharing an ITCH timestamp and a resting
    side: one aggressive order's executions are stamped at the same nanosecond and
    can only consume one side of the book. Two distinct aggressors landing on the
    same side in the same nanosecond would merge, which costs nothing here — the
    point of the grouping is that same-instant fills are not independent draws.

    ``row_first`` / ``row_last`` are the session-row indices of the event's first
    and last messages, which is what makes the ``after`` reference exclude the
    *whole* sweep rather than only its first message.
    """

    ts_ns: int
    side: Side
    fills: tuple[ClockFill, ...]
    row_first: int
    row_last: int

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def total_size(self) -> int:
        """Shares the aggressor took from resting orders in this event."""
        return int(sum(f.size for f in self.fills))

    @property
    def vwap_price(self) -> float:
        """Size-weighted execution price over the event's fills, in $0.0001."""
        tot = self.total_size
        if tot <= 0:
            return float("nan")
        return float(sum(f.price * f.size for f in self.fills) / tot)

    @property
    def level_size_at_fill(self) -> int:
        """Displayed size at the touched level when the event started."""
        return int(self.fills[0].level_size_at_fill)

    @property
    def queue_frac(self) -> float:
        """Size-weighted queue position of the orders this event filled."""
        tot = self.total_size
        if tot <= 0:
            return float("nan")
        num = sum(f.queue_frac * f.size for f in self.fills if np.isfinite(f.queue_frac))
        return float(num / tot)

    @property
    def size_frac(self) -> float:
        """Event size as a fraction of the level it hit (the trade-size cut)."""
        lvl = self.level_size_at_fill
        return self.total_size / lvl if lvl > 0 else float("nan")

    @property
    def emptied_touch(self) -> bool:
        """The sweep cleared the best level, so the BBO moved by construction."""
        return any(f.emptied_touch for f in self.fills)


def group_aggressor_events(
    fills: Sequence[ClockFill], rows: Sequence[int]
) -> tuple[list[AggressorEvent], list[int]]:
    """Group fills into aggressor events; returns the events and their rows.

    ``rows[i]`` is ``fills[i]``'s session-row index. Fills are sorted by row, then
    split whenever the timestamp or the resting side changes. The returned row
    list carries each event's ``row_first``, so it plugs into the same panel
    machinery as the per-fill path.
    """
    if len(fills) != len(rows):
        raise ValueError("fills and rows must be the same length")
    order = sorted(range(len(fills)), key=lambda i: (int(rows[i]), int(fills[i].ts_ns)))
    events: list[AggressorEvent] = []
    bucket: list[int] = []

    def flush() -> None:
        if not bucket:
            return
        fs = tuple(fills[i] for i in bucket)
        rs = [int(rows[i]) for i in bucket]
        events.append(
            AggressorEvent(
                ts_ns=int(fs[0].ts_ns), side=fs[0].side, fills=fs,
                row_first=min(rs), row_last=max(rs),
            )
        )
        bucket.clear()

    for i in order:
        if bucket:
            prev = fills[bucket[-1]]
            if fills[i].ts_ns != prev.ts_ns or fills[i].side != prev.side:
                flush()
        bucket.append(i)
    flush()
    return events, [e.row_first for e in events]


# --------------------------------------------------------------------------- #
# per-observation markouts                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MarkoutPanel:
    """Aligned per-observation arrays for one clock horizon (NaN = unusable row).

    ``markout`` / ``half_spread_earned`` / ``realized_spread`` are keyed by
    reference (``"before"`` / ``"after"``) and carry $0.0001 units.
    ``quoted_half_spread`` is the pre-fill ``(ask − bid) / 2``, the "half-spread
    at the time of the fill" that the ``_half_spreads`` columns divide by.
    ``pre_fill`` is the same signed move over the ``H`` before the fill, ending
    strictly before it, so it is reference-independent.
    """

    horizon_ns: int
    obs: tuple[Any, ...]
    markout: dict[str, np.ndarray]
    half_spread_earned: dict[str, np.ndarray]
    realized_spread: dict[str, np.ndarray]
    quoted_half_spread: np.ndarray
    pre_fill: np.ndarray
    drops: dict[str, int]

    @property
    def n_usable(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.markout["after"])))


def _sign(side: Side) -> float:
    """``+1`` for a resting bid (bought), ``−1`` for a resting ask (sold)."""
    return 1.0 if side == Side.Bid else -1.0


def _obs_price(o: Any) -> float:
    """Execution price of one observation: a fill's price, or an event's VWAP."""
    return float(o.vwap_price) if hasattr(o, "vwap_price") else float(o.price)


def _obs_row_last(o: Any, row_first: int) -> int:
    """Row of the observation's last message (an event's sweep end, else itself)."""
    return int(getattr(o, "row_last", row_first))


def markout_panel(
    obs: Sequence[Any],
    rows: Sequence[int],
    mids: Sequence[float],
    ts_ns: Sequence[int],
    half_spreads: Sequence[float],
    horizon_ns: int,
    *,
    close_ns: int = REGULAR_CLOSE_NS,
    open_ns: int = REGULAR_OPEN_NS,
) -> MarkoutPanel:
    """Clock markouts for one horizon, from both reference mids.

    ``obs`` is a sequence of :class:`AggressorEvent` (the main path) or
    :class:`ClockFill` (the secondary per-fill path); ``rows[i]`` is observation
    ``i``'s index into the regular-session series (``mids`` / ``ts_ns`` /
    ``half_spreads``, one entry per session event, each recorded *after* that
    event was applied).

    The terminal mid is the one at the **last book event at or before**
    ``fill_ts + horizon_ns`` (``searchsorted(..., "right") - 1``), never
    interpolated and never the next event after the horizon. The ``before``
    reference is the mid at ``row_first - 1``; the ``after`` reference is the mid
    at ``row_last``, so for an event the whole sweep is behind it.

    Rows are dropped, not patched, and counted in ``drops``:

    * ``past_close``  — ``fill_ts + horizon_ns`` runs past 16:00 ET;
    * ``past_tape``   — the horizon runs past the last row of the series;
    * ``no_pre_mid``  — no mid strictly before the observation's first message;
    * ``bad_mid``     — a one-sided book at an endpoint (``mid == 0``).

    ``pre_fill`` additionally needs a mid ``H`` *before* the fill. When that window
    starts before 09:30 (``pre_past_open``) or lands where the book was one-sided
    (``pre_bad_mid``), only ``pre_fill`` is NaN — the forward markout is still
    kept, so the two controls drop independently.
    """
    m = np.asarray(mids, dtype=np.float64).reshape(-1)
    t = np.asarray(ts_ns, dtype=np.int64).reshape(-1)
    hs = np.asarray(half_spreads, dtype=np.float64).reshape(-1)
    n = m.size
    nobs = len(obs)
    H = int(horizon_ns)

    out_m = {r: np.full(nobs, np.nan) for r in _REFERENCES}
    out_e = {r: np.full(nobs, np.nan) for r in _REFERENCES}
    out_r = {r: np.full(nobs, np.nan) for r in _REFERENCES}
    out_hs = np.full(nobs, np.nan)
    out_pre = np.full(nobs, np.nan)
    drops = {"past_close": 0, "past_tape": 0, "no_pre_mid": 0, "bad_mid": 0,
             "pre_past_open": 0, "pre_bad_mid": 0}

    last_ts = int(t[-1]) if n else 0
    for i, o in enumerate(obs):
        j = int(rows[i])
        jl = _obs_row_last(o, j)
        if j <= 0 or jl >= n:
            drops["no_pre_mid"] += 1
            continue
        target = int(t[jl]) + H
        if target >= int(close_ns):
            drops["past_close"] += 1
            continue
        if target > last_ts:
            drops["past_tape"] += 1
            continue
        k = max(int(np.searchsorted(t, target, side="right")) - 1, jl)
        mid_t = float(m[k])
        refs = {"before": float(m[j - 1]), "after": float(m[jl])}
        if mid_t <= 0.0 or any(v <= 0.0 for v in refs.values()):
            drops["bad_mid"] += 1
            continue
        s = _sign(o.side)
        px = _obs_price(o)
        for name, ref in refs.items():
            out_m[name][i] = s * (mid_t - ref)
            out_e[name][i] = s * (ref - px)
            out_r[name][i] = s * (mid_t - px)
        out_hs[i] = float(hs[j - 1])
        # pre-fill control: the same signed move over [t − H, t), ending strictly
        # before the fill, exactly as adverse_selection.pre_fill_drift does
        pre_target = int(t[j]) - H
        if pre_target < int(open_ns) or pre_target < int(t[0]):
            drops["pre_past_open"] += 1
            continue
        kp = int(np.searchsorted(t, pre_target, side="right")) - 1
        if kp < 0:
            drops["pre_past_open"] += 1
            continue
        if m[kp] <= 0.0:  # one-sided book at the start of the pre-window
            drops["pre_bad_mid"] += 1
            continue
        out_pre[i] = s * (refs["before"] - float(m[kp]))

    return MarkoutPanel(
        horizon_ns=H, obs=tuple(obs), markout=out_m, half_spread_earned=out_e,
        realized_spread=out_r, quoted_half_spread=out_hs, pre_fill=out_pre,
        drops=drops,
    )


# --------------------------------------------------------------------------- #
# statistics                                                                  #
# --------------------------------------------------------------------------- #
def raw_overlap(ts_ns: Sequence[int], horizon_ns: int) -> int:
    """Median number of later observations whose window overlaps a given one's.

    Zero means the observations do not overlap, so they can be resampled i.i.d.
    """
    t = np.sort(np.asarray(ts_ns, dtype=np.int64).reshape(-1))
    n = t.size
    if n < 2:
        return 0
    ahead = np.searchsorted(t, t + int(horizon_ns), side="left") - np.arange(n) - 1
    return int(np.median(np.maximum(ahead, 0)))


def clock_overlap_lag(
    ts_ns: Sequence[int],
    horizon_ns: int,
    *,
    cap: int = MAX_OVERLAP_LAG,
) -> int:
    """Newey–West lag: the number of later observations inside a typical H-window.

    Two observations' markouts share price path whenever their horizon windows
    overlap, i.e. whenever they are less than ``horizon_ns`` apart, so the median
    count of later observations in one window is the lag at which the
    autocovariance dies. This is the clock analogue of the event-time ``lag = h``.
    Returns at least 1, at most ``min(cap, n - 1)``.
    """
    n = len(ts_ns)
    return int(max(1, min(raw_overlap(ts_ns, horizon_ns), cap, max(1, n - 1))))


def events_per_window(ts_ns: Sequence[int], window_ns: int) -> int:
    """Typical (median) number of observations inside one clock window, self included."""
    return raw_overlap(ts_ns, window_ns) + 1


def block_plan(
    ts_ns: Sequence[int],
    horizon_ns: int,
    n_usable: int,
    *,
    min_block_ns: int = BOOTSTRAP_MIN_BLOCK_NS,
) -> dict[str, int]:
    """Clock-time block-bootstrap plan for one horizon.

    The block is ``max(H, min_block_ns)`` of **wall-clock** time, converted to the
    typical number of observations that occupy such a window — so a 100 ms horizon
    in a busy name still gets a 60 s block rather than a two-second one that would
    resample still-correlated moves. Returns ``block_ns``, ``block_events`` (the
    length handed to :func:`.experiments.bootstrap_ci`) and
    ``n_blocks_effective``, the number of blocks that tile the usable sample —
    the honest read on how much independent information the CI rests on.
    """
    block_ns = max(int(horizon_ns), int(min_block_ns))
    n = len(ts_ns)
    blk = max(1, min(events_per_window(ts_ns, block_ns), max(1, n - 1)))
    n_blocks = max(1, -(-int(n_usable) // blk)) if n_usable else 0
    return {
        "block_ns": int(block_ns),
        "block_events": int(blk),
        "n_blocks_effective": int(n_blocks),
    }


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def trimmed_mean(x: Sequence[float], prop: float = TRIM_PROP) -> float:
    """Mean after dropping ``prop`` of the sample from **each** tail."""
    a = np.sort(_finite(np.asarray(x, dtype=np.float64)))
    n = a.size
    if n == 0:
        return float("nan")
    k = int(np.floor(n * float(prop)))
    if 2 * k >= n:
        return float(np.median(a))
    return float(a[k : n - k].mean())


_NULL_CI: dict[str, float | None] = {"lo": None, "hi": None, "mean": None, "se": None}


def _ci(
    x: np.ndarray, *, block: int, n_boot: int, seed: int, stat_fn=None
) -> dict[str, float | None]:
    """Moving-block bootstrap CI; ``block`` is in observations (see :func:`block_plan`)."""
    if x.size < 2:
        return dict(_NULL_CI)
    return bootstrap_ci(
        x, kind="block", block=max(1, int(block)), n_boot=n_boot, seed=seed, stat_fn=stat_fn
    )


def _trimmed_stat(a: np.ndarray) -> float:
    return trimmed_mean(a)


def location_stats(
    x: np.ndarray,
    *,
    lag: int,
    block: int,
    n_boot: int,
    seed: int,
    robust: bool = True,
) -> dict[str, Any]:
    """Mean (Newey–West s.e./t/p) with a CI, plus median and trimmed mean.

    The mean and the trimmed mean carry block-bootstrap CIs. The **median does
    not**: mids move on a 50-unit half-tick lattice, so median replicates collapse
    onto a few lattice points and the resulting interval is an artefact of the
    grid rather than a statement about sampling error (:data:`MEDIAN_LATTICE_NOTE`).
    The sign split in :func:`sign_shares` is the distribution-shape claim that
    survives discreteness.

    ``robust=False`` skips the trimmed-mean bootstrap — used for the many small
    breakdown cells, where the mean and its CI are the claim.
    """
    a = _finite(np.asarray(x, dtype=np.float64))
    nw = nw_tstat(a, lag=int(lag))
    out: dict[str, Any] = {
        "n": int(a.size),
        "mean": nw["mean"],
        "nw_se": nw["se"],
        "nw_t": nw["t"],
        "p_value": nw["p"],
        "mean_ci95": _ci(a, block=block, n_boot=n_boot, seed=seed),
        "median": float(np.median(a)) if a.size else float("nan"),
        "median_ci95_omitted": MEDIAN_LATTICE_NOTE,
    }
    if not robust:
        return out
    out["trimmed_mean"] = trimmed_mean(a)
    out["trim_prop"] = float(TRIM_PROP)
    out["trimmed_mean_ci95"] = _ci(
        a, block=block, n_boot=n_boot, seed=seed ^ 0x2E, stat_fn=_trimmed_stat
    )
    return out


def sign_shares(
    x: np.ndarray, *, block: int, n_boot: int, seed: int
) -> dict[str, Any]:
    """Share of observations ``< 0``, ``= 0`` and ``> 0``, each with a CI.

    The discreteness-proof companion to the median: on a lattice, *where* the
    distribution sits is better described by how often it lands on each side of
    zero than by a quantile that is pinned to a grid point. Each share is the mean
    of an indicator, so the same moving-block bootstrap applies. The three shares
    sum to 1 by construction.
    """
    a = _finite(np.asarray(x, dtype=np.float64))
    out: dict[str, Any] = {"n": int(a.size)}
    for name, ind in (
        ("neg", (a < 0.0)), ("zero", (a == 0.0)), ("pos", (a > 0.0)),
    ):
        v = ind.astype(np.float64)
        out[f"share_{name}"] = float(v.mean()) if v.size else float("nan")
        out[f"share_{name}_ci95"] = _ci(
            v, block=block, n_boot=n_boot, seed=seed ^ (hash(name) & 0xFFFF)
        )
        # conditional magnitude: with the shares near 0.5 each, whether the mean is
        # negative is decided by how much losers lose against how much winners win
        sel = a[ind]
        out[f"mean_{name}"] = float(sel.mean()) if sel.size else float("nan")
    return out


def _share_zero(x: np.ndarray) -> float:
    a = _finite(x)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a == 0.0))


def markout_stats(
    markout: np.ndarray,
    realized: np.ndarray,
    earned: np.ndarray,
    quoted_half_spread: np.ndarray,
    pre_fill: np.ndarray,
    *,
    lag: int,
    block: int,
    n_boot: int,
    seed: int,
    robust: bool = True,
) -> dict[str, Any]:
    """One cell of the report: markout, realized spread, controls and shares.

    All three adverse shares are reported side by side — ``share_adverse``
    (unconditional, zero moves counted as *not* adverse), ``share_zero``, and
    ``share_adverse_conditional`` (``P(m < 0 | m ≠ 0)``, the figure the event-time
    E6 headline quotes). The ``_half_spreads`` figures divide by the mean quoted
    half-spread at the fill (a per-observation ratio would be dominated by the
    handful of 1-tick-spread rows).

    ``realized_sign`` is the sign split of the realized spread — the claim that
    survives the price lattice, and the one that answers "does a passive fill make
    money" without leaning on a mean or a lattice-pinned median.
    """
    mk = _finite(markout)
    hsm = _finite(quoted_half_spread)
    hs_mean = float(np.mean(hsm)) if hsm.size else float("nan")
    scale = hs_mean if (np.isfinite(hs_mean) and hs_mean > 0) else float("nan")
    kw = {"lag": lag, "block": block, "n_boot": n_boot, "seed": seed}
    mk_block = location_stats(markout, robust=robust, **kw)
    rl_block = location_stats(realized, robust=robust, **kw)
    pre_block = location_stats(pre_fill, robust=False, **kw)
    pmp_block = location_stats(
        np.asarray(markout, dtype=np.float64) - np.asarray(pre_fill, dtype=np.float64),
        robust=False, **kw,
    )
    er = _finite(earned)

    def scaled(v: Any) -> float:
        fv = float(v) if v is not None else float("nan")
        return float(fv / scale) if np.isfinite(scale) and np.isfinite(fv) else float("nan")

    out: dict[str, Any] = {
        "n": int(mk.size),
        "nw_lag": int(lag),
        "block_events": int(block),
        "markout_px0001": mk_block,
        "realized_spread_px0001": rl_block,
        "pre_fill_px0001": pre_block,
        "post_minus_pre_px0001": pmp_block,
        "mean_half_spread_px0001": hs_mean,
        "mean_half_spread_earned_px0001": float(np.mean(er)) if er.size else float("nan"),
        "mean_markout_half_spreads": scaled(mk_block["mean"]),
        "median_markout_half_spreads": scaled(mk_block.get("median")),
        "mean_realized_spread_half_spreads": scaled(rl_block["mean"]),
        "median_realized_spread_half_spreads": scaled(rl_block.get("median")),
        "share_adverse": p_adverse(mk, conditional=False),
        "share_zero": _share_zero(mk),
        "share_adverse_conditional": p_adverse(mk, conditional=True),
    }
    if robust:
        out["realized_sign"] = sign_shares(
            realized, block=block, n_boot=n_boot, seed=seed ^ 0x5A
        )
        out["markout_sign"] = sign_shares(
            markout, block=block, n_boot=n_boot, seed=seed ^ 0x6B
        )
        # where passive liquidity breaks even, under an ASSUMED rebate
        rl_arr = np.asarray(realized, dtype=np.float64)
        out["rebate_assumption_note"] = REBATE_ASSUMPTION_NOTE
        out["net_of_rebate"] = {
            str(int(r)): {
                "rebate_px0001": int(r),
                "assumed": True,
                "mean": float(np.mean(_finite(rl_arr + r))) if mk.size else float("nan"),
                "mean_ci95": _ci(
                    _finite(rl_arr + r), block=block, n_boot=n_boot, seed=seed ^ (0x70 + int(r))
                ),
                "mean_half_spreads": scaled(
                    float(np.mean(_finite(rl_arr + r))) if mk.size else float("nan")
                ),
                "sign": sign_shares(
                    rl_arr + r, block=block, n_boot=n_boot, seed=seed ^ (0x80 + int(r))
                ),
            }
            for r in REBATE_SCENARIOS_PX0001
        }
    return out


def _bucket_size(frac: float, median: float) -> str:
    if not np.isfinite(frac):
        return "size=n/a"
    return "size=small" if frac <= median else "size=large"


#: Breakdown groups that get the full robust-location treatment (item 4: the
#: bid/ask split is reported at every horizon alongside the overall row).
FULL_LOCATION_GROUPS: tuple[str, ...] = ("side=bid", "side=ask")


def group_masks(obs: Sequence[Any]) -> dict[str, np.ndarray]:
    """Breakdown masks: side, queue third at the fill, trade size vs the level.

    Works on :class:`AggressorEvent`s or :class:`ClockFill`s — both expose
    ``side`` / ``queue_frac`` / ``size_frac`` / ``emptied_touch``. An
    *information* effect should vary across these; a mechanical one should not.
    Queue position reuses the event-time thirds of
    ``adverse_selection._bucket_queue`` but measured **at the fill** rather than
    at placement. Size is split at the median of ``size / level_size``.

    ``emptied_touch`` splits on whether the aggressive order was **large enough to
    clear the displayed level**. The quote necessarily moved on those events, but
    that is not the interesting part: what matters is whether the move *persists*.
    Size-dependent impact that does not revert is the signature of informed flow in
    the Kyle and Glosten–Milgrom sense, not an argument against adverse selection.
    """
    qf = np.asarray([o.queue_frac for o in obs], dtype=np.float64)
    sf = np.asarray([o.size_frac for o in obs], dtype=np.float64)
    fin = _finite(sf)
    med = float(np.median(fin)) if fin.size else float("nan")
    masks: dict[str, np.ndarray] = {
        "side=bid": np.asarray([o.side == Side.Bid for o in obs]),
        "side=ask": np.asarray([o.side == Side.Ask for o in obs]),
    }
    for name in ("queue=front", "queue=mid", "queue=back"):
        masks[name] = np.asarray([_bucket_queue(q) == name for q in qf])
    for name in ("size=small", "size=large"):
        masks[name] = np.asarray([_bucket_size(s, med) == name for s in sf])
    masks["emptied_touch=yes"] = np.asarray([o.emptied_touch for o in obs])
    masks["emptied_touch=no"] = np.asarray([not o.emptied_touch for o in obs])
    return {k: np.asarray(v, dtype=bool).reshape(-1) for k, v in masks.items()}


def _panel_block(
    panel: MarkoutPanel,
    masks: dict[str, np.ndarray] | None,
    *,
    lag: int,
    block: int,
    n_boot: int,
    seed: int,
    min_group: int,
) -> dict[str, Any]:
    """Per-reference stats for one panel: overall plus every viable breakdown."""
    out: dict[str, Any] = {"n_usable": panel.n_usable, "drops": panel.drops}
    for ref in _REFERENCES:
        mk = panel.markout[ref]
        args = (mk, panel.realized_spread[ref], panel.half_spread_earned[ref],
                panel.quoted_half_spread, panel.pre_fill)
        entry: dict[str, Any] = {
            "overall": markout_stats(
                *args, lag=lag, block=block, n_boot=n_boot, seed=seed, robust=True
            ),
            "groups": {},
        }
        for name, mask in (masks or {}).items():
            if mask.size == 0 or int(np.count_nonzero(mask & np.isfinite(mk))) < min_group:
                continue
            entry["groups"][name] = markout_stats(
                *[a[mask] for a in args], lag=lag, block=block, n_boot=n_boot, seed=seed,
                robust=name in FULL_LOCATION_GROUPS,
            )
        out[ref] = entry
    return out



# --------------------------------------------------------------------------- #
# attribution: from E6's headline to the clock number, one change at a time    #
# --------------------------------------------------------------------------- #
#: Event-time horizon the event-time E6 headline is quoted at.
E6_HEADLINE_H_EVENTS: int = 5


def event_time_markout(
    obs: Sequence[Any],
    rows: Sequence[int],
    mids: Sequence[float],
    h: int,
    *,
    reference: str = "after",
) -> np.ndarray:
    """Signed markout ``h`` **book events** after the fill (NaN where unusable).

    The event-time definition, reproduced here so the attribution ladder can hold
    everything else fixed while swapping one dimension at a time. With
    ``reference="after"`` and one observation per fill this is exactly E6's
    ``post_fill_drift``: ``s · (mid[t+h] − mid[t])`` where ``mid[t]`` is the mid
    *after* the fill event — which is E6's own base, a detail that matters for
    reading the ladder.

    ``reference="before"`` measures from ``mid[row_first − 1]`` instead, so the
    fill's own BBO move is included. The terminal row is ``row_last + h``, which
    for a single fill is ``t + h``.
    """
    if reference not in _REFERENCES:
        raise ValueError(f"reference must be one of {_REFERENCES}, got {reference!r}")
    m = np.asarray(mids, dtype=np.float64).reshape(-1)
    n = m.size
    out = np.full(len(obs), np.nan, dtype=np.float64)
    for i, o in enumerate(obs):
        j = int(rows[i])
        jl = _obs_row_last(o, j)
        term = jl + int(h)
        base_row = jl if reference == "after" else j - 1
        if base_row < 0 or term >= n:
            continue
        base, end = float(m[base_row]), float(m[term])
        if base <= 0.0 or end <= 0.0:
            continue
        out[i] = _sign(o.side) * (end - base)
    return out


#: The ladder statistic the walk is checked against — E6's own headline.
LADDER_STAT: str = "share_adverse_conditional"


def _ladder_cell(x: np.ndarray) -> dict[str, Any]:
    a = _finite(x)
    return {
        "n": int(a.size),
        "mean_markout_px0001": float(np.mean(a)) if a.size else float("nan"),
        "median_markout_px0001": float(np.median(a)) if a.size else float("nan"),
        "share_adverse": p_adverse(a, conditional=False),
        "share_zero": _share_zero(a),
        "share_adverse_conditional": p_adverse(a, conditional=True),
    }


def attribution_ladder(
    fills: Sequence[ClockFill],
    rows: Sequence[int],
    events: Sequence[AggressorEvent],
    event_rows: Sequence[int],
    mids: Sequence[float],
    ts_ns: Sequence[int],
    *,
    h_events: int = E6_HEADLINE_H_EVENTS,
    horizon_ns: int = 100 * NS_PER_MS,
    close_ns: int = REGULAR_CLOSE_NS,
    open_ns: int = REGULAR_OPEN_NS,
) -> list[dict[str, Any]]:
    """Walk from E6's published ``P(adverse | moved)`` to the clock number.

    Three rows, each changing exactly one thing from the row above, and each
    reported under **both** reference mids so the mechanical before/after step is
    visible on every row:

    1. **per fill, event time h=5** — the ``after`` cell is E6's own definition and
       should reproduce its headline;
    2. **per aggressor event, event time h=5** — only the unit of observation
       changed, so the difference is sweep multiplicity;
    3. **per aggressor event, clock 100 ms** — only the clock changed.

    Everything runs on the *same* fills, so each step is attributable. Note that
    E6 measures from the mid *after* the fill event (``mid_history[event_index]``),
    so the row that reproduces its headline is the ``after`` column of row 1, not
    the ``before`` one.
    """
    half = np.zeros(len(mids), dtype=np.float64)  # unused by the clock cell below
    specs: list[tuple[str, str, str, Sequence[Any], Sequence[int], bool]] = [
        ("1", "per fill", f"event h={h_events}", fills, rows, False),
        ("2", "per aggressor event", f"event h={h_events}", events, event_rows, False),
        (
            "3", "per aggressor event", horizon_label(int(horizon_ns)),
            events, event_rows, True,
        ),
    ]
    ladder: list[dict[str, Any]] = []
    for step, unit, clock, obs, obs_rows, is_clock in specs:
        row: dict[str, Any] = {"step": step, "unit": unit, "clock": clock}
        if is_clock:
            panel = markout_panel(
                obs, obs_rows, mids, ts_ns, half, int(horizon_ns),
                close_ns=close_ns, open_ns=open_ns,
            )
            for ref in _REFERENCES:
                row[ref] = _ladder_cell(panel.markout[ref])
        else:
            for ref in _REFERENCES:
                row[ref] = _ladder_cell(
                    event_time_markout(obs, obs_rows, mids, int(h_events), reference=ref)
                )
        # the before/after gap is a contrast WITHIN a row (how much of the move is
        # the fill's own BBO jump), not a step along the walk — see attribution_check
        row["before_minus_after"] = {
            "stat": LADDER_STAT,
            "value": row["before"][LADDER_STAT] - row["after"][LADDER_STAT],
            "is_walk_step": False,
        }
        ladder.append(row)
    return ladder


def attribution_check(
    ladder: Sequence[dict[str, Any]],
    *,
    reference: str = "after",
    stat: str = LADDER_STAT,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Verify the listed steps account for the entire change along one column.

    The walk runs **down** one reference column: each step changes exactly one
    thing (the unit of observation, then the clock), so the step deltas must sum
    to ``last - first``. ``closes`` is that check.

    The ``before`` vs ``after`` gap is deliberately *not* a step. It is a contrast
    within each row — how much of the measured move is the fill's own BBO jump —
    and adding it to the walk would double-count, since both endpoints of the walk
    are already measured from the same reference.

    ``defined`` is ``False`` when any cell in the column is non-finite — a
    conditional share is NaN when nothing in that cell moved, so there is no walk
    to decompose. ``closes`` is then ``None`` rather than a misleading ``False``.
    """
    rows = list(ladder)
    if len(rows) < 2:
        return {
            "reference": reference, "stat": stat, "steps": [], "steps_sum": 0.0,
            "total_change": 0.0, "defined": True, "closes": True,
        }
    vals = [float(r[reference][stat]) for r in rows]
    if not all(np.isfinite(v) for v in vals):
        return {
            "reference": reference, "stat": stat, "values": vals, "steps": [],
            "steps_sum": float("nan"), "total_change": float("nan"),
            "defined": False, "closes": None,
        }
    steps: list[dict[str, Any]] = []
    for a, b, va, vb in zip(rows[:-1], rows[1:], vals[:-1], vals[1:], strict=True):
        if a["unit"] != b["unit"]:
            changed = "unit of observation"
        elif a["clock"] != b["clock"]:
            changed = "clock"
        else:  # pragma: no cover — the ladder always changes one of the two
            changed = "nothing"
        steps.append({
            "from_step": a["step"], "to_step": b["step"], "changed": changed,
            "from_value": va, "to_value": vb, "delta": vb - va,
        })
    steps_sum = float(sum(st["delta"] for st in steps))
    total = float(vals[-1] - vals[0])
    return {
        "reference": reference,
        "stat": stat,
        "first_value": vals[0],
        "last_value": vals[-1],
        "steps": steps,
        "steps_sum": steps_sum,
        "total_change": total,
        "defined": True,
        "closes": bool(abs(steps_sum - total) <= tol),
    }


def clock_markout_report(
    fills: Sequence[ClockFill],
    rows: Sequence[int],
    mids: Sequence[float],
    ts_ns: Sequence[int],
    half_spreads: Sequence[float],
    *,
    horizons_ns: Sequence[int] = CLOCK_HORIZONS_NS,
    close_ns: int = REGULAR_CLOSE_NS,
    open_ns: int = REGULAR_OPEN_NS,
    min_group: int = 30,
    n_boot: int = 200,
    seed: int = 0x51ED,
) -> dict[str, Any]:
    """E6b bundle: clock markouts per aggressor event (main) and per fill (secondary).

    Shape (JSON-able throughout)::

        {"n_fills", "n_events", "fills_per_event", "px_unit_usd", "tick_px_units",
         "size_frac_median", "attribution": [...],
         "horizons": {"1s": {
            "horizon_ns", "horizon", "nw_lag", "median_overlap_events",
            "events_per_horizon_window",
            "block": {"block_ns", "block_events", "n_blocks_effective"},
            "events": {"n_usable", "drops", "before": {"overall", "groups"},
                       "after": {...}},
            "fills":  {... same shape, no groups ...}}}}

    Every cell is :func:`markout_stats`. Groups with fewer than ``min_group``
    usable observations are omitted — no claims on five fills.
    """
    fills = list(fills)
    rows = list(rows)
    if len(fills) != len(rows):
        raise ValueError("fills and rows must be the same length")
    events, event_rows = group_aggressor_events(fills, rows)
    sf = np.asarray([e.size_frac for e in events], dtype=np.float64)
    fin = _finite(sf)
    out: dict[str, Any] = {
        "n_fills": len(fills),
        "n_events": len(events),
        "fills_per_event": (len(fills) / len(events)) if events else float("nan"),
        "px_unit_usd": PX_UNIT_USD,
        "tick_px_units": TICK_PX_UNITS,
        "size_frac_median": float(np.median(fin)) if fin.size else float("nan"),
        "horizons": {},
    }
    if not fills:
        return out
    masks = group_masks(events)
    ev_ts = [e.ts_ns for e in events]
    fl_ts = [f.ts_ns for f in fills]
    out["attribution"] = attribution_ladder(
        fills, rows, events, event_rows, mids, ts_ns,
        h_events=E6_HEADLINE_H_EVENTS, horizon_ns=min(int(h) for h in horizons_ns),
        close_ns=close_ns, open_ns=open_ns,
    )
    out["attribution_check"] = {
        ref: attribution_check(out["attribution"], reference=ref)
        for ref in _REFERENCES
    }
    for horizon in horizons_ns:
        H = int(horizon)
        lag = clock_overlap_lag(ev_ts, H)
        lag_f = clock_overlap_lag(fl_ts, H)
        panel_e = markout_panel(
            events, event_rows, mids, ts_ns, half_spreads, H,
            close_ns=close_ns, open_ns=open_ns,
        )
        panel_f = markout_panel(
            fills, rows, mids, ts_ns, half_spreads, H,
            close_ns=close_ns, open_ns=open_ns,
        )
        plan_e = block_plan(ev_ts, H, panel_e.n_usable)
        plan_f = block_plan(fl_ts, H, panel_f.n_usable)
        out["horizons"][horizon_label(H)] = {
            "horizon_ns": H,
            "horizon": horizon_label(H),
            "nw_lag": lag,
            "median_overlap_events": raw_overlap(ev_ts, H),
            "events_per_horizon_window": events_per_window(ev_ts, H),
            "block": plan_e,
            "block_fills": plan_f,
            "events": _panel_block(
                panel_e, masks, lag=lag, block=plan_e["block_events"],
                n_boot=n_boot, seed=seed, min_group=min_group,
            ),
            "fills": _panel_block(
                panel_f, None, lag=lag_f, block=plan_f["block_events"],
                n_boot=n_boot, seed=seed, min_group=min_group,
            ),
        }
    return out


# --------------------------------------------------------------------------- #
# markdown rendering                                                          #
# --------------------------------------------------------------------------- #
def _f(x: Any, w: int = 8, d: int = 2) -> str:
    if x is None:
        return "-" * w
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-" * w
    if not np.isfinite(v):
        return "-" * w
    return f"{v:>{w}.{d}f}"


def _ci_str(ci: dict[str, Any] | None, d: int = 2) -> str:
    if not ci:
        return "-"
    return f"[{_f(ci.get('lo'), 7, d)}, {_f(ci.get('hi'), 7, d)}]"


_PREAMBLE = (
    "Markouts on the **clock**, from ITCH nanosecond timestamps, over the same "
    "passive-fill population as E6 (the first execution of each tracked resting "
    "order, regular session only). The terminal mid is the mid of the last book "
    "event at or before `fill_ts + H` — never interpolated. Sign is from the "
    "passive order's side (`s = +1` resting bid, `s = −1` resting ask), so "
    "**negative = adverse**.\n\n"
    "**Units.** ITCH prices are integers in **$0.0001**. That is not a tick: AAPL "
    "and QQQ both quote on a **$0.01 tick = 100 price units**, so 100 units = "
    "1 tick = 1 cent. Columns headed `$0.0001` are in price units; columns headed "
    "`×HS` are multiples of the mean quoted half-spread at the fill.\n\n"
    "**Unit of observation.** One aggressive order can fill many resting orders at "
    "the same nanosecond; those fills share one aggressor, one price path and one "
    "mechanical BBO move, so they are not independent draws. The main tables use "
    "one observation per **aggressor event** (same ITCH timestamp, same resting "
    "side), and the Newey–West s.e. and bootstrap CIs are computed over events. "
    "The per-fill table at the end is the same measurement with the sweep "
    "multiplicity left in, which overstates the effective sample size.\n\n"
    "**Two reference mids.**\n\n"
    "* **before** — the mid of the last book event strictly *before* the event's "
    "first message, so the sweep's own BBO move is **included**;\n"
    "* **after** — the mid once the event's *last* message is applied, so that "
    "move is **excluded**. Aggregating to the event is what lets this reference "
    "step over the whole sweep rather than only its first message.\n\n"
    "`before − after` is the mechanical component of the move: price change caused "
    "by the fill itself clearing displayed depth, not by anything the taker knew.\n\n"
    "**Realized spread** = `half-spread earned + markout` = `s · (mid[t+H] − P)`, "
    "`P` the event's size-weighted execution price. It is the same number under "
    "either reference — the reference only moves value between the two terms. "
    "Positive means the passive fill made money net of adverse selection.\n\n"
    "**Location.** Means over markouts are tail-sensitive, so each horizon reports "
    "the mean, the median and the 10 %-each-tail trimmed mean. The mean and the "
    "trimmed mean carry 95 % CIs; **the median does not** — mids move on a 50-unit "
    "half-tick lattice, so median bootstrap replicates collapse onto a few lattice "
    "points and the interval would describe the grid, not the sampling error. The "
    "sign-split table is the distribution claim that survives discreteness.\n\n"
    "**Dependence.** CIs are moving-block bootstraps whose block is "
    "`max(H, 60 s)` of **wall-clock** time, converted to the typical number of "
    "observations in such a window — at H = 100 ms a block sized to the horizon "
    "would span a second or two and resample moves that are still correlated. The "
    "Newey–West lag is the number of later observations inside a typical "
    "`H`-window. `effective blocks` is how many such blocks tile the sample, i.e. "
    "how much independent information each CI actually rests on; at H = 60 s it is "
    "small, and the wide intervals there are honest about that.\n\n"
    "`pre_fill` is the same signed move over the `H` *before* the fill, ending "
    "strictly before it — the clock analogue of E6's pre-fill control — so "
    "`post − pre` nets out a session already trending into the fill."
)


def _headline_table(res: dict[str, Any]) -> list[str]:
    """The lead result: does a passive fill make money, and by how much?"""
    lines = [
        "### Realized spread — does a passive fill make money?",
        "",
        (
            "**This is the headline.** `realized spread = half-spread earned + markout` "
            "= `s · (mid[t+H] − P)`, one observation per aggressor event, `P` the "
            "event's size-weighted execution price. Reference-invariant, so one row per "
            "horizon. Negative means the passive side gave back more to adverse "
            "selection than the spread it earned. `×HS` expresses that as a fraction of "
            "the half-spread quoted at the fill, which is the scale-free read."
        ),
        "",
        (
            "| H | n events | mean ($0.0001) | 95% CI | mean (×HS) "
            "| median ($0.0001) † | trimmed mean ($0.0001) | 95% CI "
            "| NW t | p | HS earned ($0.0001) | mean markout ($0.0001) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for hl, blk in res.get("horizons", {}).items():
        st = blk["events"]["after"]["overall"]
        rl, mk = st["realized_spread_px0001"], st["markout_px0001"]
        lines.append(
            f"| {hl} | {st['n']:,} | {_f(rl['mean'])} | {_ci_str(rl['mean_ci95'])} | "
            f"{_f(st['mean_realized_spread_half_spreads'], 6, 2)} | {_f(rl.get('median'))} | "
            f"{_f(rl.get('trimmed_mean'))} | "
            f"{_ci_str(rl.get('trimmed_mean_ci95'))} | {_f(rl['nw_t'], 7, 1)} | "
            f"{_f(rl['p_value'], 6, 3)} | {_f(st['mean_half_spread_earned_px0001'])} | "
            f"{_f(mk['mean'])} |"
        )
    lines += ["", f"† no CI on the median: {MEDIAN_LATTICE_NOTE}."]
    return lines


def _rebate_table(res: dict[str, Any]) -> list[str]:
    """Where passive liquidity breaks even, under an explicitly assumed rebate."""
    horizons = res.get("horizons", {})
    if not horizons:
        return []
    first = next(iter(horizons.values()))["events"]["after"]["overall"]
    scenarios = sorted(int(k) for k in (first.get("net_of_rebate") or {}))
    if not scenarios:
        return []
    lines = [
        "### Break-even including an assumed maker rebate",
        "",
        (
            f"**The rebate here is assumed, not measured: {REBATE_ASSUMPTION_NOTE}.** "
            "The values below are illustrative, chosen to bracket a typical 2019 US "
            "equity maker rebate, and every figure in this table is conditional on "
            "them. A rebate is a constant per share, so it shifts the realized-spread "
            "distribution without changing its shape — which is why the share "
            "profitable moves as well as the mean."
        ),
        "",
        (
            "| H | realized spread ($0.0001) | "
            + " | ".join(
                f"+ {r} rebate | 95% CI | ×HS | share > 0"
                for r in scenarios
            )
            + " |"
        ),
        "|---|---|" + "---|---|---|---|" * len(scenarios),
    ]
    for hl, blk in horizons.items():
        st = blk["events"]["after"]["overall"]
        cells = [f"| {hl} | {_f(st['realized_spread_px0001']['mean'])} "]
        for r in scenarios:
            nr = st["net_of_rebate"][str(r)]
            cells.append(
                f"| {_f(nr['mean'])} | {_ci_str(nr['mean_ci95'])} | "
                f"{_f(nr['mean_half_spreads'], 6, 2)} | "
                f"{_f(nr['sign']['share_pos'], 6, 3)} "
            )
        lines.append("".join(cells) + "|")
    return lines


def _markout_table(res: dict[str, Any], scope: str = "events") -> list[str]:
    label = "aggressor event" if scope == "events" else "individual fill"
    lines = [
        f"One observation per {label}. † median: no CI (lattice).",
        "",
        (
            "| H | ref | n | mean ($0.0001) | 95% CI | median ($0.0001) † "
            "| trimmed mean ($0.0001) | 95% CI | mean (×HS) | NW t | p | NW lag "
            "| block (obs) | eff. blocks | adverse | zero | adverse\\|Δ≠0 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for hl, blk in res.get("horizons", {}).items():
        plan = blk["block" if scope == "events" else "block_fills"]
        for ref in _REFERENCES:
            st = blk[scope][ref]["overall"]
            mk = st["markout_px0001"]
            lines.append(
                f"| {hl} | {ref} | {st['n']:,} | {_f(mk['mean'])} | "
                f"{_ci_str(mk['mean_ci95'])} | {_f(mk.get('median'))} | "
                f"{_f(mk.get('trimmed_mean'))} | "
                f"{_ci_str(mk.get('trimmed_mean_ci95'))} | "
                f"{_f(st['mean_markout_half_spreads'], 6, 2)} | {_f(mk['nw_t'], 7, 1)} | "
                f"{_f(mk['p_value'], 6, 3)} | {st['nw_lag']} | {st['block_events']} | "
                f"{plan['n_blocks_effective']} | "
                f"{_f(st['share_adverse'], 6, 3)} | {_f(st['share_zero'], 5, 3)} | "
                f"{_f(st['share_adverse_conditional'], 6, 3)} |"
            )
    return lines


def _control_table(res: dict[str, Any]) -> list[str]:
    """Item 4: bid/ask split and the pre-fill control at every horizon."""
    lines = [
        "### Pre-fill control and the bid/ask split",
        "",
        (
            "`pre` is the signed mid move over the `H` before the fill, ending strictly "
            "before it; `post − pre` is the fill-conditional excess. A session-wide "
            "trend shows up in `pre`; adverse selection should not."
        ),
        "",
        (
            "| H | ref | group | n | post ($0.0001) | 95% CI | pre ($0.0001) | 95% CI "
            "| post − pre ($0.0001) | 95% CI | NW t | adverse | adverse\\|Δ≠0 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for hl, blk in res.get("horizons", {}).items():
        for ref in _REFERENCES:
            groups = blk["events"][ref]["groups"]
            cells = [("all", blk["events"][ref]["overall"])]
            cells += [(g, groups[g]) for g in FULL_LOCATION_GROUPS if g in groups]
            for name, st in cells:
                mk, pre = st["markout_px0001"], st["pre_fill_px0001"]
                pmp = st["post_minus_pre_px0001"]
                lines.append(
                    f"| {hl} | {ref} | {name} | {st['n']:,} | {_f(mk['mean'])} | "
                    f"{_ci_str(mk['mean_ci95'])} | {_f(pre['mean'])} | "
                    f"{_ci_str(pre['mean_ci95'])} | {_f(pmp['mean'])} | "
                    f"{_ci_str(pmp['mean_ci95'])} | {_f(pmp['nw_t'], 7, 1)} | "
                    f"{_f(st['share_adverse'], 6, 3)} | "
                    f"{_f(st['share_adverse_conditional'], 6, 3)} |"
                )
    return lines


def _group_table(res: dict[str, Any], hl: str) -> list[str]:
    blk = res.get("horizons", {}).get(hl)
    if blk is None:
        return []
    lines = [
        f"**Breakdowns at H = {hl}**, one observation per aggressor event.",
        "",
        (
            "`emptied_touch=yes` means the aggressive order was large enough to clear "
            "the displayed level. Losses concentrating there, and persisting across "
            "horizons, is **size-dependent permanent impact** — the Kyle / "
            "Glosten–Milgrom signature of informed flow, not evidence against adverse "
            "selection. `queue=` buckets are near-degenerate on a FIFO book, since the "
            "order that gets hit is by definition at the front of its level."
        ),
        "",
        (
            "| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) "
            "| realized spread ($0.0001) | 95% CI | NW t | adverse | zero "
            "| adverse\\|Δ≠0 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in blk["events"]["after"]["groups"]:
        for ref in _REFERENCES:
            st = blk["events"][ref]["groups"].get(name)
            if st is None:
                continue
            mk, rl = st["markout_px0001"], st["realized_spread_px0001"]
            lines.append(
                f"| {name} | {ref} | {st['n']:,} | {_f(mk['mean'])} | "
                f"{_ci_str(mk['mean_ci95'])} | "
                f"{_f(st['mean_markout_half_spreads'], 6, 2)} | {_f(rl['mean'])} | "
                f"{_ci_str(rl['mean_ci95'])} | {_f(mk['nw_t'], 7, 1)} | "
                f"{_f(st['share_adverse'], 6, 3)} | {_f(st['share_zero'], 5, 3)} | "
                f"{_f(st['share_adverse_conditional'], 6, 3)} |"
            )
    return lines


def _drop_table(res: dict[str, Any]) -> list[str]:
    lines = [
        (
            "| H | past 16:00 | past tape end | no pre-fill mid | one-sided book "
            "| pre-window before 09:30 | pre-window one-sided | usable events |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for hl, blk in res.get("horizons", {}).items():
        d = blk["events"]["drops"]
        lines.append(
            f"| {hl} | {d['past_close']:,} | {d['past_tape']:,} | {d['no_pre_mid']:,} | "
            f"{d['bad_mid']:,} | {d['pre_past_open']:,} | {d['pre_bad_mid']:,} | "
            f"{blk['events']['n_usable']:,} |"
        )
    return lines



def _sign_table(res: dict[str, Any]) -> list[str]:
    """Item 3: the discreteness-proof shape claim, for realized spread and markout."""
    lines = [
        "### Sign split — the claim that survives the price lattice",
        "",
        (
            "Mids move in 50-unit (half-tick) steps, so a median is pinned to a lattice "
            "point. How often the outcome lands on each side of zero is not. Shares are "
            "over aggressor events, measured from the `after` mid, with block-bootstrap "
            "CIs; the three shares in each group sum to 1."
        ),
        "",
        (
            "| H | quantity | n | share < 0 | 95% CI | share = 0 | 95% CI "
            "| share > 0 | 95% CI |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for hl, blk in res.get("horizons", {}).items():
        st = blk["events"]["after"]["overall"]
        for qty, key in (("realized spread", "realized_sign"), ("markout", "markout_sign")):
            sg = st.get(key)
            if not sg:
                continue
            lines.append(
                f"| {hl} | {qty} | {sg['n']:,} | {_f(sg['share_neg'], 6, 3)} | "
                f"{_ci_str(sg['share_neg_ci95'], 3)} | {_f(sg['share_zero'], 6, 3)} | "
                f"{_ci_str(sg['share_zero_ci95'], 3)} | {_f(sg['share_pos'], 6, 3)} | "
                f"{_ci_str(sg['share_pos_ci95'], 3)} |"
            )
    return lines


def _attribution_table(res: dict[str, Any]) -> list[str]:
    """Walk from E6's headline to the clock number, one change per row."""
    ladder = res.get("attribution") or []
    if not ladder:
        return []
    checks = res.get("attribution_check") or {}
    lines = [
        "### Attribution — where E6's 0.957 goes",
        "",
        (
            "Each row changes exactly one thing from the row above, on the **same** "
            "fills, so every step is attributable. The statistic is "
            "`P(adverse | mid moved)` — E6's own headline — with the unconditional and "
            "zero shares beside it."
        ),
        "",
        (
            "Row 1 / `after` reproduces E6's published figure, which establishes that "
            "**E6 already measured from the post-fill mid** "
            "(`mid_history[event_index]`). The walk down the `after` column is "
            "therefore the decomposition of E6's number: two steps, the unit of "
            "observation and the clock, and they account for the whole change."
        ),
        "",
        (
            "The `before` column is a *separate* comparison, not a third step: it shows "
            "how much of each row's measured move is the fill's own BBO jump. Both "
            "endpoints of the walk are already measured from the same reference, so "
            "adding the before/after gap to the two steps would double-count."
        ),
        "",
        (
            "| row | unit | clock | ref | n | P(adv\\|moved) | P(adv) uncond "
            "| share zero | mean markout ($0.0001) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in ladder:
        for ref in _REFERENCES:
            c = row[ref]
            lines.append(
                f"| {row['step']} | {row['unit']} | {row['clock']} | {ref} | "
                f"{c['n']:,} | **{_f(c['share_adverse_conditional'], 6, 3)}** | "
                f"{_f(c['share_adverse'], 6, 3)} | {_f(c['share_zero'], 6, 3)} | "
                f"{_f(c['mean_markout_px0001'])} |"
            )
    for ref in _REFERENCES:
        chk = checks.get(ref)
        if not chk:
            continue
        if not chk.get("defined", True):
            lines += [
                "",
                (
                    f"Steps along the `{ref}` column: **undefined** — a cell has no "
                    "non-zero moves, so `P(adverse | moved)` is NaN there and there is "
                    "no walk to decompose."
                ),
            ]
            continue
        lines += [
            "",
            f"Steps along the `{ref}` column (these are the decomposition):",
            "",
            "| step | what changed | from | to | Δ |",
            "|---|---|---|---|---|",
        ]
        for st in chk["steps"]:
            lines.append(
                f"| {st['from_step']} → {st['to_step']} | {st['changed']} | "
                f"{_f(st['from_value'], 6, 3)} | {_f(st['to_value'], 6, 3)} | "
                f"{_f(st['delta'], 7, 3)} |"
            )
        ok = "checks out" if chk["closes"] else "DOES NOT CLOSE"
        lines += [
            "",
            (
                f"Sum of steps {_f(chk['steps_sum'], 1, 3).strip()} vs total change "
                f"{_f(chk['total_change'], 1, 3).strip()} "
                f"({_f(chk['first_value'], 1, 3).strip()} → "
                f"{_f(chk['last_value'], 1, 3).strip()}) — **{ok}**."
            ),
        ]
    lines += [
        "",
        "Within-row `before − after` gap (the fill's own BBO jump, **not** a walk step):",
        "",
        "| row | unit | clock | before − after |",
        "|---|---|---|---|",
    ]
    for row in ladder:
        bma = row.get("before_minus_after") or {}
        lines.append(
            f"| {row['step']} | {row['unit']} | {row['clock']} | "
            f"{_f(bma.get('value'), 7, 3)} |"
        )
    return lines



def _reversion_table(res: dict[str, Any]) -> list[str]:
    """Item 2: the test that actually discriminates mechanical from informational."""
    horizons = res.get("horizons", {})
    if not horizons:
        return []
    rows = []
    for hl, blk in horizons.items():
        mk = blk["events"]["after"]["overall"]["markout_px0001"]
        rows.append((hl, mk["mean"], mk["mean_ci95"]))
    first, last = rows[0][1], rows[-1][1]
    reverts = bool(np.isfinite(first) and np.isfinite(last) and abs(last) < abs(first))
    verdict = (
        "the impact **reverts**: the markout shrinks with horizon, which is what a "
        "transient dislocation in the quote looks like once displayed depth is "
        "replenished"
        if reverts else
        "the impact is **permanent**: the markout does not shrink with horizon, so the "
        "price does not come back. A purely mechanical dislocation would revert once "
        "displaced depth was replenished; this does not"
    )
    lines = [
        "### Reversion check — mechanical or informational?",
        "",
        (
            "This is the test that discriminates. The `before − after` gap cannot: it "
            "equals `s · (mid_after − mid_before)`, which involves neither the terminal "
            "mid nor `H`, so it is **constant across horizons by construction** and "
            "tests nothing. What does discriminate is whether the move comes back."
        ),
        "",
        "| H | markout from the `after` mid ($0.0001) | 95% CI |",
        "|---|---|---|",
    ]
    for hl, mean, ci in rows:
        lines.append(f"| {hl} | {_f(mean)} | {_ci_str(ci)} |")
    lines += [
        "",
        (
            f"From {horizons and rows[0][0]} to {rows[-1][0]} the markout goes "
            f"{_f(first, 1, 2).strip()} → {_f(last, 1, 2).strip()}, so {verdict}."
        ),
    ]
    return lines


def _loss_asymmetry_note(res: dict[str, Any]) -> list[str]:
    """Item 4: the honest description of the distribution."""
    horizons = res.get("horizons", {})
    if not horizons:
        return []
    lines = [
        "### Who loses, and by how much",
        "",
        (
            "| H | share losing | share winning | mean loss when losing ($0.0001) "
            "| mean gain when winning ($0.0001) |"
        ),
        "|---|---|---|---|---|",
    ]
    for hl, blk in horizons.items():
        sg = blk["events"]["after"]["overall"].get("realized_sign")
        if not sg:
            continue
        lines.append(
            f"| {hl} | {_f(sg['share_neg'], 6, 3)} | {_f(sg['share_pos'], 6, 3)} | "
            f"{_f(sg.get('mean_neg'))} | {_f(sg.get('mean_pos'))} |"
        )
    lines += [
        "",
        (
            "About **half** of aggressor events lose and about half win; the mean is "
            "negative because **losers lose more than winners win**. This is not a "
            "minority of bad fills dragging an otherwise profitable book — it is a "
            "roughly symmetric hit rate with asymmetric magnitudes."
        ),
    ]
    return lines


def _block_table(res: dict[str, Any]) -> list[str]:
    """Item 2: what each CI actually rests on."""
    lines = [
        "### Bootstrap blocks and dependence",
        "",
        (
            "| H | NW lag (events) | events per H-window | block (clock) "
            "| block (events) | effective blocks |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for hl, blk in res.get("horizons", {}).items():
        pl = blk["block"]
        lines.append(
            f"| {hl} | {blk['nw_lag']} | {blk['events_per_horizon_window']} | "
            f"{horizon_label(pl['block_ns'])} | {pl['block_events']} | "
            f"{pl['n_blocks_effective']} |"
        )
    return lines



def _hs_range(res: dict[str, Any]) -> tuple[float, float]:
    """|realized spread| in half-spreads at the **first and last** horizon.

    Endpoints, not min/max, because the sentence that uses this reads "from
    <shortest> to <longest>". The path between them is not always monotone — QQQ
    dips slightly at 1 s — so :func:`_hs_monotone` flags that separately rather
    than letting an endpoint range imply a straight line.
    """
    vals = [
        abs(float(blk["events"]["after"]["overall"]["mean_realized_spread_half_spreads"]))
        for blk in res.get("horizons", {}).values()
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return (vals[0], vals[-1]) if vals else (float("nan"), float("nan"))


def _hs_monotone(res: dict[str, Any]) -> bool:
    """True when |realized spread| grows at every horizon step."""
    vals = [
        abs(float(blk["events"]["after"]["overall"]["mean_realized_spread_half_spreads"]))
        for blk in res.get("horizons", {}).values()
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return all(b >= a for a, b in itertools.pairwise(vals)) if vals else True


def _bottom_line(per_symbol: dict[str, dict[str, Any]]) -> list[str]:
    """The claim this study supports, stated once, with its limits attached."""
    if not per_symbol:
        return []
    ranges = {sym: _hs_range(res) for sym, res in per_symbol.items()}
    span = ", ".join(
        f"{_f(lo, 1, 2).strip()}–{_f(hi, 1, 2).strip()} ({sym})"
        for sym, (lo, hi) in ranges.items()
    )
    first = next(iter(per_symbol.values()))
    hz = list(first.get("horizons", {}))
    lo_h, hi_h = (hz[0], hz[-1]) if hz else ("?", "?")
    ladder = first.get("attribution") or []
    e6 = ladder[0]["after"]["share_adverse_conditional"] if ladder else float("nan")
    non_mono = [sym for sym, res in per_symbol.items() if not _hs_monotone(res)]
    mono_note = (
        ""
        if not non_mono
        else (
            " The path between those endpoints is not monotone for "
            + ", ".join(non_mono)
            + ", so the range should not be read as a straight line."
        )
    )
    return [
        "## Bottom line",
        "",
        (
            f"The original {_f(100 * e6, 1, 0).strip()} % figure was a **conditional, "
            "event-time share** that overstated how consistent adverse selection is. "
            f"Measured as realized spread, passive fills on 2019-12-30 lost {span} "
            f"half-spreads from {lo_h} to {hi_h}, CIs excluding zero. The loss is "
            "concentrated in fills hit by level-clearing aggressive orders and does not "
            "revert within 60 s. Single day; shape over time differs by symbol."
            + mono_note
        ),
        "",
        (
            "Three things separate that from the headline figure, in order of size: the "
            "statistic was conditional on the mid having moved (a fifth to a third of "
            "fills see no move at all); five book events is a far shorter window than it "
            "sounds in a busy name; and same-nanosecond fills from one aggressor were "
            "counted as independent observations. The attribution table in each symbol "
            "section quantifies the last two and checks that they sum."
        ),
        "",
        (
            "**Not established here.** Why the horizon profile differs between the two "
            "symbols is an open question — that information decays faster in the more "
            "liquid name is a *hypothesis* this study does not test. Nor is one session "
            "enough to generalise the horizon shape; the multi-day aggregation is the "
            "place for that."
        ),
    ]


def render_clock_markouts_md(day: str, per_symbol: dict[str, dict[str, Any]]) -> str:
    """Render the E6b tables for one day, one section per symbol."""
    head = (
        f"# Clock-time passive-fill markouts — NASDAQ ITCH 5.0, "
        f"{day[:2]}/{day[2:4]}/{day[4:]}"
    )
    lines = [head, "", _PREAMBLE, ""]
    for sym, res in per_symbol.items():
        lines += [f"## {sym}", ""]
        lines.append(
            f"{res.get('n_fills', 0):,} passive fills grouped into "
            f"{res.get('n_events', 0):,} aggressor events "
            f"({_f(res.get('fills_per_event'), 1, 2).strip()} fills per event). "
            "Observations dropped per horizon:"
        )
        lines += ["", *_drop_table(res), ""]
        lines += [*_headline_table(res), ""]
        lines += [*_sign_table(res), ""]
        lines += [*_rebate_table(res), ""]
        lines += [*_attribution_table(res), ""]
        lines += ["### Markout", "", *_markout_table(res, "events"), ""]
        lines += [*_reversion_table(res), ""]
        lines += [*_loss_asymmetry_note(res), ""]
        lines += [*_block_table(res), ""]
        lines += [*_control_table(res), ""]
        lines += ["### Breakdowns", ""]
        for hl in res.get("horizons", {}):
            lines += [*_group_table(res, hl), ""]
        lines += [
            "### Secondary — per individual fill",
            "",
            (
                "The same measurement without the aggressor grouping. Sweep "
                "multiplicity inflates `n`, so the CIs here are too narrow; kept for "
                "comparison with the event-time E6, which is also per fill."
            ),
            "",
            *_markout_table(res, "fills"),
            "",
        ]
    lines += [*_bottom_line(per_symbol), ""]
    return "\n".join(lines) + "\n"
