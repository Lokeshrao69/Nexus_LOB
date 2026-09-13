"""Order-level queue tracking + fill-probability research (Phase 3).

Answers the microstructure question *for a passive order*:

> Given my queue position and market state at decision time ``t``, what is
> ``P(within-window fill)``, and (in ``adverse_selection.py``) how much
> unfavourable mid-price movement follows the fill?

What is OBSERVED vs INFERRED (the honesty contract)
---------------------------------------------------
* **Observed** from the order-level feed (real ITCH ``NormalizedEvent`` or our
  synthetic ``SyntheticFlow``): the identity of every order (``order_id``),
  the side/price/size a new order joins with, which orders execute and how
  much, which orders (partially) cancel or delete, and the resulting aggregate
  ladder. ``QueueTracker`` reconstructs a per-``(side, price)`` FIFO from
  exactly these events, so the reconstruction is diff-testable against the
  generator's internal book (``reconstruct() == SyntheticFlow.internal_book``).
* **Inferred**: that the venue matches in price-time priority (the fifo order
  is the arrival order). A real L2 snapshot alone could not split a level's
  size into per-order queue positions — ITCH's order-level messages can, which
  is why the tracker consumes events, not snapshots. ``REPLACE`` inherits the
  simplification from ``ReplayEngine.apply`` (new order id joins the back of
  its price level) rather than the full NASDAQ-U priority rules.
* **Never** used as a feature: anything not knowable at decision time. Fill
  probability features are read from the tracker state *before* any look-ahead
  walk; the (future) outcome is the label only. Tests enforce this.

Two complementary answer populations:
* **Hypothetical drops** (``fill_dataset``): repeatedly "join the touch" with a
  passive order of a fixed size and walk the tape forward — no cancellation
  selection bias, this is the controlled P(fill) experiment.
* **Real standing orders** (the ``fills`` the tracker recorded): every executed
  order is a passive fill with known queue position — the population for
  adverse-selection (``adverse_selection.py``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..book_port import View
from ..book_state import DEPTH, Side, empty_state
from ..itch_parser import EventType, NormalizedEvent

from .experiments import brier_score, calibration_curve
from .features import (
    flow_intensity,
    lob_imbalance,
    momentum,
    ofi,
    realized_vol,
    spread_bps,
)
from .synthetic_flow import SyntheticFlow

_View = Any


@dataclass(slots=True)
class RestingOrder:
    """A live resting order (``size`` = current remaining; history frozen)."""

    oid: int
    side: Side
    price: int
    size: int
    born_size: int
    born_index: int
    born_ts_ns: int
    ahead_at_birth: int


@dataclass(frozen=True, slots=True)
class FillRecord:
    """A real passive order that executed (the adverse-selection population)."""

    oid: int
    side: Side
    price: int
    filled: int
    event_index: int
    ts_ns: int
    size_at_birth: int
    ahead_at_birth: int
    ahead_at_fill: int


@dataclass(slots=True)
class _Level:
    side: Side
    price: int
    order2size: dict[int, int] = field(default_factory=dict)
    fifo: deque[int] = field(default_factory=deque)

    def total(self) -> int:
        return sum(self.order2size.values())

    def positions(self) -> list[tuple[int, int]]:
        """[(oid, size)] in FIFO order (front first)."""
        return [(oid, self.order2size[oid]) for oid in self.fifo]

    def ahead_of(self, oid: int) -> int:
        """Sum of sizes strictly ahead of ``oid`` in FIFO order (0 if absent)."""
        if oid not in self.order2size:
            return 0
        ahead = 0
        for o in self.fifo:
            if o == oid:
                break
            ahead += self.order2size[o]
        return ahead


class QueueTracker:
    """Reconstruct per-``(side, price)`` FIFO queues from an order-level tape.

    Mirrors ``ReplayEngine.apply`` exactly: partial fills keep the order at its
    queue priority; ``REPLACE`` removes the old id and the new id joins the
    back of its price level.
    """

    def __init__(self) -> None:
        self._orders: dict[int, RestingOrder] = {}
        self._levels: dict[Side, dict[int, _Level]] = {Side.Bid: {}, Side.Ask: {}}
        self._side_levels_owner = self._levels  # (kept for clarity/tests)
        self.fills: list[FillRecord] = []
        self.issues: list[str] = []
        self.mid_history: list[float] = []
        self.seq = 0
        self.ts_ns = 0
        self._last_mid = 0.0

    # ------------------------------------------------------------------ #
    # event handling (mirrors ReplayEngine.apply)                        #
    # ------------------------------------------------------------------ #
    def on_event(self, ev: NormalizedEvent) -> None:
        kind = ev.kind
        if kind in (EventType.ADD, EventType.ADD_MPID):
            self._add_order(ev.order_id, ev.side, ev.price_ticks, ev.size, ev.ts_ns)
        elif kind in (EventType.EXECUTE, EventType.EXECUTE_PX):
            self._execute(ev)
        elif kind == EventType.CANCEL:
            self._cancel(ev.order_id, ev.size)
        elif kind == EventType.DELETE:
            self._cancel(ev.order_id, ev.size if ev.size else None)
        elif kind == EventType.REPLACE:
            self._replace(ev)
        elif kind == EventType.TRADE:
            pass  # a print — no queue effect
        else:  # pragma: no cover — defensive
            self.issues.append(f"seq{self.seq}: unhandled kind {kind}")
        self.seq += 1
        self.ts_ns = ev.ts_ns
        m = self.mid()
        self._last_mid = m
        self.mid_history.append(m)

    def _add_order(self, oid: int, side: Side, price: int, size: int, ts_ns: int) -> None:
        if oid <= 0 or size <= 0:
            self.issues.append(f"seq{self.seq}: bad add oid={oid} size={size}")
            return
        lvl = self._levels[side].setdefault(price, _Level(side, price))
        ahead = lvl.total()
        order = RestingOrder(oid, side, price, size, size, self.seq, ts_ns, ahead)
        self._orders[oid] = order
        lvl.order2size[oid] = size
        lvl.fifo.append(oid)

    def _execute(self, ev: NormalizedEvent) -> None:
        live = self._orders.get(ev.order_id)
        if live is None:
            self.issues.append(f"seq{self.seq}: execute untracked oid {ev.order_id}")
            return
        filled = min(live.size, ev.size)
        if filled <= 0:
            return
        lvl = self._levels[live.side].get(live.price)
        px = ev.price_ticks if (ev.kind == EventType.EXECUTE_PX and ev.price_ticks) else live.price
        ahead_at_fill = lvl.ahead_of(live.oid) if lvl is not None else 0
        self.fills.append(
            FillRecord(
                oid=live.oid, side=live.side, price=px, filled=filled,
                event_index=self.seq, ts_ns=ev.ts_ns,
                size_at_birth=live.born_size,
                ahead_at_birth=live.ahead_at_birth, ahead_at_fill=ahead_at_fill,
            )
        )
        self._reduce(live.oid, filled)

    def _cancel(self, oid: int, size: int | None) -> None:
        live = self._orders.get(oid)
        if live is None:
            self.issues.append(f"seq{self.seq}: cancel untracked oid {oid}")
            return
        cut = live.size if size is None else min(live.size, size)
        self._reduce(oid, cut)

    def _replace(self, ev: NormalizedEvent) -> None:
        live = self._orders.get(ev.order_id)
        if live is None:
            self.issues.append(f"seq{self.seq}: replace untracked oid {ev.order_id}")
            return
        self._reduce(ev.order_id, live.size)
        if ev.new_order_id:
            self._add_order(ev.new_order_id, live.side, ev.price_ticks, ev.size, ev.ts_ns)

    def _reduce(self, oid: int, cut: int) -> None:
        """Reduce (or fully consume) an order in place; partial keeps priority."""
        live = self._orders.get(oid)
        if live is None or cut <= 0:
            return
        nxt = live.size - cut
        if nxt > 0:
            live.size = nxt
            lvl = self._levels[live.side].get(live.price)
            if lvl is not None:
                lvl.order2size[oid] = nxt
            return
        # consumed / cancelled entirely → remove
        lvl = self._levels[live.side].get(live.price)
        if lvl is not None:
            lvl.order2size.pop(oid, None)
            if oid in lvl.fifo:
                lvl.fifo.remove(oid)
            if not lvl.fifo:
                self._levels[live.side].pop(live.price, None)
        self._orders.pop(oid, None)

    # ------------------------------------------------------------------ #
    # read surface                                                       #
    # ------------------------------------------------------------------ #
    def queue(self, side: Side, price: int) -> list[RestingOrder]:
        lvl = self._levels[side].get(price)
        if lvl is None:
            return []
        out: list[RestingOrder] = []
        for oid in lvl.fifo:
            ro = self._orders.get(oid)
            if ro is not None:
                out.append(ro)
        return out

    def level_size(self, side: Side, price: int) -> int:
        lvl = self._levels[side].get(price)
        return lvl.total() if lvl is not None else 0

    def ahead_of(self, oid: int) -> int:
        live = self._orders.get(oid)
        if live is None:
            return 0
        lvl = self._levels[live.side].get(live.price)
        return lvl.ahead_of(oid) if lvl is not None else 0

    def best(self, side: Side) -> int:
        lvl = self._levels[side]
        if side == Side.Bid:
            return max((px for px, l in lvl.items() if l.fifo), default=0)
        return min((px for px, l in lvl.items() if l.fifo), default=0)

    def mid(self) -> float:
        bb, ba = self.best(Side.Bid), self.best(Side.Ask)
        if not bb or not ba:
            return 0.0
        return (bb + ba) / 2.0

    def reconstruct(self) -> dict[Side, dict[int, list[tuple[int, int]]]]:
        """FIFO-serialized (oid, size) per (side, price) — diff-test shape."""
        out: dict[Side, dict[int, list[tuple[int, int]]]] = {Side.Bid: {}, Side.Ask: {}}
        for side in (Side.Bid, Side.Ask):
            for px, lvl in sorted(self._levels[side].items()):
                if lvl.fifo:
                    out[side][px] = lvl.positions()
        return out

    def view(self) -> View:
        """Aggregate ladder shaped like the frozen ``BOOK_STATE_DTYPE`` mirror.

        Research ``features`` (``lob_imbalance``, ``spread``, ``mid``, …)
        operate on it directly.
        """
        st = empty_state()
        for i, (px, sz, ct) in enumerate(self._ladder(Side.Bid)):
            st["bid_px"][i], st["bid_sz"][i], st["bid_ct"][i] = px, sz, ct
        for i, (px, sz, ct) in enumerate(self._ladder(Side.Ask)):
            st["ask_px"][i], st["ask_sz"][i], st["ask_ct"][i] = px, sz, ct
        return {
            "bid_px": st["bid_px"], "bid_sz": st["bid_sz"], "bid_ct": st["bid_ct"],
            "ask_px": st["ask_px"], "ask_sz": st["ask_sz"], "ask_ct": st["ask_ct"],
            "seq": self.seq, "ts_ns": self.ts_ns,
            "cum_volume": 0, "last_trade_px": 0, "last_trade_sz": 0,
            "last_trade_side": Side.NONE, "version": self.seq,
        }

    def _ladder(self, side: Side) -> list[tuple[int, int, int]]:
        prices = sorted(self._levels[side].keys(), reverse=(side == Side.Bid))
        out: list[tuple[int, int, int]] = []
        for px in prices[:DEPTH]:
            lvl = self._levels[side][px]
            out.append((px, lvl.total(), len(lvl.order2size)))
        return out

    def positions_by_side(self, side: Side) -> list[tuple[int, list[tuple[int, int]]]]:
        """[(price, [(oid, size), ...])] FIFO per level, best price first."""
        prices = sorted(self._levels[side].keys(), reverse=(side == Side.Bid))
        return [(px, self._levels[side][px].positions()) for px in prices]


# --------------------------------------------------------------------------- #
# Hypothetical passive-fill counterfactual (the "queue-ahead walk")           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FillOutcome:
    """Result of dropping a hypothetical passive order and walking forward."""

    decision_index: int
    side: Side
    price: int
    qty: int
    filled: bool
    fill_qty: int
    tau: int | None  # events from decision to COMPLETE fill (None = window-end)
    queue_ahead: int  # shares ahead of the dropped order at decision time
    level_size: int   # total displayed size at its price at decision time


def queue_ahead_walk(
    queue: list[RestingOrder],
    events: list[NormalizedEvent],
    *,
    decision_index: int,
    side: Side,
    price: int,
    qty: int,
    window: int,
    queue_ahead: int = 0,  # shares ahead of the dropped order at decision time
    level_size: int = 0,   # total displayed size at its price at decision time
) -> FillOutcome:
    """Replay the tape's at-price events against a copy of the level's queue.

    ``queue`` is the tracker's ``queue(side, price)`` snapshot at
    ``decision_index`` (front→back). The hypothetical order ``qty`` joins the
    **back** of that queue and the events ``[decision_index+1, …+window)`` are
    walked once. Marketable flow that the tape shows reaching *our* price
    consumes the level's queue from the front; the hypothetical fills when that
    flow reaches its slot:

    * ``EXECUTE(oid, n)`` on a snapshot order consumes ``n`` from its remaining
      size; if the order is fully consumed and the flow continues ``>n``, the
      leftover cascades through the next front orders and finally into the
      hypothetical (partial fills keep priority);
    * ``CANCEL``/``DELETE`` of a snapshot order frees queue ahead; behind the
      hypothetical → ignored (never moves us);
    * an ``ADD`` at this price after ``t`` joins **behind** the hypothetical
      (never moves us); events at any *other* (side, price) are ignored — the
      tape is ground truth about which level marketable flow reached, so
      better-price depth appearing after ``t`` is handled implicitly;
    * a ``REPLACE`` removes the old id (shrinking what's ahead if it was) and
      the new id joins the back of its price.

    Leak-free by construction: only the queue known at ``t`` plus strictly
    forward tape events are used; the (future) outcome is a label, never a
    feature.
    """
    # Per-order remaining sizes of the snapshot orders (all ahead of H) plus the
    # cumulative flow that has already reached H's slot.
    rem: dict[int, int] = {o.oid: o.size for o in queue}
    ahead = sum(rem.values())
    h_rem = qty
    h_filled = 0
    last = decision_index + 1 + window
    end = min(last, len(events))
    for k in range(decision_index + 1, end):
        ev = events[k]
        kind = ev.kind
        if kind in (EventType.ADD, EventType.ADD_MPID, EventType.TRADE):
            continue  # at our price joins behind H; other prices irrelevant
        if kind in (EventType.EXECUTE, EventType.EXECUTE_PX):
            oid, n = ev.order_id, ev.size
            cur = rem.get(oid)
            if cur is None:
                # order joined after t (behind H) or a priority-exception id —
                # never in front of H, so it cannot draw flow past us.
                continue
            if n >= cur:
                # flow fully consumes this order and continues past it
                leftover = n - cur
                ahead -= cur
                del rem[oid]
                # propagate the leftover through the new front until it stops
                while leftover > 0 and rem:
                    foid = next(iter(rem))  # dict preserves FIFO insertion order
                    fs = rem[foid]
                    take = min(fs, leftover)
                    fs -= take
                    leftover -= take
                    ahead -= take
                    if fs <= 0:
                        del rem[foid]
                    else:
                        rem[foid] = fs
                        break
                if leftover > 0 and h_rem > 0:
                    fill = min(h_rem, leftover)
                    h_filled += fill
                    h_rem -= fill
                    if h_rem <= 0:
                        return _outcome(decision_index, side, price, qty, True,
                                        h_filled, k - decision_index, queue_ahead, level_size)
            else:
                # partial fill of this order — flow stops here
                rem[oid] = cur - n
                ahead -= n
            continue
        if kind in (EventType.CANCEL, EventType.DELETE):
            oid = ev.order_id
            cur = rem.get(oid)
            if cur is None:
                continue
            cut = cur if (kind == EventType.DELETE or ev.size <= 0) else min(cur, ev.size)
            nxt = cur - cut
            ahead -= cut
            if nxt <= 0:
                del rem[oid]
            else:
                rem[oid] = nxt
            continue
        if kind == EventType.REPLACE:
            oid = ev.order_id
            cur = rem.get(oid)
            if cur is None:
                continue
            ahead -= cur
            del rem[oid]
            # new id rejoins the back of its (new) price → never in front of H
            continue
    # window exhausted (or tape ended) with H still resting
    return _outcome(decision_index, side, price, qty, h_rem <= 0,
                    h_filled, None, queue_ahead, level_size)


def _outcome(
    decision_index: int, side: Side, price: int, qty: int,
    filled: bool, h_filled: int, tau: int | None,
    queue_ahead: int, level_size: int,
) -> FillOutcome:
    return FillOutcome(
        decision_index=decision_index, side=side, price=price, qty=qty,
        filled=filled, fill_qty=h_filled, tau=tau,
        queue_ahead=queue_ahead, level_size=level_size,
    )

# --------------------------------------------------------------------------- #
# (WS-1) Real standing orders: lifecycles + Kaplan-Meier survival              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OrderLife:
    """One real standing order's lifecycle (replayed from the order-level tape)."""

    oid: int
    side: Side
    price: int
    born_index: int
    birth_size: int
    end_index: int | None   # None = still resting at tape end (right-censored)
    end_kind: str           # "fill"|"cancel"|"delete"|"replace"|"open"
    filled: int

    @property
    def tau(self) -> int | None:
        """Events from birth to end (None for open orders)."""
        return None if self.end_index is None else self.end_index - self.born_index


def standing_order_lifetimes(events: Sequence[NormalizedEvent]) -> list[OrderLife]:
    """One forward pass over the tape producing every order's lifecycle.

    EXECUTE reduces remaining; when it reaches 0 the order ends "fill".
    CANCEL/DELETE end it "cancel"/"delete" (the competing risks); REPLACE ends the
    old oid "replace" and starts the new id; orders still resting at tape end are
    "open" (right-censored at the end of study).
    """
    live: dict[int, tuple[int, Side, int, int, int]] = {}
    out: list[OrderLife] = []
    for idx, ev in enumerate(events):
        k = ev.kind
        if k in (EventType.ADD, EventType.ADD_MPID):
            live[ev.order_id] = (idx, ev.side, ev.price_ticks, ev.size, ev.size)
        elif k in (EventType.EXECUTE, EventType.EXECUTE_PX):
            rec = live.get(ev.order_id)
            if rec is None:
                continue
            born, side, px, born_size, rem = rec
            nxt = rem - ev.size
            if nxt <= 0:
                live.pop(ev.order_id, None)
                out.append(OrderLife(ev.order_id, side, px, born, born_size,
                                     idx, "fill", min(rem, ev.size)))
            else:
                live[ev.order_id] = (born, side, px, born_size, nxt)
        elif k in (EventType.CANCEL, EventType.DELETE):
            rec = live.get(ev.order_id)
            if rec is None:
                continue
            born, side, px, born_size, rem = rec
            cut = rem if (k == EventType.DELETE or ev.size <= 0) else min(rem, ev.size)
            nxt = rem - cut
            kind = "cancel" if k == EventType.CANCEL else "delete"
            if nxt <= 0:
                live.pop(ev.order_id, None)
                out.append(OrderLife(ev.order_id, side, px, born, born_size,
                                     idx, kind, cut))
            else:
                live[ev.order_id] = (born, side, px, born_size, nxt)
        elif k == EventType.REPLACE:
            rec = live.get(ev.order_id)
            if rec is None:
                continue
            born, side, px, born_size, rem = rec
            live.pop(ev.order_id, None)
            out.append(OrderLife(ev.order_id, side, px, born, born_size,
                                 idx, "replace", rem))
            if ev.new_order_id:
                live[ev.new_order_id] = (idx, ev.side, ev.price_ticks, ev.size, ev.size)
        # TRADE has no queue effect.
    for oid, (born, side, px, born_size, _rem) in live.items():
        out.append(OrderLife(oid, side, px, born, born_size, None, "open", 0))
    return out


def fill_prob_survival(
    levels: Sequence[OrderLife],
    *,
    hazard_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Kaplan-Meier survival of the time-to-FILL event over real standing orders.

    Cancels/deletes/replaces are a COMPETING RISK: they right-censor the fill arm
    (and a parallel survival_cancel curve is returned with fills censored) so both
    causes are visible. hazard_fn optionally injects a baseline hazard vector;
    None uses the empirical KM increments. cum_fill/cum_cancel are the cumulative
    incidences (1 - S). JSON-able.
    """
    levels = list(levels)
    fills = [ol for ol in levels if ol.end_kind == "fill"]
    others = [ol for ol in levels if ol.end_kind in ("cancel", "delete", "replace")]
    H = max([ol.end_index for ol in levels if ol.end_index is not None], default=0)
    tau = np.arange(1, int(H) + 1)
    n_vec: list[int] = []
    d_vec: list[int] = []
    c_vec: list[int] = []
    for t in tau:
        n_risk = 0
        for ol in levels:
            if ol.end_index is not None:
                if ol.end_index - ol.born_index >= t:
                    n_risk += 1
            else:
                if H - ol.born_index >= t:
                    n_risk += 1
        n_vec.append(n_risk)
        d_vec.append(sum(1 for ol in fills if ol.end_index - ol.born_index == t))
        c_vec.append(sum(1 for ol in others if ol.end_index - ol.born_index == t))
    n_arr = np.asarray(n_vec, dtype=np.float64)
    d_arr = np.asarray(d_vec, dtype=np.float64)
    c_arr = np.asarray(c_vec, dtype=np.float64)
    if hazard_fn is not None:
        h_fill = np.asarray(hazard_fn(n_arr), dtype=np.float64)
        h_cancel = np.asarray(hazard_fn(n_arr), dtype=np.float64)
    else:
        h_fill = np.where(n_arr > 0, d_arr / np.maximum(n_arr, 1.0), 0.0)
        h_cancel = np.where(n_arr > 0, c_arr / np.maximum(n_arr, 1.0), 0.0)
    s_fill = np.cumprod(np.clip(1.0 - h_fill, 0.0, 1.0))
    s_cancel = np.cumprod(np.clip(1.0 - h_cancel, 0.0, 1.0))
    return {
        "tau": tau.tolist(),
        "survival_fill": s_fill.tolist(),
        "survival_cancel": s_cancel.tolist(),
        "at_risk": [int(v) for v in n_arr],
        "cum_fill": (1.0 - s_fill).tolist(),
        "cum_cancel": (1.0 - s_cancel).tolist(),
        "n_orders": len(levels),
        "n_fills": len(fills),
        "n_censored": len(others) + sum(1 for ol in levels if ol.end_kind == "open"),
    }
