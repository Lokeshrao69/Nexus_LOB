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

import heapq
import math
import types
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Self is annotation-only (3.11+); harmless on Python 3.10
    from typing import Self

import numpy as np

from ..book_port import View
from ..book_state import DEPTH, Side, empty_state
from ..itch_parser import EventType, NormalizedEvent
from .experiments import bootstrap_ci, rank_ic
from .features import (
    flow_intensity,
    lob_imbalance,
    mid,
    momentum,
    ofi,
    realized_vol,
    spread_bps,
    spread_ticks,
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
    # (WS-2/E6, additive defaults) — level size at the moment of the fill and the
    # order-flow imbalance *of the take that filled us*, both knowable at fill time.
    level_size_at_fill: int = 0
    ofi: float = 0.0


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
        self._prev_view: _View | None = None  # view BEFORE the current event (E6 ofi)

    # ------------------------------------------------------------------ #
    # event handling (mirrors ReplayEngine.apply)                        #
    # ------------------------------------------------------------------ #
    def on_event(self, ev: NormalizedEvent) -> None:
        # Capture the view BEFORE this event's mutation, but only for EXECUTE
        # events (the only ones E6 reads) — ``view()`` is a per-call alloc, so
        # skipping it on the ~94% non-take events keeps replay cheap.
        if ev.kind in (EventType.EXECUTE, EventType.EXECUTE_PX):
            self._prev_view = self.view()
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
        level_size_at_fill = lvl.total() if lvl is not None else 0
        # Reduce first so ``ofi(self._prev_view, view())`` below sees the book
        # AFTER this fill's take — the flow of the take that filled us (E6).
        self._reduce(live.oid, filled)
        ofi_val = ofi(self._prev_view, self.view())
        self.fills.append(
            FillRecord(
                oid=live.oid, side=live.side, price=px, filled=filled,
                event_index=self.seq, ts_ns=ev.ts_ns,
                size_at_birth=live.born_size,
                ahead_at_birth=live.ahead_at_birth, ahead_at_fill=ahead_at_fill,
                level_size_at_fill=level_size_at_fill, ofi=ofi_val,
            )
        )

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
      size; a fully-consumed order's excess cascades through the next front
      orders (partial fills keep priority). Once every snapshot order ahead of
      the hypothetical is gone the take has reached its slot: it books
      ``min(qty, leftover)`` of known surplus, and — because the generator
      emits one take's EXECUTEs contiguously — if the *next* event is another
      EXECUTE the take walked deeper and absorbs the whole resting ``qty``
      (see ``_take_continues``). A take that stops exactly at the last order's
      back (no surplus, no continuation) leaves the hypothetical unfilled;
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
                # The EXECUTE event's size is capped at the order's remaining, so
                # ``leftover`` is ~always 0 even when the take is hungrier than the
                # level. The take's excess then shows up as the *next* event being
                # another EXECUTE (this generator emits one take's executes
                # contiguously). Exposure rule (Level-drain + take continuation):
                # once every snapshot order ahead of the drop is gone, the take
                # has reached our slot; it books min(qty, leftover) from known
                # surplus, and if it walks deeper (next event is an EXECUTE) its
                # ongoing hunger absorbs our whole resting qty.
                if not rem:
                    fill_here = min(h_rem, leftover) if leftover > 0 else 0
                    h_filled += fill_here
                    h_rem -= fill_here
                    if h_rem > 0 and _take_continues(k, events):
                        h_filled += h_rem
                        h_rem = 0
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


def _take_continues(k: int, events: Sequence[NormalizedEvent]) -> bool:
    """True when the take that just emitted an EXECUTE at ``k`` walks deeper.

    This generator emits one take's EXECUTEs contiguously (no intervening ADD/
    CANCEL/REPLACE/TRADE; only the post-take ``_normalize_book`` ADD/DELETEs,
    which follow a take that fully stopped). So ``events[k+1]`` being an EXECUTE
    means the same marketable flow is hungry past the level the drop sits on.
    """
    return k + 1 < len(events) and events[k + 1].kind in (EventType.EXECUTE, EventType.EXECUTE_PX)

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


def _fill_prob_survival_lifetimes(
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


# --------------------------------------------------------------------------- #
# (WS-1) Controlled-fill population + the logistic fill model                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FillRow:
    """One controlled passive-fill trial: prediction state at ``t``, label on ``t+w``.

    ``decision_index`` is the number of tape events consumed at the decision;
    the prediction state reflects ``events[:decision_index]`` ONLY (the leak
    lock — ``features`` hold just that past state) and the outcome walk covers
    ``events[decision_index : decision_index + window]`` (``filled`` /
    ``fill_qty`` / ``tau`` are the future label). A back-of-touch join means
    ``queue_ahead == level_size`` — the whole displayed level is ahead of us;
    documented, not a defect.
    """

    decision_index: int
    side: Side
    price: int
    qty: int
    filled: bool
    fill_qty: int
    tau: int | None          # events from decision to complete fill (None = window-end)
    queue_ahead: int
    level_size: int
    mid_ticks: float
    features: dict[str, float]


# Default per-row feature keys. Note: "log_level" (= log1p(level_size)) and
# "queue_ahead" (= level_size) are KEPT BOTH per the frozen API though they are
# collinear — fit with an explicit feature_names=("log_level",) when a single
# clean queue-size coefficient is what you want to report.
_DEFAULT_FEATURE_KEYS: tuple[str, ...] = (
    "log_level", "queue_ahead", "spread_bps", "lob_imbalance", "ofi",
    "realized_vol", "flow_intensity", "momentum",
)


def _decision_features(
    tracker: QueueTracker,
    events: Sequence[NormalizedEvent],
    p: int,
    side: Side,
    price: int,
    prev_view: _View | None,
) -> dict[str, float]:
    """Default decision features for a decision at prefix length ``p``.

    A pure function of (the tracker state after ``events[:p]``, ``prev_view``,
    and the past window of events) — it reads nothing from ``events[p:]``.
    ``prev_view`` is the book view at the *previous* scheduled decision point
    (None before the first, giving OFI 0).
    """
    lvl = tracker.level_size(side, price)
    view = tracker.view()
    return {
        "log_level": float(np.log1p(max(0, lvl))),
        "queue_ahead": float(lvl),
        "spread_bps": spread_bps(view),
        "lob_imbalance": lob_imbalance(view),
        "ofi": ofi(prev_view, view),
        "realized_vol": realized_vol(tracker.mid_history, n=10),
        "flow_intensity": flow_intensity(events[max(0, p - 10):p], tau=10),
        "momentum": momentum(tracker.mid_history),
    }


def fill_dataset(
    flow: SyntheticFlow,
    *,
    each_tau_drop: int = 40,
    qty: int = 100,
    window: int = 200,
    sides: Sequence[Side] = (Side.Bid, Side.Ask),
    seed: int = 0x51ED,
) -> list[FillRow]:
    """Run the controlled passive-fill experiment over a synthetic tape.

    A fresh ``QueueTracker`` consumes the tape one event at a time. Every
    ``each_tau_drop`` events is a *decision point*; for each ``side`` with a
    live touch we hypothetically join the back of the touch level with ``qty``
    shares and walk the next ``window`` events (``queue_ahead_walk``) for the
    outcome. No cancel-selection bias: a trial fires at every scheduled point
    regardless of what the tape then does — only a live touch gates it.

    Semantics (consistent with ``queue_ahead_walk``): a row with
    ``decision_index = p`` predicts from ``events[:p]`` and its label is read
    from ``events[p : p+window]`` (tau counts events from ``p``). ``seed`` is
    reserved for future controlled-drop jitter; generation is deterministic.
    """
    events = list(flow.events) if flow.events else list(flow.generate())
    if not events:
        raise ValueError("fill_dataset needs a non-empty tape")
    if each_tau_drop < 1 or qty < 1 or window < 1:
        raise ValueError("each_tau_drop/qty/window must all be >= 1")
    tracker = QueueTracker()
    prev_view = None
    rows: list[FillRow] = []
    for p, ev in enumerate(events, start=1):
        tracker.on_event(ev)
        if p % each_tau_drop:
            continue
        cur_view = tracker.view()
        for side in sides:
            price = tracker.best(side)
            if not price:
                continue
            lvl_size = tracker.level_size(side, price)
            if lvl_size <= 0:
                continue
            queue = tracker.queue(side, price)
            outcome = queue_ahead_walk(
                queue, events,
                decision_index=p - 1, side=side, price=price, qty=qty,
                window=window, queue_ahead=lvl_size, level_size=lvl_size,
            )
            feats = _decision_features(tracker, events, p, side, price, prev_view)
            rows.append(FillRow(
                decision_index=p, side=side, price=price, qty=qty,
                filled=outcome.filled, fill_qty=outcome.fill_qty, tau=outcome.tau,
                queue_ahead=lvl_size, level_size=lvl_size,
                mid_ticks=float(tracker.mid()), features=feats,
            ))
        prev_view = cur_view
    return rows


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic (no overflow on large positive ``z``)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticFillModel:
    """Ridge-regularised logistic model of ``P(within-window fill)``.

    Pure-NumPy Newton-IRLS, deterministic (no RNG). Each raw feature is
    z-scored at fit time with the stored ``mean``/``std``; the design matrix is
    ``[1, Z]`` and the L2 ridge penalises the slope coefficients only
    (intercept unpenalised). Predictions on [0, 1].
    """

    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray        # (k+1,); coef[0] = intercept

    def __init__(
        self,
        feature_names: Sequence[str],
        mean: np.ndarray,
        std: np.ndarray,
        coef: np.ndarray,
    ) -> None:
        self.feature_names = list(feature_names)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)

    @classmethod
    def fit(
        cls,
        rows: Sequence[FillRow],
        *,
        feature_names: Sequence[str] | None = None,
        l2: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> LogisticFillModel:
        if not rows:
            raise ValueError("LogisticFillModel.fit needs at least one FillRow")
        names = list(feature_names) if feature_names is not None else sorted(rows[0].features)
        for r in rows:
            missing = [n for n in names if n not in r.features]
            if missing:
                raise ValueError(f"FillRow missing feature(s) {missing}")
        X = np.asarray([[r.features[n] for n in names] for r in rows], dtype=np.float64)
        y = np.asarray([1.0 if r.filled else 0.0 for r in rows], dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std <= 0.0, 1.0, std)
        Z = (X - mean) / std
        M = np.column_stack([np.ones(len(rows)), Z])
        w = np.zeros(M.shape[1], dtype=np.float64)
        ridge = np.zeros((M.shape[1], M.shape[1]), dtype=np.float64)
        ridge[1:, 1:] = l2
        for _ in range(int(max_iter)):
            p = _sigmoid(M @ w)
            pc = np.clip(p, 1e-9, 1.0 - 1e-9)
            g = M.T @ (pc - y)
            wgt = pc * (1.0 - pc)
            H = (M * wgt[:, None]).T @ M + ridge
            step = np.linalg.solve(H, g)
            w = w - step
            if np.max(np.abs(step)) <= tol:
                break
        return cls(names, mean, std, w)

    def _design(self, X: dict | np.ndarray) -> np.ndarray:
        if isinstance(X, dict):
            row = np.asarray([[X[n] for n in self.feature_names]], dtype=np.float64)
        else:
            row = np.asarray(X, dtype=np.float64)
            if row.ndim == 1:
                row = row[None, :]
        Z = (row - self.mean) / self.std
        return np.column_stack([np.ones(len(row)), Z])

    def predict_proba(self, X: dict | np.ndarray) -> np.ndarray:
        return _sigmoid(self._design(X) @ self.coef)

    def __call__(self, features: dict | np.ndarray) -> float:
        return float(self.predict_proba(features)[0])

    def brier(self, X: dict | np.ndarray, y) -> float:
        p = self.predict_proba(X)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        if p.shape != yv.shape:
            raise ValueError("LogisticFillModel.brier: predictions and y shapes differ")
        return float(np.mean((p - yv) ** 2))


# --------------------------------------------------------------------------- #
# (WS-1) Full-episode flow-tape accumulator: compute_metrics                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class StepFlow:
    """One recorded per-step flow snapshot (book state AFTER the event).

    ``step`` is the tracker sequence count after consuming the event (1-based).
    ``mid_prev`` / ``*_touch_prev`` are the state BEFORE the event so per-step
    deltas (queue decay, flow) are computable without lookahead. Prices are
    integer ticks; ``spread`` 0 means no two-sided BBO at that instant (an
    empty-side gap mid-tape), which the decay/CAR math skips rather than NaNs.
    """

    step: int
    kind: EventType
    ts_ns: int
    mid_prev: float
    mid: float
    spread: int               # best spread in ticks AFTER the event (0 = no BBO)
    bid_touch: int            # displayed size at best bid AFTER the event (0 = empty)
    bid_touch_prev: int       # ...BEFORE the event
    ask_touch: int            # displayed size at best ask AFTER the event
    ask_touch_prev: int       # ...BEFORE the event
    ofi: float                # order-flow imbalance contributed by this step (prev_view→view)
    imbalance: float          # lob_imbalance after the event


class compute_metrics:
    """Context-manager **flow tape** that accumulates per-step flow into an episode.

    Open it in a ``with`` statement and drag it across a step loop; every call
    records the step's ``NormalizedEvent`` plus the queue/book state it produced,
    so the object is a structured **per-episode accumulator**:

        with compute_metrics() as m:
            f = SyntheticFlow(FlowConfig(n_events=800, seed=7)); f.generate()
            for ev in f.events:
                m(ev)                # record the step's flow
        out = m.metrics()            # full-episode summary at closure

    ``__exit__`` computes (and freezes onto ``result``) the summary, so calling
    ``metrics()`` after the block is stable; ``metrics()`` also works mid-loop on
    a partial tape (it recomputes from the records each call). An existing
    ``QueueTracker`` may be passed in to keep replaying an already-built book.

    Closure-time **summary metrics** (full-episode evaluation — forward labels
    like the CAR response are computed only here, never exposed during the loop):

    * ``queue_decay`` — per side, the decay of the touch (best) level's displayed
      size, i.e. the queue a *back-of-touch* passive order joins behind: per-event
      attrition/growth shares (fraction of the ahead-queue each event eats /
      restocks) and the ``consumed_fraction`` of the standing queue by episode end.
    * ``latency_attrition`` — the same queue eaten **per unit of wall-clock time**
      (s⁻¹ rate + half-life): how much queue position is lost to intervening flow
      while a passive order waits (its latency), vs the per-event view above.
    * ``car`` — conditional average response: the mean ``h``-event mid move
      following each recorded step, conditioned on the step's event kind and on
      terciles of the step's order-flow imbalance (the E3/E6 conditioning idea),
      with a rank-IC of flow→response.

    Plus flow composition (``kind_counts``), net/path mid move, spread, and flow
    totals. Everything is deterministic for a given tape (CIs use a fixed seed).
    The honesty contract: only per-step **posted** book state is recorded; the
    forward response is a closure-time label, exactly like the ``FillRow`` label.
    """

    tracker: QueueTracker
    h: int
    seed: int
    records: list[StepFlow]
    result: dict[str, Any] | None   # frozen full-episode summary (set at exit)

    def __init__(
        self,
        tracker: QueueTracker | None = None,
        *,
        h: int = 5,
        seed: int = 0x51ED,
    ) -> None:
        self.tracker = tracker if tracker is not None else QueueTracker()
        self.h = max(1, int(h))
        self.seed = int(seed)
        self.records = []
        self.result = None
        self._initial: dict[str, Any] = {"mid": 0.0, "bid": 0, "ask": 0, "ts": 0}

    # ------------------------------------------------------------------ #
    # context-manager protocol                                           #
    # ------------------------------------------------------------------ #
    def __enter__(self) -> Self:
        # Pre-loop snapshot so decay from the very first touch counts (a fresh
        # tracker starts empty → zeros, which the decay math skips).
        v = self.tracker.view()
        m0 = mid(v)
        self._initial = {
            "mid": m0,
            "bid": int(v["bid_sz"][0]) if m0 > 0 else 0,
            "ask": int(v["ask_sz"][0]) if m0 > 0 else 0,
            "ts": int(self.tracker.ts_ns),
        }
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        # Freeze the full-episode summary at closure (even if the loop raised, so
        # the partial tape is inspectable); exceptions still propagate.
        self.result = self._compute()

    # ------------------------------------------------------------------ #
    # the step loop: one call per flow event                             #
    # ------------------------------------------------------------------ #
    def __call__(self, ev: NormalizedEvent) -> None:
        self.step(ev)

    def step(self, ev: NormalizedEvent) -> None:
        """Record one step: feed ``ev`` to the tracker and snapshot the flow."""
        prev = self.tracker.view()
        prev_mid = mid(prev)
        prev_bid = int(prev["bid_sz"][0])
        prev_ask = int(prev["ask_sz"][0])
        self.tracker.on_event(ev)
        cur = self.tracker.view()
        self.records.append(StepFlow(
            step=self.tracker.seq,
            kind=ev.kind,
            ts_ns=int(ev.ts_ns),
            mid_prev=prev_mid,
            mid=mid(cur),
            spread=spread_ticks(cur),
            bid_touch=int(cur["bid_sz"][0]),
            bid_touch_prev=prev_bid,
            ask_touch=int(cur["ask_sz"][0]),
            ask_touch_prev=prev_ask,
            ofi=ofi(prev, cur),
            imbalance=lob_imbalance(cur),
        ))

    # ------------------------------------------------------------------ #
    # closure summary                                                    #
    # ------------------------------------------------------------------ #
    def metrics(self) -> dict[str, Any]:
        """The full-episode summary (frozen ``result`` after the ``with`` block,
        freshly recomputed if called mid-loop on a partial tape)."""
        return self.result if self.result is not None else self._compute()

    def _compute(self) -> dict[str, Any]:
        rec = self.records
        n = len(rec)

        kind_counts: dict[str, int] = {}
        for k in rec:
            kind_counts[k.kind.name] = kind_counts.get(k.kind.name, 0) + 1

        # full mid series anchored on the pre-loop snapshot
        mid_series = [float(self._initial["mid"])] + [float(r.mid) for r in rec]
        mids_pos = [m for m in mid_series if m > 0]
        mid_start = mids_pos[0] if mids_pos else 0.0
        mid_end = mids_pos[-1] if mids_pos else 0.0
        abs_path = 0.0
        for a, b in pairwise(mid_series):
            if a > 0 and b > 0:
                abs_path += abs(b - a)

        spreads = [float(r.spread) for r in rec if r.spread > 0]

        queue_decay = {
            side: self._queue_decay(side) for side in ("bid", "ask")
        }
        latency = {
            side: self._latency_attrition(side) for side in ("bid", "ask")
        }

        return {
            "n": n,
            "h": self.h,
            "kind_counts": kind_counts,
            "mid_start": round(mid_start, 10),
            "mid_end": round(mid_end, 10),
            "mid_total_ticks": round(mid_end - mid_start, 10),
            "mid_abs_ticks": round(abs_path, 10),
            "spread_mean_ticks": round(float(np.mean(spreads)), 10) if spreads else 0.0,
            "n_spread_steps": len(spreads),
            "flow_total": round(float(sum(r.ofi for r in rec)), 10),
            "flow_abs_mean": round(float(np.mean([abs(r.ofi) for r in rec])), 10) if rec else 0.0,
            "queue_decay": queue_decay,
            "latency_attrition": latency,
            "car": self._car(mid_series),
        }

    def _touch_series(self, side: str) -> list[int]:
        return [int(self._initial[side])] + [
            int(getattr(r, f"{side}_touch")) for r in self.records
        ]

    def _ts_series(self) -> list[int]:
        return [int(self._initial["ts"])] + [int(r.ts_ns) for r in self.records]

    def _queue_decay(self, side: str) -> dict[str, Any]:
        """Per-event decay/growth of the touch queue a back-of-touch order joins.

        ``mean_attrition`` / ``mean_growth`` are the **per-event expectations**:
        summed over every pair with a live queue (flat steps count 0), i.e. how
        much of the ahead-queue each event eats/restocks on average.
        ``mean_eaten_severity`` is the conditional average over shrinking steps
        only (the size of a typical decay event, ignoring quiet steps).
        """
        series = self._touch_series(side)
        attrs: list[float] = []
        grows: list[float] = []
        pairs = eaten = 0
        for a, b in pairwise(series):
            if a <= 0:
                continue
            pairs += 1
            if b < a:
                attrs.append((a - b) / a)
                eaten += 1
            elif b > a:
                grows.append((b - a) / a)
        mean_attr = sum(attrs) / pairs if pairs else 0.0
        mean_grow = sum(grows) / pairs if pairs else 0.0
        first, last = series[0], series[-1]
        endured = float(last / first) if first > 0 else 0.0
        return {
            "n_pairs": pairs,
            "n_eaten": eaten,
            "mean_attrition": round(mean_attr, 10),
            "mean_growth": round(mean_grow, 10),
            "net_decay": round(mean_attr - mean_grow, 10),
            "mean_eaten_severity": round(float(np.mean(attrs)), 10) if attrs else 0.0,
            "endured_fraction": round(endured, 10),
            "consumed_fraction": round(1.0 - endured, 10),
            "initial": first,
            "final": last,
        }

    def _latency_attrition(self, side: str) -> dict[str, Any]:
        """The touch queue eaten per second of resting latency (wait), not per
        event: Σ attrition / Σ wall-clock, so a take that eats a whole queue in a
        millisecond counts as far higher attrition than one spread over minutes."""
        touches = self._touch_series(side)
        ts = self._ts_series()
        num = 0.0          # Σ attrition (fraction of the ahead-queue eaten)
        den_s = 0.0        # Σ wall-clock over the pairs used, seconds
        pairs = eaten = 0
        for a, b, t0, t1 in zip(touches, touches[1:], ts, ts[1:]):
            if a <= 0:
                continue
            pairs += 1
            att = max(0.0, (a - b) / a)
            num += att
            if b < a:
                eaten += 1
            if t1 > t0:
                den_s += (t1 - t0) / 1e9
        total_wait_s = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0
        rate = num / den_s if den_s > 0 else None
        return {
            "n_pairs": pairs,
            "n_eaten": eaten,
            "rate_s": round(rate, 10) if rate is not None else None,
            "halflife_s": round(math.log(2.0) / rate, 10) if rate and rate > 0 else None,
            "total_wait_s": round(max(0.0, total_wait_s), 10),
            "clock_cadence_s": round(den_s / pairs, 10) if pairs and den_s > 0 else None,
        }

    def _car(self, mid_series: list[float]) -> dict[str, Any]:
        """Conditional average response: mean ``h``-event mid move following each
        recorded step, conditioned on event kind and flow (OFI) terciles.

        Response is a purely forward label (closure-time only): for step ``k`` it
        is ``mid_series[k+h] - mid_series[k]``, read →h→ events. Steps whose either
        endpoint mid is 0 (empty-side gap) are skipped, never NaNs.
        """
        h, n = self.h, len(self.records)
        xs: list[float] = []
        resp: list[float] = []
        ks: list[EventType] = []
        for k in range(1, n - h + 1):
            m0, m1 = mid_series[k], mid_series[k + h]
            if m0 <= 0 or m1 <= 0:
                continue
            r0 = self.records[k - 1]
            xs.append(r0.ofi)
            resp.append(m1 - m0)
            ks.append(r0.kind)

        out: dict[str, Any] = {
            "h": h,
            "n": len(resp),
            "mean_response_ticks": round(float(np.mean(resp)), 10) if resp else 0.0,
            "rank_ic": round(float(rank_ic(xs, resp)), 10) if len(resp) >= 2 else 0.0,
        }

        by_kind: dict[str, dict[str, Any]] = {}
        for name in sorted({k.name for k in ks}):
            vals = [rv for rv, kd in zip(resp, ks) if kd.name == name]
            entry: dict[str, Any] = {
                "n": len(vals),
                "mean_ticks": round(float(np.mean(vals)), 10),
            }
            ci = _ci95(vals, self.seed)
            if ci is not None:
                entry["ci95"] = ci
            by_kind[name] = entry
        out["by_kind"] = by_kind
        out["by_flow_tercile"] = self._car_terciles(xs, resp)
        return out

    def _car_terciles(
        self, xs: list[float], resp: list[float]
    ) -> list[dict[str, Any]]:
        """Tercile-CAR table: mean response inside each flow (OFI) tercile.

        Terciles are the {⅓, ⅔} quantiles (deterministic). Ties collapse to fewer
        bins; empty bins are omitted; per-bin 95% CI via a seeded iid bootstrap
        (only for bins with ≥ 2 observations).
        """
        arr = np.asarray(xs, dtype=np.float64)
        if arr.size == 0:
            return []
        q = [float(v) for v in np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])]
        idx = np.searchsorted(q, arr, side="right")
        resp_arr = np.asarray(resp, dtype=np.float64)
        edges = [None] + q + [None]
        out: list[dict[str, Any]] = []
        for b in range(int(idx.max()) + 1):
            sel = idx == b
            if int(sel.sum()) == 0:
                continue
            bxs = arr[sel]
            bres = resp_arr[sel]
            entry: dict[str, Any] = {
                "bin": int(b),
                "lo": edges[b],
                "hi": edges[b + 1],
                "n": int(sel.sum()),
                "mean_x": round(float(bxs.mean()), 10),
                "mean_response_ticks": round(float(bres.mean()), 10),
            }
            ci = _ci95(list(bres), self.seed)
            if ci is not None:
                entry["ci95"] = ci
            out.append(entry)
        return out


def _ci95(values: Sequence[float], seed: int) -> dict[str, float] | None:
    """Seeded 95% bootstrap CI of the mean (None when < 2 obs)."""
    if len(values) < 2:
        return None
    b = bootstrap_ci(values, kind="iid", n_boot=1000, seed=seed)
    return {"lo": round(b["lo"], 10), "hi": round(b["hi"], 10)}


# ---------------------------------------------------------------------------
# OrderLevelTracker and Empirical Research Components (Person B)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrackedOrder:
    """One resting order and its queue context (all sizes in shares)."""

    order_id: int
    side: Side
    price: int
    size: int  # REMAINING displayed size (0 once the order has left the book)
    size0: int  # original size at placement
    ts_add: int
    idx_add: int
    ahead_at_add: int  # displayed size ahead in the FIFO when the order arrived
    ahead: int  # displayed size currently ahead (decreases as older orders leave)
    same_best_dist: int = 0  # ticks behind the same-side best at placement (<0 = new best)
    opp_dist: int = -1  # ticks to the opposite best at placement (-1 = no opposite side)
    filled: int = 0
    ts_first_fill: int | None = None
    idx_first_fill: int | None = None
    ts_end: int | None = None  # when the order left the book (fill-out / cancel)
    idx_end: int | None = None
    outcome: str = "resting"  # "resting" | "filled" (>=1 execution) | "cancelled" (never traded)


@dataclass(slots=True)
class _TrackedLevel:
    """FIFO queue at one (side, price): order ids in arrival order."""

    queue: list[int] = field(default_factory=list)
    size: int = 0


class OrderLevelTracker:
    """Track queue position of every resting order on an ITCH stream.

    Feed events in tape order via ``on_event`` (or the typed ``on_add`` /
    ``on_execute`` / ``on_cancel_delete`` / ``on_replace`` / ``on_trade``).
    ``ahead_qty(oid)`` / ``behind_qty(oid)`` answer the queue question for any
    live order; ``completed`` accumulates the (filled | cancelled) records the
    survival and logistic models consume.
    """

    def __init__(self) -> None:
        self.orders: dict[int, TrackedOrder] = {}
        self.levels: dict[tuple[int, int], _TrackedLevel] = {}
        # lazy-deletion heaps for O(log n) best-price lookups: bids keyed by -price
        self._heaps: dict[int, list[int]] = {int(Side.Bid): [], int(Side.Ask): []}
        self.completed: list[TrackedOrder] = []
        self.index = 0  # event counter (monotone tape position)
        self.stats = {"adds": 0, "executes": 0, "cancels": 0, "deletes": 0,
                      "replaces": 0, "trades": 0, "unknown_id": 0}

    # ------------------------------------------------------------------ api
    def on_event(self, ev: NormalizedEvent) -> None:
        k = ev.kind
        if k in (EventType.ADD, EventType.ADD_MPID):
            self.on_add(ev)
        elif k in (EventType.EXECUTE, EventType.EXECUTE_PX):
            self.on_execute(ev)
        elif k in (EventType.CANCEL, EventType.DELETE):
            self.on_cancel_delete(ev)
        elif k == EventType.REPLACE:
            self.on_replace(ev)
        elif k == EventType.TRADE:
            self.on_trade(ev)
        self.index += 1

    def on_add(self, ev: NormalizedEvent) -> TrackedOrder:
        self.stats["adds"] += 1
        side = Side(int(ev.side))
        price = int(ev.price_ticks)
        opp = Side.Ask if side == Side.Bid else Side.Bid
        same_best = self.best_price(side)
        opp_best = self.best_price(opp)
        sgn = 1 if side == Side.Bid else -1
        same_dist = sgn * (same_best - price) if same_best else 0
        opp_dist = sgn * (opp_best - price) if opp_best else -1
        key = (int(side), price)
        lvl = self.levels.get(key)
        if lvl is None:
            lvl = self.levels[key] = _TrackedLevel()
            heapq.heappush(self._heaps[int(side)], -price if side == Side.Bid else price)
        o = TrackedOrder(
            order_id=int(ev.order_id), side=side, price=price,
            size=int(ev.size), size0=int(ev.size), ts_add=int(ev.ts_ns), idx_add=self.index,
            ahead_at_add=lvl.size, ahead=lvl.size,
            same_best_dist=int(same_dist), opp_dist=int(opp_dist),
        )
        lvl.queue.append(o.order_id)
        lvl.size += o.size
        self.orders[o.order_id] = o
        return o

    def on_execute(self, ev: NormalizedEvent) -> None:
        self.stats["executes"] += 1
        o = self.orders.get(int(ev.order_id))
        if o is None:
            self.stats["unknown_id"] += 1
            return
        qty = min(o.size, int(ev.size))
        if o.filled == 0 and qty > 0:
            o.ts_first_fill = int(ev.ts_ns)
            o.idx_first_fill = self.index
        o.filled += qty
        self._shrink(o, qty)
        if o.size <= 0:
            self._finish(o, "filled", ev.ts_ns)

    def on_cancel_delete(self, ev: NormalizedEvent) -> None:
        o = self.orders.get(int(ev.order_id))
        if ev.kind == EventType.DELETE:
            self.stats["deletes"] += 1
        else:
            self.stats["cancels"] += 1
        if o is None:
            self.stats["unknown_id"] += 1
            return
        qty = o.size if ev.kind == EventType.DELETE or ev.size <= 0 else min(o.size, int(ev.size))
        self._shrink(o, qty)
        if o.size <= 0:
            # a fully-cancelled order that never traded is the competing risk
            self._finish(o, "cancelled" if o.filled == 0 else "filled", ev.ts_ns)

    def on_replace(self, ev: NormalizedEvent) -> None:
        """ITCH ``U``: old order leaves its queue, new id joins the back of the new level."""
        self.stats["replaces"] += 1
        old = self.orders.get(int(ev.order_id))
        if old is None:
            self.stats["unknown_id"] += 1
            return
        side = old.side
        self._shrink(old, old.size)
        self._finish(old, "cancelled" if old.filled == 0 else "filled", ev.ts_ns)
        new_ev = NormalizedEvent(
            EventType.ADD, ev.ts_ns, ev.new_order_id or ev.order_id + 1, side,
            ev.price_ticks, ev.size, raw_type="U",
        )
        self.on_add(new_ev)

    def on_trade(self, ev: NormalizedEvent) -> None:
        """``P`` prints hit hidden liquidity — no displayed queue changes."""
        self.stats["trades"] += 1

    # ------------------------------------------------------------- queries
    def ahead_qty(self, order_id: int) -> int:
        o = self.orders.get(int(order_id))
        return 0 if o is None else int(o.ahead)

    def behind_qty(self, order_id: int) -> int:
        o = self.orders.get(int(order_id))
        if o is None:
            return 0
        lvl = self.levels.get((int(o.side), o.price))
        if lvl is None:
            return 0
        return int(lvl.size - o.ahead - o.size)

    def level_size(self, side: Side, price: int) -> int:
        lvl = self.levels.get((int(side), int(price)))
        return 0 if lvl is None else int(lvl.size)

    def best_price(self, side: Side) -> int:
        """Best displayed price on ``side`` (0 if empty). Amortized O(log n)."""
        heap = self._heaps[int(side)]
        while heap:
            price = -heap[0] if side == Side.Bid else heap[0]
            lvl = self.levels.get((int(side), price))
            if lvl is not None and lvl.size > 0:
                return price
            heapq.heappop(heap)
        return 0

    # ------------------------------------------------------------ internals
    def _shrink(self, o: TrackedOrder, qty: int) -> None:
        if qty <= 0:
            return
        lvl = self.levels.get((int(o.side), o.price))
        o.size -= qty
        if lvl is None:
            return
        lvl.size -= qty
        # everything behind ``o`` in the FIFO has ``qty`` less ahead of it
        try:
            pos = lvl.queue.index(o.order_id)
        except ValueError:
            return
        for oid in lvl.queue[pos + 1 :]:
            other = self.orders.get(oid)
            if other is not None:
                other.ahead = max(0, other.ahead - qty)
        if o.size <= 0:
            lvl.queue.pop(pos)
            if lvl.size <= 0 and not lvl.queue:
                self.levels.pop((int(o.side), o.price), None)

    def _finish(self, o: TrackedOrder, outcome: str, ts: int) -> None:
        o.outcome = outcome
        o.ts_end = int(ts)
        o.idx_end = self.index
        self.completed.append(o)
        self.orders.pop(o.order_id, None)


# ---------------------------------------------------------------------------
# E5 — passive fill probability
# ---------------------------------------------------------------------------

def _fill_prob_survival_tracked(
    orders: Iterable[TrackedOrder],
    *,
    horizons: Sequence[float],
    clock: str = "events",
    cancel_is_censoring: bool = True,
) -> dict[str, Any]:
    """Kaplan–Meier P(fill by τ) with cancels as the competing risk.

    ``clock="events"`` measures time-to-outcome in tape events (``idx_end −
    idx_add``); ``"ns"`` uses the ITCH timestamp. With
    ``cancel_is_censoring=True`` a cancelled order is right-censored at its
    cancel time for the *fill* hazard — the standard competing-risk treatment
    that gives the fill probability *conditional on the order staying in the
    book*. Orders still resting at the end of the stream are censored at the
    stream end (their ``idx_end``/``ts_end`` is None → excluded; pass them
    through ``censor_open_orders`` first if you want them counted).

    Returns ``{"tau": [...], "survival": [...], "p_fill": [...], "n": int,
    "n_fill": int, "n_cancel": int, "median_fill_time": float | None}``.
    """
    orders = list(orders)
    times: list[float] = []
    fills: list[bool] = []
    n_cancel = 0
    for o in orders:
        if o.idx_end is None or o.ts_end is None:
            continue
        if o.outcome == "filled":
            # time-to-FIRST-fill: a partial fill is the event of interest
            i_end = o.idx_first_fill if o.idx_first_fill is not None else o.idx_end
            t_end = o.ts_first_fill if o.ts_first_fill is not None else o.ts_end
            t = float(i_end - o.idx_add) if clock == "events" else float(t_end - o.ts_add)
            times.append(t)
            fills.append(True)
            continue
        t = float(o.idx_end - o.idx_add) if clock == "events" else float(o.ts_end - o.ts_add)
        if o.outcome == "cancelled":
            n_cancel += 1
            if cancel_is_censoring:
                times.append(t)
                fills.append(False)
    t_arr = np.asarray(times, dtype=np.float64)
    f_arr = np.asarray(fills, dtype=bool)
    n = int(t_arr.size)
    cif = cumulative_incidence_competing_risks(orders, horizons=horizons, clock=clock)
    out: dict[str, Any] = {"tau": [float(h) for h in horizons], "n": n,
                           "n_fill": int(f_arr.sum()), "n_cancel": n_cancel,
                           "cif_fill": cif["cif_fill"], "cif_cancel": cif["cif_cancel"]}
    if n == 0:
        out.update({"survival": [1.0] * len(horizons), "p_fill": [0.0] * len(horizons),
                    "median_fill_time": None})
        return out
    order = np.argsort(t_arr, kind="mergesort")
    t_sorted = t_arr[order]
    f_sorted = f_arr[order]
    uniq = np.unique(t_sorted[f_sorted]) if f_sorted.any() else np.empty(0)
    surv_steps: list[tuple[float, float]] = []
    s = 1.0
    for t in uniq:
        at_risk = int(np.sum(t_sorted >= t))
        d = int(np.sum((t_sorted == t) & f_sorted))
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        surv_steps.append((float(t), s))
    surv = []
    for h in horizons:
        val = 1.0
        for t, sv in surv_steps:
            if t <= h:
                val = sv
            else:
                break
        surv.append(float(val))
    median = None
    for t, sv in surv_steps:
        if sv <= 0.5:
            median = t
            break
    out.update({"survival": surv, "p_fill": [1.0 - v for v in surv], "median_fill_time": median})
    return out


def cumulative_incidence_competing_risks(
    orders: Iterable[Any],
    *,
    horizons: Sequence[float],
    clock: str = "events",
) -> dict[str, Any]:
    """Compute the Cumulative Incidence Function (CIF) under competing risks (Aalen-Johansen).

    In an order book queue, an order lifecycle terminates either via execution (Fill)
    or via cancellation / deletion (Cancel). Treating cancellations as standard non-informative
    right-censoring under Kaplan-Meier estimates a counterfactual:
    "What would fill probability be if cancelling were prohibited?"
    Because cancellations are endogenous and frequent (94-98% of orders), Kaplan-Meier
    systematically overestimates the real-world probability of execution.

    The Aalen-Johansen estimator tracks Fill (cause 1) and Cancel (cause 2) as distinct
    competing endpoints alongside right-censored resting orders (cause 0).
    The resulting CIF satisfy the exact identity:
        CIF_fill(t) + CIF_cancel(t) = 1 - S(t)
    and CIF_fill(t) <= P_KM(t).

    Returns:
        {
            "tau": list[float],
            "cif_fill": list[float],
            "cif_cancel": list[float],
            "survival": list[float],
            "n": int,
            "n_fill": int,
            "n_cancel": int,
            "n_censored": int,
        }
    """
    times: list[float] = []
    events: list[int] = []

    for o in orders:
        if isinstance(o, TrackedOrder) or (hasattr(o, "outcome") and hasattr(o, "idx_add")):
            if o.idx_end is None and o.ts_end is None:
                continue
            if o.outcome == "filled":
                i_end = o.idx_first_fill if o.idx_first_fill is not None else o.idx_end
                t_end = o.ts_first_fill if o.ts_first_fill is not None else o.ts_end
                t = float(i_end - o.idx_add) if clock == "events" else float(t_end - o.ts_add)
                times.append(max(0.0, t))
                events.append(1)
            elif o.outcome == "cancelled":
                t = float(o.idx_end - o.idx_add) if clock == "events" else float(o.ts_end - o.ts_add)
                times.append(max(0.0, t))
                events.append(2)
            else:
                t = float(o.idx_end - o.idx_add) if clock == "events" else float(o.ts_end - o.ts_add)
                times.append(max(0.0, t))
                events.append(0)
        elif isinstance(o, OrderLife) or (hasattr(o, "end_kind") and hasattr(o, "born_index")):
            if o.end_index is None:
                continue
            t = float(o.end_index - o.born_index)
            times.append(max(0.0, t))
            if o.end_kind == "fill":
                events.append(1)
            elif o.end_kind in ("cancel", "delete", "replace"):
                events.append(2)
            else:
                events.append(0)
        elif isinstance(o, (tuple, list)) and len(o) >= 2:
            t = float(o[0])
            times.append(max(0.0, t))
            ev = o[1]
            if ev in (1, "fill", "filled"):
                events.append(1)
            elif ev in (2, "cancel", "cancelled", "delete", "replace"):
                events.append(2)
            else:
                events.append(0)
        elif isinstance(o, dict) and "time" in o:
            t = float(o["time"])
            times.append(max(0.0, t))
            ev = o.get("event", o.get("outcome", o.get("status", 0)))
            if ev in (1, "fill", "filled"):
                events.append(1)
            elif ev in (2, "cancel", "cancelled", "delete", "replace"):
                events.append(2)
            else:
                events.append(0)

    t_arr = np.asarray(times, dtype=np.float64)
    ev_arr = np.asarray(events, dtype=np.int64)
    n = int(t_arr.size)
    n_fill = int(np.sum(ev_arr == 1))
    n_cancel = int(np.sum(ev_arr == 2))
    n_censored = int(np.sum(ev_arr == 0))

    hz = [float(h) for h in horizons]
    out: dict[str, Any] = {
        "tau": hz,
        "n": n,
        "n_fill": n_fill,
        "n_cancel": n_cancel,
        "n_censored": n_censored,
    }

    if n == 0:
        out.update({
            "cif_fill": [0.0] * len(hz),
            "cif_cancel": [0.0] * len(hz),
            "survival": [1.0] * len(hz),
        })
        return out

    ev_mask = (ev_arr == 1) | (ev_arr == 2)
    uniq_times = np.sort(np.unique(t_arr[ev_mask])) if ev_mask.any() else np.empty(0)

    s_cur = 1.0
    cif1_cur = 0.0
    cif2_cur = 0.0
    steps: list[tuple[float, float, float, float]] = []

    for t in uniq_times:
        at_risk = int(np.sum(t_arr >= t))
        if at_risk <= 0:
            continue
        d1 = int(np.sum((t_arr == t) & (ev_arr == 1)))
        d2 = int(np.sum((t_arr == t) & (ev_arr == 2)))
        d = d1 + d2

        h1 = d1 / at_risk
        h2 = d2 / at_risk
        cif1_cur += s_cur * h1
        cif2_cur += s_cur * h2
        s_cur *= (1.0 - d / at_risk)

        steps.append((float(t), s_cur, cif1_cur, cif2_cur))

    cif_fill_list = []
    cif_cancel_list = []
    survival_list = []

    for h in hz:
        s_val = 1.0
        c1_val = 0.0
        c2_val = 0.0
        for t, s_st, c1_st, c2_st in steps:
            if t <= h:
                s_val = s_st
                c1_val = c1_st
                c2_val = c2_st
            else:
                break
        survival_list.append(float(s_val))
        cif_fill_list.append(float(c1_val))
        cif_cancel_list.append(float(c2_val))

    out.update({
        "cif_fill": cif_fill_list,
        "cif_cancel": cif_cancel_list,
        "survival": survival_list,
    })
    return out




def fill_prob_survival(
    orders_or_levels: Any,
    *,
    horizons: Sequence[float] | None = None,
    clock: str = "events",
    cancel_is_censoring: bool = True,
    hazard_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Kaplan-Meier survival of the time-to-fill event.

    Supports both:
    1. OrderLife sequence from standing_order_lifetimes (with optional hazard_fn)
    2. TrackedOrder iterable from OrderLevelTracker (with horizons, clock, cancel_is_censoring)
    """
    if horizons is not None:
        return _fill_prob_survival_tracked(
            orders_or_levels,
            horizons=horizons,
            clock=clock,
            cancel_is_censoring=cancel_is_censoring,
        )
    first = None
    if isinstance(orders_or_levels, Sequence):
        if len(orders_or_levels) > 0:
            first = orders_or_levels[0]
    elif isinstance(orders_or_levels, Iterable):
        orders_or_levels = list(orders_or_levels)
        if len(orders_or_levels) > 0:
            first = orders_or_levels[0]
    if isinstance(first, TrackedOrder):
        return _fill_prob_survival_tracked(
            orders_or_levels,
            horizons=[1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
            clock=clock,
            cancel_is_censoring=cancel_is_censoring,
        )
    return _fill_prob_survival_lifetimes(orders_or_levels, hazard_fn=hazard_fn)

def censor_open_orders(tracker: OrderLevelTracker, ts_end: int) -> list[TrackedOrder]:
    """Snapshot copies of still-resting orders, censored at ``ts_end`` / current index."""
    out: list[TrackedOrder] = []
    for o in tracker.orders.values():
        c = TrackedOrder(**{f: getattr(o, f) for f in TrackedOrder.__slots__})
        c.ts_end = int(ts_end)
        c.idx_end = tracker.index
        c.outcome = "cancelled"  # censored: contributes to the at-risk set only
        out.append(c)
    return out


def order_features(o: TrackedOrder, *, level_size_at_add: int | None = None) -> dict[str, float]:
    """Placement-time features for the logistic fill model (all observable at add).

    ``same_best_dist`` / ``opp_dist`` are signed-log-compressed tick distances
    (an order deep in the book fills far less often than one at the touch).
    """
    ahead = float(o.ahead_at_add)
    lvl = float(level_size_at_add if level_size_at_add is not None else o.ahead_at_add + o.size0)
    d_same = float(o.same_best_dist)
    d_opp = float(o.opp_dist)
    return {
        "log_ahead": float(np.log1p(ahead)),
        "log_size": float(np.log1p(o.size0)),
        "queue_frac": ahead / max(1.0, lvl),
        "same_best_dist": float(np.sign(d_same) * np.log1p(abs(d_same))),
        "opp_dist": float(np.log1p(d_opp)) if d_opp >= 0 else 0.0,
        "no_opposite": 1.0 if d_opp < 0 else 0.0,
    }


def logistic_fill_model(
    features: Mapping[str, Sequence[float]] | np.ndarray,
    filled: Sequence[bool] | Sequence[int],
    *,
    l2: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-8,
    holdout: float = 0.3,
) -> dict[str, Any]:
    """Fit P(fill | features) by ridge-regularized logistic regression (Newton).

    Rows are time-ordered orders; the LAST ``holdout`` fraction is the
    out-of-sample block (walk-forward, never shuffled). Reports the
    out-of-sample **Brier score**, the **calibration slope** (logit of the
    prediction regressed on the outcome — 1.0 = perfectly calibrated, < 0.8 is
    the E5 failure criterion), and a decile calibration table.
    """
    if isinstance(features, np.ndarray):
        X = np.asarray(features, dtype=np.float64)
        names = [f"x{j}" for j in range(X.shape[1])]
    else:
        names = list(features.keys())
        X = np.column_stack([np.asarray(features[n], dtype=np.float64) for n in names])
    y = np.asarray(filled, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("features rows must match filled length")
    n = y.size
    if n < 20:
        raise ValueError("logistic_fill_model needs at least 20 orders")
    n_train = max(10, int(n * (1.0 - holdout)))
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd[sd <= 1e-12] = 1.0
    Z = (X - mu) / sd
    D = np.column_stack([np.ones(n), Z])
    Dtr, ytr = D[:n_train], y[:n_train]
    w = np.zeros(D.shape[1])
    reg = np.full(D.shape[1], l2)
    reg[0] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(Dtr @ w)
        g = Dtr.T @ (p - ytr) + reg * w
        W = p * (1.0 - p)
        H = (Dtr * W[:, None]).T @ Dtr + np.diag(reg)
        step = np.linalg.solve(H + 1e-9 * np.eye(H.shape[0]), g)
        w -= step
        if float(np.max(np.abs(step))) < tol:
            break
    p_all = _sigmoid(D @ w)
    p_te, y_te = p_all[n_train:], y[n_train:]
    brier = float(np.mean((p_te - y_te) ** 2)) if y_te.size else float("nan")
    base_rate = float(ytr.mean())
    brier_base = float(np.mean((base_rate - y_te) ** 2)) if y_te.size else float("nan")
    slope = _calibration_slope(p_te, y_te) if y_te.size >= 10 else float("nan")
    return {
        "coefs": dict(zip(["intercept", *names], w.tolist())),
        "n_train": int(n_train),
        "n_test": int(y_te.size),
        "base_rate": base_rate,
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill": (1.0 - brier / brier_base) if brier_base > 0 else float("nan"),
        "calibration_slope": slope,
        "calibration_table": calibration_table(p_te, y_te) if y_te.size else [],
        "predict": lambda Xnew: _sigmoid(
            np.column_stack([np.ones(len(Xnew)), (np.asarray(Xnew, dtype=np.float64) - mu) / sd]) @ w
        ),
    }


def _calibration_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Slope of a logistic recalibration ``y ~ a + b·logit(p)`` (Cox 1958). 1.0 = calibrated."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if np.std(lp) <= 1e-12:
        return float("nan")
    D = np.column_stack([np.ones(p.size), lp])
    w = np.zeros(2)
    for _ in range(50):
        q = _sigmoid(D @ w)
        g = D.T @ (q - y)
        H = (D * (q * (1 - q))[:, None]).T @ D + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return float(w[1])


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Decile calibration: mean predicted vs realized fill rate per prediction bin."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    rows = []
    for chunk in np.array_split(order, min(bins, p.size)):
        if chunk.size == 0:
            continue
        rows.append({"n": int(chunk.size), "pred": float(p[chunk].mean()), "realized": float(y[chunk].mean())})
    return rows


def brier_score(p: Sequence[float], y: Sequence[float]) -> float:
    pa = np.asarray(p, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if pa.size != ya.size or pa.size == 0:
        return float("nan")
    return float(np.mean((pa - ya) ** 2))


@dataclass(frozen=True, slots=True)
class EmpiricalQueueHazard:
    """Pre-calibrated empirical queue hazard representation for online simulation.

    Encapsulates calibrated fill hazard rates or survival probabilities
    derived from historical ITCH queue dynamics (e.g. Kaplan-Meier or logistic models),
    ensuring strictly observable features at placement time without lookahead.
    """

    base_logit: float = -1.2
    queue_coef: float = -1.5
    dist_coef: float = -0.8
    horizons: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0)
    p_fill_by_horizon: tuple[float, ...] = (0.05, 0.18, 0.35, 0.58)

    def predict_fill_prob(
        self,
        queue_ahead: int,
        level_size: int = 0,
        distance_ticks: int = 0,
    ) -> float:
        """Predict within-step fill probability using observable placement-time state.

        Zero lookahead: inputs are strictly known at decision time t before subsequent prints.
        """
        lvl = max(1, int(level_size))
        frac_ahead = float(np.clip(int(queue_ahead) / lvl, 0.0, 1.0))
        dist = max(0, int(distance_ticks))
        z = self.base_logit + self.queue_coef * frac_ahead + self.dist_coef * dist
        if z >= 0.0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = float(np.exp(z))
        return float(ez / (1.0 + ez))

    @classmethod
    def from_tracker(
        cls,
        tracker: OrderLevelTracker,
        *,
        horizons: Sequence[float] = (1.0, 5.0, 10.0, 20.0),
    ) -> EmpiricalQueueHazard:
        """Calibrate hazard representation from an empirical OrderLevelTracker stream."""
        completed = tracker.completed
        if not completed:
            return cls(horizons=tuple(horizons))
        surv = _fill_prob_survival_tracked(completed, horizons=horizons)
        p_fills = tuple(surv.get("p_fill", [0.0] * len(horizons)))
        n_tot = max(1, len(completed))
        n_filled = sum(1 for o in completed if o.outcome == "filled")
        fill_rate = max(0.01, min(0.99, n_filled / n_tot))
        base_logit = float(np.log(fill_rate / (1.0 - fill_rate)))
        return cls(
            base_logit=base_logit,
            horizons=tuple(horizons),
            p_fill_by_horizon=p_fills,
        )

    @classmethod
    def from_survival_dict(
        cls,
        surv: dict[str, Any],
    ) -> EmpiricalQueueHazard:
        """Calibrate hazard representation from a pre-computed survival dictionary."""
        tau = tuple(float(h) for h in surv.get("tau", (1.0, 5.0, 10.0, 20.0)))
        p_fills = tuple(float(p) for p in surv.get("p_fill", (0.05, 0.18, 0.35, 0.58)))
        base_p = p_fills[0] if p_fills else 0.1
        base_p = max(0.01, min(0.99, base_p))
        base_logit = float(np.log(base_p / (1.0 - base_p)))
        return cls(
            base_logit=base_logit,
            horizons=tau,
            p_fill_by_horizon=p_fills,
        )
