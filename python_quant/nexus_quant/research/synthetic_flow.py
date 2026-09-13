"""Synthetic order-level ITCH tape generator (Phase 3).

Emits a seeded, deterministic stream of ``NormalizedEvent`` objects from an
internal **price-time book**, so the tape has a ground truth the research
layers (``queue_dynamics.QueueTracker``, ``replay.ReplayEngine``) can be
self-consistency-tested against.

Why this exists
---------------
The Phase-3 research questions (P(fill), adverse selection) are about *orders in
a queue*. ``StubOrderBook`` is aggregate-only (a price level stores total size +
order count), so it cannot be the source for a queue study. Real NASDAQ ITCH is
order-level — every add/execute/cancel names an ``order_id``. This module is
the order-level stand-in: the same ``NormalizedEvent`` vocabulary a real
``iter_itch_events(path)`` tape provides, so Phase-4 swaps in a real tape with
**zero redesign** of the tracker/adverse-selection consumers.

Honesty contract
----------------
* ``FlowConfig`` defaults describe a **symmetric random-walk tape** (the null).
  ``FLOW_PRESETS[rw/drift/revert/stress]`` are documented, no-target control
  flows — they exist to *test that the analytics detect structure when it is
  present*, never to manufacture alpha.
* A take of taker side ``X`` walks the *opposite* resting book and emits exactly
  one ``EXECUTE(oid, consumed)`` per consumed resting order (partial fills
  supported) — the same semantics as ``StubBookAdapter.take``.
* Events are emitted strictly in order, oids are unique and monotonic, and an
  ``EXECUTE`` always references a resting order with ``consumed <= remaining``
  — the tape is coherent by construction (asserted by tests).
* ``SyntheticFlow`` advances **one event at a time** (``step_gen()``) so a
  tracker can consume each event and compare its own reconstruction against
  ``internal_book()`` in lockstep (the self-consistency test).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..book_state import Side
from ..itch_parser import EventType, NormalizedEvent

# Per-event additive timestamps (ns since midnight start). Bounded so ts_ns is
# strictly monotone but looks like a real ITCH cadence (≈1–5 ms apart).
_TS_LO, _TS_HI = 1_000, 5_000_000


@dataclass(frozen=True)
class FlowConfig:
    """Seeded synthetic tape parameters. Defaults = symmetric random walk."""

    n_events: int = 50_000
    seed: int = 0x51ED
    center0: float = 15_000.0      # initial mid center around which adds are placed
    anchor: float = 15_000.0       # mean-reversion anchor (used when mean_revert > 0)
    book_depth: int = 10           # adds are placed up to `book_depth` ticks off center
    # --- center process (ticks) ---
    vol_per_event: float = 1.0     # std of the center random walk per event
    drift_ticks: float = 0.0       # per-event upward drift of the center
    mean_revert: float = 0.0       # OU pull coefficient back to `anchor`
    # --- flow mix (probabilities per non-gap event, need not sum to 1) ---
    take_intensity: float = 0.06   # P(take) — marketable flow consuming resting orders
    cancel_intensity: float = 0.08 # P(cancel/delete of a resting order)
    replace_prob: float = 0.02     # P(replace of a resting order)
    cancel_partial_p: float = 0.5  # among cancels, P(partial CANCEL) vs P(delete)
    side_bias: float = 0.0         # in [-1,1]; >0 → taker/add side leans Ask (selling)
    # --- sizes (shares) ---
    add_size_min: int = 10
    add_size_max: int = 220
    take_size_min: int = 40
    take_size_max: int = 200
    # --- gap events (rare large takes) ---
    gap_prob: float = 0.0
    gap_size_min: int = 600
    gap_size_max: int = 1800

    def __post_init__(self) -> None:
        if not 0 < self.book_depth <= 50:
            raise ValueError(f"book_depth must be in (0, 50], got {self.book_depth}")
        if not -1.0 <= self.side_bias <= 1.0:
            raise ValueError(f"side_bias must be in [-1, 1], got {self.side_bias}")
        if self.take_intensity < 0 or self.cancel_intensity < 0 or self.replace_prob < 0:
            raise ValueError("flow intensities must be >= 0")


@dataclass(frozen=True)
class FLOW_PRESETS:
    """Documented control flows (no tuning to any target)."""

    rw: FlowConfig = FlowConfig()
    drift: FlowConfig = FlowConfig(
        drift_ticks=0.6, vol_per_event=0.6, n_events=20_000
    )
    revert: FlowConfig = FlowConfig(
        center0=15_400.0,
        anchor=15_000.0,
        mean_revert=0.02,
        vol_per_event=1.0,
        drift_ticks=0.0,
        n_events=20_000,
    )
    stress: FlowConfig = FlowConfig(
        take_intensity=0.16,
        take_size_min=120,
        take_size_max=500,
        gap_prob=0.01,
        gap_size_min=1000,
        gap_size_max=2600,
        cancel_intensity=0.12,
        add_size_min=8,
        add_size_max=90,
        vol_per_event=2.0,
        n_events=20_000,
    )


@dataclass(slots=True)
class _OrderIdx:
    """Price-time index for one (side, price) level.

    ``order2size`` preserves insertion order (add-time); ``fifo`` is the same
    order as a deque so the front order is O(1) to pop. Cancels/executes target
    a specific oid and modify the dict in place; the front oid is removed only
    once its size hits zero.
    """

    order2size: dict[int, int] = field(default_factory=dict)
    fifo: deque[int] = field(default_factory=deque)

    def total(self) -> int:
        return sum(self.order2size.values())

    def front(self) -> int | None:
        return self.fifo[0] if self.fifo else None

    def reduce_front(self, qty: int) -> tuple[int, int]:
        """Consume ``qty`` from the front order in FIFO order.

        Returns (consumed, remaining). If the front order is only partially
        consumed it stays in place (priority preserved); otherwise it pops.
        """
        oid = self.fifo[0]
        now = self.order2size.get(oid, 0)
        consumed = min(qty, now)
        left = now - consumed
        if left <= 0:
            self.order2size.pop(oid, None)
            self.fifo.popleft()
        else:
            self.order2size[oid] = left
        return consumed, left


class SyntheticFlow:
    """Generate a seeded order-level ITCH tape, exposing its internal book."""

    def __init__(self, cfg: FlowConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else FlowConfig()
        self._rng: np.random.Generator | None = None
        self.events: list[NormalizedEvent] = []
        self.mid_trace: list[float] = []
        self._book: dict[Side, dict[int, _OrderIdx]] = {Side.Bid: {}, Side.Ask: {}}
        self._next_oid = 1
        self._ts = 0
        self.center = float(self.cfg.center0)
        self._emitted = 0
        self._seeded = False

    def reset(self, seed: int | None = None) -> "SyntheticFlow":
        if seed is not None:
            self.cfg = FlowConfig(**{**self.cfg.__dict__, "seed": int(seed)})
        self._rng = np.random.default_rng(self.cfg.seed)
        self.events = []
        self.mid_trace = []
        self._book = {Side.Bid: {}, Side.Ask: {}}
        self._next_oid = 1
        self._ts = 0
        self.center = float(self.cfg.center0)
        self._emitted = 0
        self._seeded = False
        return self

    # ------------------------------------------------------------------ #
    # tape generation — one event at a time (lockstep-friendly)          #
    # ------------------------------------------------------------------ #
    def step_gen(self) -> NormalizedEvent | None:
        """Emit and apply the next event, or ``None`` when the tape is done.

        The seed book is emitted as real ``ADD`` events at the head of the tape
        (so a consumer starting from an empty book reconstructs identically).
        """
        if self._rng is None:
            self.reset()
        if not self._seeded:
            self._seed_book()  # appends its own ADD events
            self._seeded = True
        if self._emitted >= self.cfg.n_events:
            return None
        self._advance_center()
        ev = self._emit_one()
        self._normalize_book()  # prune stale levels, top up near-center depth
        self._emitted += 1
        self.mid_trace.append(self._mid())
        return ev

    def generate(self) -> list[NormalizedEvent]:
        """Consume the whole tape into ``self.events`` and return it."""
        self.reset()
        while self.step_gen() is not None:
            pass
        return self.events

    # --- seeding + the four mutation kinds ----------------------------- #
    def _seed_book(self) -> None:
        r = self.cfg
        for i in range(1, r.book_depth + 1):
            for side, sign in ((Side.Bid, -1), (Side.Ask, +1)):
                for _ in range(int(self._rng.integers(1, 4))):
                    size = int(self._rng.integers(r.add_size_min, r.add_size_max + 1))
                    px = round(self.center) + sign * i
                    self._add_event(side, px, size)

    def _advance_center(self) -> None:
        r = self._rng
        rw = float(r.normal(0.0, self.cfg.vol_per_event))
        revert = self.cfg.mean_revert * (self.cfg.anchor - self.center)
        self.center = self.center + self.cfg.drift_ticks + rw + revert

    def _emit_one(self) -> NormalizedEvent:
        r = self.cfg
        roll = float(self._rng.random())
        if r.gap_prob > 0 and roll < r.gap_prob:
            return self._take(int(self._rng.integers(r.gap_size_min, r.gap_size_max + 1)))
        if roll < r.gap_prob + r.take_intensity:
            return self._take(int(self._rng.integers(r.take_size_min, r.take_size_max + 1)))
        if roll < r.gap_prob + r.take_intensity + r.cancel_intensity:
            return self._cancel_or_delete()
        if roll < r.gap_prob + r.take_intensity + r.cancel_intensity + r.replace_prob:
            return self._replace()
        return self._add()

    def _add(self) -> NormalizedEvent:
        r = self.cfg
        side = self._side(self._side_p())
        offset = int(self._rng.integers(1, r.book_depth + 1))
        px = round(self.center) + (-offset if side == Side.Bid else +offset)
        size = int(self._rng.integers(r.add_size_min, r.add_size_max + 1))
        return self._add_event(side, px, size)

    def _add_event(self, side: Side, px: int, size: int) -> NormalizedEvent:
        oid = self._next_oid
        self._next_oid += 1
        self._insert(side, px, oid, size)
        return self._append(EventType.ADD, oid, side, px, size, raw_type="A")

    def _take(self, size: int) -> NormalizedEvent:
        """An aggressive taker walks the opposite book; emits one EXECUTE/order."""
        taker = self._side(self._side_p())
        passive = Side.Ask if taker == Side.Bid else Side.Bid
        remaining = int(size)
        prices = sorted(self._book[passive].keys(), reverse=(passive == Side.Bid))
        last: NormalizedEvent | None = None
        for px in prices:
            if remaining <= 0:
                break
            idx = self._book[passive][px]
            while remaining > 0 and idx.fifo:
                oid = idx.front()
                assert oid is not None
                consumed, _left = idx.reduce_front(remaining)
                remaining -= consumed
                last = self._append(EventType.EXECUTE, oid, Side.NONE, 0, consumed,
                                    raw_type="E")
        if last is None:
            # Nothing resting on the passive side to hit — fall back to an add.
            return self._add()
        return last

    def _cancel_or_delete(self) -> NormalizedEvent:
        target = self._pick_order()
        if target is None:
            return self._add()
        side, px, oid, size = target
        partial = float(self._rng.random()) < self.cfg.cancel_partial_p
        if partial and size > 1:
            cut = max(1, int(self._rng.integers(1, size + 1)))
            self._reduce(side, px, oid, cut)
            return self._append(EventType.CANCEL, oid, Side.NONE, 0, cut, raw_type="X")
        self._remove(side, px, oid)
        return self._append(EventType.DELETE, oid, Side.NONE, 0, 0, raw_type="D")

    def _replace(self) -> NormalizedEvent:
        target = self._pick_order()
        if target is None:
            return self._add()
        side, px, oid, size = target
        r = self.cfg
        new_size = max(1, int(self._rng.integers(1, size + 1)))
        new_px = px
        if float(self._rng.random()) < 0.5:
            off = int(self._rng.integers(1, r.book_depth + 1))
            new_px = round(self.center) + (-off if side == Side.Bid else +off)
        new_id = self._next_oid
        self._next_oid += 1
        self._remove(side, px, oid)
        self._insert(side, new_px, new_id, new_size)
        return self._append(EventType.REPLACE, oid, Side.NONE, new_px, new_size,
                            new_order_id=new_id, raw_type="U")

    # --- small helpers -------------------------------------------------- #
    def _side_p(self) -> float:
        return 0.5 + self.cfg.side_bias / 2.0

    def _side(self, p_ask: float) -> Side:
        return Side.Ask if float(self._rng.random()) < p_ask else Side.Bid

    def _insert(self, side: Side, px: int, oid: int, size: int) -> None:
        idx = self._book[side].setdefault(px, _OrderIdx())
        idx.order2size[oid] = size
        idx.fifo.append(oid)

    def _reduce(self, side: Side, px: int, oid: int, cut: int) -> None:
        idx = self._book[side].get(px)
        if idx is None or oid not in idx.order2size:
            return
        now = idx.order2size[oid] - cut
        if now <= 0:
            self._remove(side, px, oid)
        else:
            idx.order2size[oid] = now

    def _remove(self, side: Side, px: int, oid: int) -> None:
        idx = self._book[side].get(px)
        if idx is None:
            return
        idx.order2size.pop(oid, None)
        if oid in idx.fifo:
            idx.fifo.remove(oid)  # O(level length) — levels are small
        if not idx.fifo:
            self._book[side].pop(px, None)

    def _pick_order(self) -> tuple[Side, int, int, int] | None:
        side = self._side(0.5)
        if not self._book[side]:
            side = Side.Ask if side == Side.Bid else Side.Bid
        by_px = self._book[side]
        if not by_px:
            return None
        px = int(self._rng.choice(tuple(by_px.keys())))
        idx = by_px[px]
        oids = tuple(idx.order2size.keys())
        if not oids:
            return None
        oid = int(self._rng.choice(oids))
        return side, px, oid, int(idx.order2size[oid])

    def _append(self, kind: EventType, oid: int, side: Side, px: int, size: int,
                *, new_order_id: int = 0, raw_type: str = "") -> NormalizedEvent:
        self._ts += int(self._rng.integers(_TS_LO, _TS_HI))
        ev = NormalizedEvent(
            kind=kind, ts_ns=self._ts, order_id=oid, side=side,
            price_ticks=px, size=size, new_order_id=new_order_id,
            raw_type=raw_type,
        )
        self.events.append(ev)
        return ev

    def _normalize_book(self) -> None:
        """Keep the book a coherent band around the moving center.

        Two self-healing steps that mirror real liquidity-provider behaviour
        and keep the tape free of crossed/stale books even under drift/revert:

        * **prune** any level further than ``book_depth+1`` ticks from
          ``round(center)`` (emit ``DELETE`` per resting order — liquidity
          withdrew as the price ran away);
        * **top up** missing near-center levels with 1–2 small orders (emit
          ``ADD``), so the displayed depth is ~``book_depth`` on each side and
          queue studies always have a real queue to reason about.

        Emitted replenishment is deterministic (seeded RNG) and part of the
        tape, so the tracker's reconstruction stays in lockstep.
        """
        r = self.cfg
        c = round(self.center)
        # Levels age out only once the center has moved them well outside the
        # displayed band; the slack scales with per-event volatility so a calm
        # tape doesn't churn its own book away.
        limit = r.book_depth + 1 + int(12.0 * r.vol_per_event)
        for side, sign in ((Side.Bid, -1), (Side.Ask, +1)):
            book = self._book[side]
            # Drop levels a take fully drained (from `reduce_front`) — they are
            # no longer resting liquidity and must not pollute bbo/reconstruct.
            drained = [px for px, idx in book.items() if not idx.fifo]
            for px in drained:
                book.pop(px, None)
            stale = [px for px in book if abs(px - c) > limit]
            for px in stale:
                idx = book[px]
                for oid in list(idx.order2size.keys()):
                    self._remove(side, px, oid)
                    self._append(EventType.DELETE, oid, Side.NONE, 0, 0, raw_type="D")
            for i in range(1, r.book_depth + 1):
                px = c + sign * i
                if px in book:
                    continue
                for _ in range(int(self._rng.integers(1, 3))):
                    size = int(self._rng.integers(r.add_size_min, min(r.add_size_max, 120) + 1))
                    self._add_event(side, px, size)

    # ------------------------------------------------------------------ #
    # introspection (ground truth for the self-consistency test)         #
    # ------------------------------------------------------------------ #
    def internal_book(self) -> dict[Side, dict[int, list[tuple[int, int]]]]:
        """Live FIFO-serialized ground truth: (side) -> {price: [(oid, size)...]}."""
        out: dict[Side, dict[int, list[tuple[int, int]]]] = {Side.Bid: {}, Side.Ask: {}}
        for side in (Side.Bid, Side.Ask):
            for px, idx in sorted(self._book[side].items()):
                if not idx.fifo:
                    continue  # drained levels carry no resting orders
                out[side][px] = [(oid, idx.order2size[oid]) for oid in idx.fifo]
        return out

    def bbo(self) -> tuple[int, int]:
        """Best (bid, ask), 0 for an empty side."""
        bb = max((px for px, idx in self._book[Side.Bid].items() if idx.fifo), default=0)
        ba = min((px for px, idx in self._book[Side.Ask].items() if idx.fifo), default=0)
        return int(bb), int(ba)

    def _mid(self) -> float:
        bb, ba = self.bbo()
        if not bb or not ba:
            return float(self.center)
        return (bb + ba) / 2.0