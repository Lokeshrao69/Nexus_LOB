"""ITCH → L2 replay against an injectable book.

The replay engine never constructs a C++ Engine. It talks to
``StubBookAdapter`` today; ``adapt(nexus_engine.Engine())`` is the swap
point once Person A's module is built.

``view()`` results are not retained across ``step()``. Callers that need a
durable ladder use ``snapshot()``.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .book_port import StubBookAdapter, View, adapt
from .book_state import DEPTH, Side, StubOrderBook
from .itch_parser import EventType, NormalizedEvent


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    detail: str


def check_integrity(state: View) -> list[IntegrityIssue]:
    """BBO / size / sort checks on a view or snapshot dict."""
    issues: list[IntegrityIssue] = []
    bid_px = state["bid_px"]
    ask_px = state["ask_px"]
    bid_sz = state["bid_sz"]
    ask_sz = state["ask_sz"]
    bb = int(bid_px[0]) if len(bid_px) else 0
    ba = int(ask_px[0]) if len(ask_px) else 0
    if bb and ba:
        if bb > ba:
            issues.append(IntegrityIssue("crossed", f"best bid {bb} > best ask {ba}"))
        elif bb == ba:
            issues.append(IntegrityIssue("locked", f"locked market at {bb}"))
    elif not bb and not ba:
        issues.append(IntegrityIssue("empty_bbo", "both sides empty"))
    for i in range(DEPTH):
        if int(bid_sz[i]) < 0 or int(ask_sz[i]) < 0:
            issues.append(IntegrityIssue("neg_size", f"negative size at level {i}"))
    for i in range(1, DEPTH):
        if int(bid_px[i]) and int(bid_px[i - 1]) and int(bid_px[i]) >= int(bid_px[i - 1]):
            issues.append(IntegrityIssue("unsorted", f"bids not descending at {i}"))
        if int(ask_px[i]) and int(ask_px[i - 1]) and int(ask_px[i]) <= int(ask_px[i - 1]):
            issues.append(IntegrityIssue("unsorted", f"asks not ascending at {i}"))
    return issues


def best_bid(state: View) -> int:
    return int(state["bid_px"][0])


def best_ask(state: View) -> int:
    return int(state["ask_px"][0])


def spread_ticks(state: View) -> int | None:
    bb, ba = best_bid(state), best_ask(state)
    if not bb or not ba:
        return None
    return ba - bb


@dataclass
class ReplayFrame:
    index: int
    event: NormalizedEvent
    applied: bool
    issues: list[IntegrityIssue]
    state: View


@dataclass
class ReplayEngine:
    events: Iterable[NormalizedEvent]
    book: StubBookAdapter = field(default_factory=StubBookAdapter)

    def __post_init__(self) -> None:
        self.book = adapt(self.book)  # type: ignore[assignment]
        self._events = list(self.events) if not hasattr(self.events, "__next__") else None
        self._iter = iter(self.events) if self._events is None else iter(self._events)
        self.index = 0
        self.applied = 0
        self.skipped = 0
        self.last_issues: list[IntegrityIssue] = []

    @classmethod
    def from_stub(
        cls, events: Iterable[NormalizedEvent], stub: StubOrderBook | None = None
    ) -> ReplayEngine:
        return cls(events, StubBookAdapter(stub or StubOrderBook()))

    def step(self) -> ReplayFrame | None:
        try:
            ev = next(self._iter)
        except StopIteration:
            return None
        ok = self.apply(ev)
        if ok:
            self.applied += 1
        else:
            self.skipped += 1
        state = self.book.snapshot()
        issues = check_integrity(state)
        self.last_issues = issues
        frame = ReplayFrame(self.index, ev, ok, issues, state)
        self.index += 1
        return frame

    def step_n(self, n: int) -> ReplayFrame | None:
        last: ReplayFrame | None = None
        for _ in range(n):
            last = self.step()
            if last is None:
                break
        return last

    def frames(self) -> Iterator[ReplayFrame]:
        while True:
            f = self.step()
            if f is None:
                return
            yield f

    def apply(self, ev: NormalizedEvent) -> bool:
        b = self.book
        if ev.kind in (EventType.ADD, EventType.ADD_MPID):
            b.rest(ev.side, ev.price_ticks, ev.size, order_id=ev.order_id)
            return True
        if ev.kind in (EventType.EXECUTE, EventType.EXECUTE_PX):
            live = b.lookup(ev.order_id)
            if live is None:
                return False
            fill = min(live.size, ev.size)
            px = ev.price_ticks if ev.kind == EventType.EXECUTE_PX and ev.price_ticks else live.price
            aggressor = Side.Ask if live.side == Side.Bid else Side.Bid
            if ev.kind != EventType.EXECUTE_PX or ev.printable:
                b.record_trade(aggressor, px, fill)
            b.cancel_id(ev.order_id, fill)
            return True
        if ev.kind == EventType.CANCEL:
            return b.cancel_id(ev.order_id, ev.size) > 0
        if ev.kind == EventType.DELETE:
            return b.cancel_id(ev.order_id) > 0
        if ev.kind == EventType.REPLACE:
            live = b.lookup(ev.order_id)
            if live is None:
                return False
            side = live.side
            b.cancel_id(ev.order_id)
            b.rest(side, ev.price_ticks, ev.size, order_id=ev.new_order_id or ev.order_id + 1)
            return True
        if ev.kind == EventType.TRADE:
            b.record_trade(ev.side, ev.price_ticks, ev.size)
            return True
        return False
