"""Injectable book surface shared by replay and OrderBookEnv.

Person A's C++ ``nexus_engine.Engine`` and Person B's ``StubOrderBook`` do
not share a mutation API:

* Stub: ``add(side, price, size)`` / ``cancel(side, price, size)`` / ``record_trade``
* Engine: ``submit_limit`` / ``submit_market`` / ``cancel(order_id)`` / ``fills``

Both expose the frozen ``view()`` / ``snapshot()`` contract. This module is
the Python-side adapter so replay/env never import a concrete engine class.
Swap the adapter, not the env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from .book_state import Side, StubOrderBook


View = Mapping[str, Any]


@runtime_checkable
class BookView(Protocol):
    """Frozen observation seam — valid for StubOrderBook and Engine."""

    def view(self) -> View:
        """Live window. Engine arrays alias engine memory until the next mutation."""

    def snapshot(self) -> View:
        """Owning copy, safe to retain across steps."""


@dataclass(slots=True)
class Resting:
    order_id: int
    side: Side
    price: int
    size: int


@dataclass(slots=True)
class TakeResult:
    filled: int
    notional_ticks: int

    @property
    def avg_px(self) -> int:
        return 0 if self.filled == 0 else int(self.notional_ticks // self.filled)


@runtime_checkable
class ExecutableBook(BookView, Protocol):
    """Mutations the env needs. Implemented by adapters, not by the stub itself."""

    def take(self, side: Side, qty: int, limit_px: int | None = None) -> TakeResult:
        """Aggressive walk of the opposite book. ``side`` is the *taker* side."""

    def rest(self, side: Side, price: int, qty: int, order_id: int | None = None) -> Resting:
        """Post a resting limit. Returns the handle used for later cancel."""

    def cancel_resting(self, handle: Resting) -> int:
        """Cancel residual at ``handle``. Returns cancelled size."""

    def record_trade(self, side: Side, price: int, size: int) -> None: ...


class StubBookAdapter:
    """Replay + env adapter over the frozen ``StubOrderBook``.

    Order-id state lives here so ``book_state.py`` stays untouched. The stub
    only sees aggregate ``add`` / ``cancel`` / ``record_trade`` calls.
    """

    def __init__(self, book: StubOrderBook | None = None) -> None:
        self.book = book if book is not None else StubOrderBook()
        self._orders: dict[int, Resting] = {}
        self._next_id = 1

    def view(self) -> View:
        return self.book.view()

    def snapshot(self) -> View:
        v = self.book.snapshot()
        return {k: (np.copy(x) if isinstance(x, np.ndarray) else x) for k, x in v.items()}

    def rest(self, side: Side, price: int, qty: int, order_id: int | None = None) -> Resting:
        oid = int(order_id) if order_id is not None else self._next_id
        self._next_id = max(self._next_id, oid + 1)
        self.book.add(side, int(price), int(qty), orders=1)
        h = Resting(oid, side, int(price), int(qty))
        self._orders[oid] = h
        return h

    def cancel_resting(self, handle: Resting) -> int:
        live = self._orders.get(handle.order_id)
        if live is None or live.size <= 0:
            return 0
        cut = live.size
        self.book.cancel(live.side, live.price, cut, orders=1)
        live.size = 0
        self._orders.pop(handle.order_id, None)
        return cut

    def cancel_id(self, order_id: int, size: int | None = None) -> int:
        live = self._orders.get(int(order_id))
        if live is None:
            return 0
        cut = live.size if size is None else min(live.size, int(size))
        if cut <= 0:
            return 0
        orders = 1 if cut >= live.size else 0
        self.book.cancel(live.side, live.price, cut, orders=orders)
        live.size -= cut
        if live.size <= 0:
            self._orders.pop(live.order_id, None)
        return cut

    def lookup(self, order_id: int) -> Resting | None:
        return self._orders.get(int(order_id))

    def record_trade(self, side: Side, price: int, size: int) -> None:
        self.book.record_trade(side, int(price), int(size))

    def take(self, side: Side, qty: int, limit_px: int | None = None) -> TakeResult:
        """Walk opposite displayed levels. Does not require C++ matching."""
        remaining = int(qty)
        notional = 0
        snap = self.snapshot()  # own the ladder; we mutate after
        if side == Side.Bid:
            prices = list(snap["ask_px"])
            sizes = list(snap["ask_sz"])
            opp = Side.Ask
            walk = range(len(prices))
        else:
            prices = list(snap["bid_px"])
            sizes = list(snap["bid_sz"])
            opp = Side.Bid
            walk = range(len(prices))
        for i in walk:
            px = int(prices[i])
            sz = int(sizes[i])
            if px == 0 or sz <= 0 or remaining <= 0:
                continue
            if limit_px is not None:
                if side == Side.Bid and px > limit_px:
                    break
                if side == Side.Ask and px < limit_px:
                    break
            hit = min(remaining, sz)
            self._hit_orders(opp, px, hit)
            self.book.cancel(opp, px, hit, orders=0)
            self.book.record_trade(side, px, hit)
            remaining -= hit
            notional += hit * px
        return TakeResult(filled=int(qty) - remaining, notional_ticks=notional)

    def _hit_orders(self, side: Side, price: int, qty: int) -> None:
        left = qty
        for oid, live in list(self._orders.items()):
            if left <= 0:
                break
            if live.side != side or live.price != price or live.size <= 0:
                continue
            take = min(live.size, left)
            live.size -= take
            left -= take
            if live.size <= 0:
                self._orders.pop(oid, None)

    def reset(self) -> None:
        """Clear liquidity in-place so an injected StubOrderBook stays the same object."""
        snap = self.snapshot()
        from .book_state import DEPTH

        for i in range(DEPTH):
            bp, bs, bc = int(snap["bid_px"][i]), int(snap["bid_sz"][i]), int(snap["bid_ct"][i])
            if bs:
                self.book.cancel(Side.Bid, bp, bs, orders=bc)
            ap, a_s, ac = int(snap["ask_px"][i]), int(snap["ask_sz"][i]), int(snap["ask_ct"][i])
            if a_s:
                self.book.cancel(Side.Ask, ap, a_s, orders=ac)
        self._orders.clear()
        self._next_id = 1


class EngineAdapter:
    """Thin wrapper around a compiled ``nexus_engine.Engine``.

    Import of ``nexus_engine`` is deferred so the package imports without a
    C++ build. Construct with ``EngineAdapter(nexus_engine.Engine())``.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._engine_cls = type(engine)
        self._next_id = 1
        self._live: dict[int, Resting] = {}

    def reset(self) -> None:
        """New empty Engine so OrderBookEnv.reset keeps the C++ seam."""
        self.engine = self._engine_cls()
        self._live.clear()
        self._next_id = 1

    def view(self) -> View:
        return self.engine.view()

    def snapshot(self) -> View:
        return self.engine.snapshot()

    def rest(self, side: Side, price: int, qty: int, order_id: int | None = None) -> Resting:
        oid = int(order_id) if order_id is not None else self._next_id
        self._next_id = max(self._next_id, oid + 1)
        tif = _engine_tif(self.engine, "GTC")
        eng_side = _engine_side(self.engine, side)
        self.engine.submit_limit(oid, eng_side, int(price), int(qty), tif)
        # If the limit crossed, only the residual is live.
        live_sz = int(qty)
        try:
            fills = self.engine.fills() if hasattr(self.engine, "fills") else []
            live_sz = int(qty) - sum(int(f[3]) for f in fills)
        except Exception:
            pass
        live_sz = max(0, live_sz)
        h = Resting(oid, side, int(price), live_sz)
        if live_sz > 0:
            self._live[oid] = h
        return h

    def cancel_resting(self, handle: Resting) -> int:
        self.engine.cancel(handle.order_id)
        return self._live.pop(handle.order_id, handle).size

    def lookup(self, order_id: int) -> Resting | None:
        """Live resting handle for ``order_id``, or None if not resting."""
        return self._live.get(int(order_id))

    def cancel_id(self, order_id: int, size: int | None = None) -> int:
        """Cancel ``size`` (default: the whole order) of a resting order.

        Mirrors ``StubBookAdapter.cancel_id`` so ``ReplayEngine.apply`` drives the
        real engine the same way it drives the stub. A full residual uses
        ``engine.cancel``; a partial reduction uses ``engine.modify`` at the same
        price, which keeps time priority (matches the stub's partial-cancel
        semantics). Returns the number of shares actually cancelled.
        """
        live = self._live.get(int(order_id))
        if live is None or live.size <= 0:
            return 0
        cut = live.size if size is None else min(live.size, int(size))
        if cut <= 0:
            return 0
        remaining = live.size - cut
        if remaining <= 0:
            self.engine.cancel(int(order_id))
            self._live.pop(int(order_id), None)
        else:
            # In-place size reduction keeps the order resting with time priority.
            self.engine.modify(int(order_id), int(live.price), remaining)
            live.size = remaining
        return cut

    def record_trade(self, side: Side, price: int, size: int) -> None:
        # Engine records prints on matching; explicit tape prints are a no-op here.
        del side, price, size

    def take(self, side: Side, qty: int, limit_px: int | None = None) -> TakeResult:
        oid = self._next_id
        self._next_id += 1
        eng_side = _engine_side(self.engine, side)
        if limit_px is None:
            r = self.engine.submit_market(oid, eng_side, int(qty))
        else:
            tif = _engine_tif(self.engine, "IOC")
            r = self.engine.submit_limit(oid, eng_side, int(limit_px), int(qty), tif)
        filled = int(r.get("filled", 0)) if isinstance(r, dict) else int(getattr(r, "filled", 0))
        notional = 0
        fills = self.engine.fills() if hasattr(self.engine, "fills") else []
        for f in fills:
            # (maker, taker, px, qty, aggressor)
            notional += int(f[2]) * int(f[3])
        if filled and notional == 0:
            px = int(self.view().get("last_trade_px") or 0)
            notional = filled * px
        for f in fills:
            maker = int(f[0])
            qty = int(f[3])
            live = self._live.get(maker)
            if live is None:
                continue
            live.size -= qty
            if live.size <= 0:
                self._live.pop(maker, None)
        return TakeResult(filled=filled, notional_ticks=notional)


def _engine_side(engine: Any, side: Side) -> Any:
    enum = getattr(type(engine), "Side", None) or getattr(engine, "Side", None)
    if enum is None:
        try:
            import nexus_engine as ne

            enum = ne.Side
        except ImportError:
            return int(side)
    return enum.Bid if side == Side.Bid else enum.Ask


def _engine_tif(engine: Any, name: str) -> Any:
    try:
        import nexus_engine as ne

        return getattr(ne.TimeInForce, name)
    except ImportError:
        return name


def adapt(book: Any) -> StubBookAdapter | EngineAdapter:
    """Wrap a stub or a compiled Engine without the caller branching."""
    if isinstance(book, StubBookAdapter) or isinstance(book, EngineAdapter):
        return book
    if isinstance(book, StubOrderBook):
        return StubBookAdapter(book)
    if hasattr(book, "submit_limit") and hasattr(book, "view"):
        return EngineAdapter(book)
    if hasattr(book, "view") and hasattr(book, "add"):
        return StubBookAdapter(book)
    raise TypeError(f"cannot adapt book of type {type(book)!r}")
