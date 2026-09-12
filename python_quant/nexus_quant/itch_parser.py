"""NASDAQ TotalView-ITCH 5.0 streaming parser.

Layouts follow the official TotalView-ITCH 5.0 specification (big-endian).
Only the book-affecting / print messages required by L2 replay are decoded:

    A  Add Order
    F  Add Order with MPID Attribution
    E  Order Executed
    C  Order Executed with Price
    X  Order Cancel
    D  Order Delete
    U  Order Replace
    P  Trade (non-cross)

The payload after the 1-byte type is identical to the spec. Two on-disk
framings are accepted:

* raw concatenated payloads (type byte first);
* 2-byte big-endian length prefix then payload (common in NASDAQ file dumps
  and MoldUDP64 reconstructions).

Malformed / truncated trailing bytes are skipped (counted on
``ItchParseStats``); the generator does not raise. A multi-GB tape is never
materialized — ``iter_itch_events`` reads a fixed-size buffer from a binary
file or stream.

ITCH prices are the spec's 4-byte integer field (4 implied decimal places).
Nexus treats that integer as a tick; no float conversion is performed.
"""
from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from io import BufferedIOBase
from pathlib import Path
from typing import BinaryIO

from .book_state import Side

# Spec payload lengths (including the type byte). Source: ITCH 5.0.
ITCH_LEN: dict[bytes, int] = {
    b"S": 12,
    b"R": 39,
    b"H": 25,
    b"Y": 20,
    b"L": 26,
    b"V": 35,
    b"W": 12,
    b"K": 28,
    b"J": 35,
    b"h": 21,
    b"A": 36,
    b"F": 40,
    b"E": 31,
    b"C": 36,
    b"X": 23,
    b"D": 19,
    b"U": 35,
    b"P": 44,
    b"Q": 40,
    b"B": 19,
    b"I": 50,
    b"N": 20,
}

_HANDLED = frozenset(b"AFECXDUP")

# Big-endian layouts after the type byte.
# locate(H) tracking(H) ts(6s) ...
_HDR = struct.Struct(">HH6s")
_A = struct.Struct(">HH6sQBI8sI")      # 35 bytes after type
_F = struct.Struct(">HH6sQBI8sI4s")    # 39
_E = struct.Struct(">HH6sQIQ")         # 30
_C = struct.Struct(">HH6sQIQcI")       # 35
_X = struct.Struct(">HH6sQI")          # 22
_D = struct.Struct(">HH6sQ")           # 18
_U = struct.Struct(">HH6sQQII")        # 34
_P = struct.Struct(">HH6sQBI8sIQ")     # 43


def _ts6(raw: bytes) -> int:
    """6-byte big-endian nanoseconds since midnight."""
    return int.from_bytes(raw, "big", signed=False)


class EventType(IntEnum):
    ADD = 1
    ADD_MPID = 2
    EXECUTE = 3
    EXECUTE_PX = 4
    CANCEL = 5
    DELETE = 6
    REPLACE = 7
    TRADE = 8


@dataclass(slots=True, frozen=True)
class NormalizedEvent:
    """Replay-facing ITCH event. Prices are integer ticks."""

    kind: EventType
    ts_ns: int
    order_id: int
    side: Side
    price_ticks: int
    size: int
    new_order_id: int = 0
    printable: bool = True
    raw_type: str = ""


@dataclass
class ItchParseStats:
    messages: int = 0
    emitted: int = 0
    skipped_type: int = 0
    truncated: int = 0
    bytes_read: int = 0


Source = (str | Path | BinaryIO | BufferedIOBase | bytes | bytearray | memoryview)


def iter_itch_events(
    source: Source,
    *,
    chunk_size: int = 1 << 20,
    stats: ItchParseStats | None = None,
) -> Iterator[NormalizedEvent]:
    """Lazily yield ``NormalizedEvent`` from an ITCH 5.0 stream.

    ``source`` may be a path, an already-open binary file, or an in-memory
    buffer (tests). ``chunk_size`` bounds the read-ahead window so a multi-GB
    tape is never loaded whole.
    """
    acc = stats if stats is not None else ItchParseStats()
    if isinstance(source, (bytes, bytearray, memoryview)):
        yield from _scan(memoryview(source), acc)
        return
    if isinstance(source, (str, Path)):
        with open(source, "rb") as fh:
            yield from _iter_file(fh, chunk_size, acc)
        return
    yield from _iter_file(source, chunk_size, acc)


def _iter_file(fh: BinaryIO, chunk_size: int, acc: ItchParseStats) -> Iterator[NormalizedEvent]:
    buf = bytearray()
    while True:
        block = fh.read(chunk_size)
        if not block:
            break
        acc.bytes_read += len(block)
        buf.extend(block)
        consumed, events = _drain(memoryview(buf), acc, final=False)
        del buf[:consumed]
        yield from events
    if buf:
        _, events = _drain(memoryview(buf), acc, final=True)
        yield from events


def _scan(mv: memoryview, acc: ItchParseStats) -> Iterator[NormalizedEvent]:
    acc.bytes_read += len(mv)
    _, events = _drain(mv, acc, final=True)
    yield from events


def _drain(
    mv: memoryview,
    acc: ItchParseStats,
    *,
    final: bool,
) -> tuple[int, list[NormalizedEvent]]:
    """Parse as many complete messages as possible from ``mv``.

    Returns (bytes_consumed, events). On ``final=True`` a leftover incomplete
    message is counted as truncated and consumed so the caller can drop it.
    """
    out: list[NormalizedEvent] = []
    i = 0
    n = len(mv)
    while i < n:
        _framed, typ, payload_len, hdr = _frame(mv, i)
        if typ is None:
            if final:
                acc.truncated += 1
                i = n
            break
        need = hdr + payload_len
        if i + need > n:
            if final:
                acc.truncated += 1
                i = n
            break
        payload = mv[i + hdr : i + need]
        acc.messages += 1
        ev = _decode(bytes(typ), payload)
        if ev is None:
            acc.skipped_type += 1
        else:
            acc.emitted += 1
            out.append(ev)
        i += need
    return i, out


def _frame(mv: memoryview, i: int) -> tuple[bool, bytes | None, int, int]:
    """Detect framed vs raw. Returns (framed, type, payload_len, header_bytes)."""
    remain = len(mv) - i
    if remain < 1:
        return False, None, 0, 0
    # Prefer a 2-byte length prefix when it names a known type.
    if remain >= 3:
        ln = int.from_bytes(mv[i : i + 2], "big")
        typ = bytes(mv[i + 2 : i + 3])
        spec = ITCH_LEN.get(typ)
        if spec is not None and 8 <= ln <= 64 and ln == spec:
            return True, typ, ln, 2
    typ = bytes(mv[i : i + 1])
    spec = ITCH_LEN.get(typ)
    if spec is None:
        # Unknown byte — skip one so we can resync on the next candidate.
        return False, typ, 1, 0
    return False, typ, spec, 0


def _decode(typ: bytes, payload: memoryview | bytes) -> NormalizedEvent | None:
    raw = bytes(payload)
    if typ == b"A":
        _loc, _trk, ts, oid, bs, shares, _stk, px = _A.unpack(raw[1:])
        return NormalizedEvent(
            EventType.ADD, _ts6(ts), oid, _side(bs), px, shares, raw_type="A"
        )
    if typ == b"F":
        _loc, _trk, ts, oid, bs, shares, _stk, px, _mpid = _F.unpack(raw[1:])
        return NormalizedEvent(
            EventType.ADD_MPID, _ts6(ts), oid, _side(bs), px, shares, raw_type="F"
        )
    if typ == b"E":
        _loc, _trk, ts, oid, shares, _match = _E.unpack(raw[1:])
        return NormalizedEvent(
            EventType.EXECUTE, _ts6(ts), oid, Side.NONE, 0, shares, raw_type="E"
        )
    if typ == b"C":
        _loc, _trk, ts, oid, shares, _match, printable, px = _C.unpack(raw[1:])
        return NormalizedEvent(
            EventType.EXECUTE_PX,
            _ts6(ts),
            oid,
            Side.NONE,
            px,
            shares,
            printable=printable == b"Y" or printable == 89,
            raw_type="C",
        )
    if typ == b"X":
        _loc, _trk, ts, oid, shares = _X.unpack(raw[1:])
        return NormalizedEvent(
            EventType.CANCEL, _ts6(ts), oid, Side.NONE, 0, shares, raw_type="X"
        )
    if typ == b"D":
        _loc, _trk, ts, oid = _D.unpack(raw[1:])
        return NormalizedEvent(
            EventType.DELETE, _ts6(ts), oid, Side.NONE, 0, 0, raw_type="D"
        )
    if typ == b"U":
        _loc, _trk, ts, oid, new_id, shares, px = _U.unpack(raw[1:])
        return NormalizedEvent(
            EventType.REPLACE,
            _ts6(ts),
            oid,
            Side.NONE,
            px,
            shares,
            new_order_id=new_id,
            raw_type="U",
        )
    if typ == b"P":
        _loc, _trk, ts, oid, bs, shares, _stk, px, _match = _P.unpack(raw[1:])
        return NormalizedEvent(
            EventType.TRADE, _ts6(ts), oid, _side(bs), px, shares, raw_type="P"
        )
    return None


def _side(indicator: int | bytes) -> Side:
    ch = indicator if isinstance(indicator, int) else indicator[0]
    if ch in (ord("B"), 0):
        return Side.Bid
    return Side.Ask


def encode_event(ev: NormalizedEvent, *, framed: bool = True) -> bytes:
    """Encode a normalized event to ITCH 5.0 bytes (test helper)."""
    ts = int(ev.ts_ns).to_bytes(6, "big")
    bs = b"B" if ev.side == Side.Bid else b"S"
    stock = b"AAPL    "
    if ev.kind in (EventType.ADD, EventType.ADD_MPID):
        body = b"A" + _A.pack(1, 1, ts, ev.order_id, bs[0], ev.size, stock, ev.price_ticks)
        if ev.kind == EventType.ADD_MPID:
            body = b"F" + _F.pack(
                1, 1, ts, ev.order_id, bs[0], ev.size, stock, ev.price_ticks, b"MPID"
            )
    elif ev.kind == EventType.EXECUTE:
        body = b"E" + _E.pack(1, 1, ts, ev.order_id, ev.size, ev.order_id)
    elif ev.kind == EventType.EXECUTE_PX:
        prn = b"Y" if ev.printable else b"N"
        body = b"C" + _C.pack(1, 1, ts, ev.order_id, ev.size, ev.order_id, prn, ev.price_ticks)
    elif ev.kind == EventType.CANCEL:
        body = b"X" + _X.pack(1, 1, ts, ev.order_id, ev.size)
    elif ev.kind == EventType.DELETE:
        body = b"D" + _D.pack(1, 1, ts, ev.order_id)
    elif ev.kind == EventType.REPLACE:
        body = b"U" + _U.pack(
            1, 1, ts, ev.order_id, ev.new_order_id or ev.order_id + 1, ev.size, ev.price_ticks
        )
    elif ev.kind == EventType.TRADE:
        body = b"P" + _P.pack(
            1, 1, ts, ev.order_id, bs[0], ev.size, stock, ev.price_ticks, ev.order_id
        )
    else:
        raise ValueError(ev.kind)
    if framed:
        return len(body).to_bytes(2, "big") + body
    return body
