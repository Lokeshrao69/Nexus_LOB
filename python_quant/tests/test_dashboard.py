"""Dashboard slot codec + hub JSON."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_state import BOOK_STATE_DTYPE, Side, StubOrderBook
from nexus_quant.dashboard import (
    SnapshotHub,
    decode_slot,
    latest_from_file_ring,
    record_from_view,
    view_from_record,
)


def test_roundtrip_view_record():
    book = StubOrderBook()
    book.add(Side.Bid, 100, 10)
    book.add(Side.Ask, 102, 8)
    rec = record_from_view(book.view())
    assert rec.nbytes == BOOK_STATE_DTYPE.itemsize
    v = view_from_record(rec)
    assert int(v["bid_px"][0]) == 100
    assert int(v["ask_px"][0]) == 102
    again = decode_slot(rec.tobytes())
    assert int(again["bid_sz"][0]) == 10


def test_file_ring_latest(tmp_path: Path):
    book = StubOrderBook()
    book.add(Side.Bid, 50, 3)
    p = tmp_path / "slots.bin"
    p.write_bytes(record_from_view(book.view()).tobytes())
    book.add(Side.Ask, 55, 4)
    with p.open("ab") as fh:
        fh.write(record_from_view(book.view()).tobytes())
    latest = latest_from_file_ring(p)
    assert latest is not None
    assert int(latest["ask_px"][0]) == 55


def test_hub_json():
    book = StubOrderBook()
    book.add(Side.Bid, 10, 1)
    book.add(Side.Ask, 12, 1)
    hub = SnapshotHub()
    hub.push(book.view(), source="test")
    js = hub.as_json()
    assert js["ok"] is True
    assert js["spread"] == 2
    assert len(js["bid_px"]) == 10
