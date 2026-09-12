"""Dashboard slot codec + hub JSON."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_state import BOOK_STATE_DTYPE, Side, StubOrderBook
from nexus_quant.dashboard import (
    SnapshotHub,
    decode_slot,
    latest_from_file_ring,
    read_shm_ring_latest,
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


def test_hub_history_tracks_and_dedupes():
    book = StubOrderBook()
    book.add(Side.Bid, 100, 10)
    book.add(Side.Ask, 102, 8)
    hub = SnapshotHub()
    v1 = book.view()
    hub.push(v1, source="test")
    book.add(Side.Bid, 101, 5)  # seq bumps AND moves best bid to 101
    hub.push(book.view(), source="test")
    assert len(hub.history) == 2
    assert hub.history[-1]["mid"] == (101 + 102) / 2
    # re-polling an already-seen seq must not duplicate the history sample
    hub.push(v1, source="test")
    assert len(hub.history) == 2
    js = hub.as_json()
    assert len(js["history"]) == 2
    assert js["mid"] == (100 + 102) / 2  # hub.view is back to v1
    assert js["bid0"] == 100 and js["ask0"] == 102


def test_hub_latency_histogram():
    book = StubOrderBook()
    book.add(Side.Bid, 10, 1)
    book.add(Side.Ask, 12, 1)
    hub = SnapshotHub()
    hub.push(book.view(), source="test")
    hub.record_latency(5_000)          # <10 µs
    hub.record_latency(20_000)         # 10-25 µs
    hub.record_latency(40_000_000)     # >10 ms
    L = hub.as_json()["latency"]
    assert L["n"] == 3
    assert sum(L["counts"]) == 3
    assert L["counts"][0] == 1
    assert L["counts"][1] == 1
    assert L["counts"][-1] == 1
    assert L["p50_ns"] is not None and L["p95_ns"] is not None
    assert L["p50_ns"] <= L["p95_ns"]
    # push() accepts an explicit feed-latency sample (used by serve_dashboard)
    book.add(Side.Bid, 9, 2)
    hub.push(book.view(), source="test", feed_latency_ns=60_000)
    assert hub.as_json()["latency"]["n"] == 4


def test_shm_ring_decoder_roundtrip():
    # read_shm_ring_latest attaches POSIX /dev/shm, which does not exist on
    # Windows (Path("/dev/shm") resolves to a drive-relative \dev\shm). The
    # decode logic it exercises is the same layout-checked on every platform,
    # so skip rather than fail where the POSIX segment can't exist.
    if not Path("/dev/shm").is_dir():
        pytest.skip("POSIX /dev/shm unavailable (live ring attach is POSIX-only)")
    import struct

    from nexus_quant.dashboard import _SHM_CTRL_N

    book = StubOrderBook()
    book.add(Side.Bid, 49990, 11)
    book.add(Side.Ask, 50010, 9)
    slot = record_from_view(book.view()).tobytes()
    cap, slot_n, state = 4, BOOK_STATE_DTYPE.itemsize, 1
    ctrl = struct.pack("<QQQQQII", 1, 0, 0, cap, slot_n, state, 0)
    assert len(ctrl) == _SHM_CTRL_N
    blob = ctrl + slot + b"\x00" * (slot_n * (cap - 1))
    shm = Path("/dev/shm") / "nex_test_slot"
    shm.write_bytes(blob)
    try:
        v = read_shm_ring_latest("nex_test_slot")
        assert v is not None
        assert int(v["bid_px"][0]) == 49990
        assert int(v["ask_sz"][0]) == 9
    finally:
        shm.unlink(missing_ok=True)
