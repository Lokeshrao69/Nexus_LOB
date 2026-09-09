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
