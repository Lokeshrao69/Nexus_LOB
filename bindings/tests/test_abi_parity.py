"""C++ <-> Python ABI parity + engine-seam tests for the Nexus-LOB bridge.

Requires the compiled bridge module `nexus_engine` (build `bindings/` first) and
pytest. Skips cleanly if the module isn't built yet:

    pytest bindings/tests/test_abi_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_quant"))

import numpy as np
import pytest
from nexus_quant import book_state as bs

nexus_engine = pytest.importorskip(
    "nexus_engine",
    reason="C++ bridge not built yet — build bindings/ then re-run (see CONTRACT.md).",
)


def test_constants_match():
    assert nexus_engine.DEPTH == bs.DEPTH
    assert nexus_engine.CONTRACT_VERSION == bs.CONTRACT_VERSION


def test_struct_size_matches_dtype():
    # The core ABI lock: C++ sizeof(BookStateView) == NumPy dtype itemsize.
    assert nexus_engine.STATE_NBYTES == bs.BOOK_STATE_DTYPE.itemsize


def test_view_matches_python_contract():
    v = nexus_engine.Engine().view()
    for name, dt in (
        ("bid_px", bs.PX_DTYPE), ("bid_sz", bs.SZ_DTYPE), ("bid_ct", bs.CT_DTYPE),
        ("ask_px", bs.PX_DTYPE), ("ask_sz", bs.SZ_DTYPE), ("ask_ct", bs.CT_DTYPE),
    ):
        assert v[name].shape == (bs.DEPTH,)
        assert v[name].dtype == np.dtype(dt)


def test_zero_copy_vs_snapshot_semantics():
    # Prove live-view vs owning-snapshot on the REAL engine (CONTRACT.md §4).
    e = nexus_engine.Engine()
    live = e.view()        # aliases engine memory (live window)
    frozen = e.snapshot()  # owning copy
    e.submit_limit(1, nexus_engine.Side.Bid, 10_000, 500, nexus_engine.TimeInForce.GTC)
    assert live["bid_px"][0] == 10_000  # live view reflects the mutation
    assert live["bid_sz"][0] == 500
    assert frozen["bid_px"][0] == 0     # earlier snapshot is unaffected


def test_submit_limit_rests():
    e = nexus_engine.Engine()
    r = e.submit_limit(1, nexus_engine.Side.Bid, 10_000, 500, nexus_engine.TimeInForce.GTC)
    assert r["id"] == 1
    assert r["status"] == nexus_engine.Status.Accepted
    assert r["filled"] == 0
    assert r["resting"] == 500
    assert e.best_bid() == 10_000
    assert e.live_orders() == 1
    assert e.view()["bid_px"][0] == 10_000
    # No cross => no fills from this call.
    assert e.fills() == []


def test_crossing_order_emits_fills():
    # Mirrors lob_test.cpp's test_multi_level_sweep (verified by the C++ suite).
    e = nexus_engine.Engine()
    e.submit_limit(10, nexus_engine.Side.Ask, 10_050, 100, nexus_engine.TimeInForce.GTC)
    e.submit_limit(11, nexus_engine.Side.Ask, 10_100, 100, nexus_engine.TimeInForce.GTC)
    e.submit_limit(12, nexus_engine.Side.Ask, 10_150, 100, nexus_engine.TimeInForce.GTC)
    r = e.submit_limit(13, nexus_engine.Side.Bid, 10_100, 250, nexus_engine.TimeInForce.GTC)
    # Crosses 100 @ 10_050 then 100 @ 10_100; 10_150 is above the limit → 50 rests.
    assert r["status"] == nexus_engine.Status.PartiallyFilledResting
    assert r["filled"] == 200 and r["resting"] == 50
    fills = e.fills()
    assert len(fills) == 2
    assert fills[0] == (10, 13, 10_050, 100, nexus_engine.Side.Bid)
    assert fills[1] == (11, 13, 10_100, 100, nexus_engine.Side.Bid)
    assert e.best_ask() == 10_150           # best ask advanced past the swept levels
    assert e.best_bid() == 10_100           # residual rested as the best bid
    assert e.view()["cum_volume"] == 200
    # A fresh order call replaces the previous fills (they're per-call).
    e.cancel(13)
    assert e.fills() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
