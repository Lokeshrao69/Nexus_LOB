"""Diff-test harness: real C++ ``Engine`` vs the pure-Python ``StubOrderBook`` oracle.

This is the contract-seam oracle test (CLAUDE.md §7 item 6, PROGRESS.md §8):
``StubOrderBook`` is the *reference* the C++ matching engine is diff-tested
against. The harness replays one identical ITCH-derived order stream through
both books and asserts the published L2 ladder matches, bit-for-bit.

It reuses ``ReplayEngine`` so both sides see byte-identical ``NormalizedEvent``
input, and ``EngineAdapter`` (the stub↔engine swap seam from ``book_port.py``).

Requires the compiled bridge module ``nexus_engine`` (build ``bindings/`` first)
and pytest. Skips cleanly if the module isn't built yet — same convention as
``test_abi_parity.py``.

    pytest bindings/tests/test_diff_engine_stub.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_quant"))

import numpy as np
import pytest
from nexus_quant.book_port import EngineAdapter
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.replay import ReplayEngine

nexus_engine = pytest.importorskip(
    "nexus_engine",
    reason="C++ bridge not built yet — build bindings/ then re-run (see CONTRACT.md).",
)

# The L2 ladder the oracle and the engine must agree on exactly. seq / ts_ns /
# version are engine-internal publish counters that legitimately differ between
# the two implementations (the engine counts its own mutations; the stub counts
# its own) — they are intentionally excluded. Trade fields (last_trade_*,
# cum_volume) are also excluded: the engine records prints on matching, while the
# stub's record_trade is a no-op on the engine adapter, so a liquidity-only stream
# (which this test uses) is the correct place for an exact-match assertion.
LADDER_KEYS = ("bid_px", "bid_sz", "bid_ct", "ask_px", "ask_sz", "ask_ct")


def _assert_ladder_equal(stub_state, eng_state, ctx: str) -> None:
    for k in LADDER_KEYS:
        np.testing.assert_array_equal(
            np.asarray(stub_state[k]),
            np.asarray(eng_state[k]),
            err_msg=f"{ctx}: field {k!r} diverged",
        )


def _liquidity_stream(seed: int = 0x5EED) -> list[NormalizedEvent]:
    """Deterministic ADD / CANCEL / DELETE / REPLACE stream around a fixed mid.

    Prices are integer ticks and are kept strictly non-crossing (bids < 10_000 <
    asks), so every ADD rests and the two books are always comparable. A seeded
    RNG makes the stream reproducible; live orders are tracked so CANCEL / DELETE /
    REPLACE always target a resting order.
    """
    rng = np.random.default_rng(seed)
    events: list[NormalizedEvent] = []
    live: dict[int, tuple[Side, int, int]] = {}  # id -> (side, price, size)
    next_id = 1

    def add(side: Side) -> None:
        nonlocal next_id
        if side == Side.Bid:
            px = int(rng.integers(9_950, 10_000))      # [9950, 9999]
        else:
            px = int(rng.integers(10_001, 10_051))     # [10001, 10050]
        size = int(20 + rng.integers(0, 120))
        events.append(NormalizedEvent(EventType.ADD, next_id, next_id, side, px, size))
        live[next_id] = (side, px, size)
        next_id += 1

    for _ in range(40):
        add(Side.Bid if rng.integers(2) == 0 else Side.Ask)
        if not live or rng.random() < 0.35:
            continue
        oid, (side, px, size) = next(iter(live.items()))
        roll = rng.random()
        if roll < 0.4:
            cut = int(1 + rng.integers(0, size))       # partial cancel
            events.append(NormalizedEvent(EventType.CANCEL, next_id, oid, Side.NONE, 0, cut))
            new_size = size - cut
            if new_size <= 0:                          # fully drained -> no longer live
                del live[oid]
            else:
                live[oid] = (side, px, new_size)
        elif roll < 0.7:
            events.append(NormalizedEvent(EventType.DELETE, next_id, oid, Side.NONE, 0, 0))
            del live[oid]
        else:
            # reprice on the same side, staying non-crossing
            if side == Side.Bid:
                new_px = int(rng.integers(9_950, 10_000))
            else:
                new_px = int(rng.integers(10_001, 10_051))
            events.append(
                NormalizedEvent(
                    EventType.REPLACE, next_id, oid, Side.NONE, new_px, size,
                    new_order_id=next_id,
                )
            )
            del live[oid]
            live[next_id] = (side, new_px, size)
            next_id += 1
    return events


def test_ladder_parity_across_replay():
    events = _liquidity_stream()

    stub_replay = ReplayEngine.from_stub(events)
    eng_replay = ReplayEngine(events, book=EngineAdapter(nexus_engine.Engine()))

    # Step both engines in lock-step over the identical stream.
    while True:
        sf = stub_replay.step()
        ef = eng_replay.step()
        assert (sf is None) == (ef is None), "the two replays ended out of step"
        if sf is None:
            break
        _assert_ladder_equal(sf.state, ef.state, f"step {sf.index}")
        # Neither book may be internally inconsistent (crossed/unsorted/negative).
        assert sf.issues == [], f"stub integrity @ {sf.index}: {sf.issues}"
        assert ef.issues == [], f"engine integrity @ {ef.index}: {ef.issues}"

    # Both replays must have consumed the stream identically.
    assert stub_replay.applied == len(events)
    assert eng_replay.applied == len(events)
    assert stub_replay.skipped == eng_replay.skipped == 0


def test_bbo_and_depth_agree_after_stream():
    events = _liquidity_stream(seed=0xBEAD)
    stub_replay = ReplayEngine.from_stub(events)
    eng_replay = ReplayEngine(events, book=EngineAdapter(nexus_engine.Engine()))
    sf = stub_replay.step_n(len(events))
    ef = eng_replay.step_n(len(events))
    assert sf is not None and ef is not None
    _assert_ladder_equal(sf.state, ef.state, "final")
    # Both books must agree on the BBO (ladder equality already implies this) and,
    # when both sides are present, must not be crossed.
    bb, ba = int(sf.state["bid_px"][0]), int(sf.state["ask_px"][0])
    assert int(ef.state["bid_px"][0]) == bb
    assert int(ef.state["ask_px"][0]) == ba
    if bb and ba:
        assert bb < ba, "engine/stub produced a crossed market"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
