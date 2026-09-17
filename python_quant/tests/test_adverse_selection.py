"""Unified test suite for adverse selection (combining PR #20 and Person B test suites)."""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.research.adverse_selection import (
    P_adverse,
    PassiveFill,
    adverse_groups,
    adverse_selection_report,
    drift_ticks,
    fills_from_tracker,
    filter_book_events,
    is_book_affecting,
    matched_unexecuted_control_drift,
    nw_tstat,
    p_adverse,
    p_adverse_unconditional,
    post_fill_drift,
    pre_fill_drift,
)
from nexus_quant.research.experiments import hac_se
from nexus_quant.research.queue_dynamics import (
    FillRecord,
    OrderLevelTracker,
    QueueTracker,
)
from nexus_quant.research.synthetic_flow import FlowConfig, SyntheticFlow

# ===========================================================================
# Part 1: Lokesh PR #20 tests
# ===========================================================================
_N = 8000
_RW = FlowConfig(n_events=_N, seed=123)
_DRIFT = FlowConfig(n_events=_N, seed=123, drift_ticks=0.6, vol_per_event=0.6)


def _replay(cfg: FlowConfig):
    tr = QueueTracker()
    for e in SyntheticFlow(cfg).generate():
        tr.on_event(e)
    return tr, list(tr.fills)


def _fill(event_index: int, side: Side = Side.Ask, ahead: int = 0,
          lvl: int = 0, ofi: float = 0.0) -> FillRecord:
    return FillRecord(
        oid=1, side=side, price=15_000, filled=1, event_index=event_index,
        ts_ns=0, size_at_birth=1, ahead_at_birth=ahead, ahead_at_fill=ahead,
        level_size_at_fill=lvl, ofi=ofi,
    )


def test_drift_definition_and_tape_end_none():
    tr = QueueTracker()
    tr.mid_history = [100.0, 101.0, 105.0, 99.0, 110.0]
    assert drift_ticks(_fill(event_index=0), tr, 2) == 105.0 - 100.0
    assert drift_ticks(_fill(event_index=0), tr, 1) == 101.0 - 100.0
    assert drift_ticks(_fill(event_index=1), tr, 3) == 110.0 - 101.0
    assert drift_ticks(_fill(event_index=4), tr, 1) is None
    assert drift_ticks(_fill(event_index=3), tr, 2) is None


def test_tape_end_rows_dropped():
    tr, fills = _replay(_RW)
    h = 2000
    r = post_fill_drift(fills, tr, h)
    expect = sum(1 for f in fills if f.event_index + h < len(tr.mid_history))
    assert 0 < r["n"] < len(fills)
    assert r["n"] == expect


def test_P_adverse_sign_and_none_at_end():
    tr = QueueTracker()
    tr.mid_history = [100.0, 101.0, 105.0, 90.0, 90.0]
    r = P_adverse(_fill(event_index=0, side=Side.Ask), tr, ofi=1.0, queue_pos=0.0, h=2)
    assert r == {
        "side": "ask", "ofi": 1.0, "queue_pos": 0.0, "h": 2,
        "drift_ticks": 5.0, "drift_bps": 5.0 / 100.0 * 1e4, "adverse": True,
    }
    r2 = P_adverse(_fill(event_index=0, side=Side.Bid), tr, ofi=-1.0, queue_pos=0.5, h=2)
    assert r2["side"] == "bid" and r2["adverse"] is False
    r3 = P_adverse(_fill(event_index=2, side=Side.Bid), tr, ofi=0.0, queue_pos=0.1, h=1)
    assert r3["drift_ticks"] == 90.0 - 105.0 and r3["adverse"] is True
    r4 = P_adverse(_fill(event_index=4, side=Side.Ask), tr, ofi=0.0, queue_pos=0.0, h=1)
    assert r4["drift_ticks"] is None and r4["drift_bps"] is None and r4["adverse"] is None


def test_null_on_rw_ci_straddles_zero():
    tr, fills = _replay(_RW)
    r = post_fill_drift(fills, tr, 5)
    assert r["n"] >= 100
    assert r["nw_lag"] == 4
    assert r["ci95"]["lo"] <= 0.0 <= r["ci95"]["hi"]
    assert r["p_value"] > 0.05
    assert abs(r["mean_ticks"]) < 0.5


def test_hac_se_inflates_under_overlap():
    x = np.cumsum(np.random.default_rng(6).normal(0.0, 1.0, 2000))
    assert hac_se(x, lag=3) > hac_se(x, lag=0)


def test_fill_records_carry_level_and_ofi_reproducibly():
    _, fills1 = _replay(_RW)
    _, fills2 = _replay(_RW)
    assert len(fills1) == len(fills2) >= 100
    for a, b in zip(fills1, fills2):
        assert a.level_size_at_fill == b.level_size_at_fill
        assert a.ofi == b.ofi
        assert a.level_size_at_fill > 0
    nz = sum(1 for f in fills1 if f.ofi != 0.0)
    assert nz / len(fills1) > 0.9


def test_adverse_groups_bins_partition_all_fills_and_json():
    tr, fills = _replay(_RW)
    res = adverse_groups(fills, tr, horizons=(1, 25))
    assert len(res["bins"]) == 2 * 3 * 3
    assert sum(b["n_fills"] for b in res["bins"]) == len(fills)
    assert res["horizons"] == [1, 25]
    keys = {(b["side"], b["ofi_sign"], b["queue_tercile"]) for b in res["bins"]}
    assert len(keys) == len(res["bins"])
    json.dumps(res)
    assert adverse_groups(fills, tr, horizons=(1, 25)) == res


def test_structural_drift_positive_with_CI_excluding_zero():
    tr, fills = _replay(_DRIFT)
    r = post_fill_drift(fills, tr, 5)
    assert r["mean_ticks"] > 0
    assert r["ci95"]["lo"] > 0
    assert r["p_value"] < 0.01
    res = adverse_groups(fills, tr, horizons=(25,))
    ask = [b for b in res["bins"] if b["side"] == "ask" and b["n_fills"] > 0]
    bid = [b for b in res["bins"] if b["side"] == "bid" and b["n_fills"] > 0]
    assert any(b["horizons"]["25"]["p_adverse"] > 0.8 for b in ask)
    assert any(b["horizons"]["25"]["mean_drift_bps"] > 0 for b in ask)
    assert all(b["horizons"]["25"]["p_adverse"] < 0.2 for b in bid)


# ===========================================================================
# Part 2: Person B tests
# ===========================================================================
def test_post_fill_drift_sign_convention():
    mids = [100.0, 101.0, 102.0, 103.0, 104.0]
    fills = [PassiveFill(idx=1, side=Side.Bid), PassiveFill(idx=1, side=Side.Ask)]
    d = post_fill_drift(fills, mids, h=2)
    np.testing.assert_allclose(d, [2.0, -2.0])
    assert p_adverse(d) == 0.5
    assert np.isnan(post_fill_drift([PassiveFill(idx=4, side=Side.Bid)], mids, h=1))[0]
    assert np.isnan(post_fill_drift([PassiveFill(idx=3, side=Side.Bid)], mids, h=5))[0]


def test_pre_fill_control_window_ends_before_the_fill():
    mids = [100.0, 100.0, 100.0, 105.0, 90.0, 90.0]
    f = [PassiveFill(idx=4, side=Side.Bid)]
    np.testing.assert_allclose(pre_fill_drift(f, mids, h=2), [5.0])
    assert np.isnan(post_fill_drift(f, mids, h=2))[0]
    np.testing.assert_allclose(post_fill_drift(f, mids, h=1), [0.0])
    assert np.isnan(pre_fill_drift([PassiveFill(idx=1, side=Side.Bid)], mids, h=2))[0]


def test_p_adverse_excludes_ties_and_nans():
    assert p_adverse([-1.0, -1.0, 1.0, 0.0, np.nan]) == 2 / 3
    assert np.isnan(p_adverse([0.0, np.nan]))


def test_p_adverse_conditional_and_unconditional_distinction():
    """Verify conditional (1.0) vs unconditional (0.20) on 80% zero-move and 20% adverse series."""
    # 80 zeros, 20 adverse (-1.0)
    drifts = [0.0] * 80 + [-1.0] * 20
    # Conditional on |ΔP| > 0: all 20 non-zero moves are adverse -> 1.0
    assert p_adverse(drifts, conditional=True) == 1.0
    # Unconditional: 20 adverse out of 100 total fills -> 0.20
    assert p_adverse(drifts, conditional=False) == 0.20
    assert p_adverse_unconditional(drifts) == 0.20

    # Also test in adverse_selection_report
    fills = [PassiveFill(idx=i, side=Side.Bid) for i in range(100)]
    # Construct mids such that first 80 fills have 0 change and last 20 fills have adverse drop
    mids = [100.0] * 105
    for i in range(80, 100):
        mids[i + 1] = mids[i] - 1.0  # each step drops 1 tick -> adverse for Bid
    rep = adverse_selection_report(fills, mids, horizons=(1,))
    ov = rep["horizons"][1]["overall"]
    assert ov["p_adverse"] == 1.0
    assert ov["p_adverse_conditional"] == 1.0
    assert ov["p_adverse_unconditional"] == 0.20



def test_newey_west_matches_hand_computation():
    x = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    n = x.size
    d = x - x.mean()
    g0 = float(d @ d) / n
    g1 = float(d[1:] @ d[:-1]) / n
    var = g0 + 2 * (1 - 1 / 2) * g1
    r = nw_tstat(x, lag=1)
    assert r["n"] == 6 and r["mean"] == x.mean()
    np.testing.assert_allclose(r["se"], sqrt(var / n))
    np.testing.assert_allclose(r["t"], x.mean() / sqrt(var / n))
    assert 0.0 <= r["p"] <= 1.0
    r0 = nw_tstat(x, lag=0)
    np.testing.assert_allclose(r0["se"], x.std() / sqrt(n))
    assert np.isnan(nw_tstat([1.0, 2.0], lag=1)["t"])


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(2000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, e.size):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert nw_tstat(ar, lag=10)["se"] > 2.0 * nw_tstat(ar, lag=0)["se"]


def _adverse_stream(n_orders: int = 60, seed: int = 11):
    rng = np.random.default_rng(seed)
    events: list[NormalizedEvent] = []
    mids: list[float] = []
    mid = 10_000.0
    oid = 1
    ts = 1
    for _ in range(n_orders):
        side = Side.Bid if rng.random() < 0.5 else Side.Ask
        px = int(mid) - 1 if side == Side.Bid else int(mid) + 1
        events.append(NormalizedEvent(EventType.ADD, ts, oid, side, px, 100))
        mids.append(mid)
        ts += 1
        for _ in range(int(rng.integers(0, 3))):
            events.append(NormalizedEvent(EventType.TRADE, ts, 0, Side.Bid, int(mid), 1))
            mids.append(mid)
            ts += 1
        events.append(NormalizedEvent(EventType.EXECUTE, ts, oid, Side.NONE, 0, 100))
        mids.append(mid)
        ts += 1
        move = float(rng.integers(2, 5))
        mid += -move if side == Side.Bid else move
        for _ in range(3):
            events.append(NormalizedEvent(EventType.TRADE, ts, 0, Side.Ask, int(mid), 1))
            mids.append(mid)
            ts += 1
        oid += 1
    return events, mids


def test_report_detects_constructed_adverse_selection():
    events, mids = _adverse_stream()
    tr = OrderLevelTracker()
    for e in events:
        tr.on_event(e)
    ofis = np.zeros(len(events))
    fills = fills_from_tracker(tr.completed, ofi_at=ofis)
    assert len(fills) == 60 and all(f.queue_frac == 0.0 for f in fills)
    rep = adverse_selection_report(fills, mids, horizons=(1, 3), min_group=10)
    assert rep["n_fills"] == 60
    for h in (1, 3):
        blk = rep["horizons"][h]
        ov = blk["overall"]
        assert ov["n"] == 60
        assert -4.0 <= ov["mean_drift"] <= -2.0
        assert ov["p_adverse"] == 1.0 and ov["t_nw"] < -5 and ov["p_value"] < 1e-6
        assert blk["post_minus_pre"]["mean_drift"] < -2.0
        groups = blk["groups"]
        assert "side=bid" in groups and "side=ask" in groups and "queue=front" in groups
        assert groups["side=bid"]["p_adverse"] == 1.0 and groups["side=ask"]["p_adverse"] == 1.0
        assert "ofi=0" in groups and "ofi>0" not in groups


def test_report_is_null_on_a_flat_tape():
    fills = [PassiveFill(idx=i, side=Side.Bid if i % 2 else Side.Ask) for i in range(5, 200, 3)]
    mids = [500.0] * 400
    rep = adverse_selection_report(fills, mids, horizons=(1, 5))
    for blk in rep["horizons"].values():
        assert blk["overall"]["mean_drift"] == 0.0
        assert np.isnan(blk["overall"]["p_adverse"])


def test_fills_from_tracker_uses_first_fill_index_and_queue_fraction():
    tr = OrderLevelTracker()
    tr.on_event(NormalizedEvent(EventType.ADD, 1, 1, Side.Ask, 105, 100))
    tr.on_event(NormalizedEvent(EventType.ADD, 2, 2, Side.Ask, 105, 100))
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 3, 1, Side.NONE, 0, 40))
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 4, 1, Side.NONE, 0, 60))
    tr.on_event(NormalizedEvent(EventType.DELETE, 5, 2, Side.NONE, 0, 0))
    fills = fills_from_tracker(tr.completed, ofi_at=[0.0, 0.0, -2.0, 1.0, 0.0])
    assert len(fills) == 1
    f = fills[0]
    assert f.idx == 2 and f.side == Side.Ask and f.size == 100 and f.ofi == -2.0 and f.queue_frac == 0.0


def test_fills_from_tracker_and_adverse_selection_sorted_by_execution_time():
    """Fills must be strictly sorted by execution index, even if orders completed out-of-order."""
    tr = OrderLevelTracker()
    # Order 1: added at ts=100 (idx=0), partial fill at ts=105 (idx=1), full fill at ts=500 (idx=4)
    tr.on_event(NormalizedEvent(EventType.ADD, 100, 1, Side.Bid, 100, 50))      # idx 0
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 105, 1, Side.NONE, 0, 10))  # idx 1 (first fill)

    # Order 2: added at ts=200 (idx=2), full fill at ts=210 (idx=3)
    tr.on_event(NormalizedEvent(EventType.ADD, 200, 2, Side.Ask, 101, 50))      # idx 2
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 210, 2, Side.NONE, 0, 50))  # idx 3 (Order 2 finishes here!)

    # Order 1 finally finishes at ts=500
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 500, 1, Side.NONE, 0, 40))  # idx 4 (Order 1 finishes here!)

    # Completed orders list has Order 2 first, then Order 1
    assert [o.order_id for o in tr.completed] == [2, 1]

    # fills_from_tracker must sort by execution event idx (Order 1 first at idx 1, Order 2 second at idx 3)
    fills = fills_from_tracker(tr.completed)
    assert len(fills) == 2
    assert fills[0].idx == 1 and fills[0].side == Side.Bid
    assert fills[1].idx == 3 and fills[1].side == Side.Ask

    # adverse_selection_report must also handle out-of-order fills safely
    scrambled = [fills[1], fills[0]]
    mids = [100.0] * 10
    rep = adverse_selection_report(scrambled, mids, horizons=(1,))
    assert rep["n_fills"] == 2


def test_off_market_trade_messages_do_not_advance_book_event_clock():
    """Verify that off-market trade messages (EventType.TRADE) do not advance the book event clock.

    Regression test for M03 (#37).
    """
    tr = QueueTracker()
    tr.on_event(NormalizedEvent(EventType.ADD, 1, 1, Side.Bid, 100, 10))
    seq_start = tr.seq
    mids_start = len(tr.mid_history)
    assert seq_start == 1 and mids_start == 1

    # Off-market trade print (EventType.TRADE)
    tr.on_event(NormalizedEvent(EventType.TRADE, 2, 0, Side.Bid, 100, 5))
    # Must NOT advance clock or append duplicate mid
    assert tr.seq == seq_start
    assert len(tr.mid_history) == mids_start

    # Book-affecting event (EXECUTE) advances the clock
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 3, 1, Side.Bid, 100, 10))
    assert tr.seq == seq_start + 1
    assert len(tr.mid_history) == mids_start + 1

    # OrderLevelTracker with count_trades_in_clock=False
    olt = OrderLevelTracker(count_trades_in_clock=False)
    olt.on_event(NormalizedEvent(EventType.ADD, 1, 1, Side.Bid, 100, 10))
    assert olt.index == 1
    olt.on_event(NormalizedEvent(EventType.TRADE, 2, 0, Side.Bid, 100, 5))
    assert olt.index == 1
    assert olt.stats["trades"] == 1
    olt.on_event(NormalizedEvent(EventType.EXECUTE, 3, 1, Side.Bid, 100, 10))
    assert olt.index == 2

    # is_book_affecting and filter_book_events
    events = [
        NormalizedEvent(EventType.ADD, 1, 1, Side.Bid, 100, 10),
        NormalizedEvent(EventType.TRADE, 2, 0, Side.Bid, 100, 5),
        NormalizedEvent(EventType.EXECUTE, 3, 1, Side.Bid, 100, 10),
    ]
    assert is_book_affecting(events[0]) is True
    assert is_book_affecting(events[1]) is False
    assert is_book_affecting(events[2]) is True

    filtered = filter_book_events(events)
    assert len(filtered) == 2
    assert [e.kind for e in filtered] == [EventType.ADD, EventType.EXECUTE]


def test_matched_unexecuted_control_drift():
    mids = [100.0, 101.0, 102.0, 105.0, 108.0]
    controls = [
        PassiveFill(idx=0, side=Side.Bid),
        PassiveFill(idx=1, side=Side.Ask),
    ]
    d = matched_unexecuted_control_drift(controls, mids, h=2)
    # Bid at 0 over h=2: mids[2] - mids[0] = 102 - 100 = 2.0
    # Ask at 1 over h=2: -(mids[3] - mids[1]) = -(105 - 101) = -4.0
    np.testing.assert_allclose(d, [2.0, -4.0])


