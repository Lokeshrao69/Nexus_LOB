"""Tests for the compute_metrics flow-tape accumulator (WS-1, queue_dynamics).

Covers: context-manager protocol (open + drag across a step loop + closure
summary), the per-step StepFlow record chain (no-lookahead structural lock),
exact hand-known queue_decay / latency_attrition / CAR numbers on a crafted
tape, determinism across identical tapes, external-tracker reuse, and the
realised behaviour on a synthetic drift tape.
"""
from __future__ import annotations

import json
from itertools import pairwise

import pytest
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.research.queue_dynamics import QueueTracker, compute_metrics
from nexus_quant.research.synthetic_flow import FlowConfig, SyntheticFlow

# ---- helpers ----------------------------------------------------------------- #

def _add(oid: int, side: Side, px: int, size: int, ts: int) -> NormalizedEvent:
    return NormalizedEvent(EventType.ADD, ts_ns=ts, order_id=oid, side=side,
                           price_ticks=px, size=size)


def _exe(oid: int, size: int, ts: int) -> NormalizedEvent:
    return NormalizedEvent(EventType.EXECUTE, ts_ns=ts, order_id=oid,
                           side=Side.NONE, price_ticks=0, size=size)


def _hand_tape():
    """A crafted ask-side tape with known touches, mids and OFI steps.

    Pre-seed Bid 10000 x10 / Ask 10001 x8; then inside the tape:
      1 ADD  Ask 10002 x5   (ts 1000)  → ask touch stays 8,    ofi 0
      2 EXE  oid2 x3        (ts 2000)  → ask touch 5,          ofi +3
      3 EXE  oid2 x5        (ts 3000)  → best ask 10002 (x5),  ofi 0
      4 ADD  Ask 10001 x4   (ts 4000)  → ask touch 4,          ofi +1

    OFI is the L1 window (best-level only): step 3 is a 5→5 level swap and
    step 4 pushes the 10002 level out of L1 (an apparent 5→4 ask shrink).
    Mids: 10000.5, 10000.5, 10000.5, 10001.0, 10000.5 (no empty-side 0s).
    """
    tr = QueueTracker()
    tr.on_event(_add(1, Side.Bid, 10000, 10, 0))
    tr.on_event(_add(2, Side.Ask, 10001, 8, 0))
    events = [
        _add(4, Side.Ask, 10002, 5, 1000),
        _exe(2, 3, 2000),
        _exe(2, 5, 3000),
        _add(5, Side.Ask, 10001, 4, 4000),
    ]
    return tr, events


# ---- (A) context-manager protocol + records -------------------------------- #


def test_drag_across_step_loop_and_closure_summary():
    tr, events = _hand_tape()
    with compute_metrics(tr, h=1) as m:
        for ev in events:
            m(ev)  # the "drag" — one call per step flow event
    out = m.metrics()
    # full-episode accumulation + frozen closure summary
    assert out["n"] == 4
    assert len(m.records) == 4
    assert out["kind_counts"] == {"ADD": 2, "EXECUTE": 2}
    assert m.result == out
    assert m.metrics() == out          # stable after the block
    # the accumulator drove a real tracker in lockstep
    assert m.tracker.level_size(Side.Ask, 10001) == 4
    assert m.tracker.level_size(Side.Bid, 10000) == 10
    # per-step record fields (mid/spread/touches)
    assert m.records[0].mid_prev == pytest.approx(10000.5)
    assert m.records[1].mid == pytest.approx(10000.5)
    assert m.records[2].spread == 2      # best ask moved to 10002
    assert m.records[3].bid_touch == 10 and m.records[3].ask_touch == 4


def test_records_form_no_lookahead_chain():
    """Each step's prev state is the previous step's post state (the chain lock)."""
    f = SyntheticFlow(FlowConfig(n_events=150, seed=11)); f.generate()
    with compute_metrics() as m:
        for ev in f.events:
            m(ev)
    recs = m.records
    assert recs[0].mid_prev == 0.0 and recs[0].bid_touch_prev == 0  # fresh book
    for a, b in pairwise(recs):
        assert b.mid_prev == pytest.approx(a.mid)
        assert b.bid_touch_prev == a.bid_touch
        assert b.ask_touch_prev == a.ask_touch
        assert b.step == a.step + 1


# ---- (B) hand-known queue decay / latency attrition ------------------------ #


def test_queue_decay_hand_known():
    tr, events = _hand_tape()
    with compute_metrics(tr, h=1) as m:
        for ev in events:
            m(ev)
    qb = m.metrics()["queue_decay"]["bid"]
    qa = m.metrics()["queue_decay"]["ask"]
    # bid never shrank: [10,10,10,10,10]
    assert qb["n_pairs"] == 4 and qb["n_eaten"] == 0
    assert qb["mean_attrition"] == 0.0 and qb["consumed_fraction"] == 0.0
    assert qb["endured_fraction"] == pytest.approx(1.0)
    assert (qb["initial"], qb["final"]) == (10, 10)
    # ask: [8,8,5,5,4] → eaten 3/8 then 1/5, over 4 pairs (flat steps count 0)
    assert qa["n_pairs"] == 4 and qa["n_eaten"] == 2
    assert qa["mean_attrition"] == pytest.approx((0.375 + 0.2) / 4)
    assert qa["mean_eaten_severity"] == pytest.approx(0.2875)
    assert qa["mean_growth"] == 0.0
    assert qa["endured_fraction"] == pytest.approx(0.5)
    assert qa["consumed_fraction"] == pytest.approx(0.5)


def test_latency_attrition_hand_known():
    tr, events = _hand_tape()
    with compute_metrics(tr, h=1) as m:
        for ev in events:
            m(ev)
    la = m.metrics()["latency_attrition"]["ask"]
    # Σ attrition 0.575 over Σ dt 4 ms → rate 143750 s⁻¹
    assert la["n_pairs"] == 4 and la["n_eaten"] == 2
    assert la["rate_s"] == pytest.approx(143750.0)
    assert la["halflife_s"] == pytest.approx(0.69314718056 / 143750.0, rel=1e-3)
    assert la["total_wait_s"] == pytest.approx(4e-6, abs=1e-12)
    assert la["clock_cadence_s"] == pytest.approx(1e-6, abs=1e-12)
    # the bid side never attenuated
    lb = m.metrics()["latency_attrition"]["bid"]
    assert lb["n_eaten"] == 0 and lb["rate_s"] == 0.0


# ---- (C) hand-known CAR + flow/mid/spread summaries ------------------------- #


def test_car_hand_known_and_summary_stats():
    tr, events = _hand_tape()
    with compute_metrics(tr, h=1) as m:
        for ev in events:
            m(ev)
    out = m.metrics()
    car = out["car"]
    # recorded per-step OFI = [0, +3, 0, +1] (L1 view: the 10001→10002 touch swap
    # is a size 5→5 level swap with OFI 0, and ADD 10001×4 reads as +1 after the
    # higher ask is pushed out of the L1 window). Responses (h=1):
    #   10000.5→10000.5 (0, flow 0), →10001.0 (+0.5, flow +3), →10000.5 (−0.5, flow 0)
    assert car["h"] == 1 and car["n"] == 3
    assert car["mean_response_ticks"] == 0.0
    assert car["rank_ic"] == pytest.approx(0.8660254038)  # rank x=[0,3,0], y=[0,.5,−.5]
    assert set(car["by_kind"]) == {"ADD", "EXECUTE"}
    assert car["by_kind"]["EXECUTE"]["mean_ticks"] == 0.0
    assert car["by_kind"]["EXECUTE"]["n"] == 2
    terc = car["by_flow_tercile"]
    # quantiles of [0,0,3] are {0, 1} → two populated bins: flow ≤1 (n2, −0.25)
    # and flow >1 (n1, +0.5) — high-flow step is precisely the one that moved up
    assert len(terc) == 2
    assert terc[0]["mean_response_ticks"] == pytest.approx(-0.25)
    assert terc[1]["mean_x"] == pytest.approx(3.0)
    assert terc[1]["mean_response_ticks"] == pytest.approx(0.5)
    # summary stats
    assert out["mid_start"] == pytest.approx(10000.5)
    assert out["mid_total_ticks"] == 0.0
    assert out["mid_abs_ticks"] == pytest.approx(1.0)     # 0+0+0.5+0.5
    assert out["spread_mean_ticks"] == pytest.approx(1.25)
    assert out["n_spread_steps"] == 4
    assert out["flow_total"] == pytest.approx(4.0)        # 0+3+0+1
    assert out["flow_abs_mean"] == pytest.approx(1.0)


def test_metrics_is_json_clean():
    tr, events = _hand_tape()
    with compute_metrics(tr, h=1) as m:
        for ev in events:
            m(ev)
    text = json.dumps(m.metrics())
    assert "NaN" not in text and "Infinity" not in text


# ---- (D) external-tracker reuse + partial/exception semantics -------------- #


def test_reuses_external_tracker():
    tr = QueueTracker()
    tr.on_event(_add(1, Side.Bid, 10000, 10, 0))
    tr.on_event(_add(2, Side.Ask, 10001, 8, 0))
    with compute_metrics(tr, h=1) as m:
        assert m._initial["bid"] == 10 and m._initial["ask"] == 8
        m(_exe(1, 4, 1000))           # bid touch 10 → 6
    assert m.tracker is tr
    qb = m.metrics()["queue_decay"]["bid"]
    assert qb["consumed_fraction"] == pytest.approx(0.4)
    assert m.tracker.level_size(Side.Bid, 10000) == 6


def test_partial_metrics_mid_loop_and_exception_tolerance():
    f = SyntheticFlow(FlowConfig(n_events=200, seed=1)); f.generate()
    m = compute_metrics()
    try:
        with m:
            for i, ev in enumerate(f.events):
                m(ev)
                if i == 49:
                    partial = m.metrics()          # valid mid-loop, partial tape
                    assert partial["n"] == 50
                if i == 99:
                    raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("exception did not propagate")
    # the raise lost no records; the frozen closure summary is the partial tape
    assert m.result is not None and m.result["n"] == 100
    assert len(m.records) == 100
    assert m.metrics() == m.result


# ---- (E) full-episode accumulation over a synthetic tape -------------------- #


def test_synthetic_episode_is_deterministic_and_lockstep():
    def run(seed: int):
        f = SyntheticFlow(FlowConfig(n_events=500, seed=seed)); f.generate()
        with compute_metrics() as m:
            for ev in f.events:
                m(ev)
        return f, m.metrics(), m

    f1, out1, m1 = run(7)
    _, out2, _ = run(7)
    # identical tape → byte-identical metrics (seeded CIs, sorted keys)
    assert out1 == out2
    # the accumulator's own tracker reconstructs the generator's ground truth
    assert m1.tracker.reconstruct() == f1.internal_book()
    assert out1["n"] == len(f1.events)
    assert sum(out1["kind_counts"].values()) == out1["n"]
    assert set(out1["kind_counts"]) <= {k.name for k in EventType}
    assert out1["mid_start"] > 0 and out1["mid_end"] > 0
    # the final recorded mid matches the tape's own last mid
    assert m1.records[-1].mid == pytest.approx(f1.mid_trace[-1])


def test_car_responds_to_flow_on_drift_tape():
    # a small deterministic upward-drift tape (same shape as FLOW_PRESETS.drift)
    cfg = FlowConfig(n_events=4000, seed=9, drift_ticks=0.6, vol_per_event=0.6)
    f = SyntheticFlow(cfg); f.generate()
    with compute_metrics(h=5) as m:
        for ev in f.events:
            m(ev)
    car = m.metrics()["car"]
    assert car["n"] > 500
    assert car["mean_response_ticks"] > 0.0        # upward drift shows in responses
    assert -1.0 <= car["rank_ic"] <= 1.0
    bins = car["by_flow_tercile"]
    assert len(bins) >= 2
    assert sum(b["n"] for b in bins) == car["n"]
    assert car["by_kind"]  # a real tape exercises several event kinds


def test_empty_tape_does_not_blow_up():
    with compute_metrics() as m:
        pass
    out = m.metrics()
    assert out["n"] == 0
    assert out["queue_decay"]["bid"]["mean_attrition"] == 0.0
    assert out["latency_attrition"]["ask"]["rate_s"] is None
    assert out["car"]["n"] == 0