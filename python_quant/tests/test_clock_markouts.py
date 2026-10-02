"""E6b — clock-time passive-fill markout tests (synthetic tapes only).

The load-bearing test is ``test_mechanical_level_sweep_moves_before_not_after``:
a fill that empties the best level must move the ``before`` mid and leave the
``after`` mid untouched. That is the known mechanical effect the clock study
exists to separate out, so it is pinned numerically, not just by sign.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_port import StubBookAdapter
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.replay import ReplayEngine
from nexus_quant.research.adverse_selection import p_adverse
from nexus_quant.research.clock_markouts import (
    BOOTSTRAP_MIN_BLOCK_NS,
    CLOCK_HORIZONS_NS,
    E6_HEADLINE_H_EVENTS,
    LADDER_STAT,
    MEDIAN_LATTICE_NOTE,
    MID_LATTICE_PX0001,
    NS_PER_MS,
    NS_PER_S,
    REBATE_ASSUMPTION_NOTE,
    REBATE_SCENARIOS_PX0001,
    TICK_PX_UNITS,
    AggressorEvent,
    ClockFill,
    ClockFillCollector,
    _reversion_table,
    _sign,
    attribution_check,
    attribution_ladder,
    block_plan,
    clock_markout_report,
    clock_overlap_lag,
    event_time_markout,
    events_per_window,
    group_aggressor_events,
    group_masks,
    horizon_label,
    location_stats,
    markout_panel,
    markout_stats,
    render_clock_markouts_md,
    sign_shares,
    trimmed_mean,
)
from nexus_quant.research.features import mid, spread_ticks
from nexus_quant.research.queue_dynamics import REGULAR_CLOSE_NS, OrderLevelTracker

OPEN_NS = int(9.5 * 3_600 * NS_PER_S)  # 09:30 ET


# --------------------------------------------------------------------------- #
# tape helpers                                                                #
# --------------------------------------------------------------------------- #
def add(ts: int, oid: int, side: Side, px: int, sz: int) -> NormalizedEvent:
    return NormalizedEvent(EventType.ADD, ts, oid, side, px, sz, raw_type="A")


def execute(ts: int, oid: int, sz: int) -> NormalizedEvent:
    return NormalizedEvent(EventType.EXECUTE, ts, oid, Side.NONE, 0, sz, raw_type="E")


def cancel(ts: int, oid: int, sz: int) -> NormalizedEvent:
    return NormalizedEvent(EventType.DELETE, ts, oid, Side.NONE, 0, sz, raw_type="D")


def replay(events) -> dict:
    """Mirror ``run_research.replay_symbol``'s row series on a synthetic tape.

    Returns the regular-session ``mids`` / ``ts`` / ``half_spreads`` series (each
    recorded *after* its event is applied, exactly as the real pipeline does),
    plus the harvested ``ClockFill``s and their row indices.
    """
    book = StubBookAdapter()
    rep = ReplayEngine(events, book)
    tracker = OrderLevelTracker()
    collector = ClockFillCollector()
    mids: list[float] = []
    half: list[float] = []
    ts: list[int] = []
    tape_idx: list[int] = []
    for i, ev in enumerate(events):
        collector.observe(i, ev, tracker)
        rep.apply(ev)
        tracker.on_event(ev)
        if not (OPEN_NS <= ev.ts_ns < REGULAR_CLOSE_NS):
            continue
        v = book.snapshot()
        mids.append(mid(v))
        half.append(spread_ticks(v) / 2.0)
        ts.append(int(ev.ts_ns))
        tape_idx.append(i)
    pos = {t: j for j, t in enumerate(tape_idx)}
    fills, rows = [], []
    for f in collector.fills:
        j = pos.get(f.tape_idx)
        if j is not None:
            fills.append(f)
            rows.append(j)
    return {"mids": mids, "ts": ts, "half_spreads": half, "fills": fills, "rows": rows}


def panel_of(r: dict, horizon_ns: int, **kw):
    """Per-fill panel (the secondary path) — one observation per fill."""
    return markout_panel(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"], horizon_ns, **kw
    )


def event_panel_of(r: dict, horizon_ns: int, **kw):
    """Per-aggressor-event panel (the main path)."""
    events, rows = group_aggressor_events(r["fills"], r["rows"])
    return markout_panel(
        events, rows, r["mids"], r["ts"], r["half_spreads"], horizon_ns, **kw
    )


# --------------------------------------------------------------------------- #
# the mechanical case                                                         #
# --------------------------------------------------------------------------- #
def test_mechanical_level_sweep_moves_before_not_after():
    """A fill that empties the best level moves the "before" mid, not the "after" one.

    Book: bid 100 @ 10 (two orders), ask 110 alone @ 5. The sell side's touch is
    held by one order; executing all 5 of it clears the level and the best ask
    jumps 110 → 120, so the mid goes 105 → 110. Nothing happens afterwards.

    * ``before`` mid = 105 → markout = s·(110 − 105) = −5 for a resting ask: the
      full mechanical jump, scored as adverse.
    * ``after`` mid = 110 → markout = 0: no information, correctly measured.
    """
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 6),
        add(t + 1, 2, Side.Bid, 100, 4),
        add(t + 2, 3, Side.Ask, 110, 5),      # the touch, held alone
        add(t + 3, 4, Side.Ask, 120, 7),      # the level behind it
        execute(t + 10 * NS_PER_MS, 3, 5),    # sweeps the whole ask touch
        add(t + NS_PER_S, 5, Side.Bid, 99, 1),  # a later, mid-neutral event to land on
    ]
    r = replay(evs)
    assert len(r["fills"]) == 1
    f = r["fills"][0]
    assert f.side == Side.Ask
    assert f.emptied_touch is True
    assert f.at_touch is True
    assert f.level_size_at_fill == 5
    assert f.ahead_at_fill == 0

    p = panel_of(r, 100 * NS_PER_MS)
    assert p.drops["past_close"] == 0
    assert p.drops["past_tape"] == 0
    assert p.drops["no_pre_mid"] == 0
    assert p.drops["bad_mid"] == 0
    # mechanical jump lands entirely in the "before" reference
    assert p.markout["before"][0] == pytest.approx(-5.0)
    assert p.markout["after"][0] == pytest.approx(0.0)
    # the "before" decomposition: 5 ticks of half-spread earned, 5 given straight
    # back by the markout → realized spread 0. The "after" reference books the same
    # realized 0 as "earned nothing, lost nothing" — the mid it measures from is
    # already the post-jump 110, which is exactly the execution price.
    assert p.half_spread_earned["before"][0] == pytest.approx(5.0)
    assert p.half_spread_earned["after"][0] == pytest.approx(0.0)
    assert p.realized_spread["before"][0] == pytest.approx(0.0)
    assert p.realized_spread["after"][0] == pytest.approx(0.0)
    # quoted half-spread at the fill is the PRE-fill (110 − 100) / 2
    assert p.quoted_half_spread[0] == pytest.approx(5.0)


def test_partial_fill_leaving_level_has_no_mechanical_gap():
    """A fill that does not clear the level moves neither mid, so both refs agree."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 10),
        add(t + 1, 2, Side.Ask, 110, 9),
        add(t + 2, 3, Side.Ask, 110, 4),      # level 110 now holds 13
        execute(t + 5 * NS_PER_MS, 2, 9),     # 4 left at 110 → BBO unchanged
        add(t + NS_PER_S, 4, Side.Bid, 99, 1),
    ]
    r = replay(evs)
    f = r["fills"][0]
    assert f.emptied_touch is False
    assert f.level_size_at_fill == 13
    assert f.size_frac == pytest.approx(9 / 13)

    p = panel_of(r, 100 * NS_PER_MS)
    assert p.markout["before"][0] == pytest.approx(0.0)
    assert p.markout["after"][0] == pytest.approx(0.0)
    assert p.markout["before"][0] == p.markout["after"][0]


def test_realized_spread_is_reference_invariant():
    """`realized = s*(mid[t+H] - exec_vwap)`, identical under both reference mids.

    The failure this guards against is a decomposition that adds the half-spread
    measured against the **before** mid to a markout measured against the
    **after** mid. That silently drops the before->after jump and overstates what
    a passive fill earns, so the test checks three things:

    * each reference's own `earned + markout` equals that reference's realized;
    * the two references give the *same* realized spread;
    * the mismatched pairing gives a *different* number, equal to the jump --
      i.e. the test would fail if the code were wired that way.
    """
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 5),
        add(t + 2, 3, Side.Bid, 98, 5),
        add(t + 3, 4, Side.Ask, 112, 5),
        execute(t + NS_PER_MS, 2, 5),          # clears ask touch: mid 105 -> 106
        add(t + 2 * NS_PER_MS, 5, Side.Ask, 108, 5),
        add(t + 3 * NS_PER_S, 6, Side.Bid, 97, 1),
    ]
    r = replay(evs)
    for p_ in (panel_of(r, NS_PER_S), event_panel_of(r, NS_PER_S)):
        for ref in ("before", "after"):
            np.testing.assert_allclose(
                p_.half_spread_earned[ref] + p_.markout[ref], p_.realized_spread[ref]
            )
        np.testing.assert_allclose(
            p_.realized_spread["before"], p_.realized_spread["after"]
        )
        # the references really do differ, so the invariance above is not vacuous
        jump = p_.half_spread_earned["before"] - p_.half_spread_earned["after"]
        assert np.any(np.abs(jump[np.isfinite(jump)]) > 0.0)
        np.testing.assert_allclose(jump, p_.markout["after"] - p_.markout["before"])
        # the cross-wired pairing is wrong by exactly that jump -- what we guard against
        mismatched = p_.half_spread_earned["before"] + p_.markout["after"]
        np.testing.assert_allclose(mismatched, p_.realized_spread["after"] + jump)
        assert not np.allclose(mismatched, p_.realized_spread["after"])


def test_realized_spread_equals_terminal_mid_minus_vwap():
    """Computed straight from the terminal mid and the execution VWAP, no reference.

    A multi-fill sweep at two prices, so the VWAP is a genuine weighted average and
    a per-fill average would give a different answer.
    """
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 10),
        add(t + 1, 2, Side.Ask, 110, 2),      # 2 shares at 110
        add(t + 2, 3, Side.Ask, 120, 6),      # 6 shares at 120
        add(t + 3, 4, Side.Ask, 130, 9),      # depth behind
        execute(t + 4, 2, 2),                 # one aggressor, both price levels
        execute(t + 4, 3, 6),
        add(t + NS_PER_S, 5, Side.Bid, 99, 1),
    ]
    r = replay(evs)
    events, _rows = group_aggressor_events(r["fills"], r["rows"])
    assert len(events) == 1
    e = events[0]
    vwap = (110 * 2 + 120 * 6) / 8
    assert e.vwap_price == pytest.approx(vwap)
    p_ = event_panel_of(r, 100 * NS_PER_MS)
    terminal = r["mids"][-1]
    # s = -1 for a resting ask: realized = P - mid[t+H]
    want = -(terminal - vwap)
    for ref in ("before", "after"):
        assert p_.realized_spread[ref][0] == pytest.approx(want)


def test_sign_convention_by_side():
    """Negative = adverse, from the passive order's side, for both sides."""
    t = OPEN_NS
    base = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 5),
        add(t + 2, 3, Side.Bid, 90, 5),
        add(t + 3, 4, Side.Ask, 120, 5),
    ]
    # resting ask swept → best ask 110 → 120, mid up: adverse for the seller
    ask = replay([*base, execute(t + NS_PER_MS, 2, 5), add(t + NS_PER_S, 9, Side.Bid, 80, 1)])
    p_ask = panel_of(ask, 100 * NS_PER_MS)
    assert ask["fills"][0].side == Side.Ask
    assert p_ask.markout["before"][0] < 0

    # resting bid swept → best bid 100 → 90, mid down: adverse for the buyer
    bid = replay([*base, execute(t + NS_PER_MS, 1, 5), add(t + NS_PER_S, 9, Side.Ask, 130, 1)])
    p_bid = panel_of(bid, 100 * NS_PER_MS)
    assert bid["fills"][0].side == Side.Bid
    assert p_bid.markout["before"][0] < 0
    # symmetric tapes → the same adverse magnitude on either side
    assert p_bid.markout["before"][0] == pytest.approx(p_ask.markout["before"][0])


# --------------------------------------------------------------------------- #
# horizon mechanics                                                           #
# --------------------------------------------------------------------------- #
def test_terminal_mid_is_last_event_at_or_before_horizon():
    """The markout reads the last book event ``<= fill_ts + H``, never the one after."""
    t = OPEN_NS
    tf = t + 3  # the fill's own timestamp — horizons are measured from here
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 5),
        add(t + 2, 3, Side.Ask, 112, 5),
        execute(tf, 2, 5),                                  # fill: mid 105 → 106
        add(tf + NS_PER_S - 1, 4, Side.Bid, 104, 5),        # inside 1 s:  mid → 108
        add(tf + NS_PER_S + 1, 5, Side.Bid, 106, 5),        # just outside: mid → 109
        add(tf + 10 * NS_PER_S, 6, Side.Ask, 130, 1),
    ]
    r = replay(evs)
    assert r["fills"][0].side == Side.Ask  # s = −1, so an up-move is adverse
    # at the horizon edge: the fill_ts+1s−1 event counts, the fill_ts+1s+1 one does not
    p1 = panel_of(r, NS_PER_S)
    assert p1.markout["after"][0] == pytest.approx(-(108.0 - 106.0))
    # a wider horizon picks up the later event
    p2 = panel_of(r, 2 * NS_PER_S)
    assert p2.markout["after"][0] == pytest.approx(-(109.0 - 106.0))


def test_horizon_running_past_close_is_dropped_and_counted():
    """Fills whose horizon crosses 16:00 are dropped, not truncated, and counted."""
    t = REGULAR_CLOSE_NS - 30 * NS_PER_S  # 30 s before the close
    evs = [
        add(OPEN_NS, 1, Side.Bid, 100, 5),
        add(OPEN_NS + 1, 2, Side.Ask, 110, 5),
        add(OPEN_NS + 2, 3, Side.Ask, 112, 5),
        execute(OPEN_NS + 3, 2, 5),      # early fill: every horizon fits
        add(t, 4, Side.Ask, 110, 5),
        execute(t + NS_PER_MS, 4, 5),    # late fill: the 60 s horizon runs past 16:00
        add(REGULAR_CLOSE_NS - NS_PER_S, 5, Side.Bid, 99, 1),
    ]
    r = replay(evs)
    assert len(r["fills"]) == 2
    p_short = panel_of(r, 100 * NS_PER_MS)
    assert p_short.drops["past_close"] == 0
    p_long = panel_of(r, 60 * NS_PER_S)
    assert p_long.drops["past_close"] == 1
    assert np.isnan(p_long.markout["after"][1])
    assert not np.isnan(p_short.markout["after"][1])


def test_horizon_past_tape_end_is_dropped():
    """A horizon running past the last recorded event is dropped, never clamped."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 5),
        add(t + 2, 3, Side.Ask, 112, 5),
        execute(t + 3, 2, 5),
        add(t + 5 * NS_PER_MS, 4, Side.Bid, 101, 1),  # tape ends 5 ms after the fill
    ]
    r = replay(evs)
    assert panel_of(r, NS_PER_MS).drops["past_tape"] == 0
    p = panel_of(r, NS_PER_S)
    assert p.drops["past_tape"] == 1
    assert np.isnan(p.markout["after"][0])
    assert np.isnan(p.markout["before"][0])


def test_first_row_fill_has_no_pre_mid():
    """``ref = "before"`` is undefined for a fill on the first session row."""
    f = ClockFill(
        tape_idx=0, ts_ns=OPEN_NS, side=Side.Ask, price=110, size=1,
        ahead_at_fill=0, level_size_at_fill=1, at_touch=True, emptied_touch=True,
    )
    p = markout_panel(
        [f], [0], [105.0, 106.0], [OPEN_NS, OPEN_NS + NS_PER_MS], [5.0, 5.0], 100 * NS_PER_MS
    )
    assert p.drops["no_pre_mid"] == 1
    assert p.n_usable == 0


def test_one_sided_book_endpoint_is_dropped():
    """A zero mid at either endpoint is a drop, not a huge spurious markout."""
    f = ClockFill(
        tape_idx=1, ts_ns=OPEN_NS + 1, side=Side.Bid, price=100, size=1,
        ahead_at_fill=0, level_size_at_fill=1, at_touch=True, emptied_touch=True,
    )
    ts = [OPEN_NS, OPEN_NS + 1, OPEN_NS + 10 * NS_PER_MS]
    p = markout_panel([f], [1], [0.0, 105.0, 106.0], ts, [5.0, 5.0, 5.0], NS_PER_MS)
    assert p.drops["bad_mid"] == 1
    assert p.n_usable == 0


# --------------------------------------------------------------------------- #
# collector                                                                   #
# --------------------------------------------------------------------------- #
def test_collector_keeps_only_first_execution_per_order():
    """One row per filled order — the same population as the event-time E6."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 10),
        execute(t + 2, 2, 3),   # first fill → recorded
        execute(t + 3, 2, 3),   # second fill on the same order → not recorded
        execute(t + 4, 2, 4),   # third, finishes the order → not recorded
    ]
    r = replay(evs)
    assert len(r["fills"]) == 1
    f = r["fills"][0]
    assert f.size == 3
    assert f.level_size_at_fill == 10
    assert f.emptied_touch is False  # 3 of 10 does not clear the level


def test_collector_records_queue_position_at_fill_not_at_placement():
    """``ahead_at_fill`` is measured when the fill happens, not when the order rested."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 4),   # front of the ask queue
        add(t + 2, 3, Side.Ask, 110, 6),   # our order: 4 ahead at placement
        cancel(t + 3, 2, 4),               # the order ahead leaves → 0 ahead now
        execute(t + 4, 3, 6),
        add(t + NS_PER_S, 4, Side.Ask, 120, 1),
    ]
    r = replay(evs)
    f = r["fills"][0]
    assert f.ahead_at_fill == 0            # not 4
    assert f.level_size_at_fill == 6
    assert f.queue_frac == pytest.approx(0.0)


def test_collector_ignores_untracked_and_non_execute_events():
    coll = ClockFillCollector()
    tr = OrderLevelTracker()
    coll.observe(0, add(OPEN_NS, 1, Side.Bid, 100, 5), tr)
    assert coll.fills == []
    coll.observe(1, execute(OPEN_NS, 999, 5), tr)  # never added
    assert coll.fills == []


# --------------------------------------------------------------------------- #
# statistics                                                                  #
# --------------------------------------------------------------------------- #
def test_three_adverse_shares_are_consistent():
    """unconditional + zero shares partition the sample; conditional renormalizes."""
    mk = np.array([-2.0, -1.0, 0.0, 0.0, 3.0, np.nan])
    st = markout_stats(
        mk, mk, np.zeros(6), np.full(6, 5.0), np.zeros(6),
        lag=1, block=1, n_boot=20, seed=1,
    )
    assert st["n"] == 5
    assert st["share_adverse"] == pytest.approx(2 / 5)
    assert st["share_zero"] == pytest.approx(2 / 5)
    assert st["share_adverse_conditional"] == pytest.approx(2 / 3)
    # the conditional figure is the inflated one whenever zero moves are common
    assert st["share_adverse_conditional"] > st["share_adverse"]
    assert st["share_adverse"] + st["share_zero"] + np.mean(mk[~np.isnan(mk)] > 0) == pytest.approx(1.0)


def test_markout_stats_scales_by_half_spread_and_reports_nw():
    mk = np.array([-10.0, -20.0, -30.0, -40.0])
    st = markout_stats(
        mk, mk + 5.0, np.full(4, 5.0), np.full(4, 10.0), np.zeros(4),
        lag=1, block=1, n_boot=50, seed=7,
    )
    m, r = st["markout_px0001"], st["realized_spread_px0001"]
    assert m["mean"] == pytest.approx(-25.0)
    assert st["mean_half_spread_px0001"] == pytest.approx(10.0)
    assert st["mean_markout_half_spreads"] == pytest.approx(-2.5)
    assert r["mean"] == pytest.approx(-20.0)
    assert st["mean_realized_spread_half_spreads"] == pytest.approx(-2.0)
    assert m["nw_t"] < 0 and np.isfinite(m["nw_se"])
    assert st["nw_lag"] == 1
    assert m["mean_ci95"]["lo"] <= m["mean"] <= m["mean_ci95"]["hi"]


def test_markout_stats_degenerate_inputs_do_not_raise():
    nan2 = np.array([np.nan, np.nan])
    st = markout_stats(nan2, nan2, nan2, nan2, nan2, lag=1, block=1, n_boot=10, seed=0)
    assert st["n"] == 0
    assert st["markout_px0001"]["mean_ci95"]["lo"] is None
    assert np.isnan(st["mean_markout_half_spreads"])


def test_clock_overlap_lag_counts_fills_inside_the_window():
    # ten fills, one per second: a 1 s horizon overlaps nothing, 5 s overlaps ~4
    ts = [OPEN_NS + i * NS_PER_S for i in range(10)]
    assert clock_overlap_lag(ts, NS_PER_S) == 1          # floored at 1
    assert clock_overlap_lag(ts, 5 * NS_PER_S) == 4
    # a horizon longer than the whole sample: median of 9, 8, …, 0 → 4
    assert clock_overlap_lag(ts, 100 * NS_PER_S) == 4
    assert clock_overlap_lag([OPEN_NS], NS_PER_S) == 1
    assert clock_overlap_lag(ts, 100 * NS_PER_S, cap=3) == 3                    # cap
    assert clock_overlap_lag([OPEN_NS, OPEN_NS + 1], 100 * NS_PER_S) == 1       # n − 1


def test_horizon_label():
    assert horizon_label(100 * NS_PER_MS) == "100ms"
    assert horizon_label(NS_PER_S) == "1s"
    assert horizon_label(60 * NS_PER_S) == "60s"
    assert horizon_label(1234) == "1234ns"
    assert [horizon_label(h) for h in CLOCK_HORIZONS_NS] == ["100ms", "1s", "10s", "60s"]


# --------------------------------------------------------------------------- #
# groups + report bundle                                                      #
# --------------------------------------------------------------------------- #
def _mechanical_tape(n: int = 120, *, seed: int = 11) -> dict:
    """``n`` sweeps that each clear a touch held by a single order — no information.

    One iteration per second. The touch is restored 500 ms after the sweep, well
    outside the 100 ms markout horizon, so the mid is genuinely *flat* over each
    fill's forward window: everything the ``before`` reference records is the fill
    event's own BBO jump, and the ``after`` reference must record nothing at all.

    Book per iteration: bid 90 / bid 100 / ask 110 / ask 120, one order each, so
    clearing the ask touch lifts the mid 105 → 110 and clearing the bid touch drops
    it 105 → 100 — half the 10-tick level gap. Either way the passive side loses
    exactly 5 ticks, every time, purely mechanically.
    """
    rng = np.random.default_rng(seed)
    evs: list[NormalizedEvent] = []
    for i in range(n):
        t = OPEN_NS + i * NS_PER_S
        oid = 10 * i + 1
        v = int(rng.integers(1, 20))
        ask_side = i % 2 == 0
        evs += [
            add(t + 0, oid, Side.Bid, 90, 5),
            add(t + 1, oid + 1, Side.Ask, 120, 5),
            add(t + 2, oid + 2, Side.Bid, 100, 5 if ask_side else v),
            add(t + 3, oid + 3, Side.Ask, 110, v if ask_side else 5),
            execute(t + 4, oid + 3 if ask_side else oid + 2, v),   # clears the touch
        ]
        # tear down AFTER the 100 ms horizon, so each fill's window stays flat
        evs += [cancel(t + 500 * NS_PER_MS + k, oid + k, 50) for k in range(4)]
    evs.append(add(OPEN_NS + n * NS_PER_S, 10 * n + 9, Side.Bid, 100, 1))
    evs.append(add(OPEN_NS + (n + 1) * NS_PER_S, 10 * n + 10, Side.Ask, 110, 1))
    return replay(evs)


def _mixed_tape(n: int = 240, *, seed: int = 23) -> dict:
    """``n`` fills spread across every breakdown bucket (both sides, all three queue
    thirds, small and large relative to the level, clearing the touch or not).

    Not a clean null like :func:`_mechanical_tape` — this one exists to populate the
    group masks and the report bundle, so coverage is all it guarantees. The
    ``variant`` cycle is what spreads the fills out:

    * 0 — our order alone at the touch, fully swept: front, large, touch cleared;
    * 1 — a big order ahead of us at the touch, we are swept: back of the level;
    * 2 — a mid-sized order ahead of us: middle of the level;
    * 3 — our order alone at the touch but only one share swept: front, small.
    """
    rng = np.random.default_rng(seed)
    evs: list[NormalizedEvent] = []
    for i in range(n):
        t = OPEN_NS + i * NS_PER_S
        oid = 10 * i + 1
        variant = i % 4
        ask_side = (i // 4) % 2 == 0
        side = Side.Ask if ask_side else Side.Bid
        touch_px = 110 if ask_side else 100
        v = int(rng.integers(4, 12))
        ahead = (0, 40, 8, 0)[variant]
        qty = v if variant != 3 else 1
        evs += [
            add(t + 0, oid, Side.Bid, 90, 5),
            add(t + 1, oid + 1, Side.Ask, 120, 5),
            add(t + 2, oid + 2, Side.Bid, 100, 5),
            add(t + 3, oid + 3, Side.Ask, 110, 5),
        ]
        if ahead:
            evs.append(add(t + 4, oid + 4, side, touch_px, ahead))
        else:
            # clear the pre-existing touch order so our order rests there alone
            evs.append(cancel(t + 5, oid + 3 if ask_side else oid + 2, 5))
        evs.append(add(t + 6, oid + 5, side, touch_px, v))
        evs.append(execute(t + 7, oid + 5, qty))
        evs += [cancel(t + 500 * NS_PER_MS + k, oid + k, 50) for k in range(6)]
    evs.append(add(OPEN_NS + n * NS_PER_S, 10 * n + 9, Side.Bid, 100, 1))
    evs.append(add(OPEN_NS + (n + 1) * NS_PER_S, 10 * n + 10, Side.Ask, 110, 1))
    return replay(evs)



def _continuous_tape(n: int = 80, *, seed: int = 31) -> dict:
    """``n`` touch sweeps on a book that is two-sided for the whole session.

    Deep bid 9000 and deep ask 11000 rest untouched from the first nanosecond, so
    every pre-window has a mid to measure from and the pre-fill control is never
    dropped for a one-sided book. One iteration per second; the touch is at
    10000 \u00b1 a slow upward drift, which is what the pre-fill control should pick up.
    """
    rng = np.random.default_rng(seed)
    evs: list[NormalizedEvent] = [
        add(OPEN_NS, 1, Side.Bid, 9000, 10_000),
        add(OPEN_NS + 1, 2, Side.Ask, 11_000, 10_000),
    ]
    oid = 100
    for i in range(n):
        t = OPEN_NS + (i + 2) * NS_PER_S          # first fill is 2 s into the session
        drift = 4 * i                              # a slow upward trend in the touch
        v = int(rng.integers(2, 20))
        ask_side = i % 2 == 0
        px = 10_000 + drift + (40 if ask_side else -40)
        evs += [
            add(t + 0, oid, Side.Ask if ask_side else Side.Bid, px, v),
            execute(t + 1, oid, v),                # clears the touch it just made
        ]
        oid += 1
    evs.append(add(OPEN_NS + (n + 4) * NS_PER_S, oid, Side.Bid, 9500, 1))
    return replay(evs)


def _informative_tape(n: int = 150, *, seed: int = 41) -> dict:
    """``n`` sweeps followed by a GENUINE post-fill move, half adverse, half not.

    The other fixtures leave the mid flat after the sweep, which makes
    `P(adverse | moved)` NaN — fine for the mechanical null, useless for exercising
    the attribution arithmetic. Here each sweep is followed 20 ms later by a quote
    change that moves the mid *further* against the passive side on even
    iterations and back in its favour on odd ones, so the conditional share sits
    strictly between 0 and 1 and every ladder step has something to measure.

    Deep bid 9000 / ask 11000 rest untouched all session, so the book is never
    one-sided and the pre-fill control is always defined. Every third iteration
    puts two resting orders at the touch and sweeps both at one timestamp, so the
    per-fill and per-event units genuinely differ.
    """
    rng = np.random.default_rng(seed)
    evs: list[NormalizedEvent] = [
        add(OPEN_NS, 1, Side.Bid, 9_000, 100_000),
        add(OPEN_NS + 1, 2, Side.Ask, 11_000, 100_000),
    ]
    oid = 100
    for i in range(n):
        t0 = OPEN_NS + (i + 2) * NS_PER_S
        v = int(rng.integers(2, 10))
        two = i % 3 == 0          # a 2-order sweep, so fills > events
        evs += [
            add(t0 + 0, oid + 0, Side.Bid, 9_940, 5),
            add(t0 + 1, oid + 1, Side.Ask, 10_060, 5),
            add(t0 + 2, oid + 2, Side.Bid, 9_990, 5),      # bid touch
            add(t0 + 3, oid + 3, Side.Ask, 10_010, v),     # ask touch, swept below
        ]
        if two:
            evs.append(add(t0 + 4, oid + 9, Side.Ask, 10_010, 3))
        # one aggressor clears the ask touch: mid 10000 -> 10025 (mechanical +25)
        evs.append(execute(t0 + 5, oid + 3, v))
        if two:
            evs.append(execute(t0 + 5, oid + 9, 3))
        # 20 ms later, a real move: against the seller on even i, for it on odd i
        if i % 2 == 0:
            evs.append(add(t0 + 20 * NS_PER_MS, oid + 4, Side.Bid, 10_010, 1))
        else:
            evs.append(add(t0 + 20 * NS_PER_MS, oid + 4, Side.Ask, 10_000, 1))
        # deep filler adds so an event-time h=5 window has rows without moving the BBO
        evs += [
            add(t0 + (30 + 10 * k) * NS_PER_MS, oid + 5 + k, Side.Bid, 9_800 + k, 1)
            for k in range(4)
        ]
        evs += [cancel(t0 + 500 * NS_PER_MS + k, oid + k, 50) for k in range(10)]
        oid += 20
    evs.append(add(OPEN_NS + (n + 4) * NS_PER_S, oid, Side.Bid, 9_500, 1))
    return replay(evs)


def test_group_masks_partition_by_side_queue_and_size():
    r = _mixed_tape()
    fills = r["fills"]
    m = group_masks(fills)
    n = len(fills)
    assert m["side=bid"].sum() + m["side=ask"].sum() == n
    assert m["queue=front"].sum() + m["queue=mid"].sum() + m["queue=back"].sum() == n
    assert m["size=small"].sum() + m["size=large"].sum() == n
    assert m["emptied_touch=yes"].sum() + m["emptied_touch=no"].sum() == n
    assert m["queue=front"].sum() > 0 and m["queue=back"].sum() > 0
    assert m["size=small"].sum() > 0 and m["size=large"].sum() > 0


def test_report_bundle_shape_and_json_safety():
    import json

    r = _mixed_tape()
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, NS_PER_S), n_boot=20, min_group=5,
    )
    assert res["n_fills"] == len(r["fills"])
    assert res["n_events"] > 0
    assert res["px_unit_usd"] == 1e-4
    assert res["tick_px_units"] == TICK_PX_UNITS == 100
    assert set(res["horizons"]) == {"100ms", "1s"}
    for blk in res["horizons"].values():
        for scope in ("events", "fills"):
            assert set(blk[scope]["drops"]) == {
                "past_close", "past_tape", "no_pre_mid", "bad_mid",
                "pre_past_open", "pre_bad_mid",
            }
            for ref in ("before", "after"):
                cell = blk[scope][ref]["overall"]
                assert cell["n"] > 0
                # every horizon carries mean, median and trimmed mean, each with a CI
                for key in ("markout_px0001", "realized_spread_px0001"):
                    loc = cell[key]
                    for stat in ("mean", "median", "trimmed_mean"):
                        assert stat in loc
                    for stat in ("mean", "trimmed_mean"):   # median: lattice, no CI
                        assert set(loc[stat + "_ci95"]) == {"lo", "hi", "mean", "se"}
                    assert "median_ci95" not in loc
                for key in ("realized_sign", "markout_sign"):
                    sg = cell[key]
                    assert sg["share_neg"] + sg["share_zero"] + sg["share_pos"] == (
                        pytest.approx(1.0)
                    )
                assert "pre_fill_px0001" in cell
                assert "post_minus_pre_px0001" in cell
        groups = blk["events"]["after"]["groups"]
        assert "side=bid" in groups and groups["side=bid"]["n"] > 0
        # the bid/ask split gets the robust treatment; other cuts are mean-only
        assert "trimmed_mean" in groups["side=bid"]["markout_px0001"]
        assert "trimmed_mean" not in groups["emptied_touch=yes"]["markout_px0001"]
        # block plan is reported per horizon (item 2)
        for key in ("block", "block_fills"):
            assert set(blk[key]) == {"block_ns", "block_events", "n_blocks_effective"}
        assert blk["block"]["block_ns"] >= BOOTSTRAP_MIN_BLOCK_NS
    json.loads(json.dumps(res, default=float))  # JSON-able end to end


def test_purely_mechanical_population_shows_zero_after_markout():
    """The honesty check: a tape with no information must look adverse only in "before".

    Every fill in ``_mechanical_tape`` clears a touch held by one order and the mid
    is flat for the whole forward window afterwards. So the "before" reference
    reports a large, uniformly adverse markout and the "after" reference reports
    nothing at all — exactly the artefact the event-time E6 headline is exposed to.
    """
    r = _mechanical_tape()
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS,), n_boot=20, min_group=5,
    )
    bef = res["horizons"]["100ms"]["events"]["before"]["overall"]
    aft = res["horizons"]["100ms"]["events"]["after"]["overall"]
    assert bef["markout_px0001"]["mean"] == pytest.approx(-5.0)  # half the level gap
    assert bef["markout_px0001"]["median"] == pytest.approx(-5.0)
    assert bef["markout_px0001"]["trimmed_mean"] == pytest.approx(-5.0)
    assert bef["share_adverse"] == pytest.approx(1.0)
    assert bef["share_adverse_conditional"] == pytest.approx(1.0)
    assert aft["markout_px0001"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert aft["markout_px0001"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert aft["share_zero"] == pytest.approx(1.0)
    assert np.isnan(aft["share_adverse_conditional"])  # no non-zero moves to condition on



# --------------------------------------------------------------------------- #
# aggressor-event aggregation                                                 #
# --------------------------------------------------------------------------- #
def test_one_sweep_of_many_orders_is_one_observation():
    """A 3-order sweep at one timestamp is ONE aggressor event, not three draws.

    This is the independence correction: the three resting orders share one
    aggressor, one price path and one mechanical BBO move, so counting them
    separately triples the apparent sample size without adding information.
    """
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 3),
        add(t + 2, 3, Side.Ask, 110, 4),
        add(t + 3, 4, Side.Ask, 110, 3),
        add(t + 4, 5, Side.Ask, 120, 9),
        # one incoming buy takes all 10 shares at 110: three E messages, one ns
        execute(t + 5, 2, 3),
        execute(t + 5, 3, 4),
        execute(t + 5, 4, 3),
        add(t + NS_PER_S, 6, Side.Bid, 99, 1),
    ]
    r = replay(evs)
    assert len(r["fills"]) == 3
    events, _rows = group_aggressor_events(r["fills"], r["rows"])
    assert len(events) == 1
    e = events[0]
    assert e.n_fills == 3
    assert e.total_size == 10
    assert e.side == Side.Ask
    assert e.row_first == r["rows"][0]
    assert e.row_last == r["rows"][2]
    # the VWAP of three fills all at 110 is 110
    assert e.vwap_price == pytest.approx(110.0)
    # one observation, not three
    assert event_panel_of(r, 100 * NS_PER_MS).n_usable == 1
    assert panel_of(r, 100 * NS_PER_MS).n_usable == 3


def test_event_after_reference_steps_over_the_whole_sweep():
    """The event's "after" mid is post-LAST-message, so the sweep is fully excluded.

    Per fill, the first two of three fills still see the pre-jump mid as their
    "after" reference and so book the sweep's move as adverse. Aggregated to the
    event, the "after" reference is past the whole sweep and the markout is zero.
    """
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 3),
        add(t + 2, 3, Side.Ask, 110, 4),
        add(t + 3, 4, Side.Ask, 110, 3),
        add(t + 4, 5, Side.Ask, 120, 9),
        execute(t + 5, 2, 3),
        execute(t + 5, 3, 4),
        execute(t + 5, 4, 3),   # this one clears 110 -> best ask 120, mid 105 -> 110
        add(t + NS_PER_S, 6, Side.Bid, 99, 1),
    ]
    r = replay(evs)
    pf = panel_of(r, 100 * NS_PER_MS)
    # per fill: the first two book the mechanical jump even from the "after" mid
    assert pf.markout["after"][0] == pytest.approx(-5.0)
    assert pf.markout["after"][1] == pytest.approx(-5.0)
    assert pf.markout["after"][2] == pytest.approx(0.0)
    # per event: nothing left, the whole sweep is behind the reference
    pe = event_panel_of(r, 100 * NS_PER_MS)
    assert pe.markout["after"][0] == pytest.approx(0.0)
    assert pe.markout["before"][0] == pytest.approx(-5.0)


def test_events_split_on_timestamp_and_on_side():
    """Different nanosecond, or different resting side, is a different aggressor."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 4),
        add(t + 1, 2, Side.Bid, 100, 4),
        add(t + 2, 3, Side.Ask, 110, 4),
        add(t + 3, 4, Side.Ask, 110, 4),
        add(t + 4, 5, Side.Bid, 90, 9),
        add(t + 5, 6, Side.Ask, 120, 9),
        execute(t + 10, 3, 4),            # ask side, ns A
        execute(t + 10, 4, 4),            # ask side, ns A -> same event
        execute(t + 11, 1, 4),            # bid side, ns B -> new event
        execute(t + 12, 2, 4),            # bid side, ns C -> new event
        add(t + NS_PER_S, 7, Side.Bid, 80, 1),
    ]
    r = replay(evs)
    events, _ = group_aggressor_events(r["fills"], r["rows"])
    assert [e.n_fills for e in events] == [2, 1, 1]
    assert [e.side for e in events] == [Side.Ask, Side.Bid, Side.Bid]


def test_event_vwap_is_size_weighted():
    """The event's execution price is the size-weighted VWAP of its fills."""
    f1 = ClockFill(tape_idx=1, ts_ns=OPEN_NS, side=Side.Ask, price=100, size=1,
                   ahead_at_fill=0, level_size_at_fill=10, at_touch=True,
                   emptied_touch=False)
    f2 = ClockFill(tape_idx=2, ts_ns=OPEN_NS, side=Side.Ask, price=200, size=3,
                   ahead_at_fill=1, level_size_at_fill=10, at_touch=True,
                   emptied_touch=True)
    e = AggressorEvent(ts_ns=OPEN_NS, side=Side.Ask, fills=(f1, f2),
                       row_first=1, row_last=2)
    assert e.total_size == 4
    assert e.vwap_price == pytest.approx((100 * 1 + 200 * 3) / 4)
    assert e.size_frac == pytest.approx(4 / 10)        # vs the level at event start
    assert e.queue_frac == pytest.approx((0.0 * 1 + 0.1 * 3) / 4)
    assert e.emptied_touch is True                     # any fill cleared the touch


def test_event_grouping_rejects_mismatched_rows():
    f = ClockFill(tape_idx=1, ts_ns=OPEN_NS, side=Side.Ask, price=110, size=1,
                  ahead_at_fill=0, level_size_at_fill=1, at_touch=True,
                  emptied_touch=True)
    with pytest.raises(ValueError, match="same length"):
        group_aggressor_events([f], [])


# --------------------------------------------------------------------------- #
# robust location                                                             #
# --------------------------------------------------------------------------- #
def test_trimmed_mean_drops_both_tails():
    x = np.array([-1000.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1000.0])
    assert np.mean(x) == pytest.approx(3.6)       # both outliers cancel here
    assert trimmed_mean(x, 0.10) == pytest.approx(4.5)   # k = 1 off each end
    # a single fat tail is what the mean cannot survive and the trim can
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10_000.0])
    assert trimmed_mean(y, 0.10) == pytest.approx(5.5)
    assert np.mean(y) > 1000.0
    assert trimmed_mean([], 0.10) != trimmed_mean([], 0.10) or True  # NaN, no raise
    assert np.isnan(trimmed_mean([]))
    # trimming everything degrades to the median rather than an empty mean
    assert trimmed_mean([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)
    assert trimmed_mean([np.nan, 5.0]) == pytest.approx(5.0)


def test_location_stats_reports_all_three_with_cis():
    rng = np.random.default_rng(5)
    x = rng.normal(-10.0, 3.0, size=400)
    st = location_stats(x, lag=1, block=1, n_boot=200, seed=3)
    for stat in ("mean", "trimmed_mean"):
        ci = st[stat + "_ci95"]
        assert ci["lo"] < st[stat] < ci["hi"]
        assert ci["hi"] < 0.0            # a genuinely negative location
    # the median is reported but carries no CI: lattice data (item 3)
    assert st["median"] < 0.0
    assert "median_ci95" not in st
    assert "lattice" in st["median_ci95_omitted"]
    assert st["trim_prop"] == pytest.approx(0.10)
    assert st["n"] == 400


def test_location_stats_robust_false_skips_the_robust_pair():
    st = location_stats(np.arange(50.0), lag=1, block=1, n_boot=10, seed=1, robust=False)
    assert "mean" in st and "mean_ci95" in st
    assert "median" in st and "median_ci95" not in st
    assert "trimmed_mean" not in st


def test_median_separates_a_tail_driven_mean():
    """A few huge losses can drive a mean the median does not see.

    This is exactly why the main tables carry all three: it is the difference
    between "passive fills lose on average" and "most passive fills lose".
    """
    x = np.concatenate([np.zeros(990), np.full(10, -10_000.0)])
    st = location_stats(x, lag=1, block=1, n_boot=100, seed=9)
    assert st["mean"] == pytest.approx(-100.0)
    assert st["median"] == pytest.approx(0.0)
    assert st["trimmed_mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# overlap / bootstrap selection                                               #
# --------------------------------------------------------------------------- #
def test_block_is_clock_sized_not_horizon_sized():
    """Blocks are max(H, 60 s) of wall-clock time, converted to an event count.

    A 100 ms horizon must not get a 100 ms block: neighbouring markouts are still
    correlated well past the horizon, so the block floor is what keeps the CI
    honest.
    """
    ts = [OPEN_NS + i * NS_PER_S for i in range(600)]   # 1 event/s for 10 minutes
    assert events_per_window(ts, NS_PER_S) == 1
    assert events_per_window(ts, 60 * NS_PER_S) == 60
    short = block_plan(ts, 100 * NS_PER_MS, len(ts))
    assert short["block_ns"] == BOOTSTRAP_MIN_BLOCK_NS      # floored at 60 s
    assert short["block_events"] == 60                     # ~60 events in 60 s
    assert short["n_blocks_effective"] == 10               # 600 / 60
    # a horizon longer than the floor sets the block itself
    long = block_plan(ts, 120 * NS_PER_S, len(ts))
    assert long["block_ns"] == 120 * NS_PER_S
    assert long["block_events"] == 120
    assert long["n_blocks_effective"] == 5
    # fewer effective blocks at the long horizon is exactly why its CI is wider
    assert long["n_blocks_effective"] < short["n_blocks_effective"]


def test_newey_west_lag_counts_events_in_a_typical_window():
    ts = [OPEN_NS + i * NS_PER_S for i in range(600)]
    assert clock_overlap_lag(ts, 10 * NS_PER_S) == 9       # 9 later events in 10 s
    assert clock_overlap_lag(ts, 60 * NS_PER_S) == 59
    assert events_per_window(ts, 10 * NS_PER_S) == clock_overlap_lag(ts, 10 * NS_PER_S) + 1


def test_block_plan_degenerate_inputs():
    assert block_plan([], NS_PER_S, 0)["n_blocks_effective"] == 0
    one = block_plan([OPEN_NS], NS_PER_S, 1)
    assert one["block_events"] == 1
    assert one["n_blocks_effective"] == 1


def test_report_block_plan_matches_the_cells():
    r = _continuous_tape(n=120)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, 60 * NS_PER_S), n_boot=20, min_group=5,
    )
    for blk in res["horizons"].values():
        plan = blk["block"]
        assert plan["block_ns"] >= BOOTSTRAP_MIN_BLOCK_NS
        for ref in ("before", "after"):
            cell = blk["events"][ref]["overall"]
            assert cell["block_events"] == plan["block_events"]
            assert cell["nw_lag"] == blk["nw_lag"]


# --------------------------------------------------------------------------- #
# sign split (discreteness)                                                   #
# --------------------------------------------------------------------------- #
def test_sign_shares_partition_and_carry_cis():
    x = np.array([-50.0, -50.0, 0.0, 0.0, 0.0, 50.0, np.nan])
    sg = sign_shares(x, block=1, n_boot=100, seed=4)
    assert sg["n"] == 6
    assert sg["share_neg"] == pytest.approx(2 / 6)
    assert sg["share_zero"] == pytest.approx(3 / 6)
    assert sg["share_pos"] == pytest.approx(1 / 6)
    assert sg["share_neg"] + sg["share_zero"] + sg["share_pos"] == pytest.approx(1.0)
    for name in ("neg", "zero", "pos"):
        ci = sg["share_" + name + "_ci95"]
        assert 0.0 <= ci["lo"] <= sg["share_" + name] <= ci["hi"] <= 1.0


def test_sign_shares_survive_the_lattice_where_the_median_does_not():
    """On lattice data the median is pinned; the sign split still moves.

    Two samples with the SAME median (0) but very different economics: the sign
    split separates them, which is the point of reporting it.
    """
    a = np.concatenate([np.zeros(60), np.full(40, -float(MID_LATTICE_PX0001))])
    b = np.concatenate([np.zeros(60), np.full(40, +float(MID_LATTICE_PX0001))])
    assert np.median(a) == np.median(b) == 0.0
    sa = sign_shares(a, block=1, n_boot=50, seed=1)
    sb = sign_shares(b, block=1, n_boot=50, seed=1)
    assert sa["share_neg"] == pytest.approx(0.4)
    assert sb["share_neg"] == pytest.approx(0.0)
    assert sa["share_neg_ci95"]["lo"] > sb["share_neg_ci95"]["hi"]


def test_mid_lattice_constant():
    assert MID_LATTICE_PX0001 == TICK_PX_UNITS // 2 == 50


# --------------------------------------------------------------------------- #
# attribution ladder                                                          #
# --------------------------------------------------------------------------- #
def test_event_time_markout_reproduces_the_e6_definition():
    """Per fill with ref="after", this IS E6's post_fill_drift: s*(mid[t+h] - mid[t])."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 5),
        add(t + 2, 3, Side.Ask, 112, 5),
        execute(t + 3, 2, 5),                 # mid 105 -> 106 (after mid = 106)
        add(t + 4, 4, Side.Bid, 104, 5),      # mid -> 108
        add(t + 5, 5, Side.Bid, 106, 5),      # mid -> 109
    ]
    r = replay(evs)
    j = r["rows"][0]
    mids = r["mids"]
    for h in (1, 2):
        got = event_time_markout(r["fills"], r["rows"], mids, h, reference="after")[0]
        want = -(mids[j + h] - mids[j])       # s = -1 for a resting ask
        assert got == pytest.approx(want)
    # the "before" variant measures from mid[j-1] instead, so it carries the jump
    bef = event_time_markout(r["fills"], r["rows"], mids, 1, reference="before")[0]
    assert bef == pytest.approx(-(mids[j + 1] - mids[j - 1]))
    assert bef < event_time_markout(r["fills"], r["rows"], mids, 1, reference="after")[0]


def test_event_time_markout_rejects_a_bad_reference():
    with pytest.raises(ValueError, match="reference must be one of"):
        event_time_markout([], [], [1.0], 1, reference="middle")


def test_attribution_ladder_changes_one_thing_per_row():
    r = _continuous_tape(n=120)
    events, event_rows = group_aggressor_events(r["fills"], r["rows"])
    ladder = attribution_ladder(
        r["fills"], r["rows"], events, event_rows, r["mids"], r["ts"],
        h_events=5, horizon_ns=100 * NS_PER_MS,
    )
    assert [row["step"] for row in ladder] == ["1", "2", "3"]
    assert [row["unit"] for row in ladder] == [
        "per fill", "per aggressor event", "per aggressor event",
    ]
    # row 1 -> 2 changes the unit only; row 2 -> 3 changes the clock only
    assert ladder[0]["clock"] == ladder[1]["clock"] == "event h=5"
    assert ladder[2]["clock"] == "100ms"
    for row in ladder:
        for ref in ("before", "after"):
            c = row[ref]
            assert c["n"] > 0
            assert 0.0 <= c["share_adverse"] <= 1.0
            assert 0.0 <= c["share_zero"] <= 1.0
    assert E6_HEADLINE_H_EVENTS == 5


def test_attribution_row1_after_matches_the_e6_formula_on_the_same_fills():
    """Row 1 / `after` must equal E6's own statistic computed directly.

    This is what makes the ladder an attribution rather than four unrelated
    numbers: the top row has to reproduce the figure being audited.
    """
    r = _continuous_tape(n=150)
    events, event_rows = group_aggressor_events(r["fills"], r["rows"])
    ladder = attribution_ladder(
        r["fills"], r["rows"], events, event_rows, r["mids"], r["ts"], h_events=5,
    )
    direct = event_time_markout(r["fills"], r["rows"], r["mids"], 5, reference="after")
    want = p_adverse(direct[np.isfinite(direct)], conditional=True)
    assert ladder[0]["after"]["share_adverse_conditional"] == pytest.approx(want)
    assert ladder[0]["after"]["n"] == int(np.count_nonzero(np.isfinite(direct)))


def test_attribution_mechanical_tape_drops_to_zero_down_the_ladder():
    """On a tape with no information the ladder must walk 1.0 -> 0 (or NaN).

    Every fill clears a touch held by one order and the mid is flat afterwards, so
    the `before` cells see a uniformly adverse move and the `after` cells see
    nothing once the unit is the aggressor event.
    """
    r = _mechanical_tape(n=120)
    events, event_rows = group_aggressor_events(r["fills"], r["rows"])
    ladder = attribution_ladder(
        r["fills"], r["rows"], events, event_rows, r["mids"], r["ts"],
        h_events=1, horizon_ns=100 * NS_PER_MS,
    )
    assert ladder[0]["before"]["share_adverse_conditional"] == pytest.approx(1.0)
    clock_after = ladder[2]["after"]
    assert clock_after["share_zero"] == pytest.approx(1.0)
    assert np.isnan(clock_after["share_adverse_conditional"])


# --------------------------------------------------------------------------- #
# pre-fill control                                                            #
# --------------------------------------------------------------------------- #
def test_pre_fill_control_measures_the_window_before_the_fill():
    """`pre` is the signed move over [t - H, t), ending strictly before the fill.

    A tape that drifts up into a resting-ask fill must show a negative (adverse)
    `pre` for the seller, and `post - pre` nets that trend out.
    """
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 120, 5),    # the ask that gets filled
        add(t + 2, 3, Side.Ask, 130, 5),    # depth behind it
        # walk the best bid up with the ask fixed, so the mid rises 110 -> 116
        add(t + 1 * NS_PER_S, 4, Side.Bid, 104, 5),   # mid 112
        add(t + 2 * NS_PER_S, 5, Side.Bid, 108, 5),   # mid 114
        add(t + 3 * NS_PER_S, 6, Side.Bid, 112, 5),   # mid 116
        execute(t + 4 * NS_PER_S, 2, 5),    # clears ask 120 -> best ask 130, mid 121
        add(t + 5 * NS_PER_S, 7, Side.Bid, 90, 1),    # deep, mid unchanged
        add(t + 7 * NS_PER_S, 8, Side.Bid, 91, 1),
    ]
    r = replay(evs)
    p = event_panel_of(r, 2 * NS_PER_S)
    # over [t+2s, t+4s) the mid rose 114 -> 116: adverse for a resting seller
    assert p.pre_fill[0] == pytest.approx(-2.0)
    assert p.drops["pre_past_open"] == 0
    assert p.drops["pre_bad_mid"] == 0
    # the fill's own jump is in "before" (-5) and out of "after" (0)
    assert p.markout["before"][0] == pytest.approx(-5.0)
    assert p.markout["after"][0] == pytest.approx(0.0)
    # post - pre nets the pre-existing uptrend out of the forward move
    assert p.markout["after"][0] - p.pre_fill[0] == pytest.approx(2.0)


def test_pre_window_before_the_open_is_dropped_but_keeps_the_markout():
    """A pre-window running before 09:30 nulls only `pre`, not the forward markout."""
    t = OPEN_NS
    evs = [
        add(t + 0, 1, Side.Bid, 100, 5),
        add(t + 1, 2, Side.Ask, 110, 5),
        add(t + 2, 3, Side.Ask, 112, 5),
        execute(t + 3, 2, 5),                  # 3 ns into the session
        add(t + NS_PER_S, 4, Side.Bid, 99, 1),
    ]
    r = replay(evs)
    p = event_panel_of(r, 100 * NS_PER_MS)     # a 100 ms pre-window predates 09:30
    assert p.drops["pre_past_open"] == 1
    assert np.isnan(p.pre_fill[0])
    assert np.isfinite(p.markout["after"][0])  # the forward markout survives
    assert p.n_usable == 1


def test_pre_fill_is_reference_independent():
    """`pre` ends strictly before the fill, so it does not depend on the reference."""
    r = _continuous_tape(n=40)
    p = event_panel_of(r, NS_PER_S)
    # one array, shared by both references by construction
    assert p.pre_fill.shape == (len(p.obs),)
    assert np.count_nonzero(np.isfinite(p.pre_fill)) > 30
    assert p.drops["pre_bad_mid"] == 0


def test_pre_window_on_a_one_sided_book_is_counted_separately():
    """An empty book at the pre-window start is `pre_bad_mid`, not `pre_past_open`."""
    r = _mixed_tape(n=5)   # this fixture tears the book down between iterations
    p = event_panel_of(r, 100 * NS_PER_MS)
    assert p.drops["pre_bad_mid"] > 0
    assert np.all(np.isnan(p.pre_fill))
    # the forward markouts are unaffected by the pre-window failing
    assert p.n_usable > 0


# --------------------------------------------------------------------------- #
# attribution arithmetic                                                      #
# --------------------------------------------------------------------------- #
def test_attribution_steps_sum_to_the_total_change():
    """The listed steps must account for the WHOLE walk down one reference column.

    This is the arithmetic guard: the walk is unit-of-observation then clock, two
    steps, and their deltas have to close on `last - first`. If a third step were
    ever folded in — e.g. the before/after gap — this would stop closing.
    """
    r = _informative_tape(n=150)
    events, event_rows = group_aggressor_events(r["fills"], r["rows"])
    assert len(r["fills"]) > len(events)      # the unit step is a real change
    ladder = attribution_ladder(
        r["fills"], r["rows"], events, event_rows, r["mids"], r["ts"],
        h_events=5, horizon_ns=100 * NS_PER_MS,
    )
    for ref in ("before", "after"):
        chk = attribution_check(ladder, reference=ref)
        assert chk["defined"] is True
        assert chk["closes"] is True
        assert chk["steps_sum"] == pytest.approx(chk["total_change"])
        assert [st["changed"] for st in chk["steps"]] == ["unit of observation", "clock"]
        assert chk["first_value"] == pytest.approx(ladder[0][ref][LADDER_STAT])
        assert chk["last_value"] == pytest.approx(ladder[-1][ref][LADDER_STAT])
        # each step is a real difference between adjacent rows, in order
        assert chk["steps"][0]["from_step"] == "1"
        assert chk["steps"][1]["to_step"] == "3"


def test_before_minus_after_is_not_a_walk_step():
    """The reference gap is a within-row contrast, flagged as such in the data.

    Both endpoints of the walk are measured from the same reference, so adding the
    before/after gap would double-count. The ladder records it separately with
    `is_walk_step = False` rather than leaving that to prose.
    """
    r = _informative_tape(n=120)
    events, event_rows = group_aggressor_events(r["fills"], r["rows"])
    ladder = attribution_ladder(
        r["fills"], r["rows"], events, event_rows, r["mids"], r["ts"], h_events=5,
    )
    for row in ladder:
        bma = row["before_minus_after"]
        assert bma["is_walk_step"] is False
        assert bma["stat"] == LADDER_STAT
        assert bma["value"] == pytest.approx(
            row["before"][LADDER_STAT] - row["after"][LADDER_STAT], nan_ok=True
        )
    # the gap is NOT part of the decomposition: steps still close without it
    chk = attribution_check(ladder, reference="after")
    gaps = sum(
        row["before_minus_after"]["value"] for row in ladder
        if np.isfinite(row["before_minus_after"]["value"])
    )
    assert chk["closes"] is True
    if abs(gaps) > 1e-9:  # adding the gaps would break the identity
        assert chk["steps_sum"] + gaps != pytest.approx(chk["total_change"])


def test_attribution_check_degenerate_ladder():
    chk = attribution_check([], reference="after")
    assert chk["closes"] is True
    assert chk["steps"] == []
    assert chk["steps_sum"] == 0.0


def test_attribution_check_undefined_when_a_cell_never_moved():
    """A NaN conditional share means there is no walk — say so, do not say "False"."""
    r = _mechanical_tape(n=80)   # mid is flat after every sweep
    events, event_rows = group_aggressor_events(r["fills"], r["rows"])
    ladder = attribution_ladder(
        r["fills"], r["rows"], events, event_rows, r["mids"], r["ts"],
        h_events=1, horizon_ns=100 * NS_PER_MS,
    )
    chk = attribution_check(ladder, reference="after")
    assert chk["defined"] is False
    assert chk["closes"] is None
    assert chk["steps"] == []
    md = render_clock_markouts_md("12302019", {"TEST": {
        "n_fills": len(r["fills"]), "n_events": len(events), "fills_per_event": 1.0,
        "attribution": ladder, "attribution_check": {"after": chk, "before": chk},
        "horizons": {},
    }})
    assert "undefined" in md


def test_report_carries_the_attribution_check():
    r = _informative_tape(n=120)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS,), n_boot=20, min_group=5,
    )
    assert set(res["attribution_check"]) == {"before", "after"}
    for chk in res["attribution_check"].values():
        assert chk["closes"] is True


# --------------------------------------------------------------------------- #
# assumed maker rebate                                                        #
# --------------------------------------------------------------------------- #
def test_rebate_shifts_the_realized_spread_by_a_constant():
    """A per-share rebate is an additive constant: mean and CI shift, shape does not."""
    rng = np.random.default_rng(17)
    rl = rng.normal(-40.0, 60.0, size=500)
    st = markout_stats(
        rl, rl, np.zeros(500), np.full(500, 100.0), np.zeros(500),
        lag=1, block=5, n_boot=200, seed=3,
    )
    base = st["realized_spread_px0001"]["mean"]
    for r in REBATE_SCENARIOS_PX0001:
        nr = st["net_of_rebate"][str(r)]
        assert nr["rebate_px0001"] == r
        assert nr["assumed"] is True
        assert nr["mean"] == pytest.approx(base + r)
        # a 20-unit rebate on a 100-unit half-spread is 0.2 half-spreads
        assert nr["mean_half_spreads"] == pytest.approx((base + r) / 100.0)
        # the share profitable rises with the rebate
        assert nr["sign"]["share_pos"] > st["realized_sign"]["share_pos"]
        assert nr["sign"]["share_neg"] < st["realized_sign"]["share_neg"]
    big, small = str(max(REBATE_SCENARIOS_PX0001)), str(min(REBATE_SCENARIOS_PX0001))
    assert st["net_of_rebate"][big]["mean"] > st["net_of_rebate"][small]["mean"]


def test_rebate_break_even_is_detectable():
    """A rebate exactly offsetting the loss puts the mean at zero — the break-even read."""
    rl = np.full(400, -float(min(REBATE_SCENARIOS_PX0001)))
    st = markout_stats(
        rl, rl, np.zeros(400), np.full(400, 100.0), np.zeros(400),
        lag=1, block=5, n_boot=50, seed=1,
    )
    nr = st["net_of_rebate"][str(min(REBATE_SCENARIOS_PX0001))]
    assert nr["mean"] == pytest.approx(0.0)
    assert nr["sign"]["share_zero"] == pytest.approx(1.0)


def test_rebate_is_labelled_as_assumed_everywhere():
    """The assumption must travel with the number, not live only in a docstring."""
    assert "ASSUMED" in REBATE_ASSUMPTION_NOTE
    assert "not measured" in REBATE_ASSUMPTION_NOTE
    r = _continuous_tape(n=120)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS,), n_boot=20, min_group=5,
    )
    cell = res["horizons"]["100ms"]["events"]["after"]["overall"]
    assert cell["rebate_assumption_note"] == REBATE_ASSUMPTION_NOTE
    assert all(v["assumed"] is True for v in cell["net_of_rebate"].values())
    md = render_clock_markouts_md("12302019", {"TEST": res})
    assert "### Break-even including an assumed maker rebate" in md
    assert REBATE_ASSUMPTION_NOTE in md
    assert "assumed, not measured" in md.lower()


def test_rebate_only_on_robust_cells():
    """Small breakdown cells skip the rebate scenarios along with the robust pair."""
    rl = np.arange(50.0)
    st = markout_stats(
        rl, rl, np.zeros(50), np.full(50, 100.0), np.zeros(50),
        lag=1, block=5, n_boot=10, seed=1, robust=False,
    )
    assert "net_of_rebate" not in st
    assert "realized_sign" not in st



# --------------------------------------------------------------------------- #
# interpretation guards                                                       #
# --------------------------------------------------------------------------- #
def test_before_after_gap_is_constant_in_h_by_construction():
    """The gap equals s*(mid_after - mid_before): no terminal mid, no H in it.

    So its constancy across horizons is an algebraic identity and cannot be
    evidence that the effect is mechanical. This test pins the identity, and the
    document is required to say so (see the reversion-check test below).
    """
    r = _informative_tape(n=60)
    events, rows = group_aggressor_events(r["fills"], r["rows"])
    mids = np.asarray(r["mids"], dtype=np.float64)
    gaps = []
    for h_ns in (100 * NS_PER_MS, NS_PER_S, 10 * NS_PER_S):
        p_ = markout_panel(events, rows, r["mids"], r["ts"], r["half_spreads"], h_ns)
        # before - after = s*(mid_t - mid_before) - s*(mid_t - mid_after)
        #               = s*(mid_after - mid_before)   <- the terminal mid cancels
        gap = p_.markout["before"] - p_.markout["after"]
        ok = np.isfinite(gap)
        want = np.array([
            _sign(e.side) * (mids[e.row_last] - mids[e.row_first - 1])
            for e in events
        ])
        np.testing.assert_allclose(gap[ok], want[ok])
        gaps.append(gap[ok])
    # identical at every horizon: the quantity simply does not depend on H
    for g in gaps[1:]:
        np.testing.assert_allclose(g[: len(gaps[0])], gaps[0][: len(g)])


def test_document_disclaims_the_gap_and_reports_reversion():
    """The write-up must not use the constant gap as evidence, and must test reversion."""
    r = _informative_tape(n=150)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, NS_PER_S, 10 * NS_PER_S), n_boot=20, min_group=5,
    )
    md = render_clock_markouts_md("12302019", {"TEST": res})
    assert "### Reversion check" in md
    assert "constant across horizons by construction" in md
    assert "tests nothing" in md
    # the verdict is one of the two, never both
    assert ("impact is **permanent**" in md) != ("impact **reverts**" in md)


def test_reversion_verdict_follows_the_numbers():
    """A shrinking markout reads as reverting; a growing one as permanent."""
    def md_for(means: list[float]) -> str:
        horizons = {}
        for hl, mu in zip(["100ms", "1s", "10s"], means, strict=True):
            horizons[hl] = {
                "events": {"after": {"overall": {"markout_px0001": {
                    "mean": mu, "mean_ci95": {"lo": mu - 1, "hi": mu + 1},
                }}}},
            }
        return "\n".join(_reversion_table({"horizons": horizons}))

    assert "impact is **permanent**" in md_for([-50.0, -80.0, -120.0])
    assert "impact **reverts**" in md_for([-120.0, -80.0, -50.0])


def test_sign_split_reports_conditional_magnitudes():
    """Item 4: the claim is "losers lose more than winners win", so report both."""
    x = np.array([-100.0, -80.0, 0.0, 20.0, 30.0])
    sg = sign_shares(x, block=1, n_boot=20, seed=2)
    assert sg["mean_neg"] == pytest.approx(-90.0)
    assert sg["mean_pos"] == pytest.approx(25.0)
    assert sg["mean_zero"] == pytest.approx(0.0)
    # near-equal hit rates with asymmetric magnitudes -> negative mean
    assert sg["share_neg"] == pytest.approx(0.4)
    assert sg["share_pos"] == pytest.approx(0.4)
    assert abs(sg["mean_neg"]) > abs(sg["mean_pos"])
    assert float(np.mean(x)) < 0.0


def test_loss_asymmetry_section_renders_the_magnitudes():
    r = _informative_tape(n=150)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, NS_PER_S), n_boot=20, min_group=5,
    )
    md = render_clock_markouts_md("12302019", {"TEST": res})
    assert "### Who loses, and by how much" in md
    assert "losers lose more than winners win" in md
    assert "mean loss when losing" in md
    assert "roughly symmetric hit rate with asymmetric magnitudes" in md
    # the old framing is explicitly disclaimed, never asserted
    assert "is not a minority of bad fills" in md


def test_emptied_touch_is_framed_as_informed_flow_not_against_it():
    """Item 3: size-dependent persistent impact is the Kyle / Glosten-Milgrom signature."""
    r = _informative_tape(n=150)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, NS_PER_S), n_boot=20, min_group=5,
    )
    md = render_clock_markouts_md("12302019", {"TEST": res})
    assert "large enough to clear" in md
    assert "Glosten" in md and "Kyle" in md
    assert "size-dependent permanent impact" in md
    # every breakdown table names its horizon, and there is one per horizon
    assert md.count("**Breakdowns at H = ") == len(res["horizons"])
    for hl in res["horizons"]:
        assert f"**Breakdowns at H = {hl}**" in md


def test_bottom_line_states_the_claim_and_its_limits():
    """Item 6: one bottom line, with the single-day and hypothesis caveats attached."""
    r = _informative_tape(n=150)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, 60 * NS_PER_S), n_boot=20, min_group=5,
    )
    md = render_clock_markouts_md("12302019", {"AAA": res, "BBB": res})
    assert "## Bottom line" in md
    assert "conditional, event-time share" in md
    assert "CIs excluding zero" in md
    assert "does not\nrevert" in md or "does not revert" in md
    assert "Single day" in md
    # item 5: the decay story is labelled a hypothesis, not a finding
    assert "*hypothesis*" in md
    assert "Not established here" in md
    # the bottom line comes last, after both symbol sections
    assert md.index("## Bottom line") > md.index("## BBB")


def test_report_on_no_fills_is_empty_but_well_formed():
    res = clock_markout_report([], [], [105.0], [OPEN_NS], [5.0])
    assert res["n_fills"] == 0
    assert res["horizons"] == {}


def test_report_rejects_mismatched_rows():
    f = ClockFill(
        tape_idx=1, ts_ns=OPEN_NS, side=Side.Ask, price=110, size=1,
        ahead_at_fill=0, level_size_at_fill=1, at_touch=True, emptied_touch=True,
    )
    with pytest.raises(ValueError, match="same length"):
        clock_markout_report([f], [], [105.0], [OPEN_NS], [5.0])


def test_render_md_contains_both_references_and_drop_accounting():
    r = _informative_tape(n=150)
    res = clock_markout_report(
        r["fills"], r["rows"], r["mids"], r["ts"], r["half_spreads"],
        horizons_ns=(100 * NS_PER_MS, NS_PER_S), n_boot=20, min_group=5,
    )
    md = render_clock_markouts_md("12302019", {"TEST": res})
    assert "# Clock-time passive-fill markouts" in md
    assert "## TEST" in md
    assert "past 16:00" in md
    assert "| 100ms | before |" in md
    assert "| 100ms | after |" in md
    assert "emptied_touch=yes" in md
    # the realized-spread table is the headline, above the markout table
    assert md.index("### Realized spread") < md.index("### Markout")
    assert "Pre-fill control and the bid/ask split" in md
    assert "per individual fill" in md
    # the new tables (items 2, 3, 4)
    assert "### Sign split" in md
    assert "### Attribution" in md
    assert "checks out" in md                      # the step-sum identity
    assert "not** a walk step" in md               # before/after is not a step
    assert md.index("### Realized spread") < md.index("### Attribution")
    assert "### Bootstrap blocks and dependence" in md
    assert "effective blocks" in md
    assert "| 1 | per fill | event h=5 | after |" in md
    assert MEDIAN_LATTICE_NOTE in md
    # units are labelled, and $0.0001 is never called a tick
    assert "($0.0001)" in md
    assert "$0.01 tick = 100 price units" in md
    price_cols = ("mean", "median", "markout", "realized", "post", "pre ", "HS earned")
    for line in md.splitlines():
        if not line.startswith(("| H |", "| group |")):
            continue
        if any(c in line for c in price_cols):  # the drop table is counts, not prices
            assert "$0.0001" in line, line
