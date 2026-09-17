"""Unified test suite for queue dynamics (combining PR #18/20 and Person B test suites)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_state import Side
from nexus_quant.itch_parser import (
    EventType,
    NormalizedEvent,
    encode_event,
    iter_itch_events,
)
from nexus_quant.research.queue_dynamics import (
    REGULAR_CLOSE_NS,
    REGULAR_OPEN_NS,
    FillRow,
    LogisticFillModel,
    OrderLevelTracker,
    OrderLife,
    QueueTracker,
    RestingOrder,
    _decision_features,
    brier_score,
    calibration_table,
    censor_open_orders,
    fill_dataset,
    fill_prob_survival,
    filter_session_orders,
    logistic_fill_model,
    order_features,
    queue_ahead_walk,
    standing_order_lifetimes,
)
from nexus_quant.research.synthetic_flow import FlowConfig, SyntheticFlow


# ===========================================================================
# Part 1: Lokesh PR #18 tests
# ===========================================================================
def _flow(n: int = 2000, **kw):
    cfg = FlowConfig(n_events=n, **kw)
    f = SyntheticFlow(cfg); f.generate()
    return f


def _replay_prefix(events, p):
    """Replay events[:p] into a fresh QueueTracker."""
    tr = QueueTracker()
    for e in events[:p]:
        tr.on_event(e)
    return tr


# ---- (A) lockstep reconstruct == internal book, mid-trace equality -------- #


def test_reconstruct_lockstep_and_mid_trace():
    # step the generator live and feed the tracker EVERY event it appends
    # (the returned event plus any _normalize_book ADD/DELETEs), so both
    # books advance in lockstep and reconstruct() must equal internal_book().
    f = SyntheticFlow(FlowConfig(n_events=800, seed=123))
    tr = QueueTracker()
    count = 0
    while f.step_gen() is not None:
        new_events = f.events[count:]
        for e in new_events:
            tr.on_event(e)
        count = len(f.events)
        assert tr.reconstruct() == f.internal_book()
        # once both sides have a live touch the BBO mids agree
        if tr.mid() > 0:
            assert tr.mid() == f.mid_trace[-1]


# ---- (B) rows carry known queue: best==price, queue_ahead==level_size ---- #


def test_rows_have_known_queue():
    f = _flow(3000)
    rows = fill_dataset(f, each_tau_drop=40, qty=100, window=200)
    assert rows
    for r in rows:
        assert r.queue_ahead == r.level_size
        assert r.queue_ahead > 0
        assert r.qty == 100
        assert 0.0 <= r.mid_ticks
        assert set(r.features) == {
            "log_level", "queue_ahead", "spread_bps", "lob_imbalance",
            "ofi", "realized_vol", "flow_intensity", "momentum",
        }
    # every (decision_index, side) appears at most once
    seen = set()
    for r in rows:
        key = (r.decision_index, r.side)
        assert key not in seen, "duplicate row"
        seen.add(key)


# ---- (C) no cancel-selection bias: decision at every scheduled index ------ #


def test_decide_at_every_scheduled_index():
    f = _flow(1200)
    each = 60
    rows = fill_dataset(f, each_tau_drop=each, sides=(Side.Bid,))
    idxs = {r.decision_index for r in rows}
    expected = set(range(each, len(f.events) + 1, each))
    # rows are a subset of scheduled indices only
    assert idxs <= expected
    # a live touch exists at most points → >=80% coverage
    assert len(idxs) >= 0.8 * len(expected)


# ---- (D) leak lock: features recompute exactly from prefix ---------------- #


def test_leak_lock_recompute_features_from_prefix():
    f = _flow(4000)
    each = 80
    rows = fill_dataset(f, each_tau_drop=each, qty=100, window=200)
    assert len(rows) > 10
    for r in rows[:8]:
        # rebuild via a fresh tracker on the prefix events[:decision_index]
        tr = _replay_prefix(f.events, r.decision_index)
        prev_p = r.decision_index - each
        prev_view = None if prev_p <= 0 else _replay_prefix(f.events, prev_p).view()
        feats = _decision_features(tr, f.events, r.decision_index, r.side,
                                   r.price, prev_view)
        assert feats == r.features, f"features differ at decision_index={r.decision_index}"
        assert tr.best(r.side) == r.price
        assert tr.queue(r.side, r.price)[0].oid is not None


# ---- (E) mutating a future event changes only the label ------------------- #


def test_future_mutation_changes_only_label():
    f = _flow(4000)
    each, window, qty = 80, 200, 200
    rows = fill_dataset(f, each_tau_drop=each, window=window, qty=qty)
    # Pick a row whose fill depends on the front order being consumed, so that
    # deleting the front order deterministically flips the label: exactly ONE
    # order ahead at the decision, filled in the original tape, and whose first
    # future event is the EXECUTE consuming that sole front order. Rewriting it
    # as a DELETE leaves the consuming EXECUTE pointing at a now-gone id, so the
    # walk never books our slot → filled becomes unfilled.
    victim = None
    for r in rows:
        if not r.filled:
            continue
        tr = _replay_prefix(f.events, r.decision_index)
        q = tr.queue(r.side, r.price)
        if len(q) != 1:
            continue
        fe = f.events[r.decision_index]
        if fe.kind not in (EventType.EXECUTE, EventType.EXECUTE_PX):
            continue
        if fe.order_id != q[0].oid or fe.size < q[0].size:
            continue
        victim = r
        break
    assert victim is not None, "expected a filled victim row with a lone front order"
    p = victim.decision_index
    tr = _replay_prefix(f.events, p)
    front_oid = tr.queue(victim.side, victim.price)[0].oid
    # mutate ONLY the first future event: delete the front order the fill consumed
    mut = list(f.events)
    mut[p] = NormalizedEvent(kind=EventType.DELETE, ts_ns=mut[p].ts_ns + 1,
                             order_id=front_oid, side=Side.NONE, price_ticks=0, size=0)
    dummy = SyntheticFlow(FlowConfig())
    dummy.events = mut
    rows_mut = fill_dataset(dummy, each_tau_drop=each, window=window, qty=qty)
    rm = next((rr for rr in rows_mut
               if rr.decision_index == p and rr.side == victim.side), None)
    assert rm is not None, "mutated tape must still contain the decision row"
    # leak lock: the prefix is unchanged → features identical
    assert rm.features == victim.features
    # label: deleting the front order unfills the row (the consuming EXECUTE
    # now points at a deleted id, so the walk never books our slot)
    assert (rm.filled, rm.fill_qty, rm.tau) != (victim.filled, victim.fill_qty, victim.tau)


# ---- (F) queue_ahead_walk hand cases ------------------------------------- #


def _ro(oid, px, sz):
    return RestingOrder(oid=oid, side=Side.Bid, price=px, size=sz,
                        born_size=sz, born_index=0, born_ts_ns=0, ahead_at_birth=0)


def _ev(kind, oid, sz=0, ts=0):
    return NormalizedEvent(kind=kind, ts_ns=ts, order_id=oid, side=Side.NONE,
                           price_ticks=0, size=sz)


def test_queue_ahead_walk_full_and_partial():
    px = 10000
    # ``queue_ahead_walk`` indexes into the FULL tape from ``decision_index + 1``
    # (as ``fill_dataset`` does). These hand cases pass just the future events,
    # so ``decision_index = -1`` puts ``events[0]`` at the first future event
    # and tau counts forward from 1.
    # FULL FILL: one order ahead, take clears it + a surplus inside the event
    o = queue_ahead_walk([_ro(1, px, 100)], [_ev(EventType.EXECUTE, 1, 150, 1)],
                         decision_index=-1, side=Side.Bid, price=px, qty=50, window=2)
    assert o.filled and o.fill_qty == 50 and o.tau == 1
    # EXACT STOP: no surplus, no continuation → not filled
    o = queue_ahead_walk([_ro(2, px, 100)], [_ev(EventType.EXECUTE, 2, 100, 1)],
                         decision_index=-1, side=Side.Bid, price=px, qty=50, window=2)
    assert not o.filled and o.fill_qty == 0 and o.tau is None
    # LEVEL-DRAIN + TAKE CONTINUATION: last ahead cleared, next event is EXECUTE → full fill
    events = [
        _ev(EventType.EXECUTE, 3, 100, 1),   # clears the only ahead order, leftover 0
        _ev(EventType.EXECUTE, 99, 200, 2),  # same take walks deeper
    ]
    o = queue_ahead_walk([_ro(3, px, 100)], events,
                         decision_index=-1, side=Side.Bid, price=px, qty=80, window=3)
    assert o.filled and o.fill_qty == 80 and o.tau == 1
    # CASCADE: multi-order level cleared by two listed executes, surplus into us
    events = [
        _ev(EventType.EXECUTE, 4, 50, 1),
        _ev(EventType.EXECUTE, 5, 30, 2),
        _ev(EventType.EXECUTE, 6, 9, 3),   # not in rem → ignored (behind/other level)
    ]
    o = queue_ahead_walk([_ro(4, px, 50), _ro(5, px, 30)], events,
                         decision_index=-1, side=Side.Bid, price=px, qty=20, window=4)
    assert o.filled and o.fill_qty == 20 and o.tau == 2
    assert o.tau is not None and o.tau > 0   # tau forward-only


# ---- (G) KM hand case ----------------------------------------------------- #


def test_km_hand_case():
    b, side, px = 0, Side.Bid, 10000
    levels = [
        OrderLife(1, side, px, b, 100, 2, "fill",    100),  # tau 2
        OrderLife(2, side, px, b, 100, 1, "fill",    100),  # tau 1
        OrderLife(3, side, px, b, 100, 3, "cancel",  100),  # tau 3 (competing risk)
        OrderLife(4, side, px, 1, 100, None, "open",   0),  # right-censored, H-b=3
        OrderLife(5, side, px, 2, 100, 4, "fill",    100),  # tau 2
    ]
    res = fill_prob_survival(levels)
    assert res["n_orders"] == 5
    assert res["n_fills"] == 3
    assert res["n_censored"] == 2
    assert res["tau"] == [1, 2, 3, 4]
    # t1: 1-1/5=0.8; t2: 0.8*(1-2/4)=0.4; t3: 0.4*(1-0/2)=0.4; t4: 0.4
    assert np.allclose(res["survival_fill"], [0.8, 0.4, 0.4, 0.4])
    assert np.allclose(res["at_risk"], [5, 4, 2, 0])


# ---- (H) cancel is a competing risk: parallel survival_cancel ------------- #


def test_cancel_competing_risk_parallel_curve():
    b, side, px = 0, Side.Bid, 10000
    levels = [
        OrderLife(1, side, px, b, 100, 2, "fill",   100),
        OrderLife(2, side, px, b, 100, 1, "fill",   100),
        OrderLife(3, side, px, b, 100, 3, "cancel", 100),
        OrderLife(4, side, px, 1, 100, None, "open",  0),
        OrderLife(5, side, px, 2, 100, 4, "fill",   100),
    ]
    res = fill_prob_survival(levels)
    # cancels happen only at t=3, n(3)=2 → h=0.5 there; elsewhere 0
    assert np.allclose(res["survival_cancel"], [1.0, 1.0, 0.5, 0.5])
    assert res["n_fills"] == 3              # the cancel is censored, not a fill
    assert res["n_censored"] == 2


# ---- (I) REPLACE ends old oid as "replace" -------------------------------- #


def test_replace_ends_old_oid():
    events = [
        NormalizedEvent(EventType.ADD, 0, 1, Side.Bid, 10000, 100),
        NormalizedEvent(EventType.ADD, 1, 2, Side.Ask, 10001, 50),
        NormalizedEvent(EventType.REPLACE, 2, 1, Side.Bid, 10000, 80,
                        new_order_id=3),
    ]
    lifetimes = standing_order_lifetimes(events)
    by_oid = {ol.oid: ol for ol in lifetimes}
    assert by_oid[1].end_kind == "replace" and by_oid[1].end_index == 2
    assert by_oid[1].tau == 2
    assert by_oid[3].born_index == 2 and by_oid[3].end_kind == "open"


# ---- (J) logistic recovers monotone queue hazard + Brier < 0.25 ---------- #


def test_logistic_monotone_hazard():
    f = _flow(4000)
    rows = fill_dataset(f, each_tau_drop=40, qty=100, window=250)
    assert len(rows) > 30
    m = LogisticFillModel.fit(rows, feature_names=["log_level"])
    log_coef = float(m.coef[1])
    assert log_coef < -0.1, f"log_level coef {log_coef} not negative"
    X = np.asarray([[r.features["log_level"]] for r in rows], dtype=np.float64)
    y = np.asarray([1.0 if r.filled else 0.0 for r in rows])
    b = m.brier(X, y)
    assert b < 0.25, f"Brier {b} >= 0.25"
    # nonparametric monotonicity: top-level-size tercile fills below bottom
    ls = np.asarray([r.features["log_level"] for r in rows])
    qt = np.quantile(ls, [1 / 3, 2 / 3])
    p_bot = float(y[ls < qt[0]].mean())
    p_top = float(y[ls > qt[1]].mean())
    assert p_bot > p_top


# ---- (K) predict in [0,1] + deterministic -------------------------------- #


def test_predict_range_and_determinism():
    f = _flow(2000)
    rows = fill_dataset(f, each_tau_drop=40, qty=100, window=200)
    m = LogisticFillModel.fit(rows)
    for r in rows[:20]:
        p1 = m(r.features)
        p2 = m(r.features)
        assert 0.0 <= p1 <= 1.0
        assert p1 == p2
        rp = m.predict_proba(r.features)
        assert rp.shape == (1,) and rp[0] == p1
    X = np.array([[r.features[n] for n in m.feature_names] for r in rows[:5]])
    pa = m.predict_proba(X)
    assert pa.shape == (5,)
    assert (0.0 <= pa).all() and (pa <= 1.0).all()


# ---- (L) calibration slope ~= 1 for a perfect (logistic) model ------------ #


def test_calibration_slope_perfect_model():
    from nexus_quant.research.experiments import calibration_curve

    rng = np.random.default_rng(42)
    z_vals = np.linspace(0.05, 0.95, 12)
    rows = []
    for z in z_vals:
        p_true = 1.0 / (1.0 + np.exp(-4.0 * (z - 0.5)))
        for _ in range(20):
            filled = float(rng.random()) < p_true
            rows.append(FillRow(
                decision_index=0, side=Side.Bid, price=10000, qty=100,
                filled=bool(filled), fill_qty=100 if filled else 0, tau=None,
                queue_ahead=100, level_size=100, mid_ticks=10000.0,
                features={"x": float(z)},
            ))
    m = LogisticFillModel.fit(rows, feature_names=["x"])
    X = np.array([[r.features["x"]] for r in rows])
    y_pred = m.predict_proba(X)
    y_true = np.array([float(r.filled) for r in rows])
    cal = calibration_curve(y_true, y_pred, n_bins=6)
    assert len(cal["bins"]) == 6
    assert 0.85 < cal["slope"] < 1.15, f"calibration slope {cal['slope']} not ~1"

# ===========================================================================
# Part 2: Person B tests
# ===========================================================================
def _ev_tracked(kind, oid, *, side=Side.Bid, px=0, sz=0, ts=0, new_id=0) -> NormalizedEvent:
    return NormalizedEvent(kind, ts, oid, side, px, sz, new_order_id=new_id, raw_type="")


def _feed(tr: OrderLevelTracker, events) -> None:
    for e in events:
        tr.on_event(e)


# --------------------------------------------------------------------------- tracker
def test_fifo_queue_position_is_exact():
    tr = OrderLevelTracker()
    _feed(tr, [
        _ev_tracked(EventType.ADD, 1, px=100, sz=100, ts=1),
        _ev_tracked(EventType.ADD, 2, px=100, sz=50, ts=2),
        _ev_tracked(EventType.ADD, 3, px=100, sz=30, ts=3),
    ])
    assert (tr.ahead_qty(1), tr.ahead_qty(2), tr.ahead_qty(3)) == (0, 100, 150)
    assert (tr.behind_qty(1), tr.behind_qty(2), tr.behind_qty(3)) == (80, 30, 0)
    assert tr.level_size(Side.Bid, 100) == 180
    assert tr.best_price(Side.Bid) == 100 and tr.best_price(Side.Ask) == 0

    # partial execution at the front shrinks everyone's "ahead" behind it
    _feed(tr, [_ev_tracked(EventType.EXECUTE, 1, sz=40, ts=4)])
    assert (tr.ahead_qty(2), tr.ahead_qty(3)) == (60, 110)
    assert tr.orders[1].filled == 40 and tr.orders[1].size == 60
    assert tr.orders[1].idx_first_fill == 3  # 4th event, zero-based

    # deleting the middle order removes its remaining 50 from those behind
    _feed(tr, [_ev_tracked(EventType.DELETE, 2, ts=5)])
    assert tr.ahead_qty(3) == 60
    assert 2 not in tr.orders and tr.completed[-1].outcome == "cancelled"

    # finishing order 1 puts order 3 at the front
    _feed(tr, [_ev_tracked(EventType.EXECUTE, 1, sz=60, ts=6)])
    assert tr.ahead_qty(3) == 0 and tr.behind_qty(3) == 0
    done = {o.order_id: o for o in tr.completed}
    assert done[1].outcome == "filled" and done[1].filled == 100
    assert tr.level_size(Side.Bid, 100) == 30


def test_partial_cancel_replace_and_unknown_ids():
    tr = OrderLevelTracker()
    _feed(tr, [
        _ev_tracked(EventType.ADD, 7, side=Side.Ask, px=105, sz=80, ts=1),
        _ev_tracked(EventType.ADD, 8, side=Side.Ask, px=105, sz=20, ts=2),
        _ev_tracked(EventType.CANCEL, 7, sz=30, ts=3),  # partial: 80 -> 50
    ])
    assert tr.orders[7].size == 50 and tr.ahead_qty(8) == 50
    # replace moves 7 to a new price with a new id, at the back of that level
    _feed(tr, [_ev_tracked(EventType.ADD, 9, side=Side.Ask, px=106, sz=10, ts=4)])
    _feed(tr, [_ev_tracked(EventType.REPLACE, 7, px=106, sz=25, ts=5, new_id=70)])
    assert 7 not in tr.orders and 70 in tr.orders
    assert tr.orders[70].price == 106 and tr.ahead_qty(70) == 10 and tr.ahead_qty(8) == 0
    assert tr.completed[-1].order_id == 7 and tr.completed[-1].outcome == "cancelled"
    assert tr.level_size(Side.Ask, 105) == 20 and tr.level_size(Side.Ask, 106) == 35
    # events for ids we never saw (e.g. a sliced tape) are counted, never raise
    _feed(tr, [_ev_tracked(EventType.EXECUTE, 999, sz=5, ts=6), _ev_tracked(EventType.DELETE, 998, ts=7)])
    assert tr.stats["unknown_id"] == 2
    # trades (P) never touch displayed queues
    _feed(tr, [_ev_tracked(EventType.TRADE, 0, side=Side.Bid, px=105, sz=5, ts=8)])
    assert tr.level_size(Side.Ask, 105) == 20 and tr.stats["trades"] == 1


def test_touch_distance_features_at_placement():
    tr = OrderLevelTracker()
    _feed(tr, [
        _ev_tracked(EventType.ADD, 1, side=Side.Bid, px=100, sz=10, ts=1),
        _ev_tracked(EventType.ADD, 2, side=Side.Ask, px=104, sz=10, ts=2),
        _ev_tracked(EventType.ADD, 3, side=Side.Bid, px=98, sz=10, ts=3),   # 2 ticks behind best bid
        _ev_tracked(EventType.ADD, 4, side=Side.Bid, px=101, sz=10, ts=4),  # new best bid
        _ev_tracked(EventType.ADD, 5, side=Side.Ask, px=103, sz=10, ts=5),  # new best ask
    ])
    o3, o4, o5 = tr.orders[3], tr.orders[4], tr.orders[5]
    assert o3.same_best_dist == 2 and o3.opp_dist == 6
    assert o4.same_best_dist == -1 and o4.opp_dist == 3
    assert o5.same_best_dist == -1 and o5.opp_dist == 2
    assert tr.best_price(Side.Bid) == 101 and tr.best_price(Side.Ask) == 103
    f = order_features(o3)
    assert set(f) == {"log_ahead", "log_size", "queue_frac", "same_best_dist", "opp_dist", "no_opposite"}
    assert f["queue_frac"] == 0.0 and f["no_opposite"] == 0.0
    # the very first order had no opposite side at all
    assert tr.orders[1].opp_dist == -1 and order_features(tr.orders[1])["no_opposite"] == 1.0


def test_tracker_consumes_encoded_itch_bytes():
    """End-to-end: encode → parse → track, the path the real tape takes."""
    evs = [
        _ev_tracked(EventType.ADD, 11, px=15000, sz=100, ts=1),
        _ev_tracked(EventType.ADD_MPID, 12, px=15000, sz=40, ts=2),
        _ev_tracked(EventType.EXECUTE, 11, sz=100, ts=3),
        _ev_tracked(EventType.DELETE, 12, ts=4),
    ]
    blob = b"".join(encode_event(e) for e in evs)
    tr = OrderLevelTracker()
    for e in iter_itch_events(blob):
        tr.on_event(e)
    outcomes = {o.order_id: o.outcome for o in tr.completed}
    assert outcomes == {11: "filled", 12: "cancelled"}
    assert tr.stats["adds"] == 2 and tr.stats["executes"] == 1 and tr.stats["deletes"] == 1


# --------------------------------------------------------------------------- Kaplan–Meier
def test_kaplan_meier_matches_hand_computation():
    """5 orders with known event-clock outcomes; KM computed by hand below."""
    tr = OrderLevelTracker()
    tr.on_event(_ev_tracked(EventType.ADD, 1, px=100, sz=10, ts=1))    # idx 0
    tr.on_event(_ev_tracked(EventType.ADD, 2, px=100, sz=10, ts=2))    # idx 1
    tr.on_event(_ev_tracked(EventType.EXECUTE, 1, sz=10, ts=3))        # idx 2 -> order1 fill t=2
    tr.on_event(_ev_tracked(EventType.ADD, 3, px=100, sz=10, ts=4))    # idx 3
    tr.on_event(_ev_tracked(EventType.DELETE, 2, ts=5))                # idx 4 -> order2 cancel t=3
    tr.on_event(_ev_tracked(EventType.ADD, 4, px=100, sz=10, ts=6))    # idx 5
    tr.on_event(_ev_tracked(EventType.ADD, 5, px=100, sz=10, ts=7))    # idx 6
    tr.on_event(_ev_tracked(EventType.EXECUTE, 3, sz=10, ts=8))        # idx 7 -> order3 fill t=4
    tr.on_event(_ev_tracked(EventType.DELETE, 4, ts=9))                # idx 8 -> order4 cancel t=3
    tr.on_event(_ev_tracked(EventType.EXECUTE, 5, sz=10, ts=10))       # idx 9 -> order5 fill t=3
    times = {o.order_id: (o.idx_end - o.idx_add, o.outcome) for o in tr.completed}
    assert times == {1: (2, "filled"), 2: (3, "cancelled"), 3: (4, "filled"),
                     4: (3, "cancelled"), 5: (3, "filled")}
    # sorted: 2F, 3C, 3C, 3F, 4F  -> S(2)=4/5=0.8 ; at t=3 at-risk=4, d=1 -> 0.8*3/4=0.6 ;
    # at t=4 at-risk=1, d=1 -> 0
    km = fill_prob_survival(tr.completed, horizons=[1, 2, 3, 4, 10])
    assert km["n"] == 5 and km["n_fill"] == 3 and km["n_cancel"] == 2
    np.testing.assert_allclose(km["survival"], [1.0, 0.8, 0.6, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(km["p_fill"], [0.0, 0.2, 0.4, 1.0, 1.0], atol=1e-12)
    assert km["median_fill_time"] == 4


def test_kaplan_meier_uses_first_fill_time_and_censors_open_orders():
    tr = OrderLevelTracker()
    tr.on_event(_ev_tracked(EventType.ADD, 1, px=100, sz=10, ts=1))  # idx 0
    tr.on_event(_ev_tracked(EventType.ADD, 2, px=100, sz=10, ts=2))  # idx 1 (stays open)
    tr.on_event(_ev_tracked(EventType.EXECUTE, 1, sz=4, ts=3))       # idx 2 -> first fill at t=2
    tr.on_event(_ev_tracked(EventType.ADD, 3, px=100, sz=10, ts=4))  # idx 3
    tr.on_event(_ev_tracked(EventType.EXECUTE, 1, sz=6, ts=5))       # idx 4 -> fully filled at t=4
    open_ = censor_open_orders(tr, ts_end=99)
    assert {o.order_id for o in open_} == {2, 3} and all(o.outcome == "cancelled" for o in open_)
    km = fill_prob_survival(list(tr.completed) + open_, horizons=[1, 2, 3, 4])
    # order1 event of interest at t=2 (first fill), orders 2/3 censored at t=4 and t=2
    assert km["n"] == 3 and km["n_fill"] == 1
    # at t=2: at risk = 3 (t>=2 for all), d=1 -> S=2/3
    np.testing.assert_allclose(km["survival"], [1.0, 2 / 3, 2 / 3, 2 / 3], atol=1e-12)
    # hard exclusion of cancels (not treated as censoring) changes the denominator
    km2 = fill_prob_survival(tr.completed, horizons=[2], cancel_is_censoring=False)
    assert km2["n"] == 1 and km2["p_fill"] == [1.0]


def test_kaplan_meier_empty_input_is_safe():
    km = fill_prob_survival([], horizons=[1, 5])
    assert km["n"] == 0 and km["p_fill"] == [0.0, 0.0] and km["median_fill_time"] is None


def test_censor_open_orders_never_exceeds_session_close():
    """Survival duration must never exceed REGULAR_CLOSE_NS - ts_add, even if tape continues after hours."""
    tr = OrderLevelTracker()
    ts1 = REGULAR_CLOSE_NS - 60_000_000_000  # 15:59:00
    ts2 = REGULAR_CLOSE_NS - 1_000_000_000   # 15:59:59
    ts_post = REGULAR_CLOSE_NS + 10_000_000_000  # 16:00:10 (after close add)

    tr.on_event(_ev_tracked(EventType.ADD, 10, px=100, sz=10, ts=ts1))
    tr.on_event(_ev_tracked(EventType.ADD, 20, px=100, sz=10, ts=ts2))
    tr.on_event(_ev_tracked(EventType.ADD, 30, px=100, sz=10, ts=ts_post))

    censored = censor_open_orders(tr, ts_end=REGULAR_CLOSE_NS)
    # Order 30 added after close must be ignored
    assert {o.order_id for o in censored} == {10, 20}
    for c in censored:
        assert c.ts_end == REGULAR_CLOSE_NS
        assert c.ts_end - c.ts_add <= REGULAR_CLOSE_NS - c.ts_add
        assert c.outcome == "cancelled"


def test_post_close_fills_marked_as_censored_at_regular_close():
    """Orders submitted before 16:00 that fill at 16:00:05 must NOT count as regular-session fills."""
    tr = OrderLevelTracker()
    # Order 1: placed at 15:59:59, fills at 16:00:05 (after-hours fill)
    ts_add1 = REGULAR_CLOSE_NS - 1_000_000_000  # 15:59:59
    ts_fill1 = REGULAR_CLOSE_NS + 5_000_000_000  # 16:00:05
    tr.on_event(_ev_tracked(EventType.ADD, 1, px=100, sz=10, ts=ts_add1))

    # Order 2: placed at 15:30:00, fills at 15:45:00 (regular fill)
    ts_add2 = REGULAR_OPEN_NS + 15 * 60 * 1_000_000_000
    ts_fill2 = REGULAR_OPEN_NS + 30 * 60 * 1_000_000_000
    tr.on_event(_ev_tracked(EventType.ADD, 2, px=100, sz=10, ts=ts_add2))
    tr.on_event(_ev_tracked(EventType.EXECUTE, 2, sz=10, ts=ts_fill2))

    # Order 3: placed at 15:40:00, cancels at 15:50:00 (regular cancel)
    ts_add3 = REGULAR_OPEN_NS + 20 * 60 * 1_000_000_000
    ts_cancel3 = REGULAR_OPEN_NS + 30 * 60 * 1_000_000_000
    tr.on_event(_ev_tracked(EventType.ADD, 3, px=100, sz=10, ts=ts_add3))
    tr.on_event(_ev_tracked(EventType.DELETE, 3, sz=0, ts=ts_cancel3))

    # Now Order 1 fills after close
    tr.on_event(_ev_tracked(EventType.EXECUTE, 1, sz=10, ts=ts_fill1))

    # Order 4: placed at 16:05:00, fills at 16:10:00 (entirely post-close)
    ts_add4 = REGULAR_CLOSE_NS + 300_000_000_000
    ts_fill4 = REGULAR_CLOSE_NS + 600_000_000_000
    tr.on_event(_ev_tracked(EventType.ADD, 4, px=100, sz=10, ts=ts_add4))
    tr.on_event(_ev_tracked(EventType.EXECUTE, 4, sz=10, ts=ts_fill4))

    # Partition session orders
    completed, censored = filter_session_orders(tr, ts_open=REGULAR_OPEN_NS, ts_close=REGULAR_CLOSE_NS)

    # Order 1 (filled post-close) must NOT be in completed; it must be censored at close
    completed_ids = {o.order_id for o in completed}
    assert completed_ids == {2, 3}
    assert {o.order_id: o.outcome for o in completed} == {2: "filled", 3: "cancelled"}

    censored_ids = {o.order_id for o in censored}
    assert censored_ids == {1}
    o1_censored = censored[0]
    assert o1_censored.order_id == 1
    assert o1_censored.outcome == "cancelled"
    assert o1_censored.ts_end == REGULAR_CLOSE_NS
    assert o1_censored.ts_first_fill is None
    assert o1_censored.ts_end - o1_censored.ts_add == 1_000_000_000

    # Also test censor_open_orders with include_post_cutoff=True
    post_censored = censor_open_orders(tr, ts_end=REGULAR_CLOSE_NS, include_post_cutoff=True)
    assert 1 in {o.order_id for o in post_censored}



# --------------------------------------------------------------------------- logistic
def test_logistic_fill_model_recovers_a_known_logistic_queue():
    """Truth: P(fill) = σ(1.0 − 0.9·log_ahead). Calibration slope must be ≈ 1 out of sample."""
    rng = np.random.default_rng(0x51ED)
    n = 6000
    log_ahead = rng.uniform(0.0, 6.0, size=n)
    log_size = rng.uniform(1.0, 5.0, size=n)  # pure noise feature
    p_true = 1.0 / (1.0 + np.exp(-(1.0 - 0.9 * log_ahead)))
    y = rng.random(n) < p_true
    m = logistic_fill_model({"log_ahead": log_ahead, "log_size": log_size}, y, holdout=0.3)
    assert m["n_train"] == 4200 and m["n_test"] == 1800
    assert 0.85 <= m["calibration_slope"] <= 1.15, m["calibration_slope"]
    assert m["brier"] < m["brier_base_rate"] and m["brier_skill"] > 0.15
    # standardized coefficient sign: more queue ahead -> lower fill probability
    assert m["coefs"]["log_ahead"] < 0 and abs(m["coefs"]["log_size"]) < 0.1
    table = m["calibration_table"]
    assert len(table) == 10 and sum(r["n"] for r in table) == 1800
    # monotone decile table: realized rate tracks prediction within noise
    pred = np.array([r["pred"] for r in table])
    real = np.array([r["realized"] for r in table])
    assert np.all(np.diff(pred) >= 0) and np.max(np.abs(pred - real)) < 0.12
    # the fitted predictor is usable on new rows
    p_new = m["predict"](np.column_stack([[0.0, 6.0], [3.0, 3.0]]))
    assert p_new[0] > p_new[1]


def test_logistic_fill_model_walk_forward_not_shuffled():
    """A regime shift in the last 30% must show up out of sample — proof the
    holdout is the time-ordered tail, not a random subsample."""
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.uniform(0, 4, size=n)
    p = 1.0 / (1.0 + np.exp(-(1.5 - 1.0 * x)))
    y = rng.random(n) < p
    y[int(n * 0.7):] = ~y[int(n * 0.7):]  # flip the tail
    m = logistic_fill_model({"x": x}, y)
    assert m["brier"] > m["brier_base_rate"]  # the fitted model is WORSE than base rate on the flipped tail


def test_logistic_rejects_bad_shapes_and_tiny_samples():

    with pytest.raises(ValueError):
        logistic_fill_model({"a": [0.0] * 10}, [True] * 10)
    with pytest.raises(ValueError):
        logistic_fill_model({"a": [0.0] * 30}, [True] * 29)


def test_brier_and_calibration_helpers():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert np.isnan(brier_score([], []))
    rows = calibration_table(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]), bins=2)
    assert [r["n"] for r in rows] == [2, 2] and rows[0]["realized"] == 0.0 and rows[1]["realized"] == 1.0


def test_completed_chronological_ordering():
    """Early arrival orders must appear before late arrival orders in completed_chronological."""
    tr = OrderLevelTracker()
    # Order 1 placed early at t=100
    tr.on_event(NormalizedEvent(EventType.ADD, 100, 1, Side.Bid, 100, 50))
    # Order 2 placed later at t=200
    tr.on_event(NormalizedEvent(EventType.ADD, 200, 2, Side.Bid, 100, 50))
    # Order 2 fills immediately at t=210 (finishes first)
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 210, 2, Side.NONE, 0, 50))
    # Order 1 fills late at t=500 (finishes second)
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 500, 1, Side.NONE, 0, 50))

    # Raw completed list is in completion order: [Order 2, Order 1]
    assert [o.order_id for o in tr.completed] == [2, 1]

    # completed_chronological sorts by (ts_add, idx_add): [Order 1, Order 2]
    chrono = tr.completed_chronological()
    assert [o.order_id for o in chrono] == [1, 2]
    assert chrono[0].ts_add <= chrono[1].ts_add


def test_logistic_fill_model_enforces_temporal_consistency():
    """Walk-forward splits must strictly guarantee max(ts_train) <= min(ts_test)."""
    rng = np.random.default_rng(42)
    n = 100
    # Scrambled timestamps from 1000 to 2000
    timestamps = rng.permutation(np.linspace(1000, 2000, n))
    x = rng.normal(size=n)
    y = rng.choice([True, False], size=n)

    m = logistic_fill_model({"x": x}, y, timestamps=timestamps, holdout=0.3)
    assert m["max_ts_train"] is not None and m["min_ts_test"] is not None
    assert m["max_ts_train"] <= m["min_ts_test"]

