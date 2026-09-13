"""WS-1 tests for queue_dynamics: queue reconstruction, controlled fills, KM, logistic model."""
from __future__ import annotations

import numpy as np
import pytest
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.research.queue_dynamics import (
    FillRow,
    LogisticFillModel,
    OrderLife,
    QueueTracker,
    RestingOrder,
    _decision_features,
    fill_dataset,
    fill_prob_survival,
    queue_ahead_walk,
    standing_order_lifetimes,
)
from nexus_quant.research.synthetic_flow import FlowConfig, SyntheticFlow


# ---- helpers ----------------------------------------------------------------- #

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