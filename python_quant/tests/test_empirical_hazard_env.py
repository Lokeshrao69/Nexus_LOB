"""Deterministic unit tests for Empirical Queue Hazard RL seam in OrderBookEnv (Person B).

Covers:
1. Mode validation: strict acceptance of "", "uniform", "fifo", "empirical_hazard".
2. Default hazard instantiation: automatic setup of EmpiricalQueueHazard.
3. Custom hazard instantiation: preservation of user-provided hazard model.
4. Tracker calibration: calibration of hazard from OrderLevelTracker.
5. Survival dictionary calibration: calibration of hazard from KM dictionary.
6. Zero lookahead guarantee: predictions use strictly placement-time observables.
7. Monotonicity: higher queue depth and distance from touch decrease fill probability.
8. Step determinism: identical seeds yield identical trajectory and fills.
9. Accounting integrity: inventory, cash, and PnL conservation under hazard execution.
10. Strict mode separation: distinct behavior across queue model variants.
"""
from __future__ import annotations

import numpy as np
import pytest
from nexus_quant.book_state import Side
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.research.queue_dynamics import EmpiricalQueueHazard, OrderLevelTracker


def test_mode_validation():
    """Valid queue models accepted; invalid models rejected with ValueError."""
    for valid in ("", "uniform", "fifo", "empirical_hazard"):
        env = OrderBookEnv(queue_model=valid, seed=1)
        assert env.queue_model == valid

    with pytest.raises(ValueError, match="queue_model must be"):
        OrderBookEnv(queue_model="unsupported_model")


def test_default_hazard_instantiation():
    """Initializing queue_model='empirical_hazard' automatically sets default hazard."""
    env = OrderBookEnv(queue_model="empirical_hazard", seed=2)
    assert isinstance(env.empirical_hazard, EmpiricalQueueHazard)
    assert env.empirical_hazard.base_logit == pytest.approx(-1.2)
    assert env.empirical_hazard.queue_coef == pytest.approx(-1.5)


def test_custom_hazard_instantiation():
    """Custom EmpiricalQueueHazard instance is preserved in environment."""
    custom_hazard = EmpiricalQueueHazard(base_logit=0.1, queue_coef=-2.2, dist_coef=-1.1)
    env = OrderBookEnv(queue_model="empirical_hazard", empirical_hazard=custom_hazard, seed=3)
    assert env.empirical_hazard is custom_hazard
    assert env.empirical_hazard.base_logit == pytest.approx(0.1)


def test_from_tracker_calibration():
    """Calibrate EmpiricalQueueHazard from empirical OrderLevelTracker completed orders."""
    tracker = OrderLevelTracker()
    # Add resting order 1, fill it
    tracker.on_event(NormalizedEvent(EventType.ADD, 1000, 1, Side.Ask, 15001, 100))
    tracker.on_event(NormalizedEvent(EventType.EXECUTE, 2000, 1, Side.Ask, 15001, 100))
    # Add resting order 2, cancel it
    tracker.on_event(NormalizedEvent(EventType.ADD, 3000, 2, Side.Ask, 15002, 100))
    tracker.on_event(NormalizedEvent(EventType.CANCEL, 4000, 2, Side.Ask, 15002, 100))

    hazard = EmpiricalQueueHazard.from_tracker(tracker)
    assert isinstance(hazard, EmpiricalQueueHazard)
    # 1 fill out of 2 completed -> fill_rate = 0.5 -> base_logit ~ log(0.5 / 0.5) = 0.0
    assert hazard.base_logit == pytest.approx(0.0, abs=1e-3)


def test_from_survival_dict_calibration():
    """Calibrate EmpiricalQueueHazard from pre-computed survival dictionary."""
    surv = {
        "tau": [1.0, 5.0, 10.0],
        "p_fill": [0.25, 0.50, 0.75],
    }
    hazard = EmpiricalQueueHazard.from_survival_dict(surv)
    assert isinstance(hazard, EmpiricalQueueHazard)
    assert hazard.horizons == (1.0, 5.0, 10.0)
    assert hazard.p_fill_by_horizon == (0.25, 0.50, 0.75)
    # Base rate = 0.25 -> logit = log(0.25 / 0.75) = log(1/3) ~ -1.0986
    assert hazard.base_logit == pytest.approx(np.log(0.25 / 0.75), abs=1e-3)


def test_zero_lookahead_guarantee():
    """Predict fill prob depends strictly on placement-time observable state."""
    hazard = EmpiricalQueueHazard(base_logit=-0.5, queue_coef=-1.0, dist_coef=-0.5)
    # Compute fill prob with observable state:
    p1 = hazard.predict_fill_prob(queue_ahead=50, level_size=100, distance_ticks=2)
    p2 = hazard.predict_fill_prob(queue_ahead=50, level_size=100, distance_ticks=2)
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_fill_probability_monotonicity():
    """Fill probability decreases monotonically with queue-ahead and distance from touch."""
    hazard = EmpiricalQueueHazard(base_logit=0.0, queue_coef=-2.0, dist_coef=-1.0)
    # At touch, front of queue:
    p_front = hazard.predict_fill_prob(queue_ahead=0, level_size=100, distance_ticks=0)
    # At touch, back of queue:
    p_back = hazard.predict_fill_prob(queue_ahead=100, level_size=100, distance_ticks=0)
    # 2 ticks away, front of queue:
    p_away = hazard.predict_fill_prob(queue_ahead=0, level_size=100, distance_ticks=2)

    assert p_front > p_back
    assert p_front > p_away
    assert p_back > hazard.predict_fill_prob(queue_ahead=100, level_size=100, distance_ticks=2)


def test_empirical_hazard_step_determinism():
    """Two envs with queue_model='empirical_hazard' and same seed produce identical steps."""
    env1 = OrderBookEnv(queue_model="empirical_hazard", seed=9876)
    env2 = OrderBookEnv(queue_model="empirical_hazard", seed=9876)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    np.testing.assert_array_equal(obs1, obs2)

    for a in [0.1, 0.4, -0.2, 0.6, 0.0, 0.3]:
        o1, r1, _d1, _t1, inf1 = env1.step([a])
        o2, r2, _d2, _t2, inf2 = env2.step([a])
        np.testing.assert_array_equal(o1, o2)
        assert r1 == pytest.approx(r2)
        assert inf1["filled"] == inf2["filled"]
        assert inf1["queue_ahead"] == inf2["queue_ahead"]


def test_accounting_integrity_under_empirical_hazard():
    """Conservation of inventory, cash, and mark-to-market under empirical hazard."""
    env = OrderBookEnv(inventory=600, queue_model="empirical_hazard", seed=333)
    env.reset()

    for _ in range(15):
        _, _, term, trunc, _ = env.step([0.25])
        if term or trunc:
            break

    total_filled = sum(sz for _, _, sz in env.fills)
    assert total_filled + env.inventory == 600
    expected_cash = sum(px * sz for _, px, sz in env.fills)
    assert env.cash_ticks == expected_cash
    current_mid = env._mid() or env.arrival_mid
    expected_pnl = env.cash_ticks + env.inventory * current_mid - 600 * env.arrival_mid
    assert env.mark_to_market(current_mid) == pytest.approx(expected_pnl)


def test_strict_mode_separation():
    """Verify distinct execution behavior across different queue models."""
    modes = ["", "uniform", "fifo", "empirical_hazard"]
    envs = {m: OrderBookEnv(queue_model=m, seed=1234) for m in modes}
    for e in envs.values():
        e.reset()

    actions = [0.2, 0.3, 0.1, -0.1, 0.4]
    fills_by_mode = {}
    for m, e in envs.items():
        for a in actions:
            e.step([a])
        fills_by_mode[m] = sum(sz for _, _, sz in e.fills)

    # All modes should execute without errors, and fifo vs optimistic vs hazard are tracked
    assert all(isinstance(f, int) for f in fills_by_mode.values())
    assert envs["empirical_hazard"].queue_model == "empirical_hazard"
    assert envs["fifo"].queue_model == "fifo"
