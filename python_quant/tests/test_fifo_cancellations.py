"""Deterministic unit tests for exogenous FIFO cancellations in OrderBookEnv (Person B).

Covers:
1. Existing order ahead cancellation reduces queue-ahead.
2. Partial vs full cancellations.
3. Agent resting order is never cancelled by exogenous flow.
4. Liquidity behind agent cancellation does not affect queue-ahead.
5. Multiple orders ahead preserve FIFO priority across cancellations.
6. Deterministic cancellation behavior under fixed seed with cancel_prob > 0.
7. cancel_prob=0.0 preserves legacy RNG stream bit-for-bit.
8. Exogenous cancellations actively trigger within step loop when cancel_prob > 0.
9. Cancellation ahead enables earlier agent fill upon subsequent depletion.
10. Inventory, cash, and PnL accounting integrity after cancellations and fills.
"""
from __future__ import annotations

import numpy as np
import pytest
from nexus_quant.book_state import Side
from nexus_quant.envs.order_book_env import OrderBookEnv


def test_exogenous_cancel_reduces_queue_ahead():
    """Cancelling liquidity ahead of the agent directly reduces queue-ahead."""
    env = OrderBookEnv(queue_model="fifo", seed=100)
    env.reset()
    env.book.reset()
    o_ahead = env.book.rest(Side.Ask, 15005, 100)
    env.agent_rest = env.book.rest(Side.Ask, 15005, 50)
    assert env._get_queue_ahead() == 100

    # Explicit exogenous cancel of 40 shares ahead
    cancelled = env._exogenous_cancel(o_ahead.order_id, size=40)
    assert cancelled == 40
    assert env._get_queue_ahead() == 60


def test_exogenous_cancel_partial_vs_full():
    """Verify both partial and full cancellations work correctly."""
    env = OrderBookEnv(queue_model="fifo", seed=101)
    env.reset()
    env.book.reset()
    o1 = env.book.rest(Side.Ask, 15005, 80)
    # Partial cancellation
    c1 = env._exogenous_cancel(o1.order_id, size=30)
    assert c1 == 30
    assert env.book.lookup(o1.order_id).size == 50

    # Full cancellation (size >= remaining or size=None)
    c2 = env._exogenous_cancel(o1.order_id, size=None)
    assert c2 == 50
    assert env.book.lookup(o1.order_id) is None


def test_agent_order_never_cancelled_by_exogenous_flow():
    """Agent's resting order must be protected against exogenous cancellation."""
    env = OrderBookEnv(queue_model="fifo", seed=102)
    env.reset()
    env.agent_rest = env.book.rest(Side.Ask, 15005, 50)
    agent_oid = env.agent_rest.order_id
    agent_size = env.agent_rest.size

    # Attempt exogenous cancellation targeting the agent's order ID
    cancelled = env._exogenous_cancel(agent_oid, size=agent_size)
    assert cancelled == 0
    live = env.book.lookup(agent_oid)
    assert live is not None
    assert live.size == agent_size


def test_liquidity_behind_agent_cancellation_does_not_affect_queue_ahead():
    """Liquidity arriving behind agent does not affect queue-ahead when added or cancelled."""
    env = OrderBookEnv(queue_model="fifo", seed=103)
    env.reset()
    env.book.reset()
    env.book.rest(Side.Ask, 15005, 60)
    env.agent_rest = env.book.rest(Side.Ask, 15005, 50)
    assert env._get_queue_ahead() == 60

    # Add order behind agent
    o_behind = env.book.rest(Side.Ask, 15005, 200)
    assert env._get_queue_ahead() == 60

    # Cancel order behind agent
    c = env._exogenous_cancel(o_behind.order_id, size=100)
    assert c == 100
    assert env._get_queue_ahead() == 60


def test_multiple_orders_ahead_fifo_cancellation_preserves_priority():
    """Multiple orders ahead at the same price maintain priority and cumulative queue-ahead."""
    env = OrderBookEnv(queue_model="fifo", seed=104)
    env.reset()
    env.book.reset()
    o1 = env.book.rest(Side.Ask, 15010, 50)
    o2 = env.book.rest(Side.Ask, 15010, 75)
    env.agent_rest = env.book.rest(Side.Ask, 15010, 50)
    assert env._get_queue_ahead() == 125

    # Cancel first order completely
    env._exogenous_cancel(o1.order_id)
    assert env._get_queue_ahead() == 75

    # Cancel second order partially
    env._exogenous_cancel(o2.order_id, size=25)
    assert env._get_queue_ahead() == 50


def test_deterministic_cancellation_with_cancel_prob():
    """Two envs with same seed and cancel_prob > 0 produce bit-for-bit identical executions."""
    env1 = OrderBookEnv(queue_model="fifo", cancel_prob=0.4, seed=4242)
    env2 = OrderBookEnv(queue_model="fifo", cancel_prob=0.4, seed=4242)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    np.testing.assert_array_equal(obs1, obs2)

    for a in [0.2, 0.4, 0.4, 0.1, -0.5, 0.3]:
        o1, r1, _d1, _t1, inf1 = env1.step([a])
        o2, r2, _d2, _t2, inf2 = env2.step([a])
        np.testing.assert_array_equal(o1, o2)
        assert r1 == pytest.approx(r2)
        assert inf1["queue_ahead"] == inf2["queue_ahead"]
        assert inf1["filled"] == inf2["filled"]


def test_cancel_prob_zero_preserves_legacy_rng():
    """cancel_prob=0.0 preserves identical RNG stream and outputs as baseline."""
    env_base = OrderBookEnv(queue_model="fifo", cancel_prob=0.0, seed=777)
    env_default = OrderBookEnv(queue_model="fifo", seed=777)

    env_base.reset()
    env_default.reset()

    for a in [0.3, 0.5, 0.5, -0.2, 0.0, 0.7]:
        o1, r1, _d1, _t1, inf1 = env_base.step([a])
        o2, r2, _d2, _t2, inf2 = env_default.step([a])
        np.testing.assert_array_equal(o1, o2)
        assert r1 == pytest.approx(r2)
        assert inf1["filled"] == inf2["filled"]


def test_exogenous_cancellations_active_in_step_loop():
    """With cancel_prob=1.0, exogenous cancellations fire during environment steps."""
    env = OrderBookEnv(queue_model="fifo", cancel_prob=1.0, seed=888)
    env.reset()
    # Seed book has resting orders
    initial_order_count = len(env.book._orders)
    assert initial_order_count > 0

    # Step through: cancellations will trigger each step
    cancellations_seen = 0
    for _ in range(10):
        pre_sizes = {oid: o.size for oid, o in env.book._orders.items()}
        env.step([0.5])
        post_sizes = {oid: o.size for oid, o in env.book._orders.items()}
        # Check if any pre-existing order size decreased or order vanished
        for oid, sz in pre_sizes.items():
            if oid not in post_sizes or post_sizes[oid] < sz:
                cancellations_seen += 1
                break

    assert cancellations_seen > 0


def test_cancellation_ahead_enables_earlier_agent_fill():
    """Cancelling queue ahead allows external market orders to reach the agent earlier."""
    env = OrderBookEnv(queue_model="fifo", cancel_prob=0.0, seed=123)
    env.reset()
    env.book.reset()
    o_ahead = env.book.rest(Side.Ask, 15005, 100)
    env.agent_rest = env.book.rest(Side.Ask, 15005, 50)
    assert env._get_queue_ahead() == 100

    # Cancel 70 shares ahead -> only 30 left ahead
    env._exogenous_cancel(o_ahead.order_id, size=70)
    assert env._get_queue_ahead() == 30

    # External trade of 40 shares: 30 consumes ahead, 10 fills agent
    env.book.take(Side.Bid, 40)
    assert env._get_queue_ahead() == 0
    live_agent = env.book.lookup(env.agent_rest.order_id)
    assert live_agent is not None
    assert live_agent.size == 40  # 10 shares filled


def test_inventory_and_cash_accounting_exact_after_cancellation():
    """Inventory, cash, and fills remain strictly consistent after cancellations and fills."""
    env = OrderBookEnv(inventory=500, queue_model="fifo", cancel_prob=0.5, seed=555)
    env.reset()
    for _ in range(15):
        _, _, term, trunc, _ = env.step([0.3])
        if term or trunc:
            break

    total_filled = sum(sz for _, _, sz in env.fills)
    assert total_filled + env.inventory == 500
    expected_cash = sum(px * sz for _, px, sz in env.fills)
    assert env.cash_ticks == expected_cash
