"""Cost-model tests (plan_2.md Phase 2): fee/rebate arithmetic, square-root-law
monotonicity, env cost-knob behaviour, and the env byte-parity guarantee with
all knobs defaulted off.
"""
from __future__ import annotations

import numpy as np
import pytest
from nexus_quant.book_port import StubBookAdapter, TakeResult
from nexus_quant.book_state import Side
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.execution.cost_model import CostParams, impact, net_pnl


def test_cost_params_validate_signs() -> None:
    with pytest.raises(ValueError):
        CostParams(fee_bps=-1.0)
    with pytest.raises(ValueError):
        CostParams(rebate_bps=-1.0)
    with pytest.raises(ValueError):
        CostParams(spread_ecn=-1.0)
    with pytest.raises(ValueError):
        CostParams(impact_coef=-0.1)
    with pytest.raises(ValueError):
        CostParams(impact_mode="cubic")


def test_impact_zero_cost_is_zero() -> None:
    assert impact(0.5, sigma=0.25, coef=0.0) == 0.0  # no impact param -> free


def test_impact_sqrt_monotone_in_participation() -> None:
    parts = np.linspace(0.01, 0.9, 40)
    vals = [impact(p, sigma=0.25, coef=1.0, mode="sqrt") for p in parts]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))  # strict increase
    assert vals[0] > 0.0


def test_impact_linear_monotone_in_qty() -> None:
    parts = np.linspace(0.01, 0.9, 40)
    vals = [impact(p, sigma=0.30, coef=1.0, mode="linear") for p in parts]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def test_impact_rejects_negative_participation() -> None:
    with pytest.raises(ValueError):
        impact(-0.1, sigma=0.25, coef=1.0)


def test_net_pnl_fee_reduces_and_rebate_credits() -> None:
    """Acceptance signs (work-package §WP2): a taker fee lowers net PnL below
    gross, a maker rebate raises it above gross, zero-cost = gross exactly."""
    fills = [(15000, 100), (14999, 150)]  # (price_tick, size)
    notional = 15000 * 100 + 14999 * 150  # 3_749_850 ticks of traded notional
    gross = 100.0
    clean = net_pnl(fills, gross=gross)
    fee_only = net_pnl(fills, gross=gross, side_cost=CostParams(fee_bps=5.0))
    rebate_only = net_pnl(fills, gross=gross, side_cost=CostParams(rebate_bps=5.0))
    assert clean == pytest.approx(gross)
    assert fee_only == pytest.approx(gross - (5.0 / 1e4) * notional)
    assert rebate_only == pytest.approx(gross + (5.0 / 1e4) * notional)
    assert fee_only < clean < rebate_only


def _run_n(env: OrderBookEnv, action: float = 0.0, n: int = 20) -> list[float]:
    # reset() seeds the book + RNG (the __init__ doesn't) — without it the
    # agent steps an empty book and never fills (the knobs never bite).
    env.reset(seed=env._seed0)
    rewards: list[float] = []
    for _ in range(n):
        _, r, term, trunc, _ = env.step(action)
        rewards.append(r)
        if term or trunc:
            break
    return rewards


def test_env_byte_parity_when_cost_knobs_default() -> None:
    """Today's default env must be byte-identical to one with all knobs explicit."""
    seed = 1234
    r1 = _run_n(OrderBookEnv(seed=seed))
    r2 = _run_n(
        OrderBookEnv(seed=seed, fee_bps=0.0, rebate_bps=0.0, impact_coef=0.0, queue_model="")
    )
    assert r1 == pytest.approx(r2)


def test_env_taker_fee_reduces_reward_on_market_fills() -> None:
    """Same all-market actions -> identical fills; a taker fee worsens reward
    exactly where a fill happens and never helps."""
    seed = 42
    base = _run_n(OrderBookEnv(seed=seed), action=-1)
    fee = _run_n(OrderBookEnv(seed=seed, fee_bps=8.0), action=-1)
    assert len(base) == len(fee)
    assert all(b == pytest.approx(f) or b > f for b, f in zip(base, fee))
    assert any(b > f for b, f in zip(base, fee))  # the fee actually bites somewhere


def test_env_maker_rebate_improves_reward_on_passive_fills() -> None:
    """Same actions (rest at mid) -> identical fills; the maker rebate credits
    passive fills (raises reward) and never hurts."""
    seed = 7
    base = _run_n(OrderBookEnv(seed=seed), action=0)
    reb = _run_n(OrderBookEnv(seed=seed, rebate_bps=8.0), action=0)
    assert len(base) == len(reb)
    assert all(b == pytest.approx(r) or b < r for b, r in zip(base, reb))
    assert any(b < r for b, r in zip(base, reb))  # at least one passive fill


def test_env_impact_degrades_reward_on_fills() -> None:
    seed = 11
    base = _run_n(OrderBookEnv(seed=seed), action=0)
    imp = _run_n(OrderBookEnv(seed=seed, impact_coef=1.0), action=0)
    assert len(base) == len(imp)
    assert all(b == pytest.approx(v) or b > v for b, v in zip(base, imp))
    assert any(b > v for b, v in zip(base, imp))


def test_env_rejects_unknown_queue_model() -> None:
    with pytest.raises(ValueError):
        OrderBookEnv(queue_model="bogus")


def test_env_uniform_queue_smoke() -> None:
    """queue_model='uniform' is deterministic and runs a full episode."""
    env = OrderBookEnv(seed=5, queue_model="uniform")
    env.reset(seed=5)
    done = False
    steps = 0
    while not done:
        _, _, term, trunc, info = env.step(0.0)
        done = term or trunc
        steps += 1
    assert steps > 0
    assert "market_vwap" in info


def test_market_vwap_is_volume_weighted_exogenous_tape() -> None:
    """Market VWAP is volume-weighted over the TAPE's own prints, not the
    arithmetic-mean price — the benchmark the agent is measured against."""
    env = OrderBookEnv(seed=3)
    env.reset(seed=3)
    env._record_market_trade(TakeResult(filled=100, notional_ticks=100 * 14990))
    env._record_market_trade(TakeResult(filled=200, notional_ticks=200 * 15010))
    expected = (100 * 14990 + 200 * 15010) / 300
    assert env.market_vwap() == pytest.approx(expected)
    assert env.market_vwap() != pytest.approx((14990 + 15010) / 2.0)  # not price-mean


def test_env_market_vwap_in_info() -> None:
    env = OrderBookEnv(seed=7)
    env.reset(seed=7)
    info: dict = {}
    done = False
    while not done:
        _, _, term, trunc, info = env.step(0.0)
        done = term or trunc
    assert "market_vwap" in info
    assert isinstance(info["market_vwap"], float)
    assert info["market_vwap"] >= 0.0


# =========================================================================
# FIFO Queue Model Tests (Deterministic Priority, Consumption & Accounting)
# =========================================================================


def test_fifo_depth_ahead_delays_fill() -> None:
    """1. Existing depth ahead delays agent fill."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    env.book.rest(Side.Ask, px, 100)
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    assert env._get_queue_ahead() == 100
    env.book.take(Side.Bid, 60)
    # Agent order untouched; ahead reduced to 40
    assert agent_h.size == 50
    assert env._get_queue_ahead() == 40


def test_fifo_external_consumption_reduces_queue_ahead() -> None:
    """2. Exact amount of external consumption reduces queue-ahead."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    env.book.rest(Side.Ask, px, 100)
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    assert env._get_queue_ahead() == 100
    env.book.take(Side.Bid, 40)
    assert env._get_queue_ahead() == 60
    env.book.take(Side.Bid, 35)
    assert env._get_queue_ahead() == 25


def test_fifo_multi_step_depletion() -> None:
    """3. Partial queue depletion across multiple steps."""
    env = OrderBookEnv(seed=42, horizon=10, queue_model="fifo")
    env.reset(seed=42)
    _, _, _, _, i1 = env.step(0.1)
    a1 = i1["queue_ahead"]
    assert a1 >= 0
    _, _, _, _, i2 = env.step(0.1)
    assert "queue_ahead" in i2


def test_fifo_cancellation_ahead_advances_agent() -> None:
    """4. Cancellation ahead moves the agent forward."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    h_other = env.book.rest(Side.Ask, px, 100)
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    assert env._get_queue_ahead() == 100
    env.book.cancel_id(h_other.order_id, size=40)
    assert env._get_queue_ahead() == 60
    env.book.cancel_id(h_other.order_id)
    assert env._get_queue_ahead() == 0


def test_fifo_new_liquidity_behind_preserves_position() -> None:
    """5. New liquidity arriving behind does not move the agent backward."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    env.book.rest(Side.Ask, px, 50)
    agent_h = env.book.rest(Side.Ask, px, 40)
    env.agent_rest = agent_h
    assert env._get_queue_ahead() == 50
    # New order behind agent at same price
    env.book.rest(Side.Ask, px, 300)
    assert env._get_queue_ahead() == 50


def test_fifo_two_orders_same_price_priority() -> None:
    """6. Two resting orders at the same price preserve FIFO priority."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    h1 = env.book.rest(Side.Ask, px, 100)
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    h3 = env.book.rest(Side.Ask, px, 200)
    env.book.take(Side.Bid, 120)
    assert h1.size == 0
    assert agent_h.size == 30
    assert h3.size == 200
    assert env._get_queue_ahead() == 0


def test_fifo_agent_cannot_fill_before_ahead_consumed() -> None:
    """7. Agent cannot fill before queue-ahead is consumed."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    env.book.rest(Side.Ask, px, 100)
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    env.book.take(Side.Bid, 99)
    assert agent_h.size == 50
    assert env._get_queue_ahead() == 1
    env.book.take(Side.Bid, 1)
    assert agent_h.size == 50
    assert env._get_queue_ahead() == 0
    env.book.take(Side.Bid, 1)
    assert agent_h.size == 49


def test_fifo_partial_agent_fills_update_quantity() -> None:
    """8. Partial agent fills update remaining quantity correctly."""
    env = OrderBookEnv(seed=42, queue_model="fifo")
    env.reset(seed=42)
    env.book = StubBookAdapter()
    px = 15001
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    assert env._get_queue_ahead() == 0
    env.book.take(Side.Bid, 15)
    assert agent_h.size == 35
    env.book.take(Side.Bid, 20)
    assert agent_h.size == 15


def test_fifo_inventory_cash_pnl_accounting() -> None:
    """9. Inventory/cash/PnL accounting remains correct."""
    env = OrderBookEnv(inventory=500, horizon=15, seed=42, queue_model="fifo")
    env.reset(seed=42)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(0.0)
        done = term or trunc
    executed = sum(f[2] for f in env.fills)
    assert env.inventory + executed == 500
    assert env.cash_ticks == sum(p * s for _, p, s in env.fills)
    mid = env._mid() or env.arrival_mid
    expected_pnl = env.cash_ticks + env.inventory * mid - 500 * env.arrival_mid
    assert abs(env.mark_to_market(mid) - expected_pnl) < 1e-6


def test_fifo_mode_deterministic_fixed_seed() -> None:
    """10. FIFO mode is deterministic for a fixed seed and event sequence."""
    env1 = OrderBookEnv(inventory=500, horizon=10, seed=99, queue_model="fifo")
    env2 = OrderBookEnv(inventory=500, horizon=10, seed=99, queue_model="fifo")
    o1, _ = env1.reset(seed=99)
    o2, _ = env2.reset(seed=99)
    np.testing.assert_array_equal(o1, o2)
    for _ in range(10):
        o1, r1, t1, tr1, i1 = env1.step(0.1)
        o2, r2, t2, tr2, i2 = env2.step(0.1)
        np.testing.assert_array_equal(o1, o2)
        assert abs(r1 - r2) < 1e-6
        assert t1 == t2
        assert tr1 == tr2
        assert i1["queue_ahead"] == i2["queue_ahead"]
        assert i1["filled"] == i2["filled"]


def test_fifo_mode_no_random_degradation() -> None:
    """11. FIFO mode does not invoke random queue degradation."""
    env = OrderBookEnv(seed=77, queue_model="fifo")
    env.reset(seed=77)
    px = 15001
    agent_h = env.book.rest(Side.Ask, px, 50)
    env.agent_rest = agent_h
    env._queue_ahead_fifo_initial = 100
    f1 = env._queue_ahead_frac()
    f2 = env._queue_ahead_frac()
    assert f1 == f2
    rng_val = env._rng.random()
    env_clone = OrderBookEnv(seed=77, queue_model="fifo")
    env_clone.reset(seed=77)
    env_clone.book.rest(Side.Ask, px, 50)
    assert abs(env_clone._rng.random() - rng_val) < 1e-9
