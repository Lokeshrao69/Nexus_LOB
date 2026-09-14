"""Hand-constructed test cases for E7 execution metrics: max_drawdown and fill_rate."""

from __future__ import annotations

import numpy as np
import pytest

from nexus_quant.agents.evaluate import _episode_rows
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.execution.metrics import fill_rate, max_drawdown


# ---------------------------------------------------------------------------
# max_drawdown hand-constructed tests
# ---------------------------------------------------------------------------
def test_max_drawdown_strictly_increasing():
    # Never draws down; peak updates at every step
    path = [0.0, 5.0, 10.0, 15.0, 20.0]
    assert max_drawdown(path) == 0.0


def test_max_drawdown_strictly_decreasing():
    # Trough is at the very end
    path = [20.0, 15.0, 10.0, 5.0, 0.0]
    assert max_drawdown(path) == pytest.approx(20.0)


def test_max_drawdown_multi_peak_known():
    # Peak reaches 20, drops to 8 -> drawdown 12; later recovery to 15 does not exceed peak
    path = [0.0, 10.0, 5.0, 20.0, 8.0, 15.0]
    assert max_drawdown(path) == pytest.approx(12.0)

    # First peak 120 -> drop to 70 (dd 50). Later new peak 130 -> drop to 90 (dd 40).
    path2 = [100.0, 120.0, 80.0, 110.0, 70.0, 130.0, 90.0]
    assert max_drawdown(path2) == pytest.approx(50.0)


def test_max_drawdown_negative_paths():
    # Peak is -10, trough is -30 -> drawdown is 20
    path = [-10.0, -20.0, -15.0, -30.0]
    assert max_drawdown(path) == pytest.approx(20.0)


def test_max_drawdown_edge_cases():
    assert max_drawdown([]) == 0.0
    assert max_drawdown([42.0]) == 0.0
    assert max_drawdown([10.0, 10.0, 10.0]) == 0.0


# ---------------------------------------------------------------------------
# fill_rate hand-constructed tests
# ---------------------------------------------------------------------------
def test_fill_rate_boundaries_and_empty():
    assert fill_rate([], 100) == 0.0
    assert fill_rate([(100, 50)], 0) == 0.0
    assert fill_rate([(100, 50)], -10) == 0.0


def test_fill_rate_partial_and_multi_slice():
    # Single slice: 50 / 200 = 0.25
    assert fill_rate([(100, 50)], 200) == pytest.approx(0.25)

    # Multi-slice 3-tuple format (t, px, sz): 50 + 50 = 100 / 200 = 0.5
    assert fill_rate([(0, 100, 50), (1, 101, 50)], 200) == pytest.approx(0.5)


def test_fill_rate_complete_and_overfill_clamp():
    # Exact completion
    assert fill_rate([(100, 200)], 200) == pytest.approx(1.0)

    # Overfill safety clamp
    assert fill_rate([(100, 250)], 200) == pytest.approx(1.0)


def test_fill_rate_inventory_conservation():
    """Verify fill_rate vs completion mathematical equivalence when inventory is conserved."""
    env = OrderBookEnv(seed=123, inventory=50)
    obs, _ = env.reset(seed=123)
    # Take an aggressive market sell action to force fills
    for _ in range(5):
        _, _, term, trunc, _ = env.step(-1.0)
        if term or trunc:
            break

    total_filled = sum(sz for _, _, sz in env.fills)
    assert env.inventory0 - env.inventory == total_filled

    fr = fill_rate(env.fills, env.inventory0)
    completion = 1.0 - env.inventory / env.inventory0
    assert fr == pytest.approx(completion)


def test_episode_rows_includes_e7_metrics():
    """Verify _episode_rows exposes fill_rate and mdd_ticks."""
    env = OrderBookEnv(seed=7)
    rows = _episode_rows(env, lambda _env, _ob: -0.5, seeds=[7, 8])
    assert len(rows) == 2
    for r in rows:
        assert "fill_rate" in r
        assert "mdd_ticks" in r
        assert 0.0 <= r["fill_rate"] <= 1.0
        assert r["mdd_ticks"] >= 0.0
