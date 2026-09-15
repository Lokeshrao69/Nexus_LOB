"""High-volatility regime tests for OrderBookEnv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nexus_quant.baselines import compare, run_episode
from nexus_quant.envs.order_book_env import OBS_DIM, OrderBookEnv


def _highvol_env(inventory=2000, horizon=40, **kw) -> OrderBookEnv:
    return OrderBookEnv(
        inventory=inventory, horizon=horizon, seed=0x51ED,
        regime_prob=0.08, vol_decay=0.15,
        gap_prob=0.20, gap_min=800, gap_max=1800, gap_down_prob=0.75,
        vol_take_prob=0.55, vol_take_min=60, vol_take_max=220,
        vol_add_min=15, vol_add_max=70,
        vol_add_offset_min=2, vol_add_offset_max=12,
        vol_events_min=3, vol_events_max=8,
        **kw,
    )


def test_highvol_default_identity():
    """Default env (no regime params) produces same obs shape and finite values."""
    env = OrderBookEnv(inventory=400, horizon=8, seed=99)
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_regime_transitions():
    """With regime_prob=1.0 and vol_decay=0.0, _volatile flips to True on first step."""
    env = OrderBookEnv(
        inventory=200, horizon=4, seed=42,
        regime_prob=1.0, vol_decay=0.0,
        gap_prob=0.0,
    )
    env.reset()
    assert env._volatile is False
    env.step(0.0)
    assert env._volatile is True


def test_regime_persists_with_low_decay():
    """With vol_decay=0.0, once volatile it stays volatile."""
    env = OrderBookEnv(
        inventory=200, horizon=10, seed=42,
        regime_prob=1.0, vol_decay=0.0,
        gap_prob=0.0,
    )
    env.reset()
    for _ in range(5):
        env.step(0.0)
    assert env._volatile is True


def test_regime_can_clear():
    """With high vol_decay, regime can transition back to calm."""
    env = OrderBookEnv(
        inventory=200, horizon=20, seed=42,
        regime_prob=1.0, vol_decay=1.0,
        gap_prob=0.0,
    )
    env.reset()
    env.step(0.0)
    assert env._volatile is True
    env.step(0.0)
    assert env._volatile is False  # vol_decay=1.0 always clears


def test_gap_reduces_bid_depth():
    """A gap event with gap_down_prob=1.0 consumes bid levels, dropping the mid."""
    env = OrderBookEnv(
        inventory=2000, horizon=40, seed=42,
        regime_prob=1.0, vol_decay=0.0,
        gap_prob=1.0, gap_min=1500, gap_max=1500, gap_down_prob=1.0,
        vol_take_prob=0.0,  # no extra takes, only the gap
    )
    env.reset()
    s0 = env.book.view()
    mid0 = env._mid(s0)
    assert mid0 is not None
    # step — gap fires inside _exogenous_flow
    env.step(0.0)
    s1 = env.book.view()
    mid1 = env._mid(s1)
    # mid should have dropped (or at least not risen) after a 1500-share bid sweep
    assert mid1 is not None
    assert mid1 <= mid0 + 1  # allow tiny rounding, but no rise


def test_vol_feature_dim():
    """vol_feature=True → obs shape (45,), index 44 is 0.0 or 1.0."""
    env = _highvol_env(vol_feature=True)
    obs, _ = env.reset()
    assert obs.shape == (45,)
    assert obs[44] == 0.0  # starts calm
    # run a few steps to possibly enter volatile
    for _ in range(20):
        obs, r, term, trunc, _ = env.step(0.0)
        if term or trunc:
            break
        assert obs[44] in (0.0, 1.0)


def test_vol_feature_off_default():
    """vol_feature=False (default) → obs shape (44,)."""
    env = _highvol_env(vol_feature=False)
    obs, _ = env.reset()
    assert obs.shape == (44,)


def test_inventory_conservation_highvol():
    """Inventory is conserved even with aggressive gap events."""
    Q = 2000
    env = _highvol_env(inventory=Q, horizon=40, child_max=220)
    env.reset()
    while True:
        _o, _r, term, trunc, _i = env.step(-1.0)
        if term or trunc:
            break
    filled = sum(sz for _, _, sz in env.fills)
    assert filled + env.inventory == Q
    assert env.inventory >= 0


def test_obs_finite_highvol():
    """All obs values finite even in volatile regime with gaps."""
    env = _highvol_env()
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs))
    for _ in range(40):
        obs, r, term, trunc, _ = env.step(-0.5)
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {env.t}"
        if term or trunc:
            break


def test_highvol_baselines_run():
    """All 4 baselines complete without error on highvol env."""
    rows = compare(
        seed=11, inventory=2000, horizon=40, child_max=220,
        regime_prob=0.08, vol_decay=0.15,
        gap_prob=0.20, gap_min=800, gap_max=1800,
        vol_take_prob=0.55, vol_take_min=60, vol_take_max=220,
        vol_add_min=15, vol_add_max=70,
        vol_add_offset_min=2, vol_add_offset_max=12,
        vol_events_min=3, vol_events_max=8,
    )
    assert {r.name for r in rows} == {"twap", "vwap", "pov", "passive"}
    for r in rows:
        assert r.steps > 0
        assert r.filled + r.leftover == 2000
        assert np.isfinite(r.reward)


def test_highvol_shortfall_higher_than_calm():
    """VWAP shortfall should be higher (worse) in highvol vs calm env."""
    calm_rows = compare(seed=42, inventory=2000, horizon=40, child_max=220)
    vol_rows = compare(
        seed=42, inventory=2000, horizon=40, child_max=220,
        regime_prob=0.08, vol_decay=0.15,
        gap_prob=0.20, gap_min=800, gap_max=1800,
        vol_take_prob=0.55, vol_take_min=60, vol_take_max=220,
        vol_add_min=15, vol_add_max=70,
        vol_add_offset_min=2, vol_add_offset_max=12,
        vol_events_min=3, vol_events_max=8,
    )
    calm_vwap = next(r for r in calm_rows if r.name == "vwap")
    vol_vwap = next(r for r in vol_rows if r.name == "vwap")
    # highvol should make VWAP shortfall worse (higher bps)
    assert vol_vwap.shortfall_bps > calm_vwap.shortfall_bps
