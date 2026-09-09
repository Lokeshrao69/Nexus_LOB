"""EngineAdapter + OrderBookEnv against a compiled nexus_engine.

Skipped when the pybind module is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ne = pytest.importorskip("nexus_engine")

from nexus_quant.baselines import run_episode
from nexus_quant.book_port import EngineAdapter, adapt
from nexus_quant.book_state import Side
from nexus_quant.envs.order_book_env import OBS_DIM, OrderBookEnv


def test_adapt_keeps_engine_across_reset():
    env = OrderBookEnv(inventory=200, horizon=4, seed=4, child_max=50, book=ne.Engine())
    assert isinstance(env.book, EngineAdapter)
    env.reset()
    assert isinstance(env.book, EngineAdapter)
    assert int(env.book.view()["bid_px"][0]) > 0
    assert int(env.book.view()["ask_px"][0]) > 0


def test_engine_limit_market_cancel_fills():
    ad = EngineAdapter(ne.Engine())
    ad.rest(Side.Ask, 10_050, 100, order_id=2)
    t = ad.take(Side.Bid, 40)
    assert t.filled == 40
    assert t.notional_ticks == 40 * 10_050
    ad.rest(Side.Bid, 10_000, 500, order_id=1)
    assert ad.cancel_id(1, 100) == 100
    assert int(ad.view()["bid_sz"][0]) == 400
    assert ad.cancel_id(1) == 400
    assert int(ad.view()["bid_px"][0]) == 0


def test_env_inventory_and_obs_on_engine():
    Q, T = 240, 6
    env = OrderBookEnv(inventory=Q, horizon=T, seed=9, child_max=50, book=ne.Engine())
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    while True:
        obs, _r, term, trunc, info = env.step(-1.0)
        if term or trunc:
            break
    filled = sum(sz for _, _, sz in env.fills)
    assert filled + env.inventory == Q
    assert obs.shape == (44,)
    assert np.isfinite(info["shortfall_bps"])
    assert env.execution_vwap() > 0


def test_twap_on_engine():
    env = OrderBookEnv(inventory=120, horizon=4, seed=5, child_max=40, book=ne.Engine())
    res = run_episode(env, "twap", seed=5)
    assert res.filled + res.leftover == 120
    assert res.steps > 0
