"""GRPO smoke tests — interface + finite update, not headline performance."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nexus_quant.agents.grpo import GRPOConfig, train_grpo
from nexus_quant.envs.order_book_env import OrderBookEnv


def test_grpo_one_iter_finite():
    def factory():
        return OrderBookEnv(inventory=80, horizon=3, seed=1, child_max=40)

    cfg = GRPOConfig(
        iterations=1,
        episodes=4,
        group=4,
        epochs=1,
        minibatch=32,
        eval_every=0,
        seed=11,
        unit_seed=21,
    )
    pol, hist = train_grpo(factory, cfg)
    a = pol.act(np.zeros(44, dtype=np.float64), deterministic=True)
    assert -1.0 <= a <= 1.0
    assert np.isfinite(a)


def test_grpo_requires_group_multiple():
    try:
        train_grpo(cfg=GRPOConfig(episodes=5, group=2, iterations=0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
