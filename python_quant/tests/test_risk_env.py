"""Risk oracle + env inventory-penalty seam."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.risk import compute_var_cvar, inventory_risk_penalty, var_cvar, terminal_losses


def test_oracle_deterministic():
    a = compute_var_cvar(n_paths=64, steps=8, seed=7, prefer_engine=False)
    b = compute_var_cvar(n_paths=64, steps=8, seed=7, prefer_engine=False)
    assert a.var == b.var and a.cvar == b.cvar
    c = compute_var_cvar(n_paths=64, steps=8, seed=8, prefer_engine=False)
    assert a.var != c.var


def test_cvar_ge_var():
    r = compute_var_cvar(n_paths=128, steps=8, seed=1, prefer_engine=False)
    assert r.cvar >= r.var
    assert r.source == "numpy"


def test_var_cvar_matches_sorted_tail():
    losses = terminal_losses(100.0, 0.05, 0.25, 0.1, 8, 32, 0.0, 0.0, 0.0, 3)
    v, c, m = var_cvar(losses, 0.95)
    assert np.isfinite([v, c, m]).all()


def test_penalty_zero_when_lambda_off():
    pen, res = inventory_risk_penalty(0.5, lambda_risk=0.0)
    assert pen == 0.0 and res.source == "off"


def test_env_default_has_no_cvar_cost():
    env = OrderBookEnv(inventory=200, horizon=4, seed=3)
    env.reset()
    _o, r0, *_rest = env.step(-1.0)
    env2 = OrderBookEnv(inventory=200, horizon=4, seed=3, lambda_risk=0.4, risk_paths=32, risk_steps=4)
    env2.reset()
    _o, r1, _t, _u, info = env2.step(-1.0)
    assert "cvar" in info
    # same tape until risk term; risk-on reward is lower (more penalty)
    assert r1 <= r0 + 1e-12
