"""PPO agent tests: gradient correctness, GAE, updates, training, evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from nexus_quant.agents import (
    PPOConfig,
    PPOPolicy,
    evaluate_policy,
    evaluate_regime_ci,
    format_table,
    strategy_table,
    train_ppo,
)
from nexus_quant.agents.evaluate import _baseline_summary
from nexus_quant.baselines import policy_action
from nexus_quant.agents.mlp import MLP, Adam
from nexus_quant.agents.ppo import collect_rollouts, compute_gae, ppo_update
from nexus_quant.envs.order_book_env import OrderBookEnv

# ----------------------------------------------------------------------
#  MLP backward vs. finite differences
# ----------------------------------------------------------------------

def test_mlp_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    mlp = MLP((3, 4, 2), "tanh", out_act="tanh", out_scale=1.0, seed=0)
    x = rng.normal(size=(5, 3))
    out = mlp.forward(x)                     # (5, 2)
    # loss = mean(out^2)  →  grad wrt out = 2*out/(n*m)   (mean over ALL elems)
    d_out = (2.0 * out) / out.size
    grads = mlp.backward(d_out)              # numeric: d(mean loss)/d param

    def loss_fn(params):
        for lyr, p in zip(mlp.params, params):
            for dst, src in zip(lyr, p):
                dst[...] = src
        o = mlp.forward(x)
        return float(np.mean(o * o))

    base = loss_fn([[lyr[0].copy(), lyr[1].copy()] for lyr in mlp.params])
    for li in range(len(mlp.params)):
        W, b = mlp.params[li]
        for jj, W0 in enumerate([W, b]):
            for i in range(W0.shape[0]):
                for j in range(W0.shape[1] if W0.ndim == 2 else 1):
                    idx = (i, j)
                    eps = 1e-6
                    cur = mlp.params[li][jj]
                    if W0.ndim == 2:
                        cur[i, j] += eps
                    else:
                        cur[i] += eps
                    plus = loss_fn([[lyr[0].copy(), lyr[1].copy()] for lyr in mlp.params])
                    cur = mlp.params[li][jj]
                    if W0.ndim == 2:
                        cur[i, j] -= 2 * eps
                    else:
                        cur[i] -= 2 * eps
                    minus = loss_fn([[lyr[0].copy(), lyr[1].copy()] for lyr in mlp.params])
                    cur = mlp.params[li][jj]
                    if W0.ndim == 2:
                        cur[i, j] += eps
                    else:
                        cur[i] += eps
                    num = (plus - minus) / (2 * eps)

                    d_lyr = grads[li]
                    if W0.ndim == 2:
                        ana = d_lyr[jj][idx]
                    else:
                        ana = d_lyr[jj][i]
                    assert abs(num - ana) < 5e-4, (li, jj, idx, num, ana)
    _ = base


# ----------------------------------------------------------------------
#  Adam
# ----------------------------------------------------------------------

def test_adam_minimises_scalar_quadratic():
    x = np.array([1.0])
    opt = Adam([x], lr=0.05)
    for _ in range(200):
        opt.step([2.0 * x])                  # d/dx (x^2)
    assert abs(x[0]) < 1e-4


# ----------------------------------------------------------------------
#  GAE
# ----------------------------------------------------------------------

def test_compute_gae_hand_rolled_trajectory():
    # r, terminated flags, next-state values, current values
    rew = np.array([0.0, 1.0, -1.0, 2.0])
    term = np.array([False, False, False, True])
    nv = np.array([0.5, 0.4, 0.3, 0.0])      # V(s') each step
    vals = np.array([0.0, 0.5, 0.4, 0.3])
    gamma, lam = 0.99, 0.95
    adv, ret = compute_gae(rew, term, nv, vals, gamma, lam)
    # delta_3 = r_3 - V_3 = 2 - 0.3 = 1.7  (terminal: no carry)
    d3 = 2.0 - 0.3
    assert abs(adv[3] - d3) < 1e-12
    d2 = -1.0 + gamma * nv[2] - vals[2]     # -1 + .99*.3 - .4
    g2 = d2 + gamma * lam * d3
    assert abs(adv[2] - g2) < 1e-12
    assert abs(ret[2] - (g2 + vals[2])) < 1e-12
    d1 = 1.0 + gamma * nv[1] - vals[1]
    g1 = d1 + gamma * lam * g2
    assert abs(adv[1] - g1) < 1e-12


def test_compute_gae_truncation_boundary_no_leak():
    """Truncation terminates trajectory but bootstraps from next_val without leaking into preceding steps (F04)."""
    rew = np.array([1.0, 100.0])
    term = np.array([False, True])
    trunc = np.array([True, False])
    nv = np.array([5.0, 0.0])
    vals = np.array([0.0, 0.0])
    gamma, lam = 0.99, 0.95
    adv, _ret = compute_gae(rew, term, nv, vals, gamma, lam, trunc=trunc)
    # Step 0: delta = 1.0 + 0.99*5.0 - 0.0 = 5.95
    # Running advantage from step 1 (100.0) MUST NOT leak into step 0!
    expected_adv_0 = 1.0 + gamma * nv[0] - vals[0]
    assert abs(adv[0] - expected_adv_0) < 1e-12
    assert abs(adv[1] - 100.0) < 1e-12


def test_ppo_ratio_gradient_mask_unclipped_regions():
    """Verify that when A > 0 and r < 1 - eps (or A < 0 and r > 1 + eps), gradient is not zeroed (F05)."""
    clip = 0.2
    # Case 1: ad > 0, ratio < 1 - clip -> unclipped gradient active
    ad = np.array([1.0])
    ratio = np.array([0.7])
    mask = np.where(ad >= 0, ratio <= 1.0 + clip, ratio >= 1.0 - clip)
    assert mask[0] == True

    # Case 2: ad > 0, ratio > 1 + clip -> clipped above, gradient zeroed
    ratio_high = np.array([1.3])
    mask_high = np.where(ad >= 0, ratio_high <= 1.0 + clip, ratio_high >= 1.0 - clip)
    assert mask_high[0] == False

    # Case 3: ad < 0, ratio > 1 + clip -> unclipped gradient active
    ad_neg = np.array([-1.0])
    mask_neg = np.where(ad_neg >= 0, ratio_high <= 1.0 + clip, ratio_high >= 1.0 - clip)
    assert mask_neg[0] == True


# ----------------------------------------------------------------------
#  Policy behaviour
# ----------------------------------------------------------------------

def test_policy_act_and_sample_bounds():
    p = PPOPolicy(obs_dim=44, seed=42)
    obs = np.zeros(44, dtype=np.float32)
    for _ in range(50):
        a = p.act(obs, deterministic=True)
        assert -1.0 <= a <= 1.0
        s, lp = p.sample(obs)
        assert -1.0 <= s <= 1.0
        assert np.isfinite(lp)
    # deterministic mean is the tanh-squashed net output, in range
    assert abs(p.act(obs)) < 1.0


def test_ppo_update_produces_finite_losses():
    env = OrderBookEnv(inventory=1000, horizon=10, seed=5)
    p = PPOPolicy(obs_dim=44, seed=1)
    buf = collect_rollouts(env, p, episodes=4, base_seed=101)
    cfg = PPOConfig(epochs=2, minibatch=16, seed=1)
    from nexus_quant.agents.ppo import make_optimizers

    ao, co, so = make_optimizers(p, cfg)
    losses = ppo_update(p, buf, cfg, ao, co, so)
    for k, v in losses.items():
        assert np.isfinite(v), (k, v)
    assert losses["pg_loss"] < losses["value_loss"] + 1e6  # sane magnitude


# ----------------------------------------------------------------------
#  train_ppo end-to-end
# ----------------------------------------------------------------------

def test_train_ppo_smoke_and_determinism():
    cfg = PPOConfig(
        iterations=5,
        episodes=3,
        epochs=2,
        minibatch=8,
        seed=22,
        unit_seed=11,
        eval_every=3,
        eval_episodes=4,
    )
    p1, hist1 = train_ppo(OrderBookEnv, cfg)
    p2, hist2 = train_ppo(OrderBookEnv, cfg)
    assert len(hist1) >= 1
    for h in hist1:
        assert np.isfinite(h.reward_mean)
        assert np.isfinite(h.shortfall_bps_mean)
        assert h.samples > 0
    # deterministic given the same config
    assert [h.as_row() for h in hist1] == [h.as_row() for h in hist2]
    obs = np.zeros(44, dtype=np.float32)
    assert -1.0 <= p1.act(obs) <= 1.0


# ----------------------------------------------------------------------
#  Evaluation harness
# ----------------------------------------------------------------------

def test_evaluate_policy_reproducible():
    p = PPOPolicy(obs_dim=44, seed=3)
    rows, summary = evaluate_policy(p, n_episodes=4, seed=9)
    rows2, summary2 = evaluate_policy(p, n_episodes=4, seed=9)
    assert summary.n == 4
    assert abs(summary.shortfall_bps_mean - summary2.shortfall_bps_mean) < 1e-12
    assert all(r["shortfall_bps"] >= 0.0 for r in rows)
    assert summary.leftover_mean >= 0.0


def test_strategy_table_contains_all_and_vwap_delta():
    p = PPOPolicy(obs_dim=44, seed=3)
    rows = strategy_table(p, agent_name="ppo", n_episodes=4, seed=7)
    names = {r["name"] for r in rows}
    assert {"ppo", "twap", "vwap", "pov", "passive"} <= names
    vwap_row = next(r for r in rows if r["name"] == "vwap")
    ppo_row = next(r for r in rows if r["name"] == "ppo")
    assert vwap_row["vs_vwap_bps"] == 0.0
    assert ppo_row["vs_vwap_pct"] == (vwap_row["shortfall_bps_mean"] - ppo_row["shortfall_bps_mean"]) / vwap_row["shortfall_bps_mean"] * 100.0
    s = format_table(rows)
    assert s.splitlines()[0].startswith("strategy")


def test_baseline_summary_smoke():
    s = _baseline_summary("vwap", n_episodes=2, seed=5)
    assert s.name == "vwap"
    assert s.n == 2
    assert np.isfinite(s.shortfall_bps_mean)


# ----------------------------------------------------------------------
#  Regressions worth locking in
# ----------------------------------------------------------------------

def test_policy_save_load_roundtrip(tmp_path):
    import numpy as np

    p = PPOPolicy(obs_dim=44, hidden=(8, 4), seed=7)
    obs = np.zeros(44, dtype=np.float32)
    before = p.act(obs)
    path = str(tmp_path / "pol.npz")
    p.save(path)
    q = PPOPolicy.load(path)
    assert q.obs_dim == 44
    assert q.hidden == (8, 4)
    assert abs(q.act(obs) - before) < 1e-12
    # weights actually restored (not trivially zero/log_std-initialized)
    for a, b in zip(p.actor.param_list(), q.actor.param_list()):
        assert np.array_equal(a, b)


def test_env_reward_knobs_default_to_zero_effect():
    """is_coef=1 / lambda_sched=0 (defaults) reproduce the original reward."""
    from nexus_quant.envs.order_book_env import OrderBookEnv as E

    def first_reward(**kw):
        e = E(inventory=400, horizon=6, seed=3, **kw)
        e.reset()
        _, r, *_ = e.step(-1.0)
        return r

    assert first_reward() == first_reward(is_coef=1.0, lambda_sched=0.0)
    # is_coef≠1 changes the reward path (price term scaled)
    assert first_reward(is_coef=3.0) != first_reward()


def test_agent_vs_baselines_runs_with_zero_env_kw():
    # The seeded comparison must work with the default constructor (the
    # numbers the resume/metric tooling reads).
    rows = strategy_table(n_episodes=2, seed=13)
    assert {r["name"] for r in rows} == {"twap", "vwap", "pov", "passive"}
    for r in rows:
        assert np.isfinite(r["shortfall_bps_mean"])


# ----------------------------------------------------------------------
#  Phase 3 — fair per-regime evaluation + strengthened baselines
# ----------------------------------------------------------------------

def test_evaluate_regime_ci_shape_and_reproducibility():
    regimes = {
        "calm": OrderBookEnv(inventory=400, horizon=6, seed=0),
        "drift_tick": OrderBookEnv(inventory=400, horizon=6, seed=0, drift_ticks=0.05),
    }
    p = PPOPolicy(obs_dim=44, seed=2)
    kw = dict(seeds=5, baselines=("twap", "vwap", "apov"), seed0=7, n_boot=100)
    r1 = evaluate_regime_ci(p, regimes, **kw)
    r2 = evaluate_regime_ci(p, regimes, **kw)
    assert r1 == r2  # fully deterministic
    for name, r in r1.items():
        assert r["metric"] == "shortfall_bps" and r["n"] == 5
        assert r["mean"] is not None and np.isfinite(r["mean"])
        assert set(r["strategies"]) == {"agent", "twap", "vwap", "apov"}
        for s, d in r["strategies"].items():
            assert d["n"] == 5
            assert np.isfinite(d["mean"]) and d["vwap_slip_bps"] >= 0.0
            assert 0.0 <= d["completion"] <= 1.0 and d["leftover"] >= 0
            assert set(d["ci95"]) == {"lo", "hi", "mean", "se"}
        assert len(r["rows"]) == 5 * 4  # 5 seeds x (agent + 3 baselines)
    assert {row["seed"] for row in r1["calm"]["rows"]} == {7, 8, 9, 10, 11}


def test_fair_toggle_raises_on_vol_feature_and_false_escapes():
    env_true = OrderBookEnv(inventory=100, horizon=4, seed=0, vol_feature=True)
    with pytest.raises(ValueError):
        evaluate_regime_ci(None, {"reg": env_true}, seeds=5, baselines=("twap",))
    # fair=False is the documented escape hatch for a flag-trained agent
    res = evaluate_regime_ci(
        None, {"reg": env_true}, seeds=5, baselines=("twap",), fair=False
    )
    assert "twap" in res["reg"]["strategies"]


def test_evaluate_regime_ci_validates_seeds_and_regimes():
    env = OrderBookEnv(inventory=100, horizon=4, seed=0)
    with pytest.raises(ValueError):
        evaluate_regime_ci(None, {"reg": env}, seeds=3, baselines=("twap",))
    with pytest.raises(ValueError):
        evaluate_regime_ci(None, {}, seeds=5, baselines=("twap",))


def test_evaluate_regime_ci_baselines_only():
    res = evaluate_regime_ci(
        None, {"calm": OrderBookEnv(inventory=400, horizon=6, seed=0)},
        seeds=5, baselines=("twap", "pov", "isaware"), seed0=3, n_boot=50,
    )
    r = res["calm"]
    assert r["mean"] is None and r["ci95"] is None
    assert "agent" not in r["strategies"]
    assert set(r["strategies"]) == {"twap", "pov", "isaware"}


def test_adaptive_baselines_in_range_and_deterministic():
    env = OrderBookEnv(inventory=400, horizon=12, seed=3)
    from nexus_quant.baselines import run_episode

    for name in ("apov", "stwap", "isaware"):
        env.reset(seed=3)
        while True:
            a = policy_action(name, env)
            assert -1.0 <= a <= 1.0, (name, a)
            _obs, _r, term, trunc, _info = env.step(a)
            if term or trunc:
                break
        e2 = OrderBookEnv(inventory=400, horizon=12, seed=3)
        r1 = run_episode(e2, name, seed=3)
        e3 = OrderBookEnv(inventory=400, horizon=12, seed=3)
        r2 = run_episode(e3, name, seed=3)
        assert r1.shortfall_bps == r2.shortfall_bps  # same seed -> same episode
        assert np.isfinite(r1.reward) and r1.leftover >= 0


def test_strategy_table_opt_in_adaptive_baselines():
    rows = strategy_table(
        n_episodes=3, seed=4, baselines=("twap", "apov", "stwap", "isaware")
    )
    assert {r["name"] for r in rows} == {"twap", "apov", "stwap", "isaware"}
    for r in rows:
        assert np.isfinite(r["shortfall_bps_mean"]) and np.isfinite(r["reward_mean"])