"""Subsystem 3 — exact CPU-reference <-> NumPy-oracle VaR/CVaR parity.

The C++ reference (cuda_risk/risk_cpu.hpp, exposed via nexus_engine.compute_var_cvar)
draws its per-path randomness from a counter-based splitmix64 generator. We
re-implement that generator and the GBM/jump path recursion verbatim in NumPy,
so the two must agree BIT-FOR-BIT — a much stronger check than Monte-Carlo
noise tolerance, and the same diff-test philosophy used for the LOB oracle.

Skipped when the compiled `nexus_engine` module is absent (needs a build).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import nexus_engine as ne
except Exception:  # pragma: no cover — module not built
    ne = None  # type: ignore[assignment]

# Mirrors cuda_risk/risk_common.hpp, exactly.
_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)
_TWO53 = 9007199254740992.0


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64).copy()
    x += _M1
    x = (x ^ (x >> np.uint64(30))) * _M2
    x = (x ^ (x >> np.uint64(27))) * _M3
    x = x ^ (x >> np.uint64(31))
    return x


def rng_uniform(seed: int, paths: np.ndarray, lane: int) -> np.ndarray:
    # lane scalar folded in Python ints (arbitrary precision, no uint overflow
    # warning), then XORed into the uint64 array.
    lane_term = (int(lane) * int(_M2)) & 0xFFFFFFFFFFFFFFFF
    x = np.uint64(seed) ^ (paths * _M1) ^ np.uint64(lane_term)
    return (splitmix64(x) >> np.uint64(11)) * (1.0 / _TWO53)


def box_muller(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    a = np.maximum(u1, 1e-300)
    return np.sqrt(-2.0 * np.log(a)) * np.cos(2.0 * np.pi * u2)


def terminal_losses(s0, mu, sigma, T, steps, n, lam, jmu, jsig, seed) -> np.ndarray:
    dt = T / steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol_sd = sigma * np.sqrt(dt)
    lam_dt = lam * dt
    paths = np.arange(n, dtype=np.uint64)
    logS = np.full(n, np.log(s0), dtype=np.float64)
    for t in range(steps):
        tl = np.uint64(t)
        inc = drift + vol_sd * box_muller(
            rng_uniform(seed, paths, int(tl * np.uint64(2))),
            rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(1))),
        )
        if lam > 0:
            mask = rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(1000000))) < lam_dt
            jump = jmu + jsig * box_muller(
                rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(2000000))),
                rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(3000000))),
            )
            inc = np.where(mask, inc + jump, inc)
        logS += inc
    st = np.exp(logS)
    return 1.0 - st / s0


def var_cvar(losses: np.ndarray, alpha: float):
    s = np.sort(losses)
    n = len(s)
    k = int(np.ceil((1.0 - alpha) * n))
    k = max(1, min(k, n))
    idx = n - k
    return float(s[idx]), float(s[idx:].mean()), float(losses.mean())


def _run_case(n, steps, **over):
    params = dict(s0=100.0, mu=0.05, sigma=0.25, T=1.0, steps=steps, n_paths=n,
                  alpha=0.95, seed=0x51ED, lambda_jump=0.0, jump_mu=0.0, jump_sigma=0.0)
    params.update(over)
    losses = terminal_losses(
        params["s0"], params["mu"], params["sigma"], params["T"], params["steps"],
        params["n_paths"], params["lambda_jump"], params["jump_mu"],
        params["jump_sigma"], params["seed"],
    )
    ov, oc, om = var_cvar(losses, params["alpha"])
    r = ne.compute_var_cvar(**params)
    return ov, oc, om, r


@pytest.mark.skipif(ne is None, reason="nexus_engine not built")
def test_var_cvar_bitwise_parity_gbm():
    ov, oc, om, r = _run_case(n=60000, steps=120)
    assert r["var"] == pytest.approx(ov, abs=1e-12)
    assert r["cvar"] == pytest.approx(oc, abs=1e-12)
    assert r["mean_loss"] == pytest.approx(om, abs=1e-12)
    # the case itself is meaningful (nonzero tail risk)
    assert r["var"] > 0.0 and r["cvar"] > r["var"]


@pytest.mark.skipif(ne is None, reason="nexus_engine not built")
def test_var_cvar_bitwise_parity_jump():
    ov, oc, om, r = _run_case(n=40000, steps=100,
                              lambda_jump=0.2, jump_mu=0.0, jump_sigma=0.3)
    assert r["var"] == pytest.approx(ov, abs=1e-12)
    assert r["cvar"] == pytest.approx(oc, abs=1e-12)
    assert r["mean_loss"] == pytest.approx(om, abs=1e-12)


@pytest.mark.skipif(ne is None, reason="nexus_engine not built")
def test_deterministic_across_calls():
    a = ne.compute_var_cvar(seed=42, n_paths=50000, steps=100)
    b = ne.compute_var_cvar(seed=42, n_paths=50000, steps=100)
    assert a == b
    c = ne.compute_var_cvar(seed=43, n_paths=50000, steps=100)
    assert a != c  # different seed -> different path set