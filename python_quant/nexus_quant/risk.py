"""Person B risk seam — Monte-Carlo VaR/CVaR + env inventory penalty.

Mirrors ``cuda_risk/risk_common.hpp`` / ``risk_cpu.hpp`` so results match
``nexus_engine.compute_var_cvar`` bit-for-bit when the module is built.
When it is not, this NumPy oracle is the runtime used by ``OrderBookEnv``.

Losses are *fractions of initial notional* (1 − S_T/S_0). The env multiplies
CVaR by remaining inventory fraction to get a dynamic holding penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    lane_term = (int(lane) * int(_M2)) & 0xFFFFFFFFFFFFFFFF
    x = np.uint64(seed) ^ (paths * _M1) ^ np.uint64(lane_term)
    return (splitmix64(x) >> np.uint64(11)) * (1.0 / _TWO53)


def box_muller(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    a = np.maximum(u1, 1e-300)
    return np.sqrt(-2.0 * np.log(a)) * np.cos(2.0 * np.pi * u2)


def terminal_losses(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    n: int,
    lam: float,
    jmu: float,
    jsig: float,
    seed: int,
) -> np.ndarray:
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
            mask = rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(1_000_000))) < lam_dt
            jump = jmu + jsig * box_muller(
                rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(2_000_000))),
                rng_uniform(seed, paths, int(tl * np.uint64(2) + np.uint64(3_000_000))),
            )
            inc = np.where(mask, inc + jump, inc)
        logS += inc
    st = np.exp(logS)
    return 1.0 - st / s0


def var_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float, float]:
    s = np.sort(losses)
    n = len(s)
    k = int(np.ceil((1.0 - alpha) * n))
    k = max(1, min(k, n))
    idx = n - k
    return float(s[idx]), float(s[idx:].mean()), float(losses.mean())


@dataclass(frozen=True)
class RiskResult:
    var: float
    cvar: float
    mean_loss: float
    n: int
    source: str  # "engine" | "numpy"


def compute_var_cvar(
    *,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.25,
    T: float = 1.0,
    steps: int = 32,
    n_paths: int = 512,
    alpha: float = 0.95,
    seed: int = 0x51ED,
    lambda_jump: float = 0.0,
    jump_mu: float = 0.0,
    jump_sigma: float = 0.0,
    prefer_engine: bool = True,
) -> RiskResult:
    """Engine CPU reference when built; otherwise the matching NumPy oracle."""
    params = dict(
        s0=s0, mu=mu, sigma=sigma, T=T, steps=steps, n_paths=n_paths,
        alpha=alpha, seed=seed, lambda_jump=lambda_jump,
        jump_mu=jump_mu, jump_sigma=jump_sigma,
    )
    if prefer_engine:
        try:
            import nexus_engine as ne

            r = ne.compute_var_cvar(**params)
            return RiskResult(float(r["var"]), float(r["cvar"]), float(r["mean_loss"]),
                              int(r["n"]), "engine")
        except Exception:
            pass
    losses = terminal_losses(
        s0, mu, sigma, T, steps, n_paths, lambda_jump, jump_mu, jump_sigma, seed
    )
    v, c, m = var_cvar(losses, alpha)
    return RiskResult(v, c, m, n_paths, "numpy")


def inventory_risk_penalty(
    inv_frac: float,
    *,
    sigma: float = 0.25,
    seed: int = 0x51ED,
    n_paths: int = 256,
    steps: int = 16,
    lambda_risk: float = 1.0,
    horizon_frac: float = 1.0,
) -> tuple[float, RiskResult]:
    """``λ_risk * CVaR * inv_frac``. Shorter remaining horizon → smaller T."""
    if lambda_risk <= 0.0 or inv_frac <= 0.0:
        z = RiskResult(0.0, 0.0, 0.0, 0, "off")
        return 0.0, z
    T = max(1.0 / 252.0, float(horizon_frac))
    res = compute_var_cvar(
        sigma=sigma, T=T, steps=max(4, int(steps)), n_paths=int(n_paths), seed=int(seed),
        prefer_engine=False,  # hot path: keep the env free of a pybind import
    )
    return float(lambda_risk) * res.cvar * float(inv_frac), res
