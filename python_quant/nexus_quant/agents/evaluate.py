"""Evaluation harness: trained PPO agent vs. the execution baselines.

The headline resume metric is **lower implementation-shortfall slippage than
VWAP**. ``shortfall_bps`` (from ``OrderBookEnv`` info) is ``(arrival_mid −
vwap) / arrival_mid × 1e4`` — how much the child execution conceded relative
to the arrival mid, in basis points. Lower is better.

To make the comparison fair every strategy runs the **same seeded episodes**:
episode *i* = ``seed + i``, which reproduces identical initial books and
exogenous flow for TWAP/VWAP/POV/Passive and the agent alike, so the only
difference in results is the policy, not the tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

import numpy as np

from ..envs.order_book_env import OrderBookEnv

BaselineId = Literal["twap", "vwap", "pov", "passive"]


class Policy(Protocol):
    """Anything with ``act(obs, deterministic=True) -> float`` (PPOPolicy)."""

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> float: ...


@dataclass
class EvalSummary:
    name: str
    reward_mean: float
    shortfall_bps_mean: float
    shortfall_bps_std: float
    leftover_mean: float
    n: int


def _episode_shortfall(
    env: OrderBookEnv, act_fn: Callable[[np.ndarray], float], seed: int
) -> tuple[float, float, int]:
    """One full episode: reset(seed), then follow ``act_fn`` to the end."""
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
    total = 0.0
    while True:
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=np.float64)
        total += float(r)
        if term or trunc:
            return total, float(info["shortfall_bps"]), int(env.inventory)


def evaluate_policy(
    policy: Policy,
    *,
    n_episodes: int = 50,
    seed: int = 0,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
    deterministic: bool = True,
) -> tuple[list[dict], EvalSummary]:
    """Run ``policy`` over ``n_episodes`` seeded episodes; return rows + summary.

    Rows are dicts for easy tabulation; the summary carries the mean
    shortfall (the slippage metric) and its scatter.
    """
    env = env_factory()
    rows: list[dict] = []
    rewards: list[float] = []
    sfs: list[float] = []
    leftovers: list[int] = []
    for i in range(n_episodes):
        total, sf, leftover = _episode_shortfall(
            env, lambda ob: policy.act(ob, deterministic=deterministic), int(seed) + i
        )
        rewards.append(total)
        sfs.append(sf)
        leftovers.append(leftover)
        rows.append({"name": policy.__class__.__name__, "reward": total, "shortfall_bps": sf, "leftover": leftover})
    summary = EvalSummary(
        name=policy.__class__.__name__,
        reward_mean=float(np.mean(rewards)),
        shortfall_bps_mean=float(np.mean(sfs)),
        shortfall_bps_std=float(np.std(sfs)),
        leftover_mean=float(np.mean(leftovers)),
        n=n_episodes,
    )
    return rows, summary


def _baseline_summary(
    name: BaselineId,
    n_episodes: int,
    seed: int,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> EvalSummary:
    from ..baselines import run_episode

    env = env_factory()
    rewards: list[float] = []
    sfs: list[float] = []
    leftovers: list[int] = []
    for i in range(n_episodes):
        res = run_episode(env, name, seed=int(seed) + i)
        rewards.append(res.reward)
        sfs.append(res.shortfall_bps)
        leftovers.append(res.leftover)
    return EvalSummary(
        name=name,
        reward_mean=float(np.mean(rewards)),
        shortfall_bps_mean=float(np.mean(sfs)),
        shortfall_bps_std=float(np.std(sfs)),
        leftover_mean=float(np.mean(leftovers)),
        n=n_episodes,
    )


def strategy_table(
    agent: Optional[Policy] = None,
    *,
    agent_name: str = "ppo",
    n_episodes: int = 50,
    seed: int = 0,
    baselines: tuple[BaselineId, ...] = ("twap", "vwap", "pov", "passive"),
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> list[dict]:
    """Compare agent + baselines on the same seeded episodes.

    Returns one dict per strategy:
    ``name, reward_mean, shortfall_bps_mean, shortfall_bps_std, vs_vwap_bps``
    where ``vs_vwap_bps`` is the signed *reduction* in shortfall relative to
    VWAP (positive = agent/baseline is *better* than VWAP).
    """
    rows: list[dict] = []
    vwap_sf: float | None = None
    if agent is not None:
        _, a_sum = evaluate_policy(
            agent, n_episodes=n_episodes, seed=seed,
            env_factory=env_factory, deterministic=True,
        )
        a_sum.name = agent_name
        rows.append(a_sum)
    for name in baselines:
        b = _baseline_summary(name, n_episodes, seed, env_factory=env_factory)
        if name == "vwap":
            vwap_sf = b.shortfall_bps_mean
        rows.append(b)
    out = []
    for r in rows:
        d = r.__dict__.copy()
        if vwap_sf is not None:
            d["vs_vwap_bps"] = vwap_sf - r.shortfall_bps_mean
            d["vs_vwap_pct"] = (vwap_sf - r.shortfall_bps_mean) / max(vwap_sf, 1e-9) * 100.0
        else:
            d["vs_vwap_bps"] = 0.0
            d["vs_vwap_pct"] = 0.0
        out.append(d)
    return out


def format_table(rows: list[dict]) -> str:
    """Render ``strategy_table`` output as a monospace summary line per row."""
    header = f"{'strategy':<10}{'reward':>10}{'shortfall_bps':>14}{'vs_vwap%':>10}{'leftover':>10}"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['name']:<10}{r['reward_mean']:>10.2f}{r['shortfall_bps_mean']:>14.3f}"
            f"{r['vs_vwap_pct']:>9.1f}%{r['leftover_mean']:>10.2f}"
        )
    return "\n".join(lines)