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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ..envs.order_book_env import OrderBookEnv
from ..execution.metrics import completion_rate, max_drawdown, vwap_slippage
from ..research import bootstrap_ci

BaselineId = Literal["twap", "vwap", "pov", "passive", "apov", "stwap", "isaware"]


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
    agent: Policy | None = None,
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


# ---------------------------------------------------------------------------
# Phase 3 — fair per-regime evaluation (plan_2.md §6 / plan.md WS-3)
# ---------------------------------------------------------------------------
_Regime = OrderBookEnv | Callable[[], OrderBookEnv]
_AgentAct = Callable[[OrderBookEnv, np.ndarray], float]


def _fresh_env(reg: _Regime, seed: int) -> OrderBookEnv:
    """Instance → same env (reset-per-episode re-seeds it); callable → factory."""
    return reg if isinstance(reg, OrderBookEnv) else reg()


def _regime_episode(env: OrderBookEnv, act: _AgentAct, seed: int) -> dict:
    """One full episode under ``act`` (NOT ``baselines.run_episode`` — that
    exposes neither ``fills`` nor ``info["market_vwap"]``, and the agent needs
    obs anyway). Marks to market each step so MDD is over the filled path."""
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
    total = 0.0
    mtm: list[float] = []
    while True:
        a = act(env, obs)
        obs, r, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=np.float64)
        total += float(r)
        mtm.append(float(info.get("pnl_ticks", 0.0)))
        if term or trunc:
            return {
                "reward": total,
                "shortfall_bps": float(info["shortfall_bps"]),
                "vwap_slip_bps": float(vwap_slippage(env.fills, info.get("market_vwap", 0.0))),
                "completion": float(completion_rate(env.fills, env.inventory0, env.inventory)),
                "mdd_ticks": float(max_drawdown(mtm)),
                "leftover": int(env.inventory),
            }


def _summarize_rows(rows: list[dict], *, seeds: int, n_boot: int, seed: int) -> dict:
    """Mean + iid-bootstrap-95%-CI across *independent* seeds, and the
    secondary metrics averaged over the same episodes."""
    values = np.asarray([r["shortfall_bps"] for r in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "ci95": bootstrap_ci(values, n_boot=n_boot, kind="iid", seed=seed),
        "n": seeds,
        "vwap_slip_bps": float(np.mean([r["vwap_slip_bps"] for r in rows])),
        "completion": float(np.mean([r["completion"] for r in rows])),
        "mdd_ticks": float(np.mean([r["mdd_ticks"] for r in rows])),
        "leftover": float(np.mean([r["leftover"] for r in rows])),
    }


def evaluate_regime_ci(
    policy: Policy | None = None,
    regimes: dict[str, _Regime] | None = None,
    *,
    seeds: int = 5,
    baselines: tuple[str, ...] = (
        "twap", "vwap", "pov", "passive", "apov", "stwap", "isaware",
    ),
    fair: bool = True,
    seed0: int = 0x51ED,
    n_boot: int = 2000,
) -> dict[str, dict]:
    """Per-regime comparison of the agent vs baselines with honest CIs.

    Every strategy runs the *same* seeded episodes in each regime (episode
    ``s`` = ``seed0 + s``), so differences are the policy, not the tape. CIs
    are iid bootstraps over the independent seeds — wide by design at
    ``seeds = 5``; the docstring contract: headline use needs ``seeds >= 20``.

    ``regimes`` maps a name to an env **instance** (re-used, ``reset`` re-seeds
    it deterministically) or a zero-arg **factory** (freshened per episode).
    ``fair=True`` (default) refuses any regime env with ``vol_feature=True`` —
    the "remove the feature" branch of plan_2.md §6 item 3; ``fair=False`` is
    the escape hatch for a flag-trained agent and is never the headline.

    Reports BOTH the arrival-IS ``shortfall_bps`` (continuity with the Part-1
    metric) and ``vwap_slip_bps`` vs **market** VWAP (plan_2.md §6 item 5).

    Returns per regime:
    ``{"mean", "ci95", "n", "metric": "shortfall_bps", "strategies": {name:
    {"mean","ci95","n","vwap_slip_bps","completion","mdd_ticks","leftover"}},
    "rows": [{"strategy", "seed", ...metrics...}]}`` where ``mean``/``ci95``
    are the agent's.
    """
    from ..baselines import policy_action  # lazy (matches _baseline_summary)

    if not regimes:
        raise ValueError("evaluate_regime_ci needs at least one regime")
    if seeds < 5:
        raise ValueError("evaluate_regime_ci needs seeds >= 5 for defensible CIs")
    if fair:
        for rname, reg in regimes.items():
            probe = reg if isinstance(reg, OrderBookEnv) else reg()
            if probe.vol_feature:
                raise ValueError(
                    f"regime {rname!r} has vol_feature=True; build regimes with "
                    "vol_feature=False so baselines see the same state (fair mode)"
                )

    def agent_act(env: OrderBookEnv, obs: np.ndarray) -> float:
        return policy.act(obs, deterministic=True)  # type: ignore[misc]

    def baseline_act(name: str) -> _AgentAct:
        def act(env: OrderBookEnv, _obs: np.ndarray) -> float:
            return policy_action(name, env)
        return act

    out: dict[str, dict] = {}
    for rname, reg in regimes.items():
        strategies: dict[str, dict] = {}
        rows: list[dict] = []
        names = (["agent"] if policy is not None else []) + [str(n) for n in baselines]
        for s in range(seeds):
            seed = seed0 + s
            for name in names:
                env = _fresh_env(reg, seed)
                act: _AgentAct = agent_act if name == "agent" else baseline_act(name)
                row = _regime_episode(env, act, seed)
                rows.append({"strategy": name, "seed": seed, **row})
        for name in names:
            strategies[name] = _summarize_rows(
                [r for r in rows if r["strategy"] == name],
                seeds=seeds, n_boot=n_boot, seed=seed0,
            )
        ag = strategies.get("agent")
        out[rname] = {
            "metric": "shortfall_bps",
            "mean": ag["mean"] if ag else None,
            "ci95": ag["ci95"] if ag else None,
            "n": seeds,
            "strategies": strategies,
            "rows": rows,
        }
    return out