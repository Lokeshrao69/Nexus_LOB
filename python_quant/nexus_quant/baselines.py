"""Execution baselines that share OrderBookEnv's action interface."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .envs.order_book_env import OrderBookEnv

AgentId = Literal["twap", "vwap", "pov", "passive"]


def policy_action(name: AgentId, env: OrderBookEnv) -> float:
    s = env.book.view()
    spr = 2
    if int(s["bid_px"][0]) and int(s["ask_px"][0]):
        spr = int(s["ask_px"][0]) - int(s["bid_px"][0])
    t_frac = env.t / env.horizon
    last_sz = int(s.get("last_trade_sz") or 0)
    if name == "twap":
        return -1.0 if t_frac > 0.72 else 0.12
    if name == "vwap":
        vol = min(1.0, last_sz / 120.0)
        return -1.0 if t_frac > 0.8 else 0.35 - vol * 0.9
    if name == "pov":
        if spr <= 1 and t_frac > 0.25:
            return -0.55
        return -0.95 if t_frac > 0.7 else 0.2
    if name == "passive":
        return -1.0 if t_frac > 0.92 else 0.7
    raise ValueError(name)


@dataclass
class EpisodeResult:
    name: str
    reward: float
    shortfall_bps: float
    vwap: float
    arrival: int
    leftover: int
    filled: int
    steps: int


def run_episode(
    env: OrderBookEnv,
    name: AgentId,
    *,
    seed: int | None = None,
    policy: Callable[[OrderBookEnv], float] | None = None,
) -> EpisodeResult:
    env.reset(seed=seed)
    total = 0.0
    act = policy or (lambda e: policy_action(name, e))
    while True:
        a = act(env)
        _obs, r, term, trunc, info = env.step(a)
        total += float(r)
        if term or trunc:
            filled = env.inventory0 - env.inventory
            return EpisodeResult(
                name=name,
                reward=total,
                shortfall_bps=float(info["shortfall_bps"]),
                vwap=float(info["vwap"]),
                arrival=int(info["arrival_mid"]),
                leftover=int(env.inventory),
                filled=int(filled),
                steps=int(env.t),
            )


def compare(seed: int = 0x51ED, **env_kw) -> list[EpisodeResult]:
    rows = []
    for name in ("twap", "vwap", "pov", "passive"):
        rows.append(run_episode(OrderBookEnv(seed=seed, **env_kw), name, seed=seed))
    return rows
