"""Group Relative Policy Optimization for OrderBookEnv.

GRPO (Shao et al.) drops the critic. ``G`` complete episodes share a tape
seed; advantages are group-normalized episode returns broadcast onto every
timestep, then the PPO clipped-surrogate actor update runs.

``act(obs, deterministic=True)`` is the same ``PPOPolicy`` interface the
eval harness already uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..envs.order_book_env import OrderBookEnv
from .evaluate import evaluate_policy
from .mlp import Adam, clip_grad_norm
from .ppo import PPOConfig, PPOPolicy, TrainHistory, _LOG2PI, make_optimizers


@dataclass
class GRPOConfig(PPOConfig):
    group: int = 4


def train_grpo(
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
    cfg: GRPOConfig | None = None,
    *,
    tracker: Optional[Callable[[TrainHistory], None]] = None,
) -> tuple[PPOPolicy, list[TrainHistory]]:
    cfg = cfg or GRPOConfig()
    if cfg.episodes % max(1, cfg.group) != 0:
        raise ValueError("GRPOConfig.episodes must be a multiple of group")
    env = env_factory()
    obs0, _ = env.reset(seed=cfg.unit_seed)
    obs_dim = int(np.asarray(obs0).reshape(-1).shape[0])
    policy = PPOPolicy(obs_dim, cfg.hidden, seed=cfg.seed, std0=cfg.std0)
    actor_opt, _critic_opt, std_opt = make_optimizers(policy, cfg)
    history: list[TrainHistory] = []
    n_groups = cfg.episodes // cfg.group

    for it in range(cfg.iterations):
        obs_l: list[np.ndarray] = []
        act_l: list[float] = []
        lp_l: list[float] = []
        adv_l: list[float] = []
        group_ret: list[float] = []

        for g in range(n_groups):
            tape = int(cfg.unit_seed) + it * 1009 + g
            returns: list[float] = []
            trajs: list[tuple[list[np.ndarray], list[float], list[float]]] = []
            for _k in range(cfg.group):
                obs, _ = env.reset(seed=tape)
                obs = np.asarray(obs, dtype=np.float64)
                o, a, lp = [], [], []
                total = 0.0
                while True:
                    action, logp = policy.sample(obs)
                    nxt, r, term, trunc, _ = env.step(action)
                    o.append(obs.copy())
                    a.append(float(action))
                    lp.append(float(logp))
                    total += float(r)
                    obs = np.asarray(nxt, dtype=np.float64)
                    if term or trunc:
                        break
                returns.append(total)
                trajs.append((o, a, lp))
            arr = np.asarray(returns, dtype=np.float64)
            mu_g = float(arr.mean())
            sd_g = float(arr.std() + 1e-6)
            group_ret.append(mu_g)
            for k, (o, a, lp) in enumerate(trajs):
                adv = (returns[k] - mu_g) / sd_g
                obs_l.extend(o)
                act_l.extend(a)
                lp_l.extend(lp)
                adv_l.extend([adv] * len(o))

        obs_b = np.stack(obs_l)
        act_b = np.asarray(act_l, dtype=np.float64)
        old_lp = np.asarray(lp_l, dtype=np.float64)
        adv_b = np.asarray(adv_l, dtype=np.float64)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        n = len(obs_b)
        pg_total = 0.0
        batches = 0
        for _ in range(cfg.epochs):
            perm = policy.rng.permutation(n)
            for start in range(0, n, cfg.minibatch):
                idx = perm[start : start + cfg.minibatch]
                ob, ac, ad, olp = obs_b[idx], act_b[idx], adv_b[idx], old_lp[idx]
                m = len(idx)
                mu = policy.actor.forward(ob).reshape(-1)
                sigma = float(np.exp(policy.log_std[0]))
                logp = -0.5 * (((ac - mu) / sigma) ** 2) - np.log(sigma) - 0.5 * _LOG2PI
                ratio = np.exp(logp - olp)
                ratio_cl = np.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
                surr = np.minimum(ratio * ad, ratio_cl * ad)
                pg_loss = -float(np.mean(surr))
                mask = (ratio > 1.0 - cfg.clip) & (ratio < 1.0 + cfg.clip)
                dlogp = -ad * ratio * mask
                dlogp_dmu = (ac - mu) / (sigma * sigma)
                d_out = dlogp * dlogp_dmu / m
                actor_grads = policy.actor.backward(d_out)
                dlogp_ds = ((ac - mu) / sigma) ** 2 - 1.0
                grad_s = float(np.mean(dlogp * dlogp_ds)) - cfg.ent_coef
                flat = [g for pair in actor_grads for g in pair]
                clip_grad_norm(flat + [np.asarray(grad_s)], cfg.grad_norm)
                actor_opt.step(flat)
                std_opt.step([np.asarray(grad_s)])
                pg_total += pg_loss
                batches += 1

        row = TrainHistory(
            iteration=it + 1,
            reward_mean=float(np.mean(group_ret)) if group_ret else 0.0,
            pg_loss=pg_total / max(1, batches),
            samples=n,
        )
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            summary = evaluate_policy(
                policy,
                n_episodes=cfg.eval_episodes,
                seed=cfg.unit_seed,
                env_factory=env_factory,
            )[1]
            row.reward_mean = summary.reward_mean
            row.shortfall_bps_mean = summary.shortfall_bps_mean
            history.append(row)
            if tracker is not None:
                tracker(row)
        elif it == cfg.iterations - 1:
            history.append(row)
    return policy, history
