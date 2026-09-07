"""PPO execution agent (pure NumPy) for the ``OrderBookEnv``.

The classic actor–critic recipe, entirely self-contained:

* **Policy**: Gaussian actor — mean μ(s) is a squashed (tanh) MLP over the
  44-dim observation; exploration std σ is a single trainable scalar.
* **Critic**: scalar MLP V(s).
* **Critic targets**: GAE(λ) bootstrapped with V for truncated steps
  (terminal steps bootstrap with 0 — the standard renormalized PPO trick).
* **Objective**: clipped surrogate (ε) + entropy bonus; Adam for both nets.

``train_ppo`` is the end-to-end entry point: it rolls out full episodes
against an ``OrderBookEnv``, updates the policy for ``epochs`` minibatches,
and optionally records a mid-training evaluation. Everything is seeded and
deterministic, so a fixed ``PPOConfig`` reproduces the same policy and the
same trailing history on any machine.

The network is intentionally API-shaped like a torch module
(``forward`` / ``backward``) so the same env and evaluation harness can be
driven by a torch/GRPO variant later without changing the experiment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..envs.order_book_env import OBS_DIM, OrderBookEnv
from .mlp import MLP, Adam, clip_grad_norm

_LOG2PI = np.log(2.0 * np.pi)


@dataclass
class PPOConfig:
    """All training knobs. ``unit_seed + i`` seeds episode *i* of every
    iteration, so the training flow is reproducible run-to-run."""

    iterations: int = 1500
    episodes: int = 16              # full episodes rolled out per iteration
    epochs: int = 4                 # inner minibatch passes over the buffer
    minibatch: int = 128
    gamma: float = 0.99
    lam: float = 0.95               # GAE trace-decay
    clip: float = 0.2               # surrogate clip ε
    lr_actor: float = 4e-4
    lr_critic: float = 1e-3
    lr_std: float = 1e-3
    ent_coef: float = 0.01
    std0: float = 0.4               # exploration std at start (ln σ init)
    hidden: tuple[int, int] = (64, 32)
    seed: int = 0xACE
    unit_seed: int = 0x2717         # base for per-episode seeds
    grad_norm: float = 5.0
    eval_every: int = 0             # 0 = skip; else eval this often (iters)
    eval_episodes: int = 20


class PPOPolicy:
    """Gaussian actor + value critic with stochastic and greedy actions.

    ``act(obs, deterministic=...)`` is the interface the evaluation harness
    and baselines share; ``sample`` adds exploration noise during training.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden: tuple[int, ...] = (64, 32),
        *,
        seed: int = 0,
        std0: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        #  seeded separately so actor and critic don't share init weights
        self.actor = MLP([obs_dim, *self.hidden, 1], "tanh", out_act="tanh", seed=seed)
        self.critic = MLP([obs_dim, *self.hidden, 1], "tanh", out_act="linear", seed=seed + 1)
        self.log_std = np.full(1, np.log(float(std0)), dtype=np.float64)

    # ------------------------------------------------------------- #
    def dist_params(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        mu = self.actor.forward(obs).reshape(-1)
        sigma = float(np.exp(self.log_std[0]))
        return mu, sigma

    def sample(self, obs: np.ndarray) -> tuple[float, float]:
        mu, sigma = self.dist_params(obs)
        a = np.clip(mu + sigma * self.rng.standard_normal(mu.shape), -1.0, 1.0)
        a_scalar = float(np.asarray(a).reshape(-1)[0])
        lp = self.logprob(obs, np.asarray(a_scalar, dtype=np.float64))
        return a_scalar, float(np.asarray(lp).reshape(-1)[0])

    def logprob(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        mu, sigma = self.dist_params(obs)
        d = (action - mu) / sigma
        return -0.5 * d * d - np.log(sigma) - 0.5 * _LOG2PI

    def entropy(self) -> float:
        """Per-sample Gaussian entropy (σ is a scalar, so batch-independent)."""
        return float(self.log_std[0] + 0.5 * (_LOG2PI + 1.0))

    def value(self, obs: np.ndarray) -> np.ndarray:
        return self.critic.forward(obs).reshape(-1)

    def act(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> float:
        mu, sigma = self.dist_params(obs)
        if deterministic:
            a = mu
        else:
            a = mu + sigma * self.rng.standard_normal(mu.shape)
        return float(np.clip(np.asarray(a).reshape(-1)[0], -1.0, 1.0))

    # ------------------------------------------------------------- #
    #  persistence: raw NumPy arrays, loadable with no torch/npz deps
    # ------------------------------------------------------------- #
    def state_dict(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {
            "_obs_dim": np.asarray([self.obs_dim]),
            "_hidden": np.asarray(self.hidden),
        }
        for i, (W, b) in enumerate(self.actor.params):
            d[f"actor.W{i}"] = W
            d[f"actor.b{i}"] = b
        for i, (W, b) in enumerate(self.critic.params):
            d[f"critic.W{i}"] = W
            d[f"critic.b{i}"] = b
        d["log_std"] = self.log_std
        return d

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.state_dict())

    @classmethod
    def from_dict(cls, state: dict[str, np.ndarray]) -> "PPOPolicy":
        obs_dim = int(state["_obs_dim"][0])
        hidden = tuple(int(x) for x in state["_hidden"])
        p = cls(obs_dim=obs_dim, hidden=hidden)
        for i in range(len(p.actor.params)):
            p.actor.params[i][0][...] = state[f"actor.W{i}"]
            p.actor.params[i][1][...] = state[f"actor.b{i}"]
            p.critic.params[i][0][...] = state[f"critic.W{i}"]
            p.critic.params[i][1][...] = state[f"critic.b{i}"]
        p.log_std[...] = state["log_std"]
        return p

    @classmethod
    def load(cls, path: str) -> "PPOPolicy":
        return cls.from_dict(np.load(path))


# ----------------------------------------------------------------------
#  Rollout → GAE → clipped-surrogate update
# ----------------------------------------------------------------------

def collect_rollouts(
    env: OrderBookEnv,
    policy: PPOPolicy,
    episodes: int,
    base_seed: int,
) -> dict[str, np.ndarray]:
    """Full-episode buffer of one policy epoch.

    ``env`` is reused across episodes; ``reset(seed=base_seed + i)`` reseeds
    it deterministically so episode *i* sees identical exogenous flow for
    every strategy under test. Each transition stores ``next_val`` — V of the
    post-step state — used to bootstrap GAE at truncated boundaries.
    """
    obs_l, act_l, lp_l, rew_l, term_l, trunc_l, nv_l = ([] for _ in range(7))
    for i in range(episodes):
        obs, _ = env.reset(seed=int(base_seed) + i)
        obs = np.asarray(obs, dtype=np.float64)
        while True:
            a, lp = policy.sample(obs)
            n_obs, r, term, trunc, _info = env.step(a)
            n_obs = np.asarray(n_obs, dtype=np.float64)
            next_val = policy.value(n_obs)  # V(s'); bootstraps truncated ends
            obs_l.append(obs)
            act_l.append(a)
            lp_l.append(lp)
            rew_l.append(float(r))
            term_l.append(term)
            trunc_l.append(trunc)
            nv_l.append(float(np.asarray(next_val).reshape(-1)[0]))
            obs = n_obs
            if term or trunc:
                break
    return {
        "obs": np.stack(obs_l),
        "act": np.asarray(act_l, dtype=np.float64),
        "logp": np.asarray(lp_l, dtype=np.float64),
        "rew": np.asarray(rew_l, dtype=np.float64),
        "term": np.asarray(term_l, dtype=bool),
        "trunc": np.asarray(trunc_l, dtype=bool),
        "next_val": np.asarray(nv_l, dtype=np.float64),
    }


def compute_gae(
    rew: np.ndarray,
    term: np.ndarray,
    next_val: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(λ) advantages; returns = adv + V(s). Terminal → no bootstrap."""
    n = len(rew)
    adv = np.zeros(n)
    running = 0.0
    for t in range(n - 1, -1, -1):
        if term[t]:
            delta = rew[t] - values[t]
            running = delta  # don't carry advantage across a terminal boundary
        else:
            delta = rew[t] + gamma * next_val[t] - values[t]
            running = delta + gamma * lam * running
        adv[t] = running
    return adv, adv + values


def _chunks(rng: np.random.Generator, n: int, size: int):
    perm = rng.permutation(n)
    for start in range(0, n, size):
        yield perm[start : start + size]


def ppo_update(
    policy: PPOPolicy,
    buf: dict[str, np.ndarray],
    cfg: PPOConfig,
    actor_opt: Adam,
    critic_opt: Adam,
    std_opt: Adam,
) -> dict[str, float]:
    """One PPO inner pass over the rollout buffer (epochs × minibatches)."""
    n = len(buf["obs"])
    obs = buf["obs"]
    act = buf["act"]
    old_logp = buf["logp"]
    adv, returns = compute_gae(
        buf["rew"], buf["term"], buf["next_val"], policy.value(obs), cfg.gamma, cfg.lam
    )
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    actor_ws = policy.actor.param_list()
    critic_ws = policy.critic.param_list()

    pg_total = mse_total = ent_total = 0.0
    for _ in range(cfg.epochs):
        for idx in _chunks(policy.rng, n, cfg.minibatch):
            ob, ac, ad, rt, olp = obs[idx], act[idx], adv[idx], returns[idx], old_logp[idx]
            m = len(idx)

            # ---- actor: clipped surrogate + entropy bonus ----
            mu = policy.actor.forward(ob).reshape(-1)
            sigma = float(np.exp(policy.log_std[0]))
            logp = -0.5 * (((ac - mu) / sigma) ** 2) - np.log(sigma) - 0.5 * _LOG2PI
            ratio = np.exp(logp - olp)
            ratio_cl = np.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
            surr = np.minimum(ratio * ad, ratio_cl * ad)
            pg_loss = -float(np.mean(surr))
            ent = policy.entropy()
            actor_loss = pg_loss - cfg.ent_coef * ent

            mask = (ratio > 1.0 - cfg.clip) & (ratio < 1.0 + cfg.clip)
            dlogp = -ad * ratio * mask                      # dL/dlogp per sample
            dlogp_dmu = (ac - mu) / (sigma * sigma)         # dlogp/dμ
            d_out = dlogp * dlogp_dmu / m                   # dL/d(actor out), mean
            actor_grads = policy.actor.backward(d_out)
            dlogp_ds = ((ac - mu) / sigma) ** 2 - 1.0       # dlogp/d lnσ
            grad_s = float(np.mean(dlogp * dlogp_ds)) - cfg.ent_coef
            flat_grads = [g for pair in actor_grads for g in pair]
            clip_grad_norm(flat_grads + [np.asarray(grad_s)], cfg.grad_norm)
            actor_opt.step(flat_grads)
            std_opt.step([np.asarray(grad_s)])

            # ---- critic: value regression ----
            v = policy.critic.forward(ob).reshape(-1)
            v_err = v - rt
            v_loss = float(np.mean(v_err * v_err))
            d_out_c = 2.0 * v_err / m
            critic_grads = policy.critic.backward(d_out_c)
            critic_flat = [g for pair in critic_grads for g in pair]
            clip_grad_norm(critic_flat, cfg.grad_norm)
            critic_opt.step(critic_flat)

            pg_total += pg_loss
            mse_total += v_loss
            ent_total += ent

    batches = cfg.epochs * max(1, n // cfg.minibatch)
    return {
        "pg_loss": pg_total / max(1, batches),
        "value_loss": mse_total / max(1, batches),
        "entropy": ent_total / max(1, batches),
    }


def make_optimizers(
    policy: PPOPolicy, cfg: PPOConfig
) -> tuple[Adam, Adam, Adam]:
    return (
        Adam(policy.actor.param_list(), lr=cfg.lr_actor),
        Adam(policy.critic.param_list(), lr=cfg.lr_critic),
        Adam([policy.log_std], lr=cfg.lr_std),
    )


@dataclass
class TrainHistory:
    """Per-evaluation-row snapshot recorded mid-training."""

    iteration: int = 0
    reward_mean: float = 0.0
    shortfall_bps_mean: float = 0.0
    pg_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    samples: int = 0

    def as_row(self) -> tuple:
        return (
            self.iteration,
            round(self.reward_mean, 4),
            round(self.shortfall_bps_mean, 4),
            round(self.pg_loss, 4),
            round(self.value_loss, 3),
            round(self.entropy, 4),
        )


def train_ppo(
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
    cfg: PPOConfig | None = None,
    *,
    tracker: Optional[Callable[[TrainHistory], None]] = None,
) -> tuple[PPOPolicy, list[TrainHistory]]:
    """Train a ``PPOPolicy`` against ``OrderBookEnv`` episodes.

    Returns ``(policy, history)`` where ``history`` holds one ``TrainHistory``
    per ``eval_every`` iterations (plus the final one), so callers can plot
    reward / shortfall_bps / loss curves without re-running training.
    """
    cfg = cfg or PPOConfig()
    env = env_factory()  # single instance, reseeded per episode
    obs_dim = env.observation_space.shape[0]  # 44 or 45
    policy = PPOPolicy(obs_dim, hidden=cfg.hidden, seed=cfg.seed, std0=cfg.std0)
    actor_opt, critic_opt, std_opt = make_optimizers(policy, cfg)
    history: list[TrainHistory] = []

    for it in range(cfg.iterations):
        buf = collect_rollouts(env, policy, cfg.episodes, base_seed=cfg.unit_seed + it * cfg.episodes)
        losses = ppo_update(policy, buf, cfg, actor_opt, critic_opt, std_opt)
        should_eval = cfg.eval_every > 0 and (
            it % cfg.eval_every == cfg.eval_every - 1 or it == cfg.iterations - 1
        )
        if should_eval:
            rows = _quick_eval(
                policy, cfg.unit_seed + it * cfg.episodes, cfg.eval_episodes,
                env_factory=env_factory,
            )
            h = TrainHistory(
                iteration=it + 1,
                reward_mean=float(np.mean([r["reward"] for r in rows])),
                shortfall_bps_mean=float(np.mean([r["shortfall_bps"] for r in rows])),
                pg_loss=losses["pg_loss"],
                value_loss=losses["value_loss"],
                entropy=losses["entropy"],
                samples=len(buf["rew"]),
            )
            history.append(h)
            if tracker is not None:
                tracker(h)

    return policy, history


def _quick_eval(
    policy: PPOPolicy,
    base_seed: int,
    episodes: int,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> list[dict]:
    """Deterministic mid-training evaluation; returns light episode rows."""
    from .evaluate import evaluate_policy

    rows, _ = evaluate_policy(
        policy, n_episodes=episodes, seed=base_seed, env_factory=env_factory,
    )
    return rows