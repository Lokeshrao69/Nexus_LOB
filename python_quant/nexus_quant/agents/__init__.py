"""RL execution agent for Nexus-LOB (pure-NumPy PPO), Person B.

Self-contained actor–critic PPO: no torch/sb3 — a hand-rolled NumPy core
that is byte-reproducible and drop-in replaceable by a torch/GRPO variant
(experiments only touch the ``act(obs, deterministic=True)`` interface).

Public API
----------
``PPOPolicy``   trained/stochastic policy; ``act(obs) -> float`` in [-1,1].
``PPOConfig``   every training hyperparameter (deterministic runs).
``train_ppo``   end-to-end: rollouts → GAE → clipped-surrogate updates.
``evaluate_policy``  mean reward + shortfall_bps over seeded episodes.
``strategy_table``   agent vs TWAP/VWAP/POV/Passive on the same tape.
"""

from .grpo import GRPOConfig, train_grpo
from .evaluate import (
    BaselineId,
    EvalSummary,
    Policy,
    evaluate_policy,
    format_table,
    strategy_table,
)
from .mlp import MLP, Adam, clip_grad_norm
from .ppo import (
    PPOConfig,
    PPOPolicy,
    TrainHistory,
    collect_rollouts,
    compute_gae,
    train_ppo,
)

__all__ = [
    "Adam",
    "BaselineId",
    "GRPOConfig",
    "EvalSummary",
    "MLP",
    "PPOConfig",
    "PPOPolicy",
    "Policy",
    "TrainHistory",
    "clip_grad_norm",
    "collect_rollouts",
    "compute_gae",
    "evaluate_policy",
    "format_table",
    "strategy_table",
    "train_grpo",
    "train_ppo",
]