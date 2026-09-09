"""Nexus-LOB quant package (python_quant). Owner: Person B."""
from __future__ import annotations

from .book_state import (
    BOOK_STATE_DTYPE,
    CONTRACT_VERSION,
    CT_DTYPE,
    DEPTH,
    PX_DTYPE,
    SZ_DTYPE,
    Side,
    StubOrderBook,
    empty_state,
)
from .itch_parser import EventType, ItchParseStats, NormalizedEvent, iter_itch_events
from .replay import ReplayEngine, check_integrity
from .envs import OBS_DIM, OrderBookEnv

# PPO execution agent is exposed via nexus_quant.agents — pulled in lazily by
# the top-level convenience re-exports below (kept additive, no import order
# coupling with .envs).
from .agents import (
    GRPOConfig,
    PPOConfig,
    PPOPolicy,
    evaluate_policy,
    format_table,
    strategy_table,
    train_grpo,
    train_ppo,
)

# High-volatility regime presets — pass as **HIGHVOL_PRESETS["highvol"] to OrderBookEnv
HIGHVOL_PRESETS: dict[str, dict] = {
    "highvol": dict(
        regime_prob=0.08,
        vol_decay=0.15,
        gap_prob=0.20,
        gap_min=800,
        gap_max=1800,
        gap_down_prob=0.75,
        vol_take_prob=0.55,
        vol_take_min=60,
        vol_take_max=220,
        vol_add_min=15,
        vol_add_max=70,
        vol_add_offset_min=2,
        vol_add_offset_max=12,
        vol_events_min=3,
        vol_events_max=8,
        vol_feature=True,
    ),
}

__all__ = [
    "BOOK_STATE_DTYPE",
    "CONTRACT_VERSION",
    "CT_DTYPE",
    "DEPTH",
    "GRPOConfig",
    "HIGHVOL_PRESETS",
    "OBS_DIM",
    "PX_DTYPE",
    "SZ_DTYPE",
    "EventType",
    "ItchParseStats",
    "NormalizedEvent",
    "OrderBookEnv",
    "PPOConfig",
    "PPOPolicy",
    "ReplayEngine",
    "Side",
    "StubOrderBook",
    "check_integrity",
    "empty_state",
    "evaluate_policy",
    "format_table",
    "iter_itch_events",
    "strategy_table",
    "train_grpo",
    "train_ppo",
]
