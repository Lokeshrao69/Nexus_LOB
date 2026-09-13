"""Nexus-LOB quant package (python_quant). Owner: Person B."""
from __future__ import annotations

from . import (
    execution,  # Part 2 execution realism (cost model / metrics / backtest) — Phase 2
    research,  # Part 2 research layer (features / labels / dataset / experiments)
)

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
from .envs import OBS_DIM, OrderBookEnv
from .itch_parser import EventType, ItchParseStats, NormalizedEvent, iter_itch_events
from .replay import ReplayEngine, check_integrity

# High-volatility regime presets — pass as **HIGHVOL_PRESETS["highvol"] to OrderBookEnv
HIGHVOL_PRESETS: dict[str, dict] = {
    "highvol": {
        "regime_prob": 0.08,
        "vol_decay": 0.15,
        "gap_prob": 0.20,
        "gap_min": 800,
        "gap_max": 1800,
        "gap_down_prob": 0.75,
        "vol_take_prob": 0.55,
        "vol_take_min": 60,
        "vol_take_max": 220,
        "vol_add_min": 15,
        "vol_add_max": 70,
        "vol_add_offset_min": 2,
        "vol_add_offset_max": 12,
        "vol_events_min": 3,
        "vol_events_max": 8,
        "vol_feature": True,
    },
}

__all__ = [
    "BOOK_STATE_DTYPE",
    "CONTRACT_VERSION",
    "CT_DTYPE",
    "DEPTH",
    "HIGHVOL_PRESETS",
    "OBS_DIM",
    "PX_DTYPE",
    "SZ_DTYPE",
    "EventType",
    "GRPOConfig",
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
    "execution",  # execution realism layer (Part 2) — exposed one level up
    "format_table",
    "iter_itch_events",
    "research",  # research layer (Part 2) — exposed one level up
    "strategy_table",
    "train_grpo",
    "train_ppo",
]
