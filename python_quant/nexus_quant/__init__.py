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

# Regime presets — pass as **REGIME_PRESETS[k] to OrderBookEnv (Phase 3, plan.md WS-0).
# Each encodes behavioral *character* (a tape to stress-test analytics on), not any alpha
# target. NOTE: "highvol" keeps vol_feature=True for legacy compatibility (train_eval_agent
# --highvol); a symmetric-information RL comparison must build the regime env with
# vol_feature=False (evaluate_regime_ci's fair=True asserts exactly that).
REGIME_PRESETS: dict[str, dict] = {
    "calm": {},  # the default env — gentle walk, symmetric (fair-safe)
    "lowvol": {"take_intensity": 0.12},  # fewer marketable prints -> tighter book, gentler mid
    "highvol": {  # Markov vol regime + gap events (legacy HIGHVOL_PRESETS["highvol"])
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
        "vol_feature": True,  # legacy flag (see NOTE above)
    },
    "liquidity_shock": {  # persistent vol regime + heavy gaps + thin adds
        "regime_prob": 0.10,
        "vol_decay": 0.05,
        "gap_prob": 0.35,
        "gap_min": 1200,
        "gap_max": 3200,
        "vol_take_prob": 0.65,
        "vol_take_min": 120,
        "vol_take_max": 420,
        "vol_add_min": 8,
        "vol_add_max": 60,
        "vol_events_min": 4,
        "vol_events_max": 10,
        "vol_feature": False,
    },
    "trending": {"drift_ticks": 0.6},  # steady upward mid pressure
    "mean_reverting": {"mean_revert": 0.04, "mrv_anchor": 15_000.0},  # OU pull
}

# Legacy alias (Part-1 HIGHVOL_PLAN / train_eval_agent --highvol still reference it).
HIGHVOL_PRESETS: dict[str, dict] = {"highvol": REGIME_PRESETS["highvol"]}

__all__ = [
    "BOOK_STATE_DTYPE",
    "CONTRACT_VERSION",
    "CT_DTYPE",
    "DEPTH",
    "HIGHVOL_PRESETS",
    "OBS_DIM",
    "PX_DTYPE",
    "REGIME_PRESETS",
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
