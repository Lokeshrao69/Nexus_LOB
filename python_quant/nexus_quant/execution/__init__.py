"""Execution realism (plan_2.md Phase 2) — costs, metrics, backtest.

Package layout (work-package §WP2):

* ``cost_model.py``  — ``CostParams``, square-root market impact, fee/rebate net PnL.
* ``metrics.py``     — implementation shortfall, market-VWAP slippage (NOT self
  executed VWAP — the Phase-1 flaw this layer fixes), fill/completion, MDD, inv risk.
* ``backtest.py``    — run any policy over an ``OrderBookEnv`` (the synthetic tape),
  batch per-episode metrics, summarize per-regime.
"""

from __future__ import annotations

from .backtest import run_backtest, summarize
from .cost_model import CostParams, impact, net_pnl
from .metrics import (
    arrival_slippage,
    completion_rate,
    execution_vwap,
    fill_rate,
    implementation_shortfall,
    inv_risk,
    max_drawdown,
    vwap_slippage,
)
from .volume_profile import EmpiricalVolumeForecaster, VolumeProfile

__all__ = [
    "CostParams",
    "EmpiricalVolumeForecaster",
    "VolumeProfile",
    "arrival_slippage",
    "completion_rate",
    "execution_vwap",
    "fill_rate",
    "impact",
    "implementation_shortfall",
    "inv_risk",
    "max_drawdown",
    "net_pnl",
    "run_backtest",
    "summarize",
    "vwap_slippage",
]