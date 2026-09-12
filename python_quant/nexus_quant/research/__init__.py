"""Nexus-LOB research layer (Part 2).

Pure functions over a ``View`` (the ``BOOK_STATE_DTYPE`` record as a dict)
plus forward labels, walk-forward datasets, and experiment/IC tooling.

Ground rules (plan_2.md §0) enforced here:
  * features are stateless pure functions of state at ``t`` — a test mutes the
    book after the call and asserts the feature is unchanged;
  * labels use prices strictly at ``t+h`` or later, never ``t``;
  * splits are walk-forward by calendar/sequence time, never shuffled CV.
"""
from .dataset import event_frame, make_split
from .experiments import (
    bootstrap_ci,
    decile_spread,
    diebold_mariano,
    hit_rate,
    icir,
    rank_ic,
    run_experiment,
)
from .features import (
    deep_imbalance,
    flow_intensity,
    lob_imbalance,
    microprice,
    momentum,
    ofi,
    realized_vol,
    spread_bps,
)
from .labels import forward_mid_move, forward_return
from .models import fit_ols_ic, zscore

__all__ = [
    "bootstrap_ci",
    "decile_spread",
    "deep_imbalance",
    "diebold_mariano",
    "event_frame",
    "fit_ols_ic",
    "flow_intensity",
    "forward_mid_move",
    "forward_return",
    "hit_rate",
    "icir",
    "lob_imbalance",
    "make_split",
    "microprice",
    "momentum",
    "ofi",
    "rank_ic",
    "realized_vol",
    "run_experiment",
    "spread_bps",
    "zscore",
]