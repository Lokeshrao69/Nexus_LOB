"""Nexus-LOB research layer (Part 2).

Pure functions over a ``View`` (the ``BOOK_STATE_DTYPE`` record as a dict)
plus forward labels, walk-forward datasets, and experiment/IC tooling.

Ground rules (plan_2.md §0) enforced here:
  * features are stateless pure functions of state at ``t`` — a test mutes the
    book after the call and asserts the feature is unchanged;
  * labels use prices strictly at ``t+h`` or later, never ``t``;
  * splits are walk-forward by calendar/sequence time, never shuffled CV.
"""
from .adverse_selection import (
    P_adverse,
    PassiveFill,
    adverse_groups,
    adverse_selection_report,
    drift_ticks,
    fills_from_tracker,
    nw_tstat,
    p_adverse,
    post_fill_drift,
    pre_fill_drift,
)
from .dataset import event_frame, make_split
from .experiments import (
    bootstrap_ci,
    brier_score,
    calibration_curve,
    decile_spread,
    diebold_mariano,
    hac_se,
    hit_rate,
    icir,
    normal_cdf,
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
from .multi_day_aggregation import (
    aggregate_multi_day_results,
    extract_day_study_record,
    render_multi_day_aggregation_md,
)
from .queue_dynamics import (
    EmpiricalQueueHazard,
    FillOutcome,
    FillRecord,
    FillRow,
    LogisticFillModel,
    OrderLevelTracker,
    OrderLife,
    QueueTracker,
    RestingOrder,
    StepFlow,
    TrackedOrder,
    calibration_table,
    censor_open_orders,
    compute_metrics,
    fill_dataset,
    fill_prob_survival,
    logistic_fill_model,
    order_features,
    queue_ahead_walk,
    standing_order_lifetimes,
)
from .synthetic_flow import FLOW_PRESETS, FlowConfig, SyntheticFlow

__all__ = [
    "FLOW_PRESETS",
    "EmpiricalQueueHazard",
    "FillOutcome",
    "FillRecord",
    "FillRow",
    "FlowConfig",
    "LogisticFillModel",
    "OrderLevelTracker",
    "OrderLife",
    "P_adverse",
    "PassiveFill",
    "QueueTracker",
    "RestingOrder",
    "StepFlow",
    "SyntheticFlow",
    "TrackedOrder",
    "adverse_groups",
    "adverse_selection_report",
    "aggregate_multi_day_results",
    "bootstrap_ci",
    "brier_score",
    "calibration_curve",
    "calibration_table",
    "censor_open_orders",
    "compute_metrics",
    "decile_spread",
    "deep_imbalance",
    "diebold_mariano",
    "drift_ticks",
    "event_frame",
    "extract_day_study_record",
    "fill_dataset",
    "fill_prob_survival",
    "fills_from_tracker",
    "fit_ols_ic",
    "flow_intensity",
    "forward_mid_move",
    "forward_return",
    "hac_se",
    "hit_rate",
    "icir",
    "lob_imbalance",
    "logistic_fill_model",
    "make_split",
    "microprice",
    "momentum",
    "normal_cdf",
    "nw_tstat",
    "ofi",
    "order_features",
    "p_adverse",
    "post_fill_drift",
    "pre_fill_drift",
    "queue_ahead_walk",
    "rank_ic",
    "realized_vol",
    "render_multi_day_aggregation_md",
    "run_experiment",
    "spread_bps",
    "standing_order_lifetimes",
    "zscore",
]
