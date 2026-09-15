"""Deterministic unit tests for the cross-session panel statistics (multi-day ITCH campaign)."""
from __future__ import annotations

import math

import pytest
from nexus_quant.research.multi_day_aggregation import (
    PANEL_METRICS,
    PANEL_PAIRED,
    extract_session_metrics,
    render_session_panel_md,
    session_panel_statistics,
)


def _payload(ic5: float, *, ic_micro5: float | None = None, drift5: float = -40.0, pre5: float = -3.0,
             p_fill_50: float = 0.03, n_orders: int = 1000, n_fill: int = 50, slope: float | None = 1.05) -> dict:
    rows = []
    for feature in ("lob_imbalance", "microprice_off", "deep_imbalance", "spread_bps", "ofi_l2", "ofi_order", "ofi_order_w20"):
        for h in (1, 5, 10, 25):
            val = ic5 if feature == "lob_imbalance" else (ic_micro5 if feature == "microprice_off" and ic_micro5 is not None else 0.05)
            rows.append({"feature": feature, "horizon_h": h, "ic_test": val, "n_test": 500, "ci95_lo": val - 0.02, "ci95_hi": val + 0.02})
    payload = {
        "tape": {"messages": 2000, "events": 1990, "truncated": 0, "regular_events": 1500, "tracker_unknown_id": 0, "integrity_issues": {}},
        "ic": {"rows": rows, "combined": [{"horizon_h": h, "ic_test": ic5 + 0.01, "n_test": 500} for h in (1, 5, 10, 25)]},
        "fill": {
            "n_orders_regular": n_orders,
            "km": {"tau_events": [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0], "p_fill": [0.01, p_fill_50, 0.05, 0.09, 0.11, 0.16],
                   "n": n_orders, "n_fill": n_fill, "n_cancel": n_orders - n_fill},
        },
        "adverse": {
            "n_fills": n_fill,
            "horizons": {
                "5": {"overall": {"n": n_fill, "mean_drift": drift5, "t_nw": -10.0, "p_adverse": 0.95},
                      "pre_fill": {"n": n_fill, "mean_drift": pre5},
                      "post_minus_pre": {"n": n_fill, "mean_drift": drift5 - pre5}},
            },
        },
    }
    if slope is not None:
        payload["fill"]["logistic"] = {"calibration_slope": slope, "brier_skill": 0.05, "base_rate": 0.05, "n_test": 300}
    return payload


def test_extract_session_metrics_flattens_and_never_imputes() -> None:
    row = extract_session_metrics(_payload(0.2, ic_micro5=0.21, slope=None), day="12302019", symbol="AAPL")
    assert row["day"] == "12302019" and row["symbol"] == "AAPL"
    assert row["ic_lob_imbalance_h5"] == pytest.approx(0.2)
    assert row["n_ic_lob_imbalance_h5"] == 500
    assert row["ic_combined_h5"] == pytest.approx(0.21)
    assert row["km_p_fill_50"] == pytest.approx(0.03)
    assert row["ever_filled_rate"] == pytest.approx(0.05)
    assert row["adverse_drift_h5"] == pytest.approx(-40.0)
    assert row["adverse_post_minus_pre_h5"] == pytest.approx(-37.0)
    # no logistic block -> None, not a number
    assert row["logistic_calibration_slope"] is None and row["logistic_brier_skill"] is None
    # horizons the payload lacks stay None
    assert row["adverse_drift_h1"] is None and row["adverse_drift_h25"] is None


def test_panel_statistics_mean_sd_ci_and_sign_consistency() -> None:
    rows = [
        extract_session_metrics(_payload(0.10), day="d1", symbol="AAPL"),
        extract_session_metrics(_payload(0.20), day="d2", symbol="AAPL"),
        extract_session_metrics(_payload(0.30), day="d3", symbol="AAPL"),
        extract_session_metrics(_payload(-0.05), day="d4", symbol="AAPL"),
    ]
    panel = session_panel_statistics(rows, seed=7)
    m = panel["metrics"]["ic_lob_imbalance_h5"]["pooled"]
    assert m["n_sessions"] == 4
    assert m["mean"] == pytest.approx(0.1375)
    assert m["median"] == pytest.approx(0.15)
    assert m["std_between_sessions"] == pytest.approx(0.149304, abs=1e-5)
    assert m["ci95"]["lo"] <= m["mean"] <= m["ci95"]["hi"]
    assert m["sessions_with_hypothesised_sign"] == 3 and m["sign_consistency"] == pytest.approx(0.75)
    # equal n_test everywhere -> weighted mean equals the plain mean
    assert m["mean_weighted"] == pytest.approx(m["mean"])
    # per-day CI [ic-0.02, ic+0.02] excludes 0 for 0.10/0.20/0.30, not for -0.05
    assert m["per_session_significance"] == {"sessions_with_estimate": 4, "sessions_individually_significant": 3}
    loo = m["leave_one_day_out"]
    assert loo["most_influential_day"] == "d4"  # the negative outlier moves the mean most
    assert loo["leave_one_day_out_mean_range"][0] == pytest.approx((0.10 + 0.20 - 0.05) / 3)
    assert loo["leave_one_day_out_mean_range"][1] == pytest.approx(0.20)
    assert loo["sign_stable_under_leave_one_out"] is True
    # deterministic under the same seed
    again = session_panel_statistics(rows, seed=7)
    assert again["metrics"]["ic_lob_imbalance_h5"]["pooled"]["ci95"] == m["ci95"]


def test_panel_weighted_mean_uses_sample_sizes() -> None:
    p1 = _payload(0.10)
    p2 = _payload(0.30)
    for r in p2["ic"]["rows"]:
        r["n_test"] = 1500  # 3x the weight of p1's 500
    rows = [extract_session_metrics(p1, day="d1", symbol="AAPL"), extract_session_metrics(p2, day="d2", symbol="AAPL")]
    m = session_panel_statistics(rows)["metrics"]["ic_lob_imbalance_h5"]["pooled"]
    assert m["mean"] == pytest.approx(0.20)
    assert m["mean_weighted"] == pytest.approx((0.10 * 500 + 0.30 * 1500) / 2000)
    # adverse drift: per-day NW t of -10 counts as individually significant in the hypothesised (negative) direction
    adv = session_panel_statistics(rows)["metrics"]["adverse_drift_h5"]["pooled"]
    assert adv["per_session_significance"] == {"sessions_with_estimate": 2, "sessions_individually_significant": 2}
    assert adv["mean_weighted"] == pytest.approx(-40.0)


def test_panel_paired_differences_use_same_session() -> None:
    rows = [
        extract_session_metrics(_payload(0.10, ic_micro5=0.12), day="d1", symbol="QQQ"),
        extract_session_metrics(_payload(0.20, ic_micro5=0.18), day="d2", symbol="QQQ"),
    ]
    panel = session_panel_statistics(rows, seed=1)
    paired = panel["paired"]["microprice_minus_imbalance_h5"]["pooled"]
    assert paired["n_sessions"] == 2
    assert paired["mean"] == pytest.approx(0.0)
    assert paired["sessions_a_greater"] == 1 and paired["sessions_b_greater"] == 1
    post_pre = panel["paired"]["post_minus_pre_drift_h5"]["pooled"]
    assert post_pre["mean"] == pytest.approx(-37.0)


def test_panel_single_session_has_no_ci_and_missing_metrics_are_reported() -> None:
    rows = [extract_session_metrics(_payload(0.2, slope=None), day="d1", symbol="AAPL")]
    panel = session_panel_statistics(rows)
    assert panel["metrics"]["ic_lob_imbalance_h5"]["pooled"]["ci95"] is None
    assert panel["metrics"]["logistic_calibration_slope"]["pooled"]["n_sessions"] == 0
    assert panel["metrics"]["logistic_calibration_slope"]["pooled"]["mean"] is None
    assert panel["metrics"]["ic_lob_imbalance_h5"]["pooled"]["leave_one_day_out"] is None
    md = render_session_panel_md(panel, rows)
    assert "## Session panel" in md and "| d1 | AAPL |" in md
    assert "—" in md  # missing values render as em-dash, never as numbers


def test_panel_metric_catalogue_is_well_formed() -> None:
    keys = [k for k, *_ in PANEL_METRICS]
    assert len(keys) == len(set(keys))
    assert all(sign in (-1, 0, 1) for _, _, _, sign in PANEL_METRICS)
    for key, a, b, _ in PANEL_PAIRED:
        assert a != b and key not in keys
    row = extract_session_metrics(_payload(0.1), day="d", symbol="S")
    for key in keys:
        assert key in row, key
        assert row[key] is None or math.isfinite(float(row[key]))
