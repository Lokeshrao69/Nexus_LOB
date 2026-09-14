"""Deterministic unit tests for multi-day ITCH aggregation tooling (Person B).

Covers:
1. Empty input validation.
2. Single-day aggregation.
3. Identical multi-day aggregation (zero between-day variance).
4. Multi-day variance and dispersion computation.
5. Sample-weighted vs unweighted cross-day means.
6. Seeded bootstrap CI determinism.
7. Missing / corrupted session handling.
8. Cross-day KM fill and adverse selection drift aggregation.
9. Markdown rendering integrity.
10. Direct DayStudyRecord input support.
"""
from __future__ import annotations

import pytest
from nexus_quant.research.multi_day_aggregation import (
    DayStudyRecord,
    aggregate_multi_day_results,
    extract_day_study_record,
    render_multi_day_aggregation_md,
)


def _mock_payload(day: str, ic_h5: float, n_test: int = 1000, reg_events: int = 5000, p_fill_50: float = 0.35, drift_h5: float = 0.5) -> dict:
    return {
        "day": day,
        "symbol": "AAPL",
        "coverage": {"kind": "full_day", "is_full_day": True},
        "tape": {"regular_events": reg_events, "messages": reg_events * 2},
        "ic": {
            "rows": [
                {"feature": "lob_imbalance", "horizon_h": 5, "ic_test": ic_h5, "n_test": n_test},
                {"feature": "lob_imbalance", "horizon_h": 10, "ic_test": ic_h5 * 0.8, "n_test": n_test},
            ]
        },
        "fill": {
            "n_orders_regular": 200,
            "km": {"tau_events": [10, 50, 100], "p_fill": [0.15, p_fill_50, 0.60]},
        },
        "adverse": {
            "n_fills": 50,
            "horizons": {"5": {"overall": {"mean_drift": drift_h5, "n": 50}}},
        },
    }


def test_empty_payload_raises_value_error():
    """Empty payload sequence must raise ValueError."""
    with pytest.raises(ValueError, match="requires at least one day payload"):
        aggregate_multi_day_results([])


def test_single_day_aggregation():
    """Single-day input computes valid descriptive metrics with zero between-day variance."""
    p1 = _mock_payload("12302019", ic_h5=0.12)
    res = aggregate_multi_day_results([p1])
    assert res["n_days"] == 1
    assert res["days"] == ["12302019"]
    assert res["total_regular_events"] == 5000
    h5 = res["rank_ic_by_horizon"][5]
    assert h5["mean_unweighted"] == pytest.approx(0.12)
    assert h5["mean_weighted"] == pytest.approx(0.12)
    assert h5["std_between_days"] == 0.0
    assert h5["ci95"] is None  # Needs >= 2 days for bootstrap CI


def test_identical_multi_day_aggregation():
    """Two sessions with identical metrics produce zero between-day standard deviation."""
    p1 = _mock_payload("12302019", ic_h5=0.15)
    p2 = _mock_payload("01302020", ic_h5=0.15)
    res = aggregate_multi_day_results([p1, p2])
    assert res["n_days"] == 2
    h5 = res["rank_ic_by_horizon"][5]
    assert h5["mean_unweighted"] == pytest.approx(0.15)
    assert h5["std_between_days"] == pytest.approx(0.0)


def test_multi_day_variance_and_dispersion():
    """Sessions with varying ICs compute correct between-day sample standard deviation."""
    p1 = _mock_payload("day1", ic_h5=0.10)
    p2 = _mock_payload("day2", ic_h5=0.20)
    res = aggregate_multi_day_results([p1, p2])
    h5 = res["rank_ic_by_horizon"][5]
    assert h5["mean_unweighted"] == pytest.approx(0.15)
    # std([0.10, 0.20], ddof=1) = 0.070710678
    assert h5["std_between_days"] == pytest.approx(0.070711, abs=1e-5)


def test_weighted_vs_unweighted_means():
    """Sample-weighted mean pulls estimate toward higher-volume trading sessions."""
    p1 = _mock_payload("day1", ic_h5=0.10, n_test=100)
    p2 = _mock_payload("day2", ic_h5=0.30, n_test=900)
    res = aggregate_multi_day_results([p1, p2])
    h5 = res["rank_ic_by_horizon"][5]
    assert h5["mean_unweighted"] == pytest.approx(0.20)
    # Weighted mean: (0.10*100 + 0.30*900) / 1000 = 0.28
    assert h5["mean_weighted"] == pytest.approx(0.28)


def test_bootstrap_ci_determinism():
    """Fixed random seed ensures identical bootstrap confidence interval bounds."""
    p1 = _mock_payload("day1", ic_h5=0.10)
    p2 = _mock_payload("day2", ic_h5=0.20)
    p3 = _mock_payload("day3", ic_h5=0.18)
    res1 = aggregate_multi_day_results([p1, p2, p3], seed=999)
    res2 = aggregate_multi_day_results([p1, p2, p3], seed=999)

    ci1 = res1["rank_ic_by_horizon"][5]["ci95"]
    ci2 = res2["rank_ic_by_horizon"][5]["ci95"]
    assert ci1 == ci2
    assert ci1["lo"] <= res1["rank_ic_by_horizon"][5]["mean_unweighted"] <= ci1["hi"]


def test_missing_day_handling():
    """Sessions with errors or zero regular session events are recorded in missing_days."""
    p1 = _mock_payload("day1", ic_h5=0.15)
    p_err = {"day": "day2_failed", "error": "HTTP 404: Not Found"}
    p_empty = {
        "day": "day3_partial",
        "coverage": {"kind": "partial_gzip_prefix", "is_full_day": False},
        "tape": {"regular_events": 0, "messages": 1000},
    }
    res = aggregate_multi_day_results([p1, p_err, p_empty])
    assert res["n_days"] == 1
    assert len(res["missing_days"]) == 2
    assert any("404" in m["reason"] for m in res["missing_days"])
    assert any("no regular session" in m["reason"] for m in res["missing_days"])


def test_km_fill_and_adverse_aggregation():
    """KM fill probability and post-fill adverse selection are aggregated across sessions."""
    p1 = _mock_payload("day1", ic_h5=0.1, p_fill_50=0.30, drift_h5=0.4)
    p2 = _mock_payload("day2", ic_h5=0.1, p_fill_50=0.50, drift_h5=0.8)
    res = aggregate_multi_day_results([p1, p2])

    assert res["km_fill_50"]["mean"] == pytest.approx(0.40)
    assert res["km_fill_50"]["std_between_days"] == pytest.approx(0.141421, abs=1e-4)

    assert res["adverse_drift_h5"]["mean"] == pytest.approx(0.60)
    assert res["adverse_drift_h5"]["std_between_days"] == pytest.approx(0.2828, abs=1e-3)


def test_markdown_rendering_structure():
    """Markdown report includes summary headers, session breakdown, and empirical policy."""
    p1 = _mock_payload("12302019", ic_h5=0.12)
    p2 = _mock_payload("01302020", ic_h5=0.14)
    res = aggregate_multi_day_results([p1, p2])
    md = render_multi_day_aggregation_md(res)

    assert "# Multi-Day ITCH Empirical Aggregation Report" in md
    assert "12302019" in md
    assert "01302020" in md
    assert "Between-Day Std" in md
    assert "LOB Imbalance Rank IC" in md


def test_from_day_study_records_directly():
    """Can ingest DayStudyRecord objects directly."""
    p = _mock_payload("12302019", ic_h5=0.15)
    rec = extract_day_study_record(p)
    assert isinstance(rec, DayStudyRecord)
    res = aggregate_multi_day_results([rec])
    assert res["n_days"] == 1
    assert res["days"] == ["12302019"]
