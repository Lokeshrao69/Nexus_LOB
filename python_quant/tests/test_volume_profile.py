"""Unit tests for empirical volume profiling and historical forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nexus_quant.baselines import policy_action, run_episode, volume_curve_target
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.execution.volume_profile import (
    EmpiricalVolumeForecaster,
    VolumeProfile,
)


def test_volume_profile_normalization_and_monotonicity():
    weights = [10.0, 5.0, 2.0, 3.0, 8.0]
    prof = VolumeProfile(symbol="TEST", bucket_weights=weights)

    # Bucket weights must sum to 1.0
    assert sum(prof.bucket_weights) == pytest.approx(1.0, rel=1e-7)
    assert prof.n_buckets == 5

    # Boundary conditions
    assert prof.cumulative_fraction(0.0) == pytest.approx(0.0)
    assert prof.cumulative_fraction(1.0) == pytest.approx(1.0)
    assert prof.cumulative_fraction(-0.1) == pytest.approx(0.0)
    assert prof.cumulative_fraction(1.2) == pytest.approx(1.0)

    # Monotonicity test across fine grid
    ts = np.linspace(0.0, 1.0, 500)
    c_vals = np.array([prof.cumulative_fraction(t) for t in ts])
    assert np.all(np.diff(c_vals) >= -1e-12), "Cumulative fraction must be non-decreasing"
    assert c_vals[0] == pytest.approx(0.0)
    assert c_vals[-1] == pytest.approx(1.0)


def test_volume_profile_zero_handling():
    # Handling of all zero weights should raise ValueError
    with pytest.raises(ValueError, match="sum of bucket_weights must be positive"):
        VolumeProfile(symbol="TEST", bucket_weights=[0.0, 0.0, 0.0])

    # Handling of negative weights should raise ValueError
    with pytest.raises(ValueError, match="must be non-negative"):
        VolumeProfile(symbol="TEST", bucket_weights=[1.0, -0.5, 2.0])

    # Single non-zero bucket
    prof = VolumeProfile(symbol="TEST", bucket_weights=[0.0, 5.0, 0.0])
    assert prof.cumulative_fraction(0.1) == pytest.approx(0.0)
    assert prof.cumulative_fraction(0.9) == pytest.approx(1.0)


def test_volume_profile_json_roundtrip(tmp_path: Path):
    weights = [0.2, 0.1, 0.1, 0.1, 0.2, 0.3]
    meta = {"source": "test", "window": 5}
    prof1 = VolumeProfile(symbol="AAPL", bucket_weights=weights, metadata=meta)

    save_file = tmp_path / "aapl_profile.json"
    prof1.save(save_file)

    assert save_file.exists()
    prof2 = VolumeProfile.load(save_file)

    assert prof2.symbol == "AAPL"
    assert prof2.metadata == meta
    assert len(prof2.bucket_weights) == len(prof1.bucket_weights)
    for w1, w2 in zip(prof1.bucket_weights, prof2.bucket_weights):
        assert w1 == pytest.approx(w2)

    # Cumulative fractions must match exactly
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert prof1.cumulative_fraction(t) == pytest.approx(prof2.cumulative_fraction(t))


def test_forecaster_temporal_hygiene_no_future_leakage():
    forecaster = EmpiricalVolumeForecaster(n_buckets=4)

    # Day 1: heavy morning
    forecaster.add_daily_volume("AAPL", "2020-01-01", [100.0, 50.0, 20.0, 10.0])
    # Day 2: balanced
    forecaster.add_daily_volume("AAPL", "2020-01-02", [50.0, 50.0, 50.0, 50.0])

    # Forecast for Day 3 (2020-01-03) should use Day 1 and Day 2
    f_d3_initial = forecaster.forecast("AAPL", as_of_date="2020-01-03", history_window=5)

    # Now add Day 3 volume (heavy close) and Day 4 volume
    forecaster.add_daily_volume("AAPL", "2020-01-03", [10.0, 20.0, 50.0, 500.0])
    forecaster.add_daily_volume("AAPL", "2020-01-04", [5.0, 5.0, 5.0, 1000.0])

    # Re-evaluate forecast for Day 3: MUST BE EXACTLY IDENTICAL to initial
    f_d3_after = forecaster.forecast("AAPL", as_of_date="2020-01-03", history_window=5)

    assert f_d3_initial.bucket_weights == pytest.approx(f_d3_after.bucket_weights)
    assert f_d3_after.metadata["dates_used"] == ["2020-01-01", "2020-01-02"]
    assert "2020-01-03" not in f_d3_after.metadata["dates_used"]
    assert "2020-01-04" not in f_d3_after.metadata["dates_used"]

    # Forecast for Day 4 uses Days 1, 2, 3 (not Day 4)
    f_d4 = forecaster.forecast("AAPL", as_of_date="2020-01-04", history_window=5)
    assert f_d4.metadata["dates_used"] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert "2020-01-04" not in f_d4.metadata["dates_used"]


def test_forecaster_symbol_specificity():
    forecaster = EmpiricalVolumeForecaster(n_buckets=3)

    # AAPL: front-loaded
    forecaster.add_daily_volume("AAPL", "2020-01-01", [100.0, 20.0, 10.0])
    forecaster.add_daily_volume("AAPL", "2020-01-02", [120.0, 15.0, 5.0])

    # QQQ: back-loaded
    forecaster.add_daily_volume("QQQ", "2020-01-01", [10.0, 20.0, 150.0])
    forecaster.add_daily_volume("QQQ", "2020-01-02", [15.0, 25.0, 180.0])

    aapl_prof = forecaster.forecast("AAPL", as_of_date="2020-01-03")
    qqq_prof = forecaster.forecast("QQQ", as_of_date="2020-01-03")

    # AAPL has higher early weight, QQQ has higher late weight
    assert aapl_prof.bucket_weights[0] > aapl_prof.bucket_weights[2]
    assert qqq_prof.bucket_weights[2] > qqq_prof.bucket_weights[0]
    assert aapl_prof.cumulative_fraction(0.33) > qqq_prof.cumulative_fraction(0.33)


def test_forecaster_missing_history_guards():
    forecaster = EmpiricalVolumeForecaster(n_buckets=4)

    # No history at all
    with pytest.raises(ValueError, match="No history recorded"):
        forecaster.forecast("UNKNOWN", as_of_date="2020-01-01")

    # Only future history available
    forecaster.add_daily_volume("SPY", "2020-01-05", [10, 10, 10, 10])
    with pytest.raises(ValueError, match="No prior history available"):
        forecaster.forecast("SPY", as_of_date="2020-01-02")


def test_baselines_volume_curve_integration():
    # Cosine fallback when profile=None
    target_cosine = volume_curve_target(0.25, u_weight=0.6)
    assert target_cosine > 0.25

    # Linear / flat profile (uniform buckets)
    flat_prof = VolumeProfile(symbol="FLAT", bucket_weights=[1.0, 1.0, 1.0, 1.0])
    target_flat = volume_curve_target(0.25, profile=flat_prof)
    assert target_flat == pytest.approx(0.25)

    # Extreme front-loaded profile (90% in first quarter)
    front_prof = VolumeProfile(symbol="FRONT", bucket_weights=[9.0, 1.0, 0.0, 0.0])
    target_front = volume_curve_target(0.25, profile=front_prof)
    assert target_front == pytest.approx(0.9)

    # Compare execution with custom empirical profiles
    env1 = OrderBookEnv(seed=42)
    env2 = OrderBookEnv(seed=42)

    res_front = run_episode(env1, "schedule_twap", seed=42, volume_profile=front_prof)
    res_flat = run_episode(env2, "schedule_twap", seed=42, volume_profile=flat_prof)

    assert res_front.name == "schedule_twap"
    assert res_flat.name == "schedule_twap"
    # Both successfully executed
    assert res_front.filled > 0
    assert res_flat.filled > 0
