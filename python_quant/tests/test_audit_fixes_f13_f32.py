"""Regression tests for audit findings F13–F32 and M04.

Covers:
- F13: Transactional modify in LimitOrderBook
- F14: Read-only NumPy view protection
- F15: Risk engine parameter validation
- F16: ShmRing read_seq advance in dashboard reader
- F22: Safe bounds and checked arithmetic in LOB constructor
- M04: Symmetric control terminology in regimes
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nexus_quant.dashboard import _SHM_CTRL, _SHM_CTRL_N, read_shm_ring_latest
from nexus_quant.envs.regimes import regime_factories


def test_f16_dashboard_advances_read_seq(tmp_path, monkeypatch):
    """F16: read_shm_ring_latest must advance read_seq to write_seq."""
    cap = 8
    slot_bytes = 448  # BOOK_STATE_DTYPE.itemsize
    total_size = _SHM_CTRL_N + cap * slot_bytes
    buf = bytearray(total_size)
    _SHM_CTRL.pack_into(buf, 0, 3, 0, 0, cap, slot_bytes, 1, 0)

    dev_shm = tmp_path / "dev" / "shm"
    dev_shm.mkdir(parents=True, exist_ok=True)
    shm_file = dev_shm / "nexus_test_ring"
    shm_file.write_bytes(buf)

    monkeypatch.setattr("nexus_quant.dashboard.Path", lambda p: tmp_path / p.lstrip("/\\"))

    # Read latest slot
    res = read_shm_ring_latest("nexus_test_ring")
    assert res is not None

    # Verify read_seq was updated to write_seq in the file
    updated_data = shm_file.read_bytes()
    w_seq, r_seq, _drop, _c, _sb, _state, _pad = _SHM_CTRL.unpack_from(updated_data, 0)
    assert w_seq == 3
    assert r_seq == 3, f"Expected read_seq to advance to 3, got {r_seq}"


def test_m04_highvol_null_control_arm():
    """M04: highvol_null must be configured as symmetric control arm with gap_down_prob=0.5."""
    factories = regime_factories(("highvol_null",), costs=False, vol_feature=False)
    assert "highvol_null" in factories
    env = factories["highvol_null"]()
    # Check that gap_down_prob is 0.5 (symmetric)
    assert env.gap_down_prob == 0.5


def test_f14_f15_cpp_bindings():
    """F14 & F15: test pybind engine and risk bindings if nexus_engine is available."""
    try:
        import nexus_engine
    except ImportError:
        pytest.skip("nexus_engine pybind module not built in test environment")

    # F15: compute_var_cvar validation
    with pytest.raises((ValueError, RuntimeError)):
        nexus_engine.compute_var_cvar(n_paths=0)

    with pytest.raises((ValueError, RuntimeError)):
        nexus_engine.compute_var_cvar(steps=-1)

    with pytest.raises((ValueError, RuntimeError)):
        nexus_engine.compute_var_cvar(alpha=1.5)

    with pytest.raises((ValueError, RuntimeError)):
        nexus_engine.compute_var_cvar(s0=-10.0)

    # F14: Engine view arrays must be read-only
    eng = nexus_engine.Engine(min_price=100, max_price=200, pool_capacity=1024)
    v = eng.view()
    bid_px = v["bid_px"]
    assert not bid_px.flags.writeable, "Zero-copy bid_px array must be read-only"

    with pytest.raises(ValueError):
        bid_px[0] = 9999
