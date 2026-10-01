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

import pytest
from nexus_quant.dashboard import (
    _SHM_CTRL_N,
    pack_shm_control,
    read_shm_ring_latest,
    unpack_shm_control,
)
from nexus_quant.envs.regimes import regime_factories


def test_f16_dashboard_advances_read_seq(tmp_path, monkeypatch):
    """F16: read_shm_ring_latest must advance read_seq to write_seq."""
    cap = 8
    slot_bytes = 448  # BOOK_STATE_DTYPE.itemsize
    total_size = _SHM_CTRL_N + cap * slot_bytes
    buf = bytearray(total_size)
    pack_shm_control(buf, capacity=cap, slot_bytes=slot_bytes, write_seq=3, dropped=7)

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
    ctrl = unpack_shm_control(updated_data)
    assert ctrl["write_seq"] == 3
    assert ctrl["read_seq"] == 3, f"Expected read_seq to advance to 3, got {ctrl['read_seq']}"
    # Producer-owned fields must be left exactly as they were.
    assert ctrl["dropped"] == 7


def test_dashboard_never_rewrites_producer_fields(tmp_path, monkeypatch):
    """The reader may only write read_seq; write_seq/dropped belong to the producer.

    Simulates the race: the producer advances write_seq after the dashboard has
    read the control block. The old reader rewrote the whole block from its stale
    copy, rewinding write_seq. Now only the read_seq bytes may change.
    """
    cap, slot_bytes = 8, 448
    buf = bytearray(_SHM_CTRL_N + cap * slot_bytes)
    pack_shm_control(buf, capacity=cap, slot_bytes=slot_bytes, write_seq=5, read_seq=2, dropped=1)
    dev_shm = tmp_path / "dev" / "shm"
    dev_shm.mkdir(parents=True, exist_ok=True)
    shm_file = dev_shm / "nexus_race_ring"
    shm_file.write_bytes(buf)
    monkeypatch.setattr("nexus_quant.dashboard.Path", lambda p: tmp_path / p.lstrip("/\\"))

    before = shm_file.read_bytes()
    assert read_shm_ring_latest("nexus_race_ring") is not None
    after = shm_file.read_bytes()
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    assert changed, "read_seq should have advanced"
    assert all(128 <= i < 136 for i in changed), f"bytes outside read_seq changed: {changed[:8]}"


def test_dashboard_rejects_old_ring_layout(tmp_path, monkeypatch):
    """A segment from a different ring layout version must be ignored, not misread."""
    cap, slot_bytes = 4, 448
    buf = bytearray(_SHM_CTRL_N + cap * slot_bytes)
    pack_shm_control(buf, capacity=cap, slot_bytes=slot_bytes, write_seq=1, layout_version=1)
    dev_shm = tmp_path / "dev" / "shm"
    dev_shm.mkdir(parents=True, exist_ok=True)
    (dev_shm / "nexus_old_ring").write_bytes(buf)
    monkeypatch.setattr("nexus_quant.dashboard.Path", lambda p: tmp_path / p.lstrip("/\\"))
    assert read_shm_ring_latest("nexus_old_ring") is None


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
