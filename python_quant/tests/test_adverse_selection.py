"""WS-2 tests for adverse selection (E6): drift definition, null/structural
sign, adverse-vs-side sign, OFI/level fields on fills, and the bin partition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from nexus_quant.book_state import Side
from nexus_quant.research.adverse_selection import (
    P_adverse,
    adverse_groups,
    drift_ticks,
    post_fill_drift,
)
from nexus_quant.research.experiments import hac_se
from nexus_quant.research.queue_dynamics import FillRecord, QueueTracker
from nexus_quant.research.synthetic_flow import FlowConfig, SyntheticFlow

# Test tapes: small enough to replay fast, large enough for stable stats.
# ``seed`` is a test hygiene choice (seeded, deterministic) — the *flow* itself
# is the documented rw / drift control (FLOW_PRESETS), never tuned to a signal.
_N = 8000
_RW = FlowConfig(n_events=_N, seed=123)
_DRIFT = FlowConfig(n_events=_N, seed=123, drift_ticks=0.6, vol_per_event=0.6)


# ---- helpers ---------------------------------------------------------------- #

def _replay(cfg: FlowConfig):
    tr = QueueTracker()
    for e in SyntheticFlow(cfg).generate():
        tr.on_event(e)
    return tr, list(tr.fills)


def _fill(event_index: int, side: Side = Side.Ask, ahead: int = 0,
          lvl: int = 0, ofi: float = 0.0) -> FillRecord:
    return FillRecord(
        oid=1, side=side, price=15_000, filled=1, event_index=event_index,
        ts_ns=0, size_at_birth=1, ahead_at_birth=ahead, ahead_at_fill=ahead,
        level_size_at_fill=lvl, ofi=ofi,
    )


# ---- (A) drift is exactly mid[t+h] - mid[t]; tape-end drops ---- #


def test_drift_definition_and_tape_end_none():
    tr = QueueTracker()
    tr.mid_history = [100.0, 101.0, 105.0, 99.0, 110.0]
    assert drift_ticks(_fill(event_index=0), tr, 2) == 105.0 - 100.0
    assert drift_ticks(_fill(event_index=0), tr, 1) == 101.0 - 100.0
    assert drift_ticks(_fill(event_index=1), tr, 3) == 110.0 - 101.0
    # horizon falls off the end of mid_history -> None (never extrapolated)
    assert drift_ticks(_fill(event_index=4), tr, 1) is None
    assert drift_ticks(_fill(event_index=3), tr, 2) is None


def test_tape_end_rows_dropped():
    tr, fills = _replay(_RW)
    h = 2000
    r = post_fill_drift(fills, tr, h)
    expect = sum(1 for f in fills if f.event_index + h < len(tr.mid_history))
    assert 0 < r["n"] < len(fills)
    assert r["n"] == expect


# ---- (B) P_adverse sign: mid against the passive side ---- #


def test_P_adverse_sign_and_none_at_end():
    tr = QueueTracker()
    tr.mid_history = [100.0, 101.0, 105.0, 90.0, 90.0]
    # ask (resting seller, filled by a buy take): up-move after fill -> adverse
    r = P_adverse(_fill(event_index=0, side=Side.Ask), tr, ofi=1.0, queue_pos=0.0, h=2)
    assert r == {
        "side": "ask", "ofi": 1.0, "queue_pos": 0.0, "h": 2,
        "drift_ticks": 5.0, "drift_bps": 5.0 / 100.0 * 1e4, "adverse": True,
    }
    # bid (resting buyer): same up-move is NOT adverse
    r2 = P_adverse(_fill(event_index=0, side=Side.Bid), tr, ofi=-1.0, queue_pos=0.5, h=2)
    assert r2["side"] == "bid" and r2["adverse"] is False
    # down-move after a bid fill: adverse
    r3 = P_adverse(_fill(event_index=2, side=Side.Bid), tr, ofi=0.0, queue_pos=0.1, h=1)
    assert r3["drift_ticks"] == 90.0 - 105.0 and r3["adverse"] is True
    # tape end -> every drift field is None
    r4 = P_adverse(_fill(event_index=4, side=Side.Ask), tr, ofi=0.0, queue_pos=0.0, h=1)
    assert r4["drift_ticks"] is None and r4["drift_bps"] is None and r4["adverse"] is None


# ---- (C) honest null: rw control shows CI straddling 0 ---- #


def test_null_on_rw_ci_straddles_zero():
    tr, fills = _replay(_RW)
    r = post_fill_drift(fills, tr, 5)
    assert r["n"] >= 100
    assert r["nw_lag"] == 4
    assert r["ci95"]["lo"] <= 0.0 <= r["ci95"]["hi"]
    assert r["p_value"] > 0.05
    assert abs(r["mean_ticks"]) < 0.5  # essentially zero, not a tiny tuned number


# ---- (D) overlapping horizons inflate the HAC s.e. ---- #


def test_hac_se_inflates_under_overlap():
    x = np.cumsum(np.random.default_rng(6).normal(0.0, 1.0, 2000))
    assert hac_se(x, lag=3) > hac_se(x, lag=0)


# ---- (E) FillRecords carry level size + the take's OFI (leak-free) ---- #


def test_fill_records_carry_level_and_ofi_reproducibly():
    _, fills1 = _replay(_RW)
    _, fills2 = _replay(_RW)
    assert len(fills1) == len(fills2) >= 100
    for a, b in zip(fills1, fills2):
        # deterministic replay -> identical adverse-selection population
        assert a.level_size_at_fill == b.level_size_at_fill
        assert a.ofi == b.ofi
        # level was non-empty at the fill (the take drew from a live level)
        assert a.level_size_at_fill > 0
    # the take's flow is a real (non-degenerate) signal, not a 0 placeholder
    nz = sum(1 for f in fills1 if f.ofi != 0.0)
    assert nz / len(fills1) > 0.9


# ---- (F) adverse_groups: partition + determinism + JSON ---- #


def test_adverse_groups_bins_partition_all_fills_and_json():
    tr, fills = _replay(_RW)
    res = adverse_groups(fills, tr, horizons=(1, 25))
    assert len(res["bins"]) == 2 * 3 * 3  # side x ofi_sign x queue tercile
    assert sum(b["n_fills"] for b in res["bins"]) == len(fills)
    assert res["horizons"] == [1, 25]
    keys = {(b["side"], b["ofi_sign"], b["queue_tercile"]) for b in res["bins"]}
    assert len(keys) == len(res["bins"])  # no duplicate cell
    json.dumps(res)  # fully JSON-encodable
    assert adverse_groups(fills, tr, horizons=(1, 25)) == res  # deterministic


# ---- (G) structural drift: drift control shows the adverse side-split ---- #


def test_structural_drift_positive_with_CI_excluding_zero():
    tr, fills = _replay(_DRIFT)
    r = post_fill_drift(fills, tr, 5)
    assert r["mean_ticks"] > 0
    assert r["ci95"]["lo"] > 0
    assert r["p_value"] < 0.01
    res = adverse_groups(fills, tr, horizons=(25,))
    ask = [b for b in res["bins"] if b["side"] == "ask" and b["n_fills"] > 0]
    bid = [b for b in res["bins"] if b["side"] == "bid" and b["n_fills"] > 0]
    # sellers picked off in a rising tape suffer adverse drift; resting buyers
    # are the ones who were early (their fills are NOT adverse)
    assert any(b["horizons"]["25"]["p_adverse"] > 0.8 for b in ask)
    assert any(b["horizons"]["25"]["mean_drift_bps"] > 0 for b in ask)
    assert all(b["horizons"]["25"]["p_adverse"] < 0.2 for b in bid)