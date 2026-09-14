#!/usr/bin/env python3
"""Synthetic E5/E6 vignette (plan.md WS-5) — fill probability + adverse selection.

Computes the two Phase-3 experiment arms on the documented synthetic control
flows, NOT tuned to any target:

* **E5 — fill probability** (`fill_dataset` → `LogisticFillModel`): a
  time-ordered train/test split (walk-forward discipline) of controlled
  hypothetical-touch drops; reports the out-of-sample calibration slope and
  Brier vs the base-rate-only baseline.
* **E6 — adverse selection** (`adverse_groups` / `post_fill_drift`): signed
  mean mid-drift over the h ∈ {1,5,25} events after each real passive fill,
  with block-bootstrap CIs, split by fill side (ask/bid).

READ THE RESULT HONESTLY: `FLOW_PRESETS.rw` is a pure random walk — the E6
null is CI straddling zero. `FLOW_PRESETS.drift` is a structural up-trend —
E6 should show ask fills (resting sellers) suffering positive adverse drift
while bid fills are the ones that were early. E5 is NOT a null: a synthetic
queue with FIFO priority genuinely makes near-touch fills more likely than
deep ones, so a calibrated slope near 1 is the harness working as designed,
not a tuned headline. Claims about real markets still wait for the Phase-4
NASDAQ tape.

One honest harness limitation recorded (not papered over): the synthetic
generator always picks the queue front to take, so every recorded fill has
``ahead_at_fill == 0`` — the queue-position conditioning in ``adverse_groups``
collapses to a single tercile on this tape. The side × OFI conditioning is
well-identified; queue-position conditioning needs the real tape.

Run (from repo root):
    python python_quant/scripts/queue_adverse_vignette.py --out results.json

This script inserts ``python_quant/`` on ``sys.path`` itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safe
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ``python_quant/`` contains ``nexus_quant/`` — insert that dir directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from nexus_quant.research import (
    FlowConfig,
    LogisticFillModel,
    SyntheticFlow,
    adverse_groups,
    brier_score,
    calibration_curve,
    fill_dataset,
    post_fill_drift,
)
from nexus_quant.research.queue_dynamics import QueueTracker

HORIZONS = (1, 5, 25)


def _flows(n_events: int):
    """(name, FlowConfig) control arms — the documented presets, resized only."""
    return [
        ("rw", FlowConfig(n_events=n_events)),
        ("drift", FlowConfig(n_events=n_events, drift_ticks=0.6, vol_per_event=0.6)),
    ]


def _e5(flow: SyntheticFlow) -> dict:
    rows = fill_dataset(flow, each_tau_drop=40, qty=100, window=200)
    if not rows:
        return {"error": "no decision rows"}
    # Rows are emitted in tape (decision_index) order → time-ordered split.
    cut = max(1, int(0.7 * len(rows)))
    train, test = rows[:cut], rows[cut:]
    model = LogisticFillModel.fit(train, feature_names=sorted(train[0].features))
    X = np.asarray(
        [[r.features[n] for n in model.feature_names] for r in test], dtype=np.float64
    )
    y = np.asarray([1.0 if r.filled else 0.0 for r in test], dtype=np.float64)
    p = model.predict_proba(X)
    base = float(y.mean())
    return {
        "n_train": len(train),
        "n_test": len(test),
        "base_rate": base,
        "base_brier": float(base * (1.0 - base)),  # constant-prediction baseline
        "model_brier": float(brier_score(y, p)),
        "calib_slope": calibration_curve(y, p, n_bins=10)["slope"],
        "calib_intercept": calibration_curve(y, p, n_bins=10)["intercept"],
    }


def _e6(flow: SyntheticFlow) -> dict:
    tracker = QueueTracker()
    for ev in flow.generate():
        tracker.on_event(ev)
    fills = list(tracker.fills)
    per_h: dict[str, dict] = {}
    for h in HORIZONS:
        r = post_fill_drift(fills, tracker, h)
        per_h[str(h)] = {k: r[k] for k in ("n", "mean_ticks", "mean_bps", "ci95",
                                           "t_stat", "p_value")}
    groups = adverse_groups(fills, tracker, horizons=HORIZONS)
    sides: dict[str, dict] = {"ask": {"n": 0, "p_adverse_cells": []},
                              "bid": {"n": 0, "p_adverse_cells": []}}
    for b in groups["bins"]:
        if b["n_fills"] == 0:
            continue
        side = sides[b["side"]]
        side["n"] += b["n_fills"]
        pa = b["horizons"]["25"]["p_adverse"]
        if pa is not None:
            side["p_adverse_cells"].append(pa)
    side_means = {
        side: {
            "n": agg["n"],
            "mean_p_adverse": float(np.mean(agg["p_adverse_cells"]))
            if agg["p_adverse_cells"] else None,
        }
        for side, agg in sides.items()
    }
    return {
        "n_fills": len(fills),
        "queue_tercile_edges": groups["queue_tercile_edges"],
        "per_h": per_h,
        "side_means_h25": side_means,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=30_000,
                    help="per-flow event budget (smaller = faster)")
    ap.add_argument("--out", type=str, default=None, help="write JSON here")
    args = ap.parse_args()

    results: dict = {"flows": {}}
    for name, cfg in _flows(args.n_events):
        print(f"\n=== flow: {name} (n_events={cfg.n_events}) ===")
        flow = SyntheticFlow(cfg)
        flow.generate()

        e5 = _e5(flow)
        results["flows"][name] = {"e5": e5}
        print("[E5] fill probability (out-of-sample calibration)")
        if "error" in e5:
            print("  " + e5["error"])
        else:
            print(f"  n_train={e5['n_train']} n_test={e5['n_test']} "
                  f"base_rate={e5['base_rate']:.3f}")
            print(f"  model_brier={e5['model_brier']:.4f} "
                  f"base_brier={e5['base_brier']:.4f} "
                  f"calib_slope={e5['calib_slope']:.3f}")

        e6 = _e6(flow)
        results["flows"][name]["e6"] = e6
        print("[E6] adverse drift after passive fills (h = 1/5/25) — ci95 in ticks")
        for h in HORIZONS:
            r = e6["per_h"][str(h)]
            ci = r["ci95"]
            print(f"  h={h:2d} mean_ticks={r['mean_ticks']:8.3f} "
                  f"ci95_ticks=({ci['lo']:8.3f},{ci['hi']:8.3f}) "
                  f"mean_bps={r['mean_bps']:8.3f} p={r['p_value']:.4f} n={r['n']}")
        for side, agg in e6["side_means_h25"].items():
            pav = "n/a" if agg["mean_p_adverse"] is None else f"{agg['mean_p_adverse']:.3f}"
            print(f"  side={side:>3s} mean_p_adverse(h=25)={pav} n={agg['n']}")

        null = e6["per_h"]["25"]
        read = ("null — CI straddles 0 (random-walk arm behaves)"
                if null["ci95"]["lo"] <= 0.0 <= null["ci95"]["hi"]
                else "structural drift — CI excludes 0 (ask adverse on an up-trend)")
        print(f"  [E6 read] {read}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()