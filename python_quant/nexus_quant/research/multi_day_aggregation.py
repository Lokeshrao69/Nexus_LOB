"""Cross-day empirical aggregation engine for multi-day ITCH studies (Person B).

Aggregates per-day E1–E6 microstructure results across separate trading sessions:
- Per-day summary tables (regular session events, messages, orders, fills).
- Cross-day rank IC metrics: unweighted mean, sample-weighted mean, between-day standard deviation.
- Seeded bootstrap 95% confidence intervals for cross-day population metrics.
- Cross-day queue fill survival (KM P(fill)) and adverse-selection post-fill drift.
- Missing and partial day detection with explicit provenance tracking.
- Markdown summary report generation with honest empirical caveats.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .experiments import bootstrap_ci


@dataclass(frozen=True, slots=True)
class DayStudyRecord:
    """Normalized summary of one day's E1–E6 empirical study."""

    day: str
    symbol: str
    coverage_kind: str
    is_full_day: bool
    regular_events: int
    total_messages: int
    rank_ic_by_horizon: dict[int, float]
    n_test_by_horizon: dict[int, int]
    n_orders_regular: int
    km_p_fill_50: float | None
    n_passive_fills: int
    adverse_drift_h5: float | None


def extract_day_study_record(payload: dict[str, Any], *, default_day: str = "unknown", default_symbol: str = "AAPL") -> DayStudyRecord:
    """Extract standard DayStudyRecord from a real_tape_<day>_<symbol>.json result."""
    day = str(payload.get("day", default_day))
    symbol = str(payload.get("symbol", default_symbol))
    coverage = payload.get("coverage", {})
    cov_kind = str(coverage.get("kind", "full_day" if payload.get("is_full_day", True) else "partial"))
    is_full = bool(coverage.get("is_full_day", True))

    tape = payload.get("tape", {})
    reg_events = int(tape.get("regular_events", 0))
    tot_msgs = int(tape.get("messages", 0))

    ic_rows = payload.get("ic", {}).get("rows", [])
    rank_ic_by_h: dict[int, float] = {}
    n_test_by_h: dict[int, int] = {}
    for r in ic_rows:
        if isinstance(r, dict) and r.get("feature") == "lob_imbalance":
            h = int(r.get("horizon_h", 0))
            ic_val = r.get("ic_test")
            n_val = r.get("n_test", 0)
            if ic_val is not None and math.isfinite(float(ic_val)):
                rank_ic_by_h[h] = float(ic_val)
                n_test_by_h[h] = int(n_val)

    fill = payload.get("fill", {})
    n_orders = int(fill.get("n_orders_regular", 0))
    km = fill.get("km", {})
    tau_list = km.get("tau_events", [])
    p_fill_list = km.get("p_fill", [])
    p_fill_50: float | None = None
    if 50 in tau_list:
        idx = tau_list.index(50)
        p_fill_50 = float(p_fill_list[idx])

    adverse = payload.get("adverse", {})
    n_fills = int(adverse.get("n_fills", 0))
    h5_dict = adverse.get("horizons", {}).get("5") or adverse.get("horizons", {}).get(5, {})
    drift_h5 = None
    if isinstance(h5_dict, dict) and "overall" in h5_dict:
        m = h5_dict["overall"].get("mean_drift")
        if m is not None and math.isfinite(float(m)):
            drift_h5 = float(m)

    return DayStudyRecord(
        day=day,
        symbol=symbol,
        coverage_kind=cov_kind,
        is_full_day=is_full,
        regular_events=reg_events,
        total_messages=tot_msgs,
        rank_ic_by_horizon=rank_ic_by_h,
        n_test_by_horizon=n_test_by_h,
        n_orders_regular=n_orders,
        km_p_fill_50=p_fill_50,
        n_passive_fills=n_fills,
        adverse_drift_h5=drift_h5,
    )


def aggregate_multi_day_results(
    day_payloads: Sequence[dict[str, Any] | DayStudyRecord],
    *,
    seed: int = 0x51ED,
) -> dict[str, Any]:
    """Perform cross-day aggregation of separate single-day studies.

    Strict separation:
    - Between-day standard deviation measures dispersion across sessions.
    - Sample-weighted mean accounts for differing session event counts.
    - Bootstrap CIs resample across days to estimate cross-day uncertainty.
    """
    if not day_payloads:
        raise ValueError("aggregate_multi_day_results requires at least one day payload")

    records: list[DayStudyRecord] = []
    missing_days: list[dict[str, str]] = []

    for item in day_payloads:
        if isinstance(item, DayStudyRecord):
            rec = item
        elif isinstance(item, dict):
            if "error" in item:
                missing_days.append({"day": str(item.get("day", "unknown")), "reason": str(item["error"])})
                continue
            rec = extract_day_study_record(item)
        else:
            continue

        if rec.regular_events <= 0 and not rec.is_full_day:
            missing_days.append({"day": rec.day, "reason": f"no regular session events ({rec.coverage_kind})"})
            continue
        records.append(rec)

    days = sorted({r.day for r in records})
    symbols = sorted({r.symbol for r in records})
    total_reg_events = sum(r.regular_events for r in records)
    total_orders = sum(r.n_orders_regular for r in records)
    total_passive_fills = sum(r.n_passive_fills for r in records)

    # Cross-day IC aggregation across common horizons (1, 5, 10, 25)
    horizons = sorted({h for r in records for h in r.rank_ic_by_horizon})
    ic_summary: dict[int, dict[str, Any]] = {}

    for h in horizons:
        vals = [r.rank_ic_by_horizon[h] for r in records if h in r.rank_ic_by_horizon]
        weights = [max(1, r.n_test_by_horizon.get(h, 1)) for r in records if h in r.rank_ic_by_horizon]
        if not vals:
            continue
        v_arr = np.asarray(vals, dtype=np.float64)
        w_arr = np.asarray(weights, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)

        mean_unweighted = float(np.mean(v_arr))
        mean_weighted = float(np.sum(v_arr * w_norm))
        std_between = float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 0.0

        ci95: dict[str, float] | None = None
        if len(v_arr) >= 2:
            ci = bootstrap_ci(v_arr, kind="iid", n_boot=1000, seed=seed + h)
            ci95 = {"lo": round(ci["lo"], 6), "hi": round(ci["hi"], 6)}

        ic_summary[h] = {
            "n_days": len(vals),
            "mean_unweighted": round(mean_unweighted, 6),
            "mean_weighted": round(mean_weighted, 6),
            "std_between_days": round(std_between, 6),
            "ci95": ci95,
        }

    # Cross-day KM P(fill by 50) aggregation
    fill_vals = [r.km_p_fill_50 for r in records if r.km_p_fill_50 is not None]
    fill_summary: dict[str, Any] = {
        "n_days": len(fill_vals),
        "mean": round(float(np.mean(fill_vals)), 6) if fill_vals else None,
        "std_between_days": round(float(np.std(fill_vals, ddof=1)), 6) if len(fill_vals) > 1 else 0.0,
    }

    # Cross-day adverse selection drift (h=5) aggregation
    drift_vals = [r.adverse_drift_h5 for r in records if r.adverse_drift_h5 is not None]
    adverse_summary: dict[str, Any] = {
        "n_days": len(drift_vals),
        "mean": round(float(np.mean(drift_vals)), 4) if drift_vals else None,
        "std_between_days": round(float(np.std(drift_vals, ddof=1)), 4) if len(drift_vals) > 1 else 0.0,
    }

    per_day_table: list[dict[str, Any]] = []
    for r in records:
        per_day_table.append({
            "day": r.day,
            "symbol": r.symbol,
            "coverage": r.coverage_kind,
            "is_full_day": r.is_full_day,
            "regular_events": r.regular_events,
            "total_messages": r.total_messages,
            "rank_ic_h5": r.rank_ic_by_horizon.get(5),
            "rank_ic_h10": r.rank_ic_by_horizon.get(10),
            "n_orders": r.n_orders_regular,
            "km_p_fill_50": r.km_p_fill_50,
            "n_passive_fills": r.n_passive_fills,
            "adverse_drift_h5": r.adverse_drift_h5,
        })

    policy_statement = (
        f"Aggregated {len(days)} calendar sessions across {len(symbols)} symbol(s). "
        f"Between-day dispersion reflects genuine session variation; "
        + ("cross-day sample supports preliminary multi-day inference." if len(days) >= 5
           else "cross-day sample size is small (< 5 days); metrics remain descriptive rather than asymptotic.")
    )

    return {
        "n_days": len(days),
        "days": days,
        "symbols": symbols,
        "total_regular_events": total_reg_events,
        "total_orders": total_orders,
        "total_passive_fills": total_passive_fills,
        "per_day": per_day_table,
        "rank_ic_by_horizon": ic_summary,
        "km_fill_50": fill_summary,
        "adverse_drift_h5": adverse_summary,
        "missing_days": missing_days,
        "policy_statement": policy_statement,
    }


def render_multi_day_aggregation_md(summary: dict[str, Any]) -> str:
    """Render a GitHub-flavored Markdown report from aggregated multi-day results."""
    lines = [
        "# Multi-Day ITCH Empirical Aggregation Report",
        "",
        f"> **Policy**: {summary.get('policy_statement', '')}",
        "",
        "## Summary Overview",
        "",
        f"- **Days analyzed**: {summary.get('n_days', 0)} ({', '.join(summary.get('days', [])) or 'none'})",
        f"- **Symbols**: {', '.join(summary.get('symbols', [])) or 'none'}",
        f"- **Total regular-session events**: {summary.get('total_regular_events', 0):,}",
        f"- **Total standing orders tracked (E5)**: {summary.get('total_orders', 0):,}",
        f"- **Total passive fills analyzed (E6)**: {summary.get('total_passive_fills', 0):,}",
        "",
        "## Per-Day Session Breakdown",
        "",
        "| Day | Symbol | Coverage | Regular Events | Rank IC (h=5) | Rank IC (h=10) | Orders | KM P(fill 50) | Adverse Drift (h=5) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summary.get("per_day", []):
        ic5 = f"{r.get('rank_ic_h5'):.4f}" if r.get("rank_ic_h5") is not None else "—"
        ic10 = f"{r.get('rank_ic_h10'):.4f}" if r.get("rank_ic_h10") is not None else "—"
        km50 = f"{r.get('km_p_fill_50'):.4f}" if r.get('km_p_fill_50') is not None else "—"
        adv5 = f"{r.get('adverse_drift_h5'):.2f}" if r.get('adverse_drift_h5') is not None else "—"
        lines.append(
            f"| {r.get('day')} | {r.get('symbol')} | {r.get('coverage')} | {r.get('regular_events'):,} | "
            f"{ic5} | {ic10} | {r.get('n_orders'):,} | {km50} | {adv5} |"
        )

    lines += [
        "",
        "## Cross-Day Signal Metrics (LOB Imbalance Rank IC)",
        "",
        "| Horizon (h) | Sessions | Unweighted Mean | Weighted Mean | Between-Day Std | 95% Bootstrap CI |",
        "|---:|---:|---:|---:|---:|---|",
    ]

    for h, m in sorted(summary.get("rank_ic_by_horizon", {}).items()):
        ci_str = f"[{m['ci95']['lo']:.4f}, {m['ci95']['hi']:.4f}]" if m.get("ci95") else "—"
        lines.append(
            f"| {h} | {m.get('n_days')} | {m.get('mean_unweighted'):.4f} | {m.get('mean_weighted'):.4f} | "
            f"{m.get('std_between_days'):.4f} | {ci_str} |"
        )

    if summary.get("missing_days"):
        lines += [
            "",
            "## Missing / Incomplete Sessions",
            "",
            "| Day | Reason / Status |",
            "|---|---|",
        ]
        for m in summary["missing_days"]:
            lines.append(f"| {m.get('day')} | {m.get('reason')} |")

    lines.append("")
    return "\n".join(lines)
