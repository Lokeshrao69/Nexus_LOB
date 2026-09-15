"""Cross-day empirical aggregation engine for multi-day ITCH studies (Person B).

Aggregates per-day E1–E6 microstructure results across separate trading sessions:
- Per-day summary tables (regular session events, messages, orders, fills).
- Cross-day rank IC metrics: unweighted mean, sample-weighted mean, between-day standard deviation.
- Seeded bootstrap 95% confidence intervals for cross-day population metrics.
- Cross-day queue fill survival (KM P(fill)) and adverse-selection post-fill drift.
- Missing and partial day detection with explicit provenance tracking.
- Markdown summary report generation with honest empirical caveats.
- Session-level panel statistics (``session_panel_statistics``): per-session rows
  (one row per trading day × symbol), mean / median / between-session standard
  deviation, seeded bootstrap 95% CIs on the mean, sign consistency, paired
  per-session feature differences, and leave-one-session-out robustness.

Every per-session estimate is computed by the unchanged single-day pipeline
(``scripts/run_research.py``); this module only combines finished per-day
results and never pools event rows across days.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Cross-session statistics are reported for these per-session scalars (all
# taken from the unchanged per-day E1–E6 payloads).  ``sign`` is the direction
# each project hypothesis predicts; it is only used to report sign consistency.
PANEL_METRICS: tuple[tuple[str, str, str, int], ...] = (
    # (metric key, group, description, hypothesised sign)
    ("ic_lob_imbalance_h1", "E1", "test rank IC, L1 imbalance, h=1", +1),
    ("ic_lob_imbalance_h5", "E1", "test rank IC, L1 imbalance, h=5", +1),
    ("ic_lob_imbalance_h10", "E1", "test rank IC, L1 imbalance, h=10", +1),
    ("ic_lob_imbalance_h25", "E1", "test rank IC, L1 imbalance, h=25", +1),
    ("ic_microprice_off_h5", "E2", "test rank IC, microprice − mid, h=5", +1),
    ("ic_ofi_order_w20_h5", "E3", "test rank IC, order-level OFI (20-event), h=5", +1),
    ("ic_ofi_l2_h5", "E3", "test rank IC, L2-ladder OFI approximation, h=5", +1),
    ("ic_deep_imbalance_h5", "E1", "test rank IC, deep imbalance (k=5), h=5", +1),
    ("ic_spread_bps_h5", "E1", "test rank IC, spread (bps), h=5", 0),
    ("ic_combined_h5", "E4", "test rank IC, train-fit OLS combination, h=5", +1),
    ("km_p_fill_50", "E5", "KM P(first fill within 50 events)", 0),
    ("km_p_fill_500", "E5", "KM P(first fill within 500 events)", 0),
    ("ever_filled_rate", "E5", "share of regular-session orders ever filled", 0),
    ("logistic_calibration_slope", "E5", "logistic fill model calibration slope (hold-out)", 0),
    ("logistic_brier_skill", "E5", "logistic fill model Brier skill vs base rate", +1),
    ("adverse_drift_h1", "E6", "signed post-fill mid drift, h=1 (ticks; negative = adverse)", -1),
    ("adverse_drift_h5", "E6", "signed post-fill mid drift, h=5 (ticks)", -1),
    ("adverse_drift_h25", "E6", "signed post-fill mid drift, h=25 (ticks)", -1),
    ("adverse_p_adverse_h5", "E6", "P(adverse) after a passive fill, h=5", 0),
    ("adverse_post_minus_pre_h5", "E6", "post-fill minus matched pre-fill drift, h=5 (ticks)", -1),
)

# Paired per-session comparisons (same session, same test split): a − b.
PANEL_PAIRED: tuple[tuple[str, str, str, str], ...] = (
    ("microprice_minus_imbalance_h5", "ic_microprice_off_h5", "ic_lob_imbalance_h5", "E2: microprice − L1 imbalance rank IC, h=5"),
    ("ofi_order_minus_ofi_l2_h5", "ic_ofi_order_w20_h5", "ic_ofi_l2_h5", "E3: order-level OFI − L2 approximation rank IC, h=5"),
    ("combined_minus_imbalance_h5", "ic_combined_h5", "ic_lob_imbalance_h5", "E4: OLS combination − L1 imbalance rank IC, h=5"),
    ("post_minus_pre_drift_h5", "adverse_drift_h5", "adverse_pre_drift_h5", "E6: post-fill − pre-fill signed drift, h=5 (ticks)"),
)


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
        day_records: dict[str, list[DayStudyRecord]] = {}
        for r in records:
            if h in r.rank_ic_by_horizon:
                day_records.setdefault(r.day, []).append(r)

        unique_days_h = sorted(day_records.keys())
        n_unique_days = len(unique_days_h)
        if not unique_days_h:
            continue

        h_records = [r for d in unique_days_h for r in day_records[d]]
        vals = [r.rank_ic_by_horizon[h] for r in h_records]
        weights = [max(1, r.n_test_by_horizon.get(h, 1)) for r in h_records]
        v_arr = np.asarray(vals, dtype=np.float64)
        w_arr = np.asarray(weights, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)

        mean_unweighted = float(np.mean(v_arr))
        mean_weighted = float(np.sum(v_arr * w_norm))

        day_means = []
        for d in unique_days_h:
            d_recs = day_records[d]
            d_vals = np.asarray([r.rank_ic_by_horizon[h] for r in d_recs], dtype=np.float64)
            d_w = np.asarray([max(1, r.n_test_by_horizon.get(h, 1)) for r in d_recs], dtype=np.float64)
            d_m = float(np.sum(d_vals * d_w) / np.sum(d_w)) if np.sum(d_w) > 0 else float(np.mean(d_vals))
            day_means.append(d_m)

        std_between = float(np.std(day_means, ddof=1)) if n_unique_days > 1 else 0.0

        ci95: dict[str, float] | None = None
        if n_unique_days >= 2:
            rng = np.random.default_rng(seed + h)
            boot_means = np.empty(1000, dtype=np.float64)
            for b in range(1000):
                boot_days = rng.choice(unique_days_h, size=n_unique_days, replace=True)
                b_vals = []
                b_weights = []
                for bd in boot_days:
                    for r in day_records[bd]:
                        b_vals.append(r.rank_ic_by_horizon[h])
                        b_weights.append(max(1, r.n_test_by_horizon.get(h, 1)))
                bv_arr = np.asarray(b_vals, dtype=np.float64)
                bw_arr = np.asarray(b_weights, dtype=np.float64)
                boot_means[b] = float(np.sum(bv_arr * bw_arr) / np.sum(bw_arr))
            lo, hi = np.quantile(boot_means, [0.025, 0.975])
            ci95 = {"lo": round(float(lo), 6), "hi": round(float(hi), 6)}

        ic_summary[h] = {
            "n_days": n_unique_days,
            "n_symbol_sessions": len(vals),
            "mean_unweighted": round(mean_unweighted, 6),
            "mean_weighted": round(mean_weighted, 6),
            "std_between_days": round(std_between, 6),
            "ci95": ci95,
        }

    # Cross-day KM P(fill by 50) aggregation
    day_fill: dict[str, list[float]] = {}
    for r in records:
        if r.km_p_fill_50 is not None:
            day_fill.setdefault(r.day, []).append(r.km_p_fill_50)
    fill_day_means = [float(np.mean(vals)) for vals in day_fill.values()]
    fill_summary: dict[str, Any] = {
        "n_days": len(day_fill),
        "n_symbol_sessions": sum(len(v) for v in day_fill.values()),
        "mean": round(float(np.mean(fill_day_means)), 6) if fill_day_means else None,
        "std_between_days": round(float(np.std(fill_day_means, ddof=1)), 6) if len(fill_day_means) > 1 else 0.0,
    }

    # Cross-day adverse selection drift (h=5) aggregation
    day_drift: dict[str, list[float]] = {}
    for r in records:
        if r.adverse_drift_h5 is not None:
            day_drift.setdefault(r.day, []).append(r.adverse_drift_h5)
    drift_day_means = [float(np.mean(vals)) for vals in day_drift.values()]
    adverse_summary: dict[str, Any] = {
        "n_days": len(day_drift),
        "n_symbol_sessions": sum(len(v) for v in day_drift.values()),
        "mean": round(float(np.mean(drift_day_means)), 4) if drift_day_means else None,
        "std_between_days": round(float(np.std(drift_day_means, ddof=1)), 4) if len(drift_day_means) > 1 else 0.0,
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
        "independence_assumption": "calendar trading sessions (days) are statistically independent; multiple symbols within the same day are clustered and not independent",
        "bootstrap": {
            "kind": "date_cluster_bootstrap",
            "unit": "trading_day",
            "n_boot": 1000,
            "seed": seed,
        },
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


# ---------------------------------------------------------------------------
# Session panel statistics (one row per trading day × symbol)
# ---------------------------------------------------------------------------
def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ic_lookup(payload: dict[str, Any]) -> dict[tuple[str, int], tuple[float | None, int, float | None, float | None]]:
    """(feature, h) -> (test IC, n_test, per-day CI lo, per-day CI hi)."""
    out: dict[tuple[str, int], tuple[float | None, int, float | None, float | None]] = {}
    ic = payload.get("ic", {}) if isinstance(payload.get("ic"), dict) else {}
    for row in ic.get("rows", []):
        if isinstance(row, dict) and "feature" in row and "horizon_h" in row:
            out[(str(row["feature"]), int(row["horizon_h"]))] = (
                _finite(row.get("ic_test")), int(row.get("n_test") or 0), _finite(row.get("ci95_lo")), _finite(row.get("ci95_hi")),
            )
    for row in ic.get("combined", []):
        if isinstance(row, dict) and "horizon_h" in row:
            out[("combined", int(row["horizon_h"]))] = (
                _finite(row.get("ic_test")), int(row.get("n_test") or 0), _finite(row.get("ci95_lo")), _finite(row.get("ci95_hi")),
            )
    return out


def _weight_key(metric: str) -> str | None:
    """Sample-size field that weights ``metric`` in the weighted cross-session mean."""
    if metric.startswith("ic_"):
        return "n_" + metric
    if metric.startswith("km_") or metric == "ever_filled_rate":
        return "km_n"
    if metric.startswith("logistic_"):
        return "logistic_n_test"
    if metric.startswith("adverse_"):
        return "adverse_n_" + metric.rsplit("_", 1)[-1]
    return None


def _per_session_significance(rows: Sequence[dict[str, Any]], metric: str, sign: int) -> dict[str, int] | None:
    """How many sessions individually reject 0 in the hypothesised direction.

    IC metrics use the per-day block-bootstrap CI written by ``run_research.py``;
    adverse-drift metrics use the per-day Newey–West t-statistic (|t| > 1.96).
    """
    if sign == 0:
        return None
    n_avail = n_sig = 0
    if metric.startswith("ic_"):
        for r in rows:
            lo, hi = r.get(metric + "_ci_lo"), r.get(metric + "_ci_hi")
            if lo is None or hi is None:
                continue
            n_avail += 1
            if (sign > 0 and lo > 0) or (sign < 0 and hi < 0):
                n_sig += 1
    elif metric.startswith("adverse_drift_h"):
        t_key = metric.replace("adverse_drift_", "adverse_t_nw_")
        for r in rows:
            t = r.get(t_key)
            if t is None:
                continue
            n_avail += 1
            if abs(t) > 1.96 and np.sign(t) == sign:
                n_sig += 1
    else:
        return None
    return {"sessions_with_estimate": n_avail, "sessions_individually_significant": n_sig}


def extract_session_metrics(payload: dict[str, Any], *, day: str, symbol: str) -> dict[str, Any]:
    """Flatten one ``real_tape_<day>_<symbol>.json`` payload into panel scalars.

    Missing or non-finite values are ``None`` (never imputed).  ``n_*`` fields
    carry the sample size each scalar was estimated from so weighted statistics
    and sample-size caveats stay auditable.
    """
    tape = payload.get("tape", {}) if isinstance(payload.get("tape"), dict) else {}
    fill = payload.get("fill", {}) if isinstance(payload.get("fill"), dict) else {}
    km = fill.get("km", {}) if isinstance(fill.get("km"), dict) else {}
    logistic = fill.get("logistic", {}) if isinstance(fill.get("logistic"), dict) else {}
    adverse = payload.get("adverse", {}) if isinstance(payload.get("adverse"), dict) else {}
    horizons = adverse.get("horizons", {}) if isinstance(adverse.get("horizons"), dict) else {}
    ic = _ic_lookup(payload)

    row: dict[str, Any] = {
        "day": day,
        "symbol": symbol,
        "messages": int(tape.get("messages") or 0),
        "events": int(tape.get("events") or 0),
        "truncated": int(tape.get("truncated") or 0),
        "regular_events": int(tape.get("regular_events") or 0),
        "tracker_unknown_id": int(tape.get("tracker_unknown_id") or 0),
        "integrity_issues": tape.get("integrity_issues") or {},
        "n_orders_regular": int(fill.get("n_orders_regular") or 0),
        "n_passive_fills": int(adverse.get("n_fills") or 0),
    }
    for feature in ("lob_imbalance", "microprice_off", "deep_imbalance", "spread_bps", "ofi_l2", "ofi_order", "ofi_order_w20", "combined"):
        for h in (1, 5, 10, 25):
            val, n, lo, hi = ic.get((feature, h), (None, 0, None, None))
            row[f"ic_{feature}_h{h}"] = val
            row[f"n_ic_{feature}_h{h}"] = n
            row[f"ic_{feature}_h{h}_ci_lo"] = lo
            row[f"ic_{feature}_h{h}_ci_hi"] = hi

    tau = list(km.get("tau_events", []))
    probs = list(km.get("p_fill", []))
    for t in (10, 50, 100, 500, 1000, 5000):
        val = None
        if t in tau and tau.index(t) < len(probs):
            val = _finite(probs[tau.index(t)])
        elif float(t) in tau and tau.index(float(t)) < len(probs):
            val = _finite(probs[tau.index(float(t))])
        row[f"km_p_fill_{t}"] = val
    n_km = int(km.get("n") or 0)
    n_fill = int(km.get("n_fill") or 0)
    row["km_n"] = n_km
    row["ever_filled_rate"] = (n_fill / n_km) if n_km > 0 else None
    row["logistic_calibration_slope"] = _finite(logistic.get("calibration_slope"))
    row["logistic_brier_skill"] = _finite(logistic.get("brier_skill"))
    row["logistic_base_rate"] = _finite(logistic.get("base_rate"))
    row["logistic_n_test"] = int(logistic.get("n_test") or 0)

    for h in (1, 5, 25):
        blk = horizons.get(str(h)) or horizons.get(h) or {}
        overall = blk.get("overall", {}) if isinstance(blk, dict) else {}
        pre = blk.get("pre_fill", {}) if isinstance(blk, dict) else {}
        pmp = blk.get("post_minus_pre", {}) if isinstance(blk, dict) else {}
        row[f"adverse_drift_h{h}"] = _finite(overall.get("mean_drift"))
        row[f"adverse_t_nw_h{h}"] = _finite(overall.get("t_nw"))
        row[f"adverse_p_adverse_h{h}"] = _finite(overall.get("p_adverse"))
        row[f"adverse_pre_drift_h{h}"] = _finite(pre.get("mean_drift"))
        row[f"adverse_post_minus_pre_h{h}"] = _finite(pmp.get("mean_drift"))
        row[f"adverse_n_h{h}"] = int(overall.get("n") or 0)
    return row


def _describe(
    values: Sequence[float | None],
    *,
    seed: int,
    n_boot: int = 2000,
    sign: int = 0,
    weights: Sequence[float | None] | None = None,
    days: Sequence[str] | None = None,
) -> dict[str, Any]:
    keep = [i for i, v in enumerate(values) if v is not None]
    x = np.asarray([values[i] for i in keep], dtype=np.float64)
    n = int(x.size)
    if days is not None:
        keep_days = [str(days[i]) for i in keep]
        unique_days = sorted(set(keep_days))
        n_days = len(unique_days)
    else:
        keep_days = [str(i) for i in keep]
        unique_days = sorted(set(keep_days))
        n_days = n

    out: dict[str, Any] = {"n_sessions": n, "n_days": n_days, "independence_unit": "trading_day"}
    if n == 0 or n_days == 0:
        out.update({"mean": None, "median": None, "std_between_sessions": None, "min": None, "max": None, "ci95": None})
        return out
    out["mean"] = float(np.mean(x))
    w = None
    if weights is not None:
        w = np.asarray([float(weights[i] or 0.0) for i in keep], dtype=np.float64)
        out["mean_weighted"] = float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else None
        out["weight_total"] = float(np.sum(w))
    out["median"] = float(np.median(x))
    out["min"] = float(np.min(x))
    out["max"] = float(np.max(x))

    # Compute date-level values for between-session dispersion:
    day_indices = {d: [idx for idx, day_val in enumerate(keep_days) if day_val == d] for d in unique_days}
    day_means = []
    for d in unique_days:
        idx = day_indices[d]
        if w is not None and np.sum(w[idx]) > 0:
            day_means.append(float(np.sum(x[idx] * w[idx]) / np.sum(w[idx])))
        else:
            day_means.append(float(np.mean(x[idx])))
    day_means_arr = np.asarray(day_means, dtype=np.float64)

    out["std_between_sessions"] = float(np.std(day_means_arr, ddof=1)) if n_days > 1 else 0.0

    if n_days >= 2:
        # Trading-date cluster bootstrap: resample unique_days with replacement
        rng = np.random.default_rng(seed)
        boot_means = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            boot_days = rng.choice(unique_days, size=n_days, replace=True)
            boot_idx = []
            for bd in boot_days:
                boot_idx.extend(day_indices[bd])
            bx = x[boot_idx]
            if w is not None and np.sum(w[boot_idx]) > 0:
                bw = w[boot_idx]
                boot_means[b] = float(np.sum(bx * bw) / np.sum(bw))
            else:
                boot_means[b] = float(np.mean(bx))
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        out["ci95"] = {"lo": float(lo), "hi": float(hi)}
        se = float(np.std(boot_means))
        out["se_mean"] = se
        out["t_stat_mean_zero"] = (out["mean"] / se) if se > 0 else None
    else:
        out["ci95"] = None
        out["se_mean"] = 0.0
        out["t_stat_mean_zero"] = None

    if sign != 0:
        day_signs = np.sign(day_means_arr)
        agree = int(np.sum(day_signs == sign))
        out["hypothesised_sign"] = sign
        out["sessions_with_hypothesised_sign"] = agree
        out["sign_consistency"] = agree / n_days if n_days > 0 else 0.0
    return out


def _leave_one_out(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Mean of ``key`` when each *day* (all its symbols) is held out — the outlier check."""
    days = sorted({r["day"] for r in rows if r.get(key) is not None})
    if len(days) < 3:
        return None
    full = [float(r[key]) for r in rows if r.get(key) is not None]
    full_mean = float(np.mean(full))
    by_day: dict[str, float] = {}
    for d in days:
        kept = [float(r[key]) for r in rows if r.get(key) is not None and r["day"] != d]
        by_day[d] = float(np.mean(kept))
    shifts = {d: m - full_mean for d, m in by_day.items()}
    most = max(shifts, key=lambda d: abs(shifts[d]))
    return {
        "full_mean": full_mean,
        "leave_one_day_out_mean_range": [float(min(by_day.values())), float(max(by_day.values()))],
        "most_influential_day": most,
        "most_influential_shift": shifts[most],
        "sign_stable_under_leave_one_out": all(np.sign(m) == np.sign(full_mean) for m in by_day.values()) if full_mean != 0 else None,
    }


def session_panel_statistics(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 0x51ED,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Cross-session statistics over per-session rows from ``extract_session_metrics``.

    Statistics are computed per symbol and pooled across symbols. To prevent
    spurious statistical precision, the unit of inference is the trading day:
    cross-session bootstrap CIs resample trading days (date-level cluster bootstrap),
    never treating multiple symbols from the same date as independent sessions.
    With fewer than roughly 8 sessions the intervals are indicative only and the
    ``t_stat_mean_zero`` reference is reported alongside. Nothing is dropped
    for being unfavourable: every finite per-session value enters.
    """
    rows = [dict(r) for r in rows]
    symbols = sorted({r["symbol"] for r in rows})
    days = sorted({r["day"] for r in rows})
    metrics: dict[str, Any] = {}
    for i, (key, group, description, sign) in enumerate(PANEL_METRICS):
        wk = _weight_key(key)
        entry: dict[str, Any] = {"group": group, "description": description, "weight_key": wk, "pooled": None, "by_symbol": {}}
        entry["pooled"] = _describe(
            [r.get(key) for r in rows], seed=seed + 7 * i, n_boot=n_boot, sign=sign,
            weights=[r.get(wk) for r in rows] if wk else None,
            days=[r["day"] for r in rows],
        )
        entry["pooled"]["leave_one_day_out"] = _leave_one_out(rows, key)
        entry["pooled"]["per_session_significance"] = _per_session_significance(rows, key, sign)
        for j, sym in enumerate(symbols):
            sub_rows = [r for r in rows if r["symbol"] == sym]
            entry["by_symbol"][sym] = _describe(
                [r.get(key) for r in sub_rows], seed=seed + 7 * i + 1000 * (j + 1), n_boot=n_boot, sign=sign,
                weights=[r.get(wk) for r in sub_rows] if wk else None,
                days=[r["day"] for r in sub_rows],
            )
            entry["by_symbol"][sym]["per_session_significance"] = _per_session_significance(sub_rows, key, sign)
        metrics[key] = entry

    paired: dict[str, Any] = {}
    for i, (key, a, b, description) in enumerate(PANEL_PAIRED):
        diffs = []
        diff_days = []
        for r in rows:
            va, vb = r.get(a), r.get(b)
            if va is not None and vb is not None:
                d = float(va) - float(vb)
                r[key] = d
                diffs.append(d)
                diff_days.append(r["day"])
        stats = _describe(diffs, seed=seed + 31 * (i + 1), n_boot=n_boot, days=diff_days)
        stats["sessions_a_greater"] = int(sum(1 for d in diffs if d > 0))
        stats["sessions_b_greater"] = int(sum(1 for d in diffs if d < 0))
        by_symbol = {}
        for j, sym in enumerate(symbols):
            sub = [r.get(key) for r in rows if r["symbol"] == sym and r.get(key) is not None]
            sub_days = [r["day"] for r in rows if r["symbol"] == sym and r.get(key) is not None]
            by_symbol[sym] = _describe(sub, seed=seed + 31 * (i + 1) + 1000 * (j + 1), n_boot=n_boot, days=sub_days)
        paired[key] = {"a": a, "b": b, "description": description, "pooled": stats, "by_symbol": by_symbol}

    return {
        "n_sessions": len(rows),
        "n_days": len(days),
        "days": days,
        "symbols": symbols,
        "bootstrap": {
            "kind": "date_cluster_bootstrap",
            "unit": "trading_day",
            "independence_assumption": "trading days are statistically independent; multiple symbols within the same date are clustered and not independent",
            "n_boot": n_boot,
            "seed": seed,
        },
        "metrics": metrics,
        "paired": paired,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def render_session_panel_md(panel: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    """Markdown tables for the session panel: per-session rows, then cross-session statistics."""
    lines = [
        "## Session panel (one row per trading day × symbol)",
        "",
        f"{panel['n_sessions']} sessions over {panel['n_days']} trading days ({', '.join(panel['days'])}); symbols {', '.join(panel['symbols'])}.",
        "",
        "| Day | Sym | Regular rows | Orders (E5) | Passive fills (E6) | IC h=1 | IC h=5 | IC h=10 | IC h=25 | OLS h=5 | KM P(fill 50) | Ever filled | Calib. slope | Drift h=5 (ticks) | P(adv) h=5 | Post−pre h=5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r["day"][4:], r["day"][:4], r["symbol"])):
        lines.append(
            f"| {r['day']} | {r['symbol']} | {r['regular_events']:,} | {r['n_orders_regular']:,} | {r['n_passive_fills']:,} | "
            f"{_fmt(r.get('ic_lob_imbalance_h1'), 3)} | {_fmt(r.get('ic_lob_imbalance_h5'), 3)} | {_fmt(r.get('ic_lob_imbalance_h10'), 3)} | "
            f"{_fmt(r.get('ic_lob_imbalance_h25'), 3)} | {_fmt(r.get('ic_combined_h5'), 3)} | {_fmt(r.get('km_p_fill_50'), 4)} | "
            f"{_fmt(r.get('ever_filled_rate'), 4)} | {_fmt(r.get('logistic_calibration_slope'), 2)} | {_fmt(r.get('adverse_drift_h5'), 1)} | "
            f"{_fmt(r.get('adverse_p_adverse_h5'), 3)} | {_fmt(r.get('adverse_post_minus_pre_h5'), 1)} |"
        )
    lines += [
        "",
        "## Cross-session statistics",
        "",
        (
            "Bootstrap CIs resample trading days (date-level cluster bootstrap); symbols from the same date are clustered; "
            "`sign` = share of trading days whose estimate has the hypothesised sign; "
            "`LOO range` = range of the pooled mean when each trading day (all symbols) is left out."
        ),
        "",
        "| Metric | Group | N | Mean | Weighted mean | Median | Between-session SD | 95% bootstrap CI | t (mean=0) | Sign consistency | Per-session significant | LOO mean range | Most influential day |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|---|",
    ]
    for m in panel["metrics"].values():
        p = m["pooled"]
        ci = p.get("ci95")
        loo = p.get("leave_one_day_out") or {}
        sig = p.get("per_session_significance")
        sign = (
            f"{p['sessions_with_hypothesised_sign']}/{p.get('n_days', p['n_sessions'])}" if "sign_consistency" in p else "—"
        )
        loo_range = (
            f"[{_fmt(loo['leave_one_day_out_mean_range'][0])}, {_fmt(loo['leave_one_day_out_mean_range'][1])}]" if loo else "—"
        )
        lines.append(
            f"| {m['description']} | {m['group']} | {p['n_sessions']} | {_fmt(p['mean'])} | {_fmt(p.get('mean_weighted'))} | {_fmt(p['median'])} | "
            f"{_fmt(p['std_between_sessions'])} | {('[' + _fmt(ci['lo']) + ', ' + _fmt(ci['hi']) + ']') if ci else '—'} | "
            f"{_fmt(p.get('t_stat_mean_zero'), 2)} | {sign} | "
            f"{(str(sig['sessions_individually_significant']) + '/' + str(sig['sessions_with_estimate'])) if sig else '—'} | "
            f"{loo_range} | {loo.get('most_influential_day', '—') if loo else '—'} |"
        )
    lines += [
        "",
        "### Per-symbol means",
        "",
        "| Metric | " + " | ".join(f"{s} mean [95% CI] (N)" for s in panel["symbols"]) + " |",
        "|---|" + "---|" * len(panel["symbols"]),
    ]
    for m in panel["metrics"].values():
        cells = []
        for s in panel["symbols"]:
            b = m["by_symbol"].get(s, {})
            ci = b.get("ci95")
            cells.append(
                f"{_fmt(b.get('mean'))} {('[' + _fmt(ci['lo']) + ', ' + _fmt(ci['hi']) + ']') if ci else ''} ({b.get('n_sessions', 0)})"
            )
        lines.append(f"| {m['description']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired per-session comparisons (a − b on the same session)",
        "",
        "| Comparison | N | Mean diff | Median | Between-session SD | 95% bootstrap CI | Sessions a>b | Sessions b>a |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for m in panel["paired"].values():
        p = m["pooled"]
        ci = p.get("ci95")
        lines.append(
            f"| {m['description']} | {p['n_sessions']} | {_fmt(p['mean'])} | {_fmt(p['median'])} | {_fmt(p['std_between_sessions'])} | "
            f"{('[' + _fmt(ci['lo']) + ', ' + _fmt(ci['hi']) + ']') if ci else '—'} | {p['sessions_a_greater']} | {p['sessions_b_greater']} |"
        )
    lines.append("")
    return "\n".join(lines)
