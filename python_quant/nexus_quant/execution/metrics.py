"""Execution metrics (plan_2.md Phase 2) — the honest measurement layer.

The Phase-1 flaw (plan_2.md §5.9): "shortfall vs self VWAP". Every strategy
that manipulates its own fills can game a self-VWAP benchmark. So:

  * ``vwap_slippage`` benchmarks against the **market** VWAP of the tape, not
    the strategy's own execution VWAP;
  * ``implementation_shortfall`` uses the arrival mid (observable at ``t``),
    never a look-ahead reference.

Sign convention (identical to the env's ``info["shortfall_bps"]``): every
metric reads **positive = a cost** for the strategy — sold below the benchmark
(sell) or bought above it (buy). ``side`` is +1 sell, −1 buy.

All angles are in basis points of the benchmark mid / market VWAP so numbers
are portable across tick sizes. Fills are sequences of ``(price_tick, size)``
(or ``(t, price_tick, size)`` — the shape of the env's ``fills`` list).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


def _fills_as(fills: Sequence[object]) -> list[tuple[int, int]]:
    """Normalize (price, size) pairs — accept (px, sz) or (t, px, sz) shapes."""
    out: list[tuple[int, int]] = []
    for f in fills:
        if isinstance(f, (tuple, list)):
            if len(f) == 2:
                out.append((int(f[0]), int(f[1])))
            elif len(f) == 3:
                out.append((int(f[1]), int(f[2])))
    return out


def execution_vwap(fills: Sequence[object]) -> float:
    """The strategy's own volume-weighted average fill price (ticks). Returns 0
    for no fills. Used *internally* — never as the benchmark."""
    px_sz = _fills_as(fills)
    qty = sum(sz for _, sz in px_sz)
    if qty <= 0:
        return 0.0
    return sum(px * sz for px, sz in px_sz) / qty


def _market_vwap_guard(fills: Sequence[object], market_vwap: float) -> float:
    """0 unless the strategy has fills and the tape has a market VWAP."""
    px_sz = _fills_as(fills)
    qty = sum(sz for _, sz in px_sz)
    if qty <= 0 or market_vwap <= 0:
        return 0.0
    return market_vwap


def vwap_slippage(fills: Sequence[object], market_vwap: float, *, side: int = 1) -> float:
    """Slippage of the strategy's execution VWAP vs the **market** VWAP (bps).

    Positive = a cost. ``side`` +1 sell, −1 buy:

        sell: bps = (market_vwap − execution_vwap) / market_vwap × 1e4
        buy:  bps = (execution_vwap − market_vwap) / market_vwap × 1e4

    The benchmark is the market VWAP of the tape's own prints (tracked by the
    env), never the strategy's own execution VWAP (plan_2.md §5 leak 3).
    Returns 0 for no fills / no market VWAP.
    """
    mvwap = _market_vwap_guard(fills, market_vwap)
    if mvwap <= 0:
        return 0.0
    self_vwap = execution_vwap(fills)
    if self_vwap <= 0:
        return 0.0
    return float(side * (mvwap - self_vwap) / mvwap * 1e4)


def implementation_shortfall(
    fills: Sequence[object],
    arrival_mid: float,
    *,
    side: int = 1,
) -> float:
    """Implementation shortfall vs the arrival mid (bps). Positive = a cost.

    Same signed convention as ``vwap_slippage``: for a sell (side=1) positive
    means the execution VWAP is BELOW arrival (sold cheap); for a buy (−1)
    positive means bought ABOVE arrival. This is exactly the math of the env's
    ``info["shortfall_bps"]`` for the sell side. Observable at ``t`` only.
    """
    px_sz = _fills_as(fills)
    qty = sum(sz for _, sz in px_sz)
    if qty <= 0 or arrival_mid <= 0:
        return 0.0
    self_vwap = execution_vwap(fills)
    return float(side * (arrival_mid - self_vwap) / arrival_mid * 1e4)


def arrival_slippage(
    fills: Sequence[object],
    arrival_mid: float,
    *,
    side: int = 1,
) -> float:
    """Alias-of-record for bps slippage vs the arrival mid (kept for readability
    at call sites that read "slippage" rather than "IS"). Identical math to
    ``implementation_shortfall``."""
    return implementation_shortfall(fills, arrival_mid, side=side)


def fill_rate(fills: Sequence[object], target_qty: int) -> float:
    """Fraction of parent-order ``target_qty`` that actually filled in [0, 1].

    Measures the aggregate Parent Order Fill Fraction (filled_qty / target_qty),
    NOT child limit-order execution probability.
    When inventory is strictly conserved (inventory = inventory0 - sum(fills)),
    this is mathematically equivalent to (1.0 - leftover / target_qty).
    Returns 0.0 for target_qty <= 0 or empty fills. Clamped to [0.0, 1.0].
    """
    px_sz = _fills_as(fills)
    qty = sum(sz for _, sz in px_sz)
    if target_qty <= 0:
        return 0.0
    return float(min(1.0, max(0.0, qty / target_qty)))


def completion_rate(fills: Sequence[object], target_qty: int, leftover: int) -> float:
    """Fraction of intent completed = filled / target, with unfilled counted."""
    px_sz = _fills_as(fills)
    qty = sum(sz for _, sz in px_sz)
    total = qty + leftover
    if total <= 0:
        return 0.0
    return min(1.0, qty / max(1, total))


def inv_risk(sigma: float, inv: float, t_fraction: float) -> float:
    """Standing-inventory risk proxy: sigma * |inv| * sqrt(tfrac). Deterministic,
    unit-consistent with the CVaR penalty seam (risk.py)."""
    return float(sigma * abs(inv) * sqrt(max(0.0, t_fraction)))


def max_drawdown(pnl_path: Sequence[float]) -> float:
    """Largest peak-to-trough drawdown of a cumulative PnL path in ticks (>= 0.0).

    Computes max_{t} (max_{s <= t} pnl[s] - pnl[t]) over the marked-to-market
    PnL path. Returns 0.0 for empty or single-element paths, or strictly
    increasing paths where no drawdown occurs.
    """
    path = [float(p) for p in pnl_path]
    if not path:
        return 0.0
    peak = path[0]
    mdd = 0.0
    for p in path:
        peak = max(peak, p)
        mdd = max(mdd, peak - p)
    return float(max(0.0, mdd))
