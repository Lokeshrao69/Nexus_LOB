"""Execution cost model (plan_2.md Phase 2).

Costs that make *accuracy* (not just rank) meaningful in the RL env:

  * ``CostParams`` — validated fee / rebate / impact parameters;
  * ``impact``     — square-root-law market impact in basis points;
  * ``net_pnl``    — gross PnL net of fees/rebates on the traded notional.

The env keeps every cost **default off / 0** (byte-identical to today); a
backtest/report turns them on, which is where honest "is the agent better under
real costs" numbers come from (work-package §WP2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class CostParams:
    """Execution cost assumptions. Zero-cost default keeps today's env exact.

    fee_bps     — taker cost in basis points of traded notional (>= 0).
    rebate_bps  — maker rebate in basis points of traded notional (>= 0).
    spread_ecn  — ECN add/remove spread offset in basis points (>= 0).
    impact_coef — square-root-law market-impact coefficient (>= 0); 0 = no impact.
    impact_mode — "sqrt" (Almgren–Chriss) or "linear" impact model.

    Validation is in ``__post_init__``: a negative fee, rebate, spread, or a
    negative impact coefficient is rejected, as is an unknown ``impact_mode``.
    Impact must be monotone non-decreasing in participation — tested in
    ``test_cost_model.py``.
    """

    fee_bps: float = 0.0
    rebate_bps: float = 0.0
    spread_ecn: float = 0.0
    impact_coef: float = 0.0
    impact_mode: str = "sqrt"

    def __post_init__(self) -> None:
        if self.fee_bps < 0:
            raise ValueError(f"fee_bps must be >= 0, got {self.fee_bps}")
        if self.rebate_bps < 0:
            raise ValueError(f"rebate_bps must be >= 0, got {self.rebate_bps}")
        if self.spread_ecn < 0:
            raise ValueError(f"spread_ecn must be >= 0, got {self.spread_ecn}")
        if self.impact_coef < 0:
            raise ValueError(f"impact_coef must be >= 0, got {self.impact_coef}")
        if self.impact_mode not in ("sqrt", "linear"):
            raise ValueError(f"impact_mode must be 'sqrt' or 'linear', got {self.impact_mode!r}")


def impact(
    qty_participation: float,
    *,
    sigma: float,
    coef: float,
    mode: str = "sqrt",
) -> float:
    """Market impact in basis points for a trade at ``qty_participation``.

    ``qty_participation`` is the fraction of ambient volume traded (0..1);
    ``sigma`` is the volatility of the return (bps terms optional — keep it
    unit-consistent with the coef). Square-root law:

        impact_bps = coef * sigma * sqrt(participation)        (mode="sqrt")
        impact_bps = coef * sigma * participation              (mode="linear")

    Impact is monotone in participation for both modes (tested), which is the
    property that keeps large-child strategies from gaming the metric.
    """
    if qty_participation < 0.0:
        raise ValueError(f"participation must be >= 0, got {qty_participation}")
    if mode == "sqrt":
        return coef * sigma * sqrt(qty_participation)
    return coef * sigma * qty_participation


def _notional_zero(fills: Sequence[tuple[int, int]]) -> int:
    """Sum of px*sz over ``fills`` (each a (price_tick, size) pair)."""
    return sum(px * sz for px, sz in fills)


def net_pnl(
    fills: Sequence[tuple[int, int]],
    *,
    side_cost: CostParams | None = None,
    gross: float = 0.0,
) -> float:
    """Net PnL in ticks of ``gross`` after execution costs on traded notional.

    ``fills`` is a sequence of ``(price_tick, size)`` trades and ``gross`` is
    the pre-cost PnL in ticks (positive = profit). Costs always reduce and
    rebates always raise the result:

        net = gross − fee_ticks + rebate_ticks
        fee_ticks    = fee_bps/1e4 × traded notional      (taker cost)
        rebate_ticks = rebate_bps/1e4 × traded notional   (maker credit)

    So under a taker fee the result is LOWER than ``gross``; under a maker
    rebate it is HIGHER — the acceptance-condition signs (work-package §WP2).
    With the default zero ``CostParams`` the result is ``gross`` exactly.
    """
    params = side_cost if side_cost is not None else CostParams()
    notional = _notional_zero(fills)
    fee_ticks = params.fee_bps / 1e4 * notional
    rebate_ticks = params.rebate_bps / 1e4 * notional
    return gross - fee_ticks + rebate_ticks
