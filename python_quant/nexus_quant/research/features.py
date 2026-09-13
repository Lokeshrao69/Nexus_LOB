"""Stateless LOB features over a ``View`` (a ``BOOK_STATE_DTYPE`` record).

Contract (plan_2.md Phase 1): every function is a pure function of state at
``t``, zero instance state, and reads *only* what is observable at call time.
``test_research.py::test_features_do_not_peek`` mutates the book after the
call and asserts the returned value is unchanged — the leakage lock.

The ``View`` shape mirrors ``StubOrderBook.view()`` / the C++ ``Engine``:

    view["bid_px"], view["bid_sz"], view["bid_ct"]   np.ndarray[DEPTH]
    view["ask_px"], view["ask_sz"], view["ask_ct"]   np.ndarray[DEPTH]
    view["mid"]/["spread"] are NOT stored by the engine — mid/spread are
    derived here from the best bid/ask levels only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# A shallow copy of the view dict so pure functions never keep a reference.
from typing import Any

import numpy as np

_View = Any  # dict[str, np.ndarray | int | float ...] -- kept loose on purpose


# ---------------------------------------------------------------------------
# helpers over the view
# ---------------------------------------------------------------------------
def _best(view: _View, side: str) -> tuple[int, int]:
    """Best (price, size) for a side; (0, 0) if the level is empty."""
    px = int(view[f"{side}_px"][0])
    sz = int(view[f"{side}_sz"][0])
    return px, sz


def mid(view: _View) -> float:
    """Tick mid from the BBO (0 if either side is empty)."""
    bp, _ = _best(view, "bid")
    ap, _ = _best(view, "ask")
    if bp <= 0 or ap <= 0:
        return 0.0
    return (bp + ap) / 2.0


def spread_ticks(view: _View) -> int:
    """Best-spread in ticks (0 if either side is empty)."""
    bp, _ = _best(view, "bid")
    ap, _ = _best(view, "ask")
    if bp <= 0 or ap <= 0:
        return 0
    return ap - bp


# ---------------------------------------------------------------------------
# level-1 features
# ---------------------------------------------------------------------------
def lob_imbalance(view: _View, k: int = 1) -> float:
    """(V_bid − V_ask) / (V_bid + V_ask) over the top ``k`` levels."""
    vb = float(np.sum(view["bid_sz"][:k]))
    va = float(np.sum(view["ask_sz"][:k]))
    if vb + va <= 0:
        return 0.0
    return (vb - va) / (vb + va)


def microprice(view: _View, k: int = 1) -> float:
    """VWAP of the BBO prices sized by resting volume — a bias-corrected mid.

    microprice = (V_a·P_b + V_b·P_a) / (V_a + V_b). When ``k > 1`` the ladder
    (up to depth ``k``) is volume-weighted with price-weighted rows.
    """
    bp, _ = _best(view, "bid")
    ap, _ = _best(view, "ask")
    vb = float(np.sum(view["bid_sz"][:k]))
    va = float(np.sum(view["ask_sz"][:k]))
    if vb + va <= 0 or bp <= 0 or ap <= 0:
        return float(mid(view))
    if k == 1:
        return (va * bp + vb * ap) / (vb + va)
    bpx = view["bid_px"][:k].astype(np.float64)
    apx = view["ask_px"][:k].astype(np.float64)
    bsz = view["bid_sz"][:k].astype(np.float64)
    asz = view["ask_sz"][:k].astype(np.float64)
    return float((np.sum(asz * bpx) + np.sum(bsz * apx)) / (bsz.sum() + asz.sum()))


def deep_imbalance(view: _View, k: int = 5, weighted: bool = True) -> float:
    """Multi-level imbalance, distance-weighted if ``weighted``.

    Weighted form down-weights far levels by 1/level index:
        ΔV(t) = Σ_i (B_i − A_i) / i ;  normalized by the total weighted size.
    """
    k = max(1, min(int(k), 10))
    bsz = view["bid_sz"][:k].astype(np.float64)
    asz = view["ask_sz"][:k].astype(np.float64)
    if weighted:
        w = 1.0 / np.arange(1, k + 1, dtype=np.float64)
        num = float(np.sum((bsz - asz) * w))
        den = float(np.sum((bsz + asz) * w))
    else:
        num = float(np.sum(bsz - asz))
        den = float(np.sum(bsz + asz))
    if den <= 0:
        return 0.0
    return num / den


def spread_bps(view: _View) -> float:
    """Best spread in basis points of the mid (0 if no BBO)."""
    m = mid(view)
    s = spread_ticks(view)
    if m <= 0 or s <= 0:
        return 0.0
    return 10_000.0 * s / m


# ---------------------------------------------------------------------------
# order-flow approximation + history features
# ---------------------------------------------------------------------------
def ofi(prev_view: _View | None, cur_view: _View, k: int = 1) -> float:
    """Order-flow imbalance across a book update.

    Approximation from two L2 ladders (order-level deltas need the Phase 3
    tracker): when a level's displayed size rises the volume is treated as
    add-flow on that side, when it falls the delta is treated as cancel-flow
    (the exact add/cancel split is unknowable from an L2 snapshot alone).
    """
    if prev_view is None:
        return 0.0
    vb_prev = float(np.sum(prev_view["bid_sz"][:k]))
    va_prev = float(np.sum(prev_view["ask_sz"][:k]))
    vb_cur = float(np.sum(cur_view["bid_sz"][:k]))
    va_cur = float(np.sum(cur_view["ask_sz"][:k]))
    e_bid = max(0.0, vb_cur - vb_prev) - max(0.0, vb_prev - vb_cur)
    e_ask = max(0.0, va_cur - va_prev) - max(0.0, va_prev - va_cur)
    return e_bid - e_ask


def realized_vol(mid_hist: Sequence[float], n: int = 10) -> float:
    """Annualized realized vol of the last ``n`` mid moves (zero if < 2 pts)."""
    # Drop non-positive mids: a 0 mid (empty-side gap in a HFT-style tape) would
    # otherwise produce a divide-by-zero return and NaN the feature.
    xs = np.asarray([float(x) for x in mid_hist[-n:] if float(x) > 0], dtype=np.float64)
    if xs.size < 2:
        return 0.0
    r = np.diff(xs) / xs[:-1]
    mu = r.mean()
    sig = float(np.sqrt(np.sum((r - mu) ** 2) / max(1, r.size - 1)))
    return sig * np.sqrt(252.0)


def momentum(mid_hist: Sequence[float], k: int = 5) -> float:
    """Fractional mid move over ``k`` samples: (m_t/m_{t-k} − 1)."""
    xs = [float(x) for x in mid_hist]
    if len(xs) <= k or xs[-1 - k] <= 0:
        return 0.0
    return (xs[-1] / xs[-1 - k]) - 1.0


def flow_intensity(events: Iterable[Any], tau: int = 10) -> float:
    """Signed event rate over the last ``tau`` events.

    ``events`` are order-level tuples/records with a ``side``-like field and a
    volume; the sign is +1 for bid adds / ask cancels, −1 for the reverse.
    Falls back to 0 when no events or when the records have no side/size.
    """
    s = 0.0
    seen = 0
    for ev in events:
        side = _get_side(ev)
        if side == 0:  # bid
            s += 1.0
        else:
            s -= 1.0
        seen += 1
        if seen >= tau:
            break
    return s / max(1, seen)


def _get_side(ev: Any) -> int:
    """Bid=0 / ask=1 from a dict, dataclass, or tuple-shaped event."""
    if isinstance(ev, dict):
        return int(ev.get("side", 0))
    if hasattr(ev, "side"):
        return int(ev.side)
    # tuple/namedtuple with side as the third field, matching NormalizedEvent.
    if isinstance(ev, tuple):
        return int(ev[2]) if len(ev) >= 3 else 0
    return 0