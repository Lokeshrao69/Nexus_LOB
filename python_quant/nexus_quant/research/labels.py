"""Strictly forward labels for event-level supervised learning.

A label at event ``t`` is defined from the state at ``t`` (the reference) and
the state at ``t + h`` (the outcome). It must never touch states between the
endpoints or future features — the *features* are the leak-free side (see
``features.py``); labels are the target and use the endpoint by construction.

The most important property for the walk-forward harness: a label only depends
on ``(state_t, state_{t+h})``. Tests assert the endpoint-outcome is required
(no constant / no in-sample shortcut).
"""

from __future__ import annotations

from typing import Any

_View = Any  # dict[str, np.ndarray | int | float ...]


def forward_return(state_t: _View, future_state: _View, h: int = 1) -> float:
    """Fractional mid return from ``t`` to ``t + h``.

    Uses the mid at ``t`` as the reference (observable at ``t``) and the mid at
    ``t + h`` as the outcome. Returns 0.0 if either mid is unusable (empty BBO).
    """
    m0 = _mid(state_t)
    mh = _mid(future_state)
    if m0 <= 0 or mh <= 0:
        return 0.0
    return mh / m0 - 1.0


def forward_mid_move(state_t: _View, future_state: _View, h: int = 1) -> float:
    """Signed mid move in ticks from ``t`` to ``t + h``."""
    m0 = _mid(state_t)
    mh = _mid(future_state)
    if m0 <= 0 or mh <= 0:
        return 0.0
    return float(mh - m0)


def _mid(view: _View) -> float:
    bp = int(view["bid_px"][0]) if "bid_px" in view else 0
    ap = int(view["ask_px"][0]) if "ask_px" in view else 0
    if bp <= 0 or ap <= 0:
        return 0.0
    return (bp + ap) / 2.0