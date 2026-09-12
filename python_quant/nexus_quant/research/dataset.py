"""Event-frame dataset + leak-free walk-forward splits.

Rows are time-ordered observations: a feature dict computed from the book at
event ``t`` and a strictly forward label to ``t + h``. Splits are **walk
forward** — the train / val / test blocks are contiguous in *time* (by
sequence index, which is monotone in wall-clock on a real tape). K-fold CV is
deliberately absent: shuffled CV on an autocorrelated tape is a leak.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .labels import forward_mid_move

_View = Any


@dataclass
class Row:
    """One labeled event. ``ts`` is the monotone sequence/time key."""

    ts: int
    features: dict[str, float]
    label: float
    split: str = ""  # "train" | "val" | "test", filled by make_split


def event_frame(
    events: Sequence[Any],
    apply: Callable[[Any], _View],
    *,
    feature_fns: Mapping[str, Callable[[_View], float]],
    prev_feature_fns: Mapping[str, Callable[[_View, _View], float]] | None = None,
    h: int = 5,
    label_fn: Callable[[_View, _View, int], float] = forward_mid_move,
    ts_fn: Callable[[_View], int] = lambda v: int(v["seq"]),
) -> list[Row]:
    """Build labeled rows by replaying ``events`` through ``apply``.

    ``apply(ev)`` mutates the book for one event and returns the *current*
    view (mirrors ``ReplayEngine.apply``). The label of row ``i`` uses the view
    at ``i + h``; the last ``h`` rows have no future state and are dropped.

    ``feature_fns``: name → pure function of a single view (see ``features.py``).
    ``prev_feature_fns``: name → pure function ``(prev_view, cur_view)`` of two
    consecutive views — for order-flow features such as ``ofi``. Rows ``i`` whose
    ``prev`` view is this same row's view are NOT padded: row 0 cannot be an OFI
    observation, so it is dropped when any pairwise feature is requested.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    prev_feature_fns = prev_feature_fns or {}
    views: list[_View] = []
    for ev in events:
        views.append(apply(ev))

    rows: list[Row] = []
    for i in range(len(views) - h):
        if prev_feature_fns and i == 0:
            continue  # row 0 has no pairwise predecessor -> dropped (see docstring)
        v = views[i]
        fv = views[i + h]
        feat = {name: float(fn(v)) for name, fn in feature_fns.items()}
        if prev_feature_fns:
            prev = views[i - 1]
            for name, fn in prev_feature_fns.items():
                feat[name] = float(fn(prev, v))
        rows.append(Row(ts=ts_fn(v), features=feat, label=float(label_fn(v, fv, h))))
    return rows


def make_split(
    rows: Sequence[Row],
    *,
    mode: str = "walk_forward",
    train: float = 0.6,
    val: float = 0.2,
    gap: int = 0,
) -> dict[str, list[Row]]:
    """Time-ordered walk-forward split of ``rows`` into train / val / test.

    Blocks are ordered: train first, val next, test last, in *sequence* order
    (the rows are re-sorted by ``ts`` so the caller does not have to). ``gap``
    rows are dropped between blocks so no label horizon reaches across the
    boundary (a label at ``t`` reaches to ``t + h``; ``gap >= h`` is the safe
    recommendation, but the caller decides).

    Returns ``{"train": [...], "val": [...], "test": [...]}`` and stamps each
    Row's ``split`` field.
    """
    if mode != "walk_forward":
        raise ValueError(f"unsupported split mode: {mode!r}")
    if not 0.0 < train < 1.0 or not 0.0 < val < 1.0 or train + val >= 1.0:
        raise ValueError("train and val must satisfy 0 < train, val < 1 and train+val < 1")

    ordered = sorted(rows, key=lambda r: (r.ts, id(r)))  # stable in ts
    n = len(ordered)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    idx_train = int(n * train)
    idx_val = idx_train + int(n * val)
    # gap drops are taken from the tail of train and the head of val so the
    # remaining boundaries are at least ``gap`` rows apart.
    train_rows = ordered[: idx_train - gap]
    val_rows = ordered[idx_train + gap : idx_val - gap]
    test_rows = ordered[idx_val + gap :]

    out: dict[str, list[Row]] = {"train": [], "val": [], "test": []}
    for block, rows_in_block in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        for r in rows_in_block:
            r.split = block
            out[block].append(r)
    return out