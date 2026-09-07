#!/usr/bin/env python
"""Serve the Person B L2 dashboard.

    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py
    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --synthetic
    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --ring /tmp/nexus_slots.bin

Default ``--synthetic`` drives a StubOrderBook so the page works without C++ shm.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant.book_state import Side, StubOrderBook  # noqa: E402
from nexus_quant.dashboard import (  # noqa: E402
    SnapshotHub,
    latest_from_file_ring,
    serve,
    try_attach_posix_shm,
    decode_slot,
)
from nexus_quant.risk import compute_var_cvar  # noqa: E402


def _seed(book: StubOrderBook) -> None:
    mid = 15_000
    for i in range(10):
        book.add(Side.Bid, mid - 1 - i, 80 + 10 * i)
        book.add(Side.Ask, mid + 1 + i, 80 + 8 * i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--ring", type=str, default="")
    ap.add_argument("--shm", type=str, default="")
    args = ap.parse_args()
    hub = SnapshotHub()
    book = StubOrderBook()
    _seed(book)
    tick = {"n": 0}

    def poll() -> None:
        if args.ring:
            v = latest_from_file_ring(Path(args.ring))
            if v:
                hub.push(v, source="file-ring")
                return
        if args.shm:
            mv = try_attach_posix_shm(args.shm)
            if mv is not None and len(mv) >= 448:
                # skip a possible control header; try last 448 bytes
                hub.push(decode_slot(mv[-448:]), source="shm")
                return
        # synthetic walk
        tick["n"] += 1
        side = Side.Bid if tick["n"] % 2 == 0 else Side.Ask
        book.record_trade(side, 15_000, 5)
        hub.push(book.view(), source="synthetic")
        if tick["n"] % 8 == 0:
            r = compute_var_cvar(n_paths=128, steps=8, prefer_engine=False)
            hub.risk = {"var": r.var, "cvar": r.cvar}

    poll()
    httpd = serve(hub, args.host, args.port, poll)
    print(f"Nexus-LOB dashboard http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
