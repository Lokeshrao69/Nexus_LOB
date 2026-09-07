"""Python dashboard grain on the frozen BookStateView.

The C++ ``ShmRing`` publishes 448-byte slots. This module:

* decodes a ``BOOK_STATE_DTYPE`` record (file, bytes, or in-process view dict);
* serves a stdlib HTTP page with L2 ladder, BBO, spread, last trade, optional VaR.

It does not replace Person A's ring producer. Attach a raw dump of slots
(``capacity * 448`` bytes, little-endian record layout matching NumPy
``align=True`` dtype) or feed live ``view()`` dicts from replay/env.

OS shared-memory attach is best-effort (POSIX ``/dev/shm/<name>``); if the
segment is absent the dashboard runs off the in-process hub or a file ring.
"""
from __future__ import annotations

import json
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .book_state import BOOK_STATE_DTYPE, DEPTH, empty_state
from .replay import check_integrity, spread_ticks

View = Mapping[str, Any]


def view_from_record(rec: np.ndarray) -> dict[str, Any]:
    """Turn a 0-d BOOK_STATE_DTYPE record into a view-shaped dict."""
    r = rec.reshape(())
    return {
        "seq": int(r["seq"]),
        "ts_ns": int(r["ts_ns"]),
        "cum_volume": int(r["cum_volume"]),
        "last_trade_sz": int(r["last_trade_sz"]),
        "last_trade_px": int(r["last_trade_px"]),
        "last_trade_side": int(r["last_trade_side"]),
        "version": int(r["version"]),
        "bid_px": np.asarray(r["bid_px"]).copy(),
        "bid_sz": np.asarray(r["bid_sz"]).copy(),
        "bid_ct": np.asarray(r["bid_ct"]).copy(),
        "ask_px": np.asarray(r["ask_px"]).copy(),
        "ask_sz": np.asarray(r["ask_sz"]).copy(),
        "ask_ct": np.asarray(r["ask_ct"]).copy(),
    }


def record_from_view(view: View) -> np.ndarray:
    rec = empty_state()
    rec["seq"] = int(view.get("seq", 0))
    rec["ts_ns"] = int(view.get("ts_ns", 0))
    rec["cum_volume"] = int(view.get("cum_volume", 0))
    rec["last_trade_sz"] = int(view.get("last_trade_sz", 0))
    rec["last_trade_px"] = int(view.get("last_trade_px", 0))
    rec["last_trade_side"] = int(view.get("last_trade_side", 2))
    rec["version"] = int(view.get("version", 0))
    for name in ("bid_px", "bid_sz", "bid_ct", "ask_px", "ask_sz", "ask_ct"):
        arr = np.asarray(view[name])
        rec[name][: min(DEPTH, arr.shape[0])] = arr[:DEPTH]
    return rec


def decode_slot(buf: bytes | memoryview) -> dict[str, Any]:
    raw = bytes(buf)
    if len(raw) < BOOK_STATE_DTYPE.itemsize:
        raise ValueError(f"slot too short: {len(raw)} < {BOOK_STATE_DTYPE.itemsize}")
    rec = np.frombuffer(raw[: BOOK_STATE_DTYPE.itemsize], dtype=BOOK_STATE_DTYPE, count=1)[0]
    return view_from_record(rec)


class SnapshotHub:
    """Latest book + optional risk numbers for the HTTP layer."""

    def __init__(self) -> None:
        self.view: dict[str, Any] | None = None
        self.risk: dict[str, float] = {}
        self.source = "none"

    def push(self, view: View, *, source: str = "live") -> None:
        self.view = {
            k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
            for k, v in view.items()
        }
        self.source = source

    def as_json(self) -> dict[str, Any]:
        v = self.view
        if v is None:
            return {"ok": False, "source": self.source}
        spr = spread_ticks(v)
        issues = [i.code for i in check_integrity(v)]
        return {
            "ok": True,
            "source": self.source,
            "seq": int(v.get("seq", 0)),
            "ts_ns": int(v.get("ts_ns", 0)),
            "bid_px": [int(x) for x in np.asarray(v["bid_px"])[:DEPTH]],
            "bid_sz": [int(x) for x in np.asarray(v["bid_sz"])[:DEPTH]],
            "ask_px": [int(x) for x in np.asarray(v["ask_px"])[:DEPTH]],
            "ask_sz": [int(x) for x in np.asarray(v["ask_sz"])[:DEPTH]],
            "spread": spr,
            "cum_volume": int(v.get("cum_volume", 0)),
            "last_trade_px": int(v.get("last_trade_px", 0)),
            "last_trade_sz": int(v.get("last_trade_sz", 0)),
            "issues": issues,
            "risk": self.risk,
        }


def try_attach_posix_shm(name: str) -> memoryview | None:
    path = Path("/dev/shm") / name.lstrip("/")
    if not path.is_file():
        return None
    data = path.read_bytes()
    return memoryview(data)


def latest_from_file_ring(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    n = BOOK_STATE_DTYPE.itemsize
    if len(raw) < n:
        return None
    # last complete slot
    off = (len(raw) // n - 1) * n
    return decode_slot(raw[off : off + n])


# Person A ShmRing control block (shm_ring.hpp). Slot array starts immediately after.
_SHM_CTRL = struct.Struct("<QQQQQII")
_SHM_CTRL_N = 48


def read_shm_ring_latest(name: str) -> dict[str, Any] | None:
    """Newest published 448-byte slot from a live C++ ShmRing, or None."""
    path = Path("/dev/shm") / name.lstrip("/")
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) < _SHM_CTRL_N + BOOK_STATE_DTYPE.itemsize:
        return None
    write_seq, _read, _drop, cap, slot_bytes, state, _pad = _SHM_CTRL.unpack_from(data, 0)
    if state != 1 or cap == 0 or int(slot_bytes) != BOOK_STATE_DTYPE.itemsize:
        return None
    if write_seq == 0:
        return None
    idx = (int(write_seq) - 1) % int(cap)
    off = _SHM_CTRL_N + idx * int(slot_bytes)
    if off + int(slot_bytes) > len(data):
        return None
    return decode_slot(data[off : off + int(slot_bytes)])


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Nexus-LOB desk</title>
<style>
body{font-family:ui-monospace,monospace;background:#0b0f14;color:#d6e0ea;margin:0;padding:24px}
h1{font-size:16px;letter-spacing:.08em;color:#8ab4d8}
table{border-collapse:collapse}
td,th{padding:3px 10px;text-align:right}
.bid{color:#6ee7b7}.ask{color:#fb7185}
.meta{color:#8b9bb4;margin:12px 0}
</style></head><body>
<h1>NEXUS-LOB · L2</h1>
<div class="meta" id="meta">loading…</div>
<table id="book"><thead><tr><th>bid sz</th><th>bid</th><th>ask</th><th>ask sz</th></tr></thead>
<tbody></tbody></table>
<script>
async function tick(){
  const r = await fetch('/api/state'); const s = await r.json();
  document.getElementById('meta').textContent =
    s.ok ? ('seq '+s.seq+'  spread '+(s.spread??'—')+'  vol '+s.cum_volume+'  src '+s.source)
         : ('no snapshot ('+s.source+')');
  const tb = document.querySelector('#book tbody'); tb.innerHTML='';
  if(!s.ok) return;
  for(let i=0;i<10;i++){
    const tr=document.createElement('tr');
    tr.innerHTML = `<td class="bid">${s.bid_sz[i]}</td><td class="bid">${s.bid_px[i]}</td>
                    <td class="ask">${s.ask_px[i]}</td><td class="ask">${s.ask_sz[i]}</td>`;
    tb.appendChild(tr);
  }
}
setInterval(tick, 400); tick();
</script></body></html>
"""


def make_handler(hub: SnapshotHub, poll: Callable[[], None] | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ARG002
            return

        def do_GET(self) -> None:  # noqa: N802
            if poll is not None:
                poll()
            if self.path.startswith("/api/state"):
                body = json.dumps(hub.as_json()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = _PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(hub: SnapshotHub, host: str = "127.0.0.1", port: int = 8765,
          poll: Callable[[], None] | None = None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(hub, poll))
    return httpd
