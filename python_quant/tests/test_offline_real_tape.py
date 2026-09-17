"""Phase 4 — real NASDAQ ITCH tape smoke (plan_2.md Phase 4, work package §4.2).

Two layers:

* **Always on** (no network, no data): ``fetch_itch.py``'s framing/slicing
  logic is exercised on a hand-built length-prefixed ITCH stream that mixes
  ``S`` / ``R`` / order messages for two symbols — the per-symbol slice must
  contain exactly the session events, the symbol's directory record, and
  that symbol's order messages, byte for byte.
* **Real bytes** (skipped unless ``data/itch/<day>/<SYMBOL>.itch`` exists —
  produce it with ``python python_quant/scripts/fetch_itch.py``; ``data/`` is
  gitignored and never committed): the parser consumes the real file with
  **0 truncated** messages, the replay is integrity-clean, the order-level
  tracker sees no unknown ids, and the prefix runs in bounded time.

Measured on 12/30/2019 AAPL (full day, 1.52 M messages): 0 truncated, 1
integrity flag (``empty_bbo`` on the very first pre-market message — an empty
book, not a decode error), tracker 0 unknown ids, and Engine-vs-Stub ladder
parity exact over the pre-market prefix (7,037 frames) — see ``progress_b.md``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_port import StubBookAdapter
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import (
    EventType,
    ItchParseStats,
    NormalizedEvent,
    encode_event,
    iter_itch_events,
)
from nexus_quant.replay import ReplayEngine
from nexus_quant.research.queue_dynamics import OrderLevelTracker

_ROOT = Path(__file__).resolve().parents[2]
_DAY = os.environ.get("NEXUS_ITCH_DAY", "12302019")
_SYMBOL = os.environ.get("NEXUS_ITCH_SYMBOL", "AAPL")
_TAPE = _ROOT / "data" / "itch" / _DAY / f"{_SYMBOL}.itch"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_itch", _ROOT / "python_quant" / "scripts" / "fetch_itch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _framed(body: bytes) -> bytes:
    return len(body).to_bytes(2, "big") + body


def _system_event(code: bytes, ts: int) -> bytes:
    # S: locate(2) tracking(2) ts(6) event(1)  -> 12 bytes incl. type
    return _framed(b"S" + (0).to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big") + code)


def _directory(locate: int, stock: str, ts: int) -> bytes:
    # R: locate(2) tracking(2) ts(6) stock(8) + 20 more bytes of attributes -> 39 incl. type
    body = (b"R" + locate.to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big")
            + stock.ljust(8).encode() + b"Q" + b"N" + (100).to_bytes(4, "big") + b"N" + b"Q" + b"N" + b"  "
            + b"N" + b"N" + b" " + b"N" + (0).to_bytes(4, "big") + b"N")
    assert len(body) == 39, len(body)
    return _framed(body)


def _add(locate: int, oid: int, side: Side, px: int, sz: int, ts: int) -> bytes:
    ev = NormalizedEvent(EventType.ADD, ts, oid, side, px, sz)
    raw = bytearray(encode_event(ev, framed=False))
    raw[1:3] = locate.to_bytes(2, "big")  # encode_event hard-codes locate=1
    return _framed(bytes(raw))


def _delete(locate: int, oid: int, ts: int) -> bytes:
    raw = bytearray(encode_event(NormalizedEvent(EventType.DELETE, ts, oid, Side.NONE, 0, 0), framed=False))
    raw[1:3] = locate.to_bytes(2, "big")
    return _framed(bytes(raw))


def test_symbol_slicer_keeps_exactly_the_symbol_stream(tmp_path: Path) -> None:
    fetch = _load_fetch_module()
    ts = 34_200_000_000_000
    stream = b"".join([
        _system_event(b"O", ts - 5),
        _directory(7, "AAPL", ts - 4),
        _directory(9, "QQQ", ts - 4),
        _system_event(b"Q", ts - 1),
        _add(7, 11, Side.Bid, 2_890_000, 100, ts + 1),
        _add(9, 21, Side.Ask, 2_130_000, 50, ts + 2),
        _add(7, 12, Side.Ask, 2_890_500, 40, ts + 3),
        _delete(9, 21, ts + 4),
        _delete(7, 11, ts + 5),
        _system_event(b"M", ts + 10),
    ])
    slicer = fetch._SymbolSlicer(tmp_path, {"AAPL"})
    # feed in awkward chunk boundaries to exercise the framing buffer
    prev = 0
    for cut in (5, 17, 40, 41, 90, 150, len(stream)):
        slicer.feed(stream[prev:cut])
        prev = cut
    slicer.close()
    assert slicer.messages == 10 and slicer.locate_to_symbol == {7: "AAPL"}
    assert slicer.counts["AAPL"] == {"R": 1, "A": 2, "D": 1}
    out = (tmp_path / "AAPL.itch").read_bytes()
    expected = b"".join([
        _system_event(b"O", ts - 5),  # replayed S messages seen before the R
        _directory(7, "AAPL", ts - 4),
        _system_event(b"Q", ts - 1),
        _add(7, 11, Side.Bid, 2_890_000, 100, ts + 1),
        _add(7, 12, Side.Ask, 2_890_500, 40, ts + 3),
        _delete(7, 11, ts + 5),
        _system_event(b"M", ts + 10),
    ])
    assert out == expected
    # and the slice is a valid tape for the parser + replay + tracker
    stats = ItchParseStats()
    events = list(iter_itch_events(out, stats=stats))
    assert stats.truncated == 0 and [e.kind for e in events] == [EventType.ADD, EventType.ADD, EventType.DELETE]
    rep = ReplayEngine(events, StubBookAdapter())
    last = rep.step_n(3)
    assert last is not None and rep.applied == 3 and last.issues == []
    assert int(last.state["ask_px"][0]) == 2_890_500 and int(last.state["bid_px"][0]) == 0
    assert (tmp_path / "QQQ.itch").exists() is False


def test_tape_url_and_manifest_shape() -> None:
    fetch = _load_fetch_module()
    assert fetch.tape_url("12302019") == "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz"
    assert fetch.DEFAULT_SYMBOLS and fetch.DEFAULT_DAY == "12302019"


def test_public_sample_days_catalogue() -> None:
    fetch = _load_fetch_module()
    days = fetch.PUBLIC_SAMPLE_DAYS
    assert len(days) == 15
    assert len(set(days)) == 15  # all unique
    assert fetch.DEFAULT_DAY in days
    assert "01302020" in days
    for d in days:
        assert len(d) == 8 and d.isdigit()
        url = fetch.tape_url(d)
        assert url.startswith("https://emi.nasdaq.com/ITCH/")
        assert url.endswith(f"{d}.NASDAQ_ITCH50.gz")
        if d == "05302019":
            # Relocated on the public server (2026-09-15 audit): the NASDAQ tape for this
            # catalogued day is only published inside the PSX directory; the historical
            # `Nasdaq ITCH/` URL returns 404.
            assert url == "https://emi.nasdaq.com/ITCH/Nasdaq%20PSX%20ITCH/05302019.NASDAQ_ITCH50.gz"
        else:
            assert url.startswith("https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/")
    # A --base override (tests, mirrors) must keep controlling every URL, catalogued or not.
    assert fetch.tape_url("05302019", "http://mirror/") == "http://mirror/05302019.NASDAQ_ITCH50.gz"
    assert fetch.tape_url("10182019", "http://mirror/") == "http://mirror/10182019.NASDAQ_ITCH50.gz"


def test_extended_sample_tapes_catalogue() -> None:
    """Every extra verified tape is keyed by a real MMDDYYYY trading day and served from emi.nasdaq.com."""
    import datetime as dt

    fetch = _load_fetch_module()
    extended = fetch.EXTENDED_SAMPLE_TAPES
    assert len(extended) >= 16
    assert len(set(extended.values())) == len(extended)  # one URL per day
    for day, url in extended.items():
        assert len(day) == 8 and day.isdigit()
        dt.date(int(day[4:]), int(day[:2]), int(day[2:4]))  # valid calendar date
        assert url.startswith("https://emi.nasdaq.com/ITCH/")
        assert url.endswith(".gz")
        assert fetch.tape_url(day) == url
    # The only overlap with the historical catalogue is the relocated 05302019 tape.
    assert set(extended) & set(fetch.PUBLIC_SAMPLE_DAYS) == {"05302019"}
    # Filename shapes actually published by Nasdaq: S<MMDDYY>-v50.txt.gz, itch50_<MM>_<DD>.gz, <MMDDYYYY>.NASDAQ_ITCH50.gz
    assert extended["10182019"].endswith("/S101819-v50.txt.gz")
    assert extended["05152026"].endswith("/itch50_05_15.gz")
    assert extended["12132018"].endswith("/S121318-v50.txt.gz")

def test_replay_tracks_applied_and_failed_events(tmp_path: Path) -> None:
    ts = 34_200_000_000_000
    events = [
        NormalizedEvent(EventType.ADD, ts + 1, 100, Side.Bid, 10_000, 50),
        NormalizedEvent(EventType.DELETE, ts + 2, 100, Side.NONE, 0, 0),
        NormalizedEvent(EventType.DELETE, ts + 3, 999, Side.NONE, 0, 0),  # non-existent order
    ]
    rep = ReplayEngine(events, StubBookAdapter())
    f1 = rep.step()
    assert f1 is not None and f1.applied is True
    f2 = rep.step()
    assert f2 is not None and f2.applied is True
    f3 = rep.step()
    assert f3 is not None and f3.applied is False
    assert rep.applied == 2
    assert rep.skipped == 1

    itch_file = tmp_path / "AAPL.itch"
    itch_file.write_bytes(
        b"".join([
            encode_event(NormalizedEvent(EventType.ADD, ts + 1, 100, Side.Bid, 10_000, 50)),
            encode_event(NormalizedEvent(EventType.DELETE, ts + 2, 100, Side.NONE, 0, 0)),
            encode_event(NormalizedEvent(EventType.DELETE, ts + 3, 999, Side.NONE, 0, 0)),
        ])
    )
    spec = importlib.util.spec_from_file_location(
        "run_research", _ROOT / "python_quant" / "scripts" / "run_research.py"
    )
    assert spec is not None and spec.loader is not None
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    res = rr.replay_symbol(itch_file)
    assert res["events_applied"] == 2
    assert res["events_failed"] == 1


def test_fill_study_isolates_regular_session() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_research", _ROOT / "python_quant" / "scripts" / "run_research.py"
    )
    assert spec is not None and spec.loader is not None
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)

    tr = OrderLevelTracker()
    ts_open = rr.REGULAR_OPEN_NS
    ts_close = rr.REGULAR_CLOSE_NS

    # Order 1: added at 15:59:59, fills at 16:00:05 (post-close fill)
    tr.on_event(NormalizedEvent(EventType.ADD, ts_close - 1_000_000_000, 1, Side.Bid, 10_000, 10))
    # Order 2: added at 10:00:00, fills at 10:05:00 (regular-session fill)
    tr.on_event(NormalizedEvent(EventType.ADD, ts_open + 1_800_000_000_000, 2, Side.Bid, 10_000, 10))
    tr.on_event(NormalizedEvent(EventType.EXECUTE, ts_open + 2_100_000_000_000, 2, Side.NONE, 0, 10))
    # Order 1 fills at 16:00:05
    tr.on_event(NormalizedEvent(EventType.EXECUTE, ts_close + 5_000_000_000, 1, Side.NONE, 0, 10))

    events = [
        NormalizedEvent(EventType.ADD, ts_close - 1_000_000_000, 1, Side.Bid, 10_000, 10),
        NormalizedEvent(EventType.ADD, ts_open + 1_800_000_000_000, 2, Side.Bid, 10_000, 10),
        NormalizedEvent(EventType.EXECUTE, ts_open + 2_100_000_000_000, 2, Side.NONE, 0, 10),
        NormalizedEvent(EventType.EXECUTE, ts_close + 5_000_000_000, 1, Side.NONE, 0, 10),
    ]
    rep = {"tracker": tr, "events": events}
    res = rr.fill_study(rep, horizons_events=(1, 2, 5))

    # Order 1 must NOT count as a completed fill in regular session
    assert res["outcomes"] == {"filled": 1}  # only Order 2
    assert res["n_still_open_at_close"] == 1  # Order 1 is still open at close
    assert res["n_orders_regular"] == 2


def test_manifest_provenance_verification(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_research", _ROOT / "python_quant" / "scripts" / "run_research.py"
    )
    assert spec is not None and spec.loader is not None
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)

    day_dir = tmp_path / "12302019"
    day_dir.mkdir(parents=True)
    itch_file = day_dir / "AAPL.itch"
    itch_bytes = b"sample itch data for testing"
    itch_file.write_bytes(itch_bytes)

    valid_sha = hashlib.sha256(itch_bytes).hexdigest()
    manifest_data = {
        "day": "12302019",
        "symbols": {
            "AAPL": {
                "file": "AAPL.itch",
                "bytes": len(itch_bytes),
                "sha256": valid_sha,
            }
        },
    }
    manifest_path = day_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # 1. Valid manifest passes verification without error
    rr.verify_manifest_provenance(itch_file)

    # 2. Corrupted file content (modified bytes / mismatched hash) raises ValueError
    itch_file.write_bytes(b"corrupted itch data")
    with pytest.raises(ValueError, match="SHA-256 mismatch|File size mismatch"):
        rr.verify_manifest_provenance(itch_file)

    # 3. Size mismatch raises ValueError
    manifest_data["symbols"]["AAPL"]["sha256"] = hashlib.sha256(b"corrupted itch data").hexdigest()
    manifest_data["symbols"]["AAPL"]["bytes"] = 999999
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ValueError, match="File size mismatch"):
        rr.verify_manifest_provenance(itch_file)

    # 4. Running CLI main with mismatched hash halts with ValueError
    with pytest.raises(ValueError):
        rr.main(["--data", str(tmp_path), "--day", "12302019", "--symbols", "AAPL"])


@pytest.mark.skipif(not _TAPE.exists(), reason=f"real tape not fetched: {_TAPE} (run scripts/fetch_itch.py)")
def test_real_tape_parses_clean() -> None:
    """Parser 0 truncated · replay integrity clean (bar the empty pre-open book) ·
    tracker 0 unknown ids · bounded runtime on the first 150k events."""
    stats = ItchParseStats()
    t0 = time.time()
    events = []
    for ev in iter_itch_events(_TAPE, stats=stats):
        events.append(ev)
        if len(events) >= 150_000:
            break
    assert stats.truncated == 0
    assert len(events) >= 1_000
    assert all(e.price_ticks >= 0 and e.size >= 0 for e in events)
    assert all(events[i].ts_ns <= events[i + 1].ts_ns for i in range(len(events) - 1))  # tape order
    rep = ReplayEngine(events, StubBookAdapter())
    tracker = OrderLevelTracker()
    issue_codes: dict[str, int] = {}
    for frame in rep.frames():
        tracker.on_event(frame.event)
        for iss in frame.issues:
            issue_codes[iss.code] = issue_codes.get(iss.code, 0) + 1
    assert rep.skipped == 0, "every real event must apply to the book"
    assert set(issue_codes) <= {"empty_bbo"}, issue_codes  # never crossed / locked / negative / unsorted
    assert tracker.stats["unknown_id"] == 0
    assert time.time() - t0 < 120.0
