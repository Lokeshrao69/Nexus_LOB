#!/usr/bin/env python3
"""Classify every catalogued NASDAQ ITCH tape from a bounded GZIP prefix.

Phase 1 of the multi-day campaign: before any multi-gigabyte download, each
candidate in ``fetch_itch.PUBLIC_SAMPLE_DAYS`` ∪ ``fetch_itch.EXTENDED_SAMPLE_TAPES``
is probed with an HTTP ``Range`` request (default 4 MiB) and classified on
evidence, never on its filename:

* **format** — the prefix must inflate as gzip and frame as 2-byte big-endian
  length-prefixed ITCH 5.0 messages whose lengths equal the spec lengths in
  ``nexus_quant.itch_parser.ITCH_LEN`` (the parser's own table);
* **date fingerprint** — the complete stock directory (every ``R`` message:
  symbol → locate code) is compared pair-for-pair with Nasdaq's published
  ``Stock_Locate_Codes/ndq_stocklocate_<YYYYMMDD>.txt`` for the date inferred
  from the filename.  Neighbouring dates share only ~15–20 % of pairs, so an
  ``exact_match`` pins the trading day;
* **symbols** — AAPL / QQQ locate codes must be present;
* retired tapes are recorded with their HTTP status (``404``) and no probe.

Output is a JSON list (one record per candidate) that
``build_session_manifest.py`` consumes.  Prefixes and locate files are cached
under ``--cache-dir`` (gitignored ``data/``) so a re-run is offline.

Run (from the repository root)::

    python python_quant/scripts/audit_itch_directory.py \
        --out docs/results/multi_day/itch_directory_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT / "python_quant") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python_quant"))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_itch
from nexus_quant.itch_parser import ITCH_LEN

LOCATE_BASE = "https://emi.nasdaq.com/ITCH/Stock_Locate_Codes/"
_HEADERS = {"User-Agent": "nexus-lob-fetch-itch/1.0"}


def _get(url: str, *, byte_range: str | None = None, timeout: int = 180) -> tuple[int, bytes]:
    headers = dict(_HEADERS)
    if byte_range:
        headers["Range"] = byte_range
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return resp.status, resp.read()


def _head_status(url: str, *, timeout: int = 60) -> int:
    req = urllib.request.Request(url, headers=_HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def fetch_prefix(url: str, cache_dir: Path, n_bytes: int) -> bytes:
    path = cache_dir / (url.rsplit("/", 1)[-1] + f".prefix{n_bytes}")
    if path.exists() and path.stat().st_size == n_bytes:
        return path.read_bytes()
    _, raw = _get(url, byte_range=f"bytes=0-{n_bytes - 1}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def locate_file(yyyymmdd: str, cache_dir: Path) -> dict[str, int] | None:
    path = cache_dir / f"ndq_stocklocate_{yyyymmdd}.txt"
    if not path.exists():
        try:
            _, raw = _get(f"{LOCATE_BASE}ndq_stocklocate_{yyyymmdd}.txt")
        except urllib.error.HTTPError:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    pairs: dict[str, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "," in line:
            symbol, locate = line.rsplit(",", 1)
            try:
                pairs[symbol.strip()] = int(locate)
            except ValueError:
                continue
    return pairs


def analyse_prefix(raw: bytes) -> dict[str, Any]:
    """Frame the inflated prefix exactly as ``fetch_itch._SymbolSlicer`` does."""
    if raw[:2] != b"\x1f\x8b":
        return {"gzip": False, "head_hex": raw[:16].hex()}
    data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    types: Counter[str] = Counter()
    bad_len: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    directory: dict[str, int] = {}
    system_events: list[tuple[str, float]] = []
    i, n = 0, len(data)
    while i + 2 <= n:
        ln = (data[i] << 8) | data[i + 1]
        end = i + 2 + ln
        if end > n:
            break
        typ = data[i + 2 : i + 3]
        label = typ.decode("latin1")
        spec = ITCH_LEN.get(typ)
        if spec is None:
            unknown[f"{label}:{ln}"] += 1
        elif spec != ln:
            bad_len[f"{label}:{ln}!={spec}"] += 1
        types[label] += 1
        ts = int.from_bytes(data[i + 7 : i + 13], "big")
        if typ == b"S":
            system_events.append((chr(data[i + 13]), round(ts / 3.6e12, 4)))
        elif typ == b"R":
            directory[data[i + 13 : i + 21].decode("ascii", "replace").strip()] = (data[i + 3] << 8) | data[i + 4]
        i = end
    return {
        "gzip": True,
        "raw_bytes": n,
        "messages": sum(types.values()),
        "types": dict(types),
        "spec_length_mismatches": dict(bad_len),
        "unknown_types": dict(unknown),
        "system_events": system_events,
        "directory": directory,
    }


def audit_day(day: str, *, cache_dir: Path, n_bytes: int) -> dict[str, Any]:
    url = fetch_itch.tape_url(day)
    rec: dict[str, Any] = {
        "day": day,
        "date_iso": f"{day[4:]}-{day[:2]}-{day[2:4]}",
        "name": url.rsplit("/", 1)[-1],
        "url": url,
        "in_original_catalogue": day in fetch_itch.PUBLIC_SAMPLE_DAYS,
        "in_extended_catalogue": day in fetch_itch.EXTENDED_SAMPLE_TAPES,
    }
    rec["http_status"] = _head_status(url)
    if rec["http_status"] != 200:
        rec["classification"] = "retired_http_error"
        return rec
    analysis = analyse_prefix(fetch_prefix(url, cache_dir, n_bytes))
    directory = analysis.pop("directory", {})
    rec.update({"prefix_bytes": n_bytes, **analysis})
    if not analysis.get("gzip"):
        rec["classification"] = "not_gzip"
        return rec
    rec["n_directory"] = len(directory)
    rec["AAPL_locate"] = directory.get("AAPL")
    rec["QQQ_locate"] = directory.get("QQQ")
    rec["parser_compatible"] = not rec["spec_length_mismatches"] and not rec["unknown_types"]
    pairs = locate_file(day[4:] + day[:4], cache_dir)
    if pairs is None:
        rec["locate_fingerprint"] = {"status": "locate_file_unavailable"}
    else:
        same = sum(1 for symbol, locate in directory.items() if pairs.get(symbol) == locate)
        exact = same == len(directory) == len(pairs)
        rec["locate_fingerprint"] = {
            "status": "exact_match" if exact else "mismatch",
            "tape_directory_size": len(directory),
            "locate_file_size": len(pairs),
            "matching_pairs": same,
            "only_in_tape": sorted(set(directory) - set(pairs))[:10],
            "only_in_locate_file": sorted(set(pairs) - set(directory))[:10],
        }
    ok = (
        rec["parser_compatible"]
        and rec["AAPL_locate"] is not None
        and rec["QQQ_locate"] is not None
        and rec["locate_fingerprint"]["status"] == "exact_match"
    )
    rec["classification"] = "itch50_full_session_candidate" if ok else "needs_review"
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_ROOT / "docs" / "results" / "multi_day" / "itch_directory_audit.json")
    ap.add_argument("--cache-dir", type=Path, default=_ROOT / "data" / "probe" / "cache")
    ap.add_argument("--prefix-bytes", type=int, default=4 << 20)
    ap.add_argument("--days", default=None, help="comma-separated subset (default: both catalogues)")
    args = ap.parse_args(argv)

    if args.days:
        days = [d.strip() for d in args.days.split(",") if d.strip()]
    else:
        days = list(fetch_itch.PUBLIC_SAMPLE_DAYS)
        days.extend(d for d in fetch_itch.EXTENDED_SAMPLE_TAPES if d not in days)
    days.sort(key=lambda d: (d[4:], d[:2], d[2:4]))

    records = []
    for day in days:
        try:
            rec = audit_day(day, cache_dir=args.cache_dir, n_bytes=args.prefix_bytes)
        except Exception as exc:  # noqa: BLE001 - one unreachable tape must not abort the audit
            rec = {"day": day, "url": fetch_itch.tape_url(day), "classification": "probe_error", "error": f"{type(exc).__name__}: {exc}"}
        records.append(rec)
        fp = rec.get("locate_fingerprint", {}).get("status", "—")
        print(
            f"{rec.get('date_iso', day)}  http={rec.get('http_status', '—')}  {rec.get('classification'):<32} "
            f"msgs={rec.get('messages', '—')}  dir={rec.get('n_directory', '—')}  AAPL={rec.get('AAPL_locate', '—')}  "
            f"QQQ={rec.get('QQQ_locate', '—')}  fingerprint={fp}  {rec.get('name', '')}",
            flush=True,
        )
    payload = {
        "kind": "nexus-lob-itch-directory-audit",
        "generated_at_unix": int(time.time()),
        "prefix_bytes": args.prefix_bytes,
        "method": (
            "HEAD status; bounded Range GET of the gzip prefix; 2-byte length framing checked against "
            "nexus_quant.itch_parser.ITCH_LEN; full R-message stock directory matched pair-for-pair against "
            "Nasdaq's daily ndq_stocklocate file for the inferred date."
        ),
        "candidates": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    counts = Counter(r.get("classification") for r in records)
    print(f"wrote {args.out} — {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
