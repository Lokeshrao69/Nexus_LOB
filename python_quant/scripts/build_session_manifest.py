#!/usr/bin/env python3
"""Build the auditable multi-day dataset manifest and the cross-day aggregation.

Inputs (all produced by existing tools, none of which are modified here):

* ``data/probe/fingerprints.json`` — the NASDAQ directory audit (GZIP-prefix
  format check + stock-directory fingerprint against Nasdaq's daily locate file);
* ``data/itch/<day>/manifest.json`` — ``fetch_itch.py`` source manifests
  (bytes fetched, GZIP EOF reached, per-symbol slice hashes);
* ``docs/results/multi_day/<day>/`` — ``batch_research_itch.py`` per-day
  E1–E6 results with ``research_manifest.json`` completion markers.

Outputs:

* ``docs/results/multi_day/session_manifest.json`` — one record per candidate
  tape with download / completeness / analysis status and the exclusion
  reason when a session is not aggregated;
* ``docs/results/multi_day_aggregation.json`` / ``.md`` — the existing
  ``aggregate_multi_day_results`` report **plus** the session panel
  (``session_panel_statistics``) over every included session.

A session is included only when: the source GZIP stream reached EOF on an
unbounded fetch; both requested symbols were sliced; the per-day analysis
completed under the current pipeline fingerprint; and each symbol's regular
session is complete (open/close system events present and rows within one
minute of 09:30 and 16:00).  Nothing is included because of its results.

Run (from the repository root)::

    python python_quant/scripts/build_session_manifest.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT / "python_quant") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python_quant"))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_itch
from batch_research_itch import RESEARCH_MANIFEST_NAME, _pipeline_fingerprint
from nexus_quant.research.multi_day_aggregation import (
    aggregate_multi_day_results,
    extract_session_metrics,
    render_multi_day_aggregation_md,
    render_session_panel_md,
    session_panel_statistics,
)

SYMBOLS = ("AAPL", "QQQ")

# Directory audit of https://emi.nasdaq.com/ITCH/ (2026-09-15).  Every tape-like
# object found is listed; the eligible ones are the keys of fetch_itch's catalogues.
DIRECTORY_AUDIT: dict[str, dict[str, Any]] = {
    # ---- original 15-date catalogue -----------------------------------------
    "01302018": {"filename": "01302018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "03292018": {"filename": "03292018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "05302018": {"filename": "05302018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "07302018": {"filename": "07302018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "08302018": {"filename": "08302018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "10302018": {"filename": "10302018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "12282018": {"filename": "12282018.NASDAQ_ITCH50.gz", "http_status": 404, "note": "only the .md5sum sidecar remains in the directory"},
    "01302019": {"filename": "01302019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 4764426091},
    "03272019": {"filename": "03272019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 5510131732},
    "05302019": {
        "filename": "05302019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 4246501580,
        "note": "404 under Nasdaq ITCH/; the NASDAQ tape is published inside the Nasdaq PSX ITCH/ directory (directory fingerprint = ndq_stocklocate_20190530, not the PSX locate file)",
    },
    "07302019": {"filename": "07302019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 3662140094},
    "08302019": {"filename": "08302019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 4075649457},
    "10302019": {"filename": "10302019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 3872931242},
    "12302019": {"filename": "12302019.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 3524013057},
    "01302020": {"filename": "01302020.NASDAQ_ITCH50.gz", "http_status": 200, "listed_size_bytes": 5597158940},
    # ---- additional full-session tapes found in the directory tree -----------
    "12132018": {"filename": "GIS/Nov 18, Dec 18, Jan 19/S121318-v50.txt.gz", "http_status": 200, "listed_size_bytes": 5244679335},
    "12142018": {"filename": "GIS/Nov 18, Dec 18, Jan 19/S121418-v50.txt.gz", "http_status": 200, "listed_size_bytes": 5049507039},
    "12312018": {"filename": "GIS/Nov 18, Dec 18, Jan 19/S123118-v50.txt.gz", "http_status": 200, "listed_size_bytes": 4931688102},
    "10182019": {"filename": "S101819-v50.txt.gz", "http_status": 200, "listed_size_bytes": 3951201663},
    "07132021": {"filename": "S071321-v50.txt.gz", "http_status": 200, "listed_size_bytes": 5996745270},
    "08132021": {"filename": "S081321-v50.txt.gz", "http_status": 200, "listed_size_bytes": 4889328604},
    "11282025": {"filename": "S112825-v50.txt.gz", "http_status": 200, "listed_size_bytes": 4735308661, "note": "day after US Thanksgiving: NYSE/Nasdaq early close 13:00 ET"},
    "12082025": {"filename": "S120825-v50.txt.gz", "http_status": 200, "listed_size_bytes": 8775891119},
    "12092025": {"filename": "S120925-v50.txt.gz", "http_status": 200, "listed_size_bytes": 7929915419},
    "12102025": {"filename": "S121025-v50.txt.gz", "http_status": 200, "listed_size_bytes": 11557662295},
    "12112025": {"filename": "S121125-v50.txt.gz", "http_status": 200, "listed_size_bytes": 10471034001},
    "12122025": {"filename": "S121225-v50.txt.gz", "http_status": 200, "listed_size_bytes": 12551644054},
    "05152026": {"filename": "itch50_05_15.gz", "http_status": 200, "listed_size_bytes": 13047496628},
    "05182026": {"filename": "itch50_05_18.gz", "http_status": 200, "listed_size_bytes": 16150095810},
    "06122026": {"filename": "S061226-v50.txt.gz", "http_status": 200, "listed_size_bytes": 17894268560},
}

# Directory objects that are not NASDAQ TotalView-ITCH 5.0 full-session tapes.
NON_SESSION_OBJECTS: list[dict[str, Any]] = [
    {"filename": "tvagg.gz", "size_bytes": 294755242, "classification": "not_length_prefixed_itch", "reason": "gzip inflates to a raw (unframed) message stream whose first bytes are not a valid 2-byte length prefix; 295 MB is far below a full-day tape; no directory records recovered by the framed reader"},
    {"filename": "S010303-v2.zip", "size_bytes": 58907174, "classification": "incompatible_format", "reason": "ZIP archive containing S010303-v2.txt — a v2 (not ITCH 5.0) sample from 2003; 59 MB"},
    {"filename": "NOII/S050922-v50-NOII.txt.gz", "size_bytes": 96588423, "classification": "noii_only", "reason": "Net Order Imbalance Indicator feed sample, not the order-book message stream"},
    {"filename": "GIS/NOII 2019/12062019.NOII.gz", "size_bytes": 53963965, "classification": "noii_only", "reason": "NOII feed sample only"},
    {"filename": "Nasdaq BX ITCH/*.BX_ITCH_50.gz, GIS/BX Jan 2020/*, Nasdaq BX ITCH/March 20/*", "size_bytes": None, "classification": "different_venue", "reason": "Nasdaq BX venue tapes (different book, ~0.4–1.7 GB); not the NASDAQ book the study is about"},
    {"filename": "Nasdaq PSX ITCH/*.PSX_ITCH_50.gz", "size_bytes": None, "classification": "different_venue", "reason": "Nasdaq PSX venue tapes (different book, ~0.4–1.0 GB); the misfiled NASDAQ tape 05302019.NASDAQ_ITCH50.gz in that directory is catalogued separately"},
    {"filename": "*.md5sum sidecars", "size_bytes": 72, "classification": "checksum_unavailable", "reason": "every listed .md5sum returns HTTP 404, so no server-side checksum could be verified for any tape; fetch manifests record the SHA-256 of the bytes actually received instead"},
    {"filename": "FEB 2022 Files/, NOII Beta Files/, GIS/Aug 5-9 2019/", "size_bytes": None, "classification": "empty_directory", "reason": "directory listings contain no files"},
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iso(day: str) -> str:
    return dt.date(int(day[4:]), int(day[:2]), int(day[2:4])).isoformat()


def _fingerprint_for(url: str, fingerprints: list[dict[str, Any]]) -> dict[str, Any] | None:
    for rec in fingerprints:
        if rec.get("url") == url:
            return rec
    return None


def build_records(*, data_dir: Path, results_dir: Path, probe_path: Path) -> list[dict[str, Any]]:
    fingerprints = json.loads(probe_path.read_text()) if probe_path.exists() else []
    current_fp = _pipeline_fingerprint()
    records: list[dict[str, Any]] = []
    for day, audit in DIRECTORY_AUDIT.items():
        url = fetch_itch.tape_url(day) if audit["http_status"] == 200 else fetch_itch.DEFAULT_BASE + audit["filename"]
        rec: dict[str, Any] = {
            "date": day,
            "date_iso": _iso(day),
            "in_original_catalogue": day in fetch_itch.PUBLIC_SAMPLE_DAYS,
            "source": url,
            "filename": audit["filename"],
            "format": "NASDAQ TotalView-ITCH 5.0, 2-byte length-prefixed, gzip",
            "listed_size_bytes": audit.get("listed_size_bytes"),
            "http_status": audit["http_status"],
            "checksum_sidecar": "unavailable (HTTP 404 for every .md5sum)",
            "note": audit.get("note"),
        }
        fp = _fingerprint_for(url, fingerprints)
        if fp is not None:
            lf = fp.get("locate_fingerprint", {})
            rec["prefix_probe"] = {
                "parser_compatible": fp.get("parser_compatible"),
                "spec_length_mismatches": fp.get("spec_length_mismatches"),
                "n_directory_records": fp.get("n_directory"),
                "AAPL_locate": fp.get("AAPL_locate"),
                "QQQ_locate": fp.get("QQQ_locate"),
                "date_fingerprint": lf.get("status"),
                "date_fingerprint_matching_pairs": f"{lf.get('matching_pairs')}/{lf.get('locate_file_size')}",
            }
        if audit["http_status"] != 200:
            rec.update({
                "size_bytes": None, "download_status": "http_404", "complete_session": False,
                "regular_session_coverage": None, "AAPL_events": None, "QQQ_events": None,
                "analysis_status": "not_run", "included_in_aggregation": False,
                "exclusion_reason": "tape retired from emi.nasdaq.com (HTTP 404); " + audit.get("note", ""),
                "status_class": "404/RETIRED",
            })
            records.append(rec)
            continue

        src = _read_json(data_dir / day / "manifest.json")
        if src is None:
            rec.update({
                "size_bytes": None, "download_status": "not_downloaded", "complete_session": None,
                "regular_session_coverage": None, "AAPL_events": None, "QQQ_events": None,
                "analysis_status": "not_run", "included_in_aggregation": False,
                "exclusion_reason": "not downloaded in this campaign", "status_class": "OTHER",
            })
            records.append(rec)
            continue

        complete_stream = bool(src.get("gzip_stream_complete")) and not bool(src.get("range_limited"))
        rec["size_bytes"] = src.get("gz_bytes_fetched")
        rec["gz_sha256"] = src.get("gz_sha256")
        rec["messages_framed"] = src.get("messages_framed")
        rec["last_tape_clock_hours"] = round(src.get("last_tape_ts_ns", 0) / 3.6e12, 4)
        rec["download_status"] = "complete" if complete_stream else ("partial_prefix" if src.get("range_limited") else "interrupted")
        rec["symbols_not_found"] = src.get("symbols_not_found", [])
        sym_entries = src.get("symbols", {})
        for s in SYMBOLS:
            e = sym_entries.get(s)
            rec[f"{s}_slice_messages"] = sum(e["messages"].values()) if e else None
            rec[f"{s}_slice_sha256"] = e["sha256"] if e else None

        research = _read_json(results_dir / day / RESEARCH_MANIFEST_NAME)
        session: dict[str, Any] = {}
        events: dict[str, int | None] = {s: None for s in SYMBOLS}
        analysis_status = "not_run"
        pipeline_current = None
        if research is not None:
            analysis_status = str(research.get("status"))
            pipeline_current = research.get("pipeline", {}).get("source_sha256") == current_fp
            for s in SYMBOLS:
                payload = _read_json(results_dir / day / f"real_tape_{day}_{s}.json")
                if payload is None:
                    continue
                events[s] = int(payload.get("tape", {}).get("regular_events") or 0)
                session[s] = payload.get("session")
        rec["analysis_status"] = analysis_status
        rec["pipeline_fingerprint_current"] = pipeline_current
        rec["AAPL_events"] = events["AAPL"]
        rec["QQQ_events"] = events["QQQ"]
        rec["regular_session_coverage"] = {
            s: (
                {
                    "complete": v.get("regular_session_complete"),
                    "first_row_et": _clock(v.get("first_regular_row_ns")),
                    "last_row_et": _clock(v.get("last_regular_row_ns")),
                    "system_events_et": {k: _clock(t) for k, t in (v.get("system_events_ns") or {}).items()},
                }
                if isinstance(v, dict) else None
            )
            for s, v in session.items()
        } or None

        session_complete = bool(session) and all(
            isinstance(v, dict) and v.get("regular_session_complete") for v in session.values()
        ) and set(session) == set(SYMBOLS)
        reasons: list[str] = []
        if not complete_stream:
            reasons.append("source GZIP stream did not reach EOF on an unbounded fetch")
        if rec["symbols_not_found"]:
            reasons.append(f"symbols missing from the stock directory: {rec['symbols_not_found']}")
        if research is None:
            reasons.append("E1–E6 analysis not completed")
        elif not analysis_status.startswith("completed") or analysis_status.endswith("no_regular_session_rows"):
            reasons.append(f"analysis status {analysis_status}")
        elif pipeline_current is False:
            reasons.append("analysis predates the current pipeline fingerprint")
        if research is not None and not session_complete:
            reasons.append("regular session not complete for every symbol (see regular_session_coverage)")
        rec["complete_session"] = bool(complete_stream and session_complete) if research is not None else None
        rec["included_in_aggregation"] = not reasons
        rec["exclusion_reason"] = "; ".join(reasons) if reasons else None
        if not reasons:
            rec["status_class"] = "FULL"
        elif not complete_stream:
            rec["status_class"] = "PARTIAL"
        elif research is None:
            rec["status_class"] = "OTHER"
        elif analysis_status == "failed":
            rec["status_class"] = "PROCESSING_FAILED"
        elif not session_complete:
            rec["status_class"] = "PARTIAL"
        else:
            rec["status_class"] = "OTHER"
        records.append(rec)
    return records


def _clock(ns: int | None) -> str | None:
    if ns is None:
        return None
    s = int(ns) // 10**9
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def build_aggregation(records: list[dict[str, Any]], *, results_dir: Path) -> tuple[dict[str, Any], str]:
    included = [r for r in records if r["included_in_aggregation"]]
    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for r in included:
        day = r["date"]
        for s in SYMBOLS:
            payload = _read_json(results_dir / day / f"real_tape_{day}_{s}.json")
            assert payload is not None
            payload = dict(payload)
            payload["day"] = day
            payload["symbol"] = s
            payload["coverage"] = {"kind": "full_day", "is_full_day": True}
            payloads.append(payload)
            rows.append(extract_session_metrics(payload, day=day, symbol=s))
    for r in records:
        if not r["included_in_aggregation"]:
            payloads.append({"day": r["date"], "error": r["exclusion_reason"]})
    summary = aggregate_multi_day_results(payloads)
    panel = session_panel_statistics(rows)
    summary["session_panel"] = panel
    summary["session_rows"] = rows
    summary["generated_at_unix"] = int(time.time())
    summary["pipeline_fingerprint"] = _pipeline_fingerprint()
    summary["selection_rule"] = (
        "A session enters only if its unbounded GZIP fetch reached EOF, both symbols were sliced, the per-day "
        "E1-E6 analysis completed under the current pipeline fingerprint, and each symbol's regular session is "
        "complete (Q/M system events present; rows within 60 s of 09:30 and 16:00 ET). No session was selected or "
        "dropped on the basis of its results."
    )
    md = render_multi_day_aggregation_md(summary)
    md = md.rstrip("\n") + "\n\n" + render_session_panel_md(panel, rows)
    return summary, md


def _render_manifest_md(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Multi-day ITCH session manifest",
        "",
        (
            "Every candidate object in the public NASDAQ ITCH directory tree, with download, completeness and analysis status. "
            "Only `FULL` rows enter the cross-day aggregation. `AAPL/QQQ events` are regular-session (09:30–16:00 ET) replay rows."
        ),
        "",
        "| Date | Orig. cat. | Filename | HTTP | Size (bytes) | Download | Regular session complete | AAPL events | QQQ events | Analysis | Included | Status | Exclusion reason |",
        "|---|---|---|---:|---:|---|---|---:|---:|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: r["date_iso"]):
        cov = r.get("regular_session_coverage") or {}
        cov_txt = ", ".join(f"{s}: {'yes' if (cov.get(s) or {}).get('complete') else 'no'}" for s in SYMBOLS if cov.get(s)) or "—"
        lines.append(
            f"| {r['date_iso']} | {'yes' if r['in_original_catalogue'] else 'no'} | `{r['filename']}` | {r['http_status']} | "
            f"{r['size_bytes'] if r.get('size_bytes') is not None else (r.get('listed_size_bytes') or '—')} | {r.get('download_status')} | {cov_txt} | "
            f"{r['AAPL_events'] if r.get('AAPL_events') is not None else '—'} | {r['QQQ_events'] if r.get('QQQ_events') is not None else '—'} | "
            f"{r.get('analysis_status')} | {'yes' if r['included_in_aggregation'] else 'no'} | {r['status_class']} | {r.get('exclusion_reason') or '—'} |"
        )
    lines += ["", "## Directory objects that are not full-session NASDAQ ITCH 5.0 tapes", "", "| Object | Size | Classification | Reason |", "|---|---:|---|---|"]
    for o in NON_SESSION_OBJECTS:
        lines.append(f"| `{o['filename']}` | {o['size_bytes'] if o['size_bytes'] is not None else '—'} | {o['classification']} | {o['reason']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "itch")
    ap.add_argument("--results-dir", type=Path, default=_ROOT / "docs" / "results" / "multi_day")
    ap.add_argument("--probe", type=Path, default=_ROOT / "data" / "probe" / "fingerprints.json")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "docs" / "results")
    args = ap.parse_args(argv)

    records = build_records(data_dir=args.data_dir, results_dir=args.results_dir, probe_path=args.probe)
    manifest = {
        "kind": "nexus-lob-itch-session-manifest",
        "generated_at_unix": int(time.time()),
        "directory_root": "https://emi.nasdaq.com/ITCH/",
        "symbols": list(SYMBOLS),
        "pipeline_fingerprint": _pipeline_fingerprint(),
        "n_candidates": len(records),
        "n_included": sum(1 for r in records if r["included_in_aggregation"]),
        "status_counts": {k: sum(1 for r in records if r["status_class"] == k) for k in sorted({r["status_class"] for r in records})},
        "sessions": records,
        "non_session_objects": NON_SESSION_OBJECTS,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "session_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    (args.results_dir / "session_manifest.md").write_text(_render_manifest_md(records), encoding="utf-8")
    print(f"manifest: {manifest['n_included']}/{manifest['n_candidates']} candidates included; status counts {manifest['status_counts']}")

    if manifest["n_included"] == 0:
        print("no complete sessions — aggregation not regenerated")
        return 1
    summary, md = build_aggregation(records, results_dir=args.results_dir)
    (args.out_dir / "multi_day_aggregation.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    (args.out_dir / "multi_day_aggregation.md").write_text(md, encoding="utf-8")
    print(f"aggregation: {summary['n_days']} days, {summary['session_panel']['n_sessions']} sessions → {args.out_dir / 'multi_day_aggregation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
