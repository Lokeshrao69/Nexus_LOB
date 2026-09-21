"""Deterministic regression tests for session manifest integrity and slice validation.

Verifies that build_session_manifest.py enforces strict integrity checks:
1. Both slices valid (full session included)
2. AAPL slice missing (rejected, classified INVALID, excluded)
3. QQQ slice missing (rejected, classified INVALID, excluded)
4. Invalid/corrupt slice (hash/size mismatch, classified CORRUPT, excluded)
5. Incomplete source/fetch (GZIP stream did not reach EOF, classified PARTIAL, excluded)
6. Malformed/corrupt manifest.json (classified CORRUPT, excluded)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "python_quant" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_itch
from batch_research_itch import (
    ADVERSE_HORIZONS,
    FILL_HORIZONS,
    HORIZONS,
    N_BOOT,
    RESEARCH_MANIFEST_NAME,
    SCHEMA_VERSION,
    _pipeline_fingerprint,
    _sha256,
    check_completed_research,
    check_source_manifest,
)
from build_session_manifest import build_records

TEST_DAY = "12302019"


def _make_mock_environment(
    tmp_path: Path,
    *,
    aapl_content: bytes = b"mock_aapl_itch_stream_data_12345",
    qqq_content: bytes = b"mock_qqq_itch_stream_data_67890",
    gzip_stream_complete: bool = True,
    range_limited: bool = False,
    corrupt_aapl_hash: bool = False,
    omit_aapl_file: bool = False,
    omit_qqq_file: bool = False,
    corrupt_manifest_json: bool = False,
) -> tuple[Path, Path, Path]:
    """Create a self-contained mock directory structure for build_records testing."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    probe_path = tmp_path / "probe.json"
    probe_path.write_text("[]", encoding="utf-8")

    day_data = data_dir / TEST_DAY
    day_data.mkdir(parents=True, exist_ok=True)
    day_results = results_dir / TEST_DAY
    day_results.mkdir(parents=True, exist_ok=True)

    aapl_hash = hashlib.sha256(aapl_content).hexdigest()
    qqq_hash = hashlib.sha256(qqq_content).hexdigest()

    if not omit_aapl_file:
        (day_data / "AAPL.itch").write_bytes(aapl_content)
    if not omit_qqq_file:
        (day_data / "QQQ.itch").write_bytes(qqq_content)

    manifest_aapl_hash = "f" * 64 if corrupt_aapl_hash else aapl_hash
    src_manifest = {
        "day": TEST_DAY,
        "source_url": fetch_itch.tape_url(TEST_DAY),
        "range_limited": range_limited,
        "max_gz_bytes_requested": None if not range_limited else 50000000,
        "gzip_stream_complete": gzip_stream_complete,
        "gz_bytes_fetched": 3524013057,
        "gz_sha256": "a" * 64,
        "messages_framed": 20000,
        "last_tape_ts_ns": int(16.5 * 3.6e12),
        "symbols": {
            "AAPL": {
                "file": "AAPL.itch",
                "bytes": len(aapl_content),
                "sha256": manifest_aapl_hash,
                "messages": {"A": 500},
            },
            "QQQ": {
                "file": "QQQ.itch",
                "bytes": len(qqq_content),
                "sha256": qqq_hash,
                "messages": {"A": 500},
            },
        },
    }

    manifest_file = day_data / "manifest.json"
    if corrupt_manifest_json:
        manifest_file.write_text("{corrupt: json content", encoding="utf-8")
    else:
        manifest_file.write_text(json.dumps(src_manifest, indent=2), encoding="utf-8")

    # Create research result files
    open_ns = 9 * 3600 * 10**9 + 30 * 60 * 10**9
    close_ns = 16 * 3600 * 10**9
    for sym in ("AAPL", "QQQ"):
        res_payload = {
            "tape": {"regular_events": 1000},
            "ic": {"rows": [{"feature": "lob_imbalance", "horizon_h": 5, "ic_test": 0.15, "n_test": 500}]},
            "fill": {"n_orders_regular": 200},
            "adverse": {"n_fills": 30},
            "session": {
                "regular_session_complete": True,
                "first_regular_row_ns": open_ns + 10 * 10**9,
                "last_regular_row_ns": close_ns - 10 * 10**9,
                "system_events_ns": {"Q": open_ns, "M": close_ns},
            },
        }
        res_file = day_results / f"real_tape_{TEST_DAY}_{sym}.json"
        res_file.write_text(json.dumps(res_payload, indent=2), encoding="utf-8")

    md_file = day_results / f"real_tape_{TEST_DAY}.md"
    md_file.write_text("# Report\n\nContent", encoding="utf-8")

    coverage = {
        "kind": "full_day" if gzip_stream_complete and not range_limited else "partial_gzip_prefix",
        "is_full_day": gzip_stream_complete and not range_limited,
        "range_limited": range_limited,
        "max_gz_bytes_requested": None if not range_limited else 50000000,
        "gz_bytes_fetched": 3524013057,
        "gzip_stream_complete": gzip_stream_complete,
        "last_tape_ts_ns": int(16.5 * 3.6e12),
    }
    research_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nexus-lob-itch-day-research",
        "day": TEST_DAY,
        "requested_symbols": ["AAPL", "QQQ"],
        "status": "completed" if coverage["is_full_day"] else "completed_partial",
        "completed_at_unix": int(time.time()),
        "coverage": coverage,
        "source": {
            "source_url": fetch_itch.tape_url(TEST_DAY),
            "manifest_file": "manifest.json",
            "manifest_sha256": _sha256(manifest_file) if not corrupt_manifest_json else "0" * 64,
            "gz_sha256": "a" * 64,
            "slices": {
                "AAPL": {"file": "AAPL.itch", "bytes": len(aapl_content), "sha256": manifest_aapl_hash},
                "QQQ": {"file": "QQQ.itch", "bytes": len(qqq_content), "sha256": qqq_hash},
            },
        },
        "pipeline": {
            "source_script": "python_quant/scripts/run_research.py",
            "source_sha256": _pipeline_fingerprint(),
            "experiments": ["E1-E4", "E5", "E6"],
            "regular_session_only": True,
            "horizons": list(HORIZONS),
            "fill_horizons_events": list(FILL_HORIZONS),
            "adverse_horizons_events": list(ADVERSE_HORIZONS),
            "n_boot": N_BOOT,
            "pooled_cross_day_inference": False,
        },
        "symbols": {
            "AAPL": {
                "file": f"real_tape_{TEST_DAY}_AAPL.json",
                "sha256": _sha256(day_results / f"real_tape_{TEST_DAY}_AAPL.json"),
                "regular_events": 1000,
                "e1_e4_rows": 1,
                "e5_orders": 200,
                "e6_passive_fills": 30,
                "regular_session_complete": True,
            },
            "QQQ": {
                "file": f"real_tape_{TEST_DAY}_QQQ.json",
                "sha256": _sha256(day_results / f"real_tape_{TEST_DAY}_QQQ.json"),
                "regular_events": 1000,
                "e1_e4_rows": 1,
                "e5_orders": 200,
                "e6_passive_fills": 30,
                "regular_session_complete": True,
            },
        },
        "markdown": {"file": md_file.name, "sha256": _sha256(md_file)},
    }
    (day_results / RESEARCH_MANIFEST_NAME).write_text(json.dumps(research_manifest, indent=2), encoding="utf-8")

    return data_dir, results_dir, probe_path


def _get_test_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    for r in records:
        if r["date"] == TEST_DAY:
            return r
    raise ValueError(f"Record for {TEST_DAY} not found")


def test_manifest_integrity_both_slices_valid(tmp_path: Path) -> None:
    """Case A: When both slices exist, match hashes, and stream reached EOF, session is FULL."""
    data_dir, results_dir, probe_path = _make_mock_environment(tmp_path)
    records = build_records(data_dir=data_dir, results_dir=results_dir, probe_path=probe_path)
    rec = _get_test_record(records)

    assert rec["complete_session"] is True
    assert rec["included_in_aggregation"] is True
    assert rec["status_class"] == "FULL"
    assert rec["download_status"] == "complete"
    assert rec["exclusion_reason"] is None


def test_manifest_integrity_aapl_missing(tmp_path: Path) -> None:
    """Case B: When AAPL slice file is missing, session is rejected and classified INVALID."""
    data_dir, results_dir, probe_path = _make_mock_environment(tmp_path, omit_aapl_file=True)
    records = build_records(data_dir=data_dir, results_dir=results_dir, probe_path=probe_path)
    rec = _get_test_record(records)

    assert rec["complete_session"] is False
    assert rec["included_in_aggregation"] is False
    assert rec["status_class"] == "INVALID"
    assert rec["download_status"] == "slice_missing"
    assert "missing AAPL slice" in rec["exclusion_reason"]


def test_manifest_integrity_qqq_missing(tmp_path: Path) -> None:
    """Case C: When QQQ slice file is missing, session is rejected and classified INVALID."""
    data_dir, results_dir, probe_path = _make_mock_environment(tmp_path, omit_qqq_file=True)
    records = build_records(data_dir=data_dir, results_dir=results_dir, probe_path=probe_path)
    rec = _get_test_record(records)

    assert rec["complete_session"] is False
    assert rec["included_in_aggregation"] is False
    assert rec["status_class"] == "INVALID"
    assert rec["download_status"] == "slice_missing"
    assert "missing QQQ slice" in rec["exclusion_reason"]


def test_manifest_integrity_corrupt_slice_hash(tmp_path: Path) -> None:
    """Case D: When a slice has a hash/size mismatch, session is rejected and classified CORRUPT."""
    data_dir, results_dir, probe_path = _make_mock_environment(tmp_path, corrupt_aapl_hash=True)
    records = build_records(data_dir=data_dir, results_dir=results_dir, probe_path=probe_path)
    rec = _get_test_record(records)

    assert rec["complete_session"] is False
    assert rec["included_in_aggregation"] is False
    assert rec["status_class"] == "CORRUPT"
    assert rec["download_status"] == "corrupt"
    assert "mismatch for AAPL slice" in rec["exclusion_reason"]


def test_manifest_integrity_incomplete_source_fetch(tmp_path: Path) -> None:
    """Case E: When source fetch did not reach EOF, session is rejected and classified PARTIAL."""
    data_dir, results_dir, probe_path = _make_mock_environment(tmp_path, gzip_stream_complete=False)
    records = build_records(data_dir=data_dir, results_dir=results_dir, probe_path=probe_path)
    rec = _get_test_record(records)

    assert rec["complete_session"] is False
    assert rec["included_in_aggregation"] is False
    assert rec["status_class"] == "PARTIAL"
    assert rec["download_status"] == "interrupted"
    assert "unbounded fetch did not reach the end of the GZIP stream" in rec["exclusion_reason"]


def test_manifest_integrity_corrupt_json_file(tmp_path: Path) -> None:
    """Case F: When manifest.json contains invalid JSON, session is rejected and classified CORRUPT."""
    data_dir, results_dir, probe_path = _make_mock_environment(tmp_path, corrupt_manifest_json=True)
    records = build_records(data_dir=data_dir, results_dir=results_dir, probe_path=probe_path)
    rec = _get_test_record(records)

    assert rec["complete_session"] is False
    assert rec["included_in_aggregation"] is False
    assert rec["status_class"] == "CORRUPT"
    assert rec["download_status"] == "corrupt"
    assert "manifest corrupt or invalid JSON" in rec["exclusion_reason"]


def test_existing_validators_exported() -> None:
    """check_source_manifest and check_completed_research are exported and callable."""
    assert callable(check_source_manifest)
    assert callable(check_completed_research)
