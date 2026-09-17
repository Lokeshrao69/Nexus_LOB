"""Offline regressions for quick/full research workflow isolation."""
from __future__ import annotations

import io
import json
import subprocess
import zlib
from pathlib import Path

import pytest
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent, encode_event

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_DAY = "12302019"
_QUICK_CAP = 64 << 20


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    import run_all

    monkeypatch.setattr(run_all, "_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    return run_all


def _option(cmd: list[str], name: str, default: str) -> str:
    return cmd[cmd.index(name) + 1] if name in cmd else default


@pytest.mark.parametrize("quick", [False, True])
@pytest.mark.parametrize("skip_fetch", [False, True])
def test_modes_route_fetch_and_research_to_the_same_input_root(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quick: bool, skip_fetch: bool,
) -> None:
    commands: dict[str, list[str]] = {}

    def capture(label: str, cmd: list[str]) -> float:
        commands[label] = cmd
        return 0.0

    monkeypatch.setattr(runner, "_run", capture)
    args = ["--day", "01302020", "--symbols", "QQQ,AAPL"]
    if quick:
        args += ["--quick"]
    if skip_fetch:
        args += ["--skip", "fetch"]
    assert runner.main(args) == 0
    assert set(commands) == set(runner.STAGES) - ({"fetch"} if skip_fetch else set())

    data_root = tmp_path / "data" / ("itch_quick" if quick else "itch")
    results_root = tmp_path / "docs" / "results"
    if quick:
        results_root /= "quick"
    research = commands["research"]
    assert Path(_option(research, "--data", "data/itch")).resolve() == data_root
    assert Path(_option(research, "--out-dir", str(tmp_path / "docs" / "results"))) == results_root
    assert _option(research, "--symbols", "") == ("QQQ" if quick else "QQQ,AAPL")
    assert _option(research, "--day", "") == "01302020"
    assert ("--max-events" in research) is quick
    if not skip_fetch:
        fetch = commands["fetch"]
        assert Path(_option(fetch, "--out", "data/itch")).resolve() == data_root
        assert _option(fetch, "--symbols", "") == _option(research, "--symbols", "")
        assert _option(fetch, "--day", "") == "01302020"
        assert ("--max-gz-bytes" in fetch) is quick
        if quick:
            assert int(_option(fetch, "--max-gz-bytes", "0")) == _QUICK_CAP
    if quick:
        assert Path(_option(commands["fairness"], "--out", "")) == results_root / "rl_fairness.json"


def _framed(body: bytes) -> bytes:
    return len(body).to_bytes(2, "big") + body


def _directory(locate: int, stock: str, ts: int) -> bytes:
    body = (
        b"R" + locate.to_bytes(2, "big") + b"\0\0" + ts.to_bytes(6, "big")
        + stock.ljust(8).encode() + b"QN" + (100).to_bytes(4, "big")
        + b"NQN  NN N" + (0).to_bytes(4, "big") + b"N"
    )
    assert len(body) == 39
    return _framed(body)


def _compressed_tape() -> tuple[bytes, bytes]:
    ts = 34_200_000_000_000
    prefix = b"".join([
        _framed(b"S\0\0\0\0" + (ts - 2).to_bytes(6, "big") + b"O"),
        _directory(1, "AAPL", ts - 1),
        _directory(2, "QQQ", ts - 1),
        _framed(b"S\0\0\0\0" + ts.to_bytes(6, "big") + b"Q"),
        encode_event(NormalizedEvent(EventType.ADD, ts + 1, 1, Side.Bid, 10_000, 100)),
        encode_event(NormalizedEvent(EventType.ADD, ts + 2, 2, Side.Ask, 10_010, 100)),
    ])
    suffix = b"".join([
        encode_event(NormalizedEvent(EventType.ADD, ts + 3, 3, Side.Bid, 10_000, 200)),
        _framed(b"S\0\0\0\0" + (57_600_000_000_000).to_bytes(6, "big") + b"M"),
    ])
    compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    partial = compressor.compress(prefix) + compressor.flush(zlib.Z_SYNC_FLUSH)
    full = partial + compressor.compress(suffix) + compressor.flush()
    return full, partial


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_two_quick_runs_preserve_full_day_slices_and_manifests_byte_for_byte(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import batch_research_itch as batch
    import fetch_itch
    import run_research

    full, partial = _compressed_tape()
    ranges: list[str | None] = []

    def urlopen(request, timeout: int):
        assert request.full_url == fetch_itch.tape_url(_DAY)
        requested_range = request.get_header("Range")
        ranges.append(requested_range)
        return io.BytesIO(full if requested_range is None else partial)

    monkeypatch.setattr(fetch_itch.urllib.request, "urlopen", urlopen)
    canonical = tmp_path / "data" / "itch"
    fetch_itch.fetch(day=_DAY, symbols={"AAPL", "QQQ"}, out_root=canonical, log=lambda _: None)
    valid, errors = batch.validate_fetch_manifest(_DAY, ["AAPL", "QQQ"], canonical, max_gz_bytes=None)
    assert valid is not None and errors == []
    before = _snapshot(canonical)

    results = tmp_path / "docs" / "results"
    results.mkdir(parents=True)
    canonical_report = results / f"real_tape_{_DAY}_AAPL.json"
    canonical_report.write_bytes(b"canonical full-day report\n")
    monkeypatch.setattr(run_research, "_ROOT", tmp_path)

    def run(cmd: list[str], *, cwd: Path, check: bool):
        assert cwd == tmp_path and check is True
        stages = {"fetch_itch.py": fetch_itch.main, "run_research.py": run_research.main}
        assert Path(cmd[1]).name in stages  # Never launch training or recursive pytest.
        assert stages[Path(cmd[1]).name](cmd[2:]) == 0
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    quick_root = tmp_path / "data" / "itch_quick"
    previous_slice = None
    previous_manifest = None
    for _ in range(2):
        assert runner.main(["--quick", "--skip", "tests,vignette,fairness"]) == 0
        assert _snapshot(canonical) == before
        assert canonical_report.read_bytes() == b"canonical full-day report\n"

        quick_slice = (quick_root / _DAY / "AAPL.itch").read_bytes()
        assert quick_slice != before[Path(_DAY) / "AAPL.itch"]
        assert not (quick_root / _DAY / "QQQ.itch").exists()
        manifest = json.loads((quick_root / _DAY / "manifest.json").read_text())
        assert manifest["range_limited"] is True
        assert manifest["max_gz_bytes_requested"] == _QUICK_CAP
        assert manifest["gzip_stream_complete"] is False
        assert set(manifest["symbols"]) == {"AAPL"}
        assert manifest["symbols_not_found"] == []
        partial_valid, errors = batch.validate_fetch_manifest(
            _DAY, ["AAPL"], quick_root, max_gz_bytes=_QUICK_CAP,
        )
        assert partial_valid is not None and errors == []
        full_valid, errors = batch.validate_fetch_manifest(_DAY, ["AAPL"], quick_root, max_gz_bytes=None)
        assert full_valid is None and any("range_limited" in error for error in errors)

        report = json.loads((results / "quick" / f"real_tape_{_DAY}_AAPL.json").read_text())
        assert report["tape"]["events"] == 2  # The full-day fixture has three events.
        manifest.pop("fetched_at_unix")
        if previous_slice is not None:
            assert quick_slice == previous_slice
            assert manifest == previous_manifest
        previous_slice, previous_manifest = quick_slice, manifest

    assert ranges == [None, f"bytes=0-{_QUICK_CAP - 1}", f"bytes=0-{_QUICK_CAP - 1}"]
