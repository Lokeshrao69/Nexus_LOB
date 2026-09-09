# Person B Handoff: Python Quantitative Layer

**Updated:** 2026-09-04
**Checkout:** `C:\Users\Shrikar\nexus\parent-transfer`
**Branch:** `main` at `d40bf59` (tracking `origin/main`)

## Scope

Person B owns the Python quantitative layer: ITCH decoding, L2 replay, the
Gymnasium execution environment, book adapters, baselines, and their tests.
The C++ matching engine, pybind bridge, ABI parity, and shared contract belong
to Person A.

## Current Status

The Person-B implementation is now present on upstream `main`. PR #2 ("Add
Person B execution environment and ITCH replay") was merged into `main` on
2026-09-03 as
`bf08948c6ba48e0a76b8f2ac515be5721f33b9ca`; it is closed, with no review or
issue comments. GitHub `main` and local `origin/main` are now synchronized at
`d40bf59` (2026-09-04).

Lokesh's `d40bf59` follow-up adds the missing `EngineAdapter.lookup()` and
`EngineAdapter.cancel_id()` methods in `book_port.py`, adds
`bindings/tests/test_diff_engine_stub.py`, and updates the handoff docs. The
partial-cancel path uses same-price `engine.modify()` to preserve priority.
That fix is present in the checked-out `main` branch.

## Completed Files

The original Person-B branch contributed 11 Python-layer files and 1,511 added
lines:

- `python_quant/nexus_quant/itch_parser.py`
- `python_quant/nexus_quant/replay.py`
- `python_quant/nexus_quant/book_port.py`
- `python_quant/nexus_quant/envs/order_book_env.py`
- `python_quant/nexus_quant/envs/__init__.py`
- `python_quant/nexus_quant/baselines.py`
- `python_quant/nexus_quant/__init__.py`
- `python_quant/requirements.txt`
- `python_quant/tests/test_itch_parser.py`
- `python_quant/tests/test_replay.py`
- `python_quant/tests/test_order_book_env.py`

## Frozen Boundaries

Do not modify `cpp_engine/**`, `bindings/pybind_wrapper.cpp`,
`bindings/CONTRACT.md`, `python_quant/nexus_quant/book_state.py`,
`BOOK_STATE_DTYPE`, `DEPTH`, `Side`, field order/names, integer-tick pricing,
or the `view()` zero-copy contract. No frontend, TypeScript, or React work is
part of this repository task.

## ITCH Parser

`itch_parser.py` is a lazy, bounded-chunk NASDAQ TotalView-ITCH 5.0 parser.
It decodes A/F/E/C/X/D/U/P messages, big-endian fields, 6-byte timestamps to
`ts_ns`, 8-byte order IDs, and integer `price_ticks` without float conversion.
It accepts raw concatenated messages and 2-byte length-prefixed messages.
Unknown types and a truncated tail are counted in `ItchParseStats` and do not
raise. Events are normalized as `NormalizedEvent` values.

## Replay

`ReplayEngine` applies normalized events to an injectable adapter and emits
owning `ReplayFrame` snapshots. A/F rest orders; E/C execute and reduce/remove
orders; X partially cancels; D fully cancels; U cancels and replaces; P records
a trade without displayed-size removal. `check_integrity()` checks crossed or
locked BBO, empty BBO, negative sizes, and ladder ordering.

## Book Adapter

`book_port.py` defines the view/execution protocols, `StubBookAdapter` over the
frozen `StubOrderBook`, and the `EngineAdapter` swap seam for
`nexus_engine.Engine`. It supports rest/take/cancel, zero-copy view vs owning
snapshot semantics, and the `lookup()`/`cancel_id()` methods required by
real-engine replay. Partial cancellation uses same-price `engine.modify()` to
preserve priority.

## OrderBookEnv

`OrderBookEnv` is a long-inventory liquidation environment with defaults
`Q=2000`, `T=40`, a continuous `Box([-1, 1], shape=(1,))` action, and an exact
44-element float32 observation:

- 0-9 bid distance from mid; 10-19 bid size
- 20-29 ask distance from mid; 30-39 ask size
- 40 inventory/Q; 41 remaining time/T
- 42 mark-to-market PnL normalized by `Q*10` and clipped to `[-3, 3]`
- 43 spread normalized by 20 ticks

Sell semantics: `a <= -0.92` sends a market sell; other actions map to
`round(a*12)` ticks around mid, with crossing limits capped to market behavior.
The previous residual is canceled each step. Child size is a clamped
ceil(inventory/time-left), with minimum 20 and configurable maximum. Horizon
leftovers are market-dumped. Arrival mid at reset benchmarks implementation
shortfall; reward combines normalized shortfall, inventory/time pressure,
adverse-mid movement, and an extra leftover penalty at truncation. Metrics
include VWAP, shortfall bps, filled/leftover quantity, PnL ticks, reward, and
steps.

## Baselines

`baselines.py` supplies TWAP, VWAP, POV, and Passive policies, plus
`run_episode()` and `compare()` returning execution metrics compatible with the
environment.

## Verification

Latest checks in this shell:

- `python -m compileall -q python_quant`: **PASS**
- Direct parser/replay smoke (encode one ADD, decode, replay into stub): **PASS**
- `python -m pytest python_quant/tests -q`: **NOT RUN**; Python 3.10.2 is
  available, but `pytest` is not installed (`No module named pytest`).
- Dependency probe: `numpy` installed; `gymnasium` and `pytest` absent.

There are 26 pytest test functions across the four Python test modules,
including the existing contract smoke test. The previously reported 26-pass
result is historical and was not reproducible in this checkout without the
missing dependencies. Run `pip install -r python_quant/requirements.txt` and
pytest in WSL/venv.

## Integration Blockers and Limitations

- The compiled `nexus_engine` module and ABI parity tests still require the
  Linux/WSL CMake + pybind environment described in `CLAUDE.md`.
- Run `pytest python_quant/tests bindings/tests/test_abi_parity.py
  bindings/tests/test_diff_engine_stub.py -v` after building the module.
- The real-engine replay fix is now on `main`; the remaining requirement is to
  build `nexus_engine` and run the parity/diff tests.
- No PPO/GRPO training agent, CUDA VaR/CVaR engine, or Python dashboard is
  implemented yet.
- Replay is an aggregate L2 reconstruction and intentionally does not retain a
  full ITCH market-state model beyond the adapter's tracked resting orders.

## Next Steps

1. In WSL, build the pybind module and run ABI, Python, and Engine-vs-Stub
   diff-test suites.
2. Confirm the remote-main `EngineAdapter` fix and diff-test pass against the
   real engine without changing frozen files.
3. Continue with the separate RL agent, risk analytics, and dashboard work when
   those milestones are scheduled.

## Files Changed This Turn

- `progress_b.md` (this handoff only)

---

## Update 2026-09-07 — C++ integration on Linux

**Branch:** `feature/person-b-polish` (PR #7)
**Engine build:** `cmake -S . -B /tmp/nexus_build -DCMAKE_BUILD_TYPE=Release -DNEXUS_BUILD_PYBIND=ON -DNEXUS_ENABLE_CUDA=OFF` then `cmake --build`. Module: `bindings/nexus_engine.cpython-310-x86_64-linux-gnu.so` (gitignored).

### Completed this session
- Built `nexus_engine` on Linux (g++ 12, CMake 4.4, pybind11, Python 3.10). No C++ / pybind / contract edits.
- CTest 5/5: abi_check, lob_test, ring_test, id_map_test, risk_test.
- `abi_check`: sizeof 448, alignof 8, contract v1.
- Python+bindings pytest: **75 passed, 0 skipped** (was 59 passed / 3 skipped before the .so existed).
- Risk parity `test_risk_parity.py`: **3/3**. Project check is `pytest.approx(..., abs=1e-12)`, not raw `==`. Example GBM 60k×120: engine var `0.3253218005627372` vs NumPy `0.3253218005627373`.
- Engine-vs-Stub `test_diff_engine_stub.py`: **2/2**.
- ABI `test_abi_parity.py`: **6/6**.
- `EngineAdapter.reset()` added so `OrderBookEnv.reset()` no longer silently swaps in `StubBookAdapter`. Limit / market / cancel / fills / env inventory identity verified on the real engine.
- Dashboard `read_shm_ring_latest()` decodes Person A's control block (48 B) + latest 448 B slot. Live check against `ring_producer /nex_lob_b`: seq/BBO/spread decoded; producer unlinks the segment on exit (unchanged C++ behavior).

### GPU
Blocked. No `nvcc` / `nvidia-smi` on this machine. `risk_bench` CPU path: 200k×252 in 1603.1 ms, `var=0.324574 cvar=0.389231`. GPU line: "not compiled (no CUDA toolkit)".

### Tests run
```
ctest --test-dir /tmp/nexus_build --output-on-failure     # 5/5
PYTHONPATH=python_quant:bindings python -m pytest python_quant/tests bindings/tests -q
# 75 passed
PYTHONPATH=python_quant:bindings python -m pytest \
  python_quant/tests/test_risk_parity.py \
  bindings/tests/test_diff_engine_stub.py \
  bindings/tests/test_abi_parity.py -v
# 11 passed
/tmp/nexus_build/risk_bench   # CPU only
```

### Remaining
- GPU `risk_bench` + ~40× claim on a CUDA box.
- Keep a producer alive if the dashboard should follow a live ring (C++ demo calls `ShmRing::destroy` on exit).
- Push/merge PR #7 on `Lokeshrao69/Nexus_LOB` after Person A review.
