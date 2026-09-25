# Nexus-LOB

A hybrid **C++/Python** limit-order-book and market-microstructure research
platform for quantitative trading and execution research.

It consists of a zero-allocation C++20 matching engine, a pure-NumPy execution
and research layer (ITCH parsing, Gymnasium environment, PPO/GRPO agents), a
CUDA Monte-Carlo risk engine, and a replay-style execution desk. Every subsystem
is test-verified; every research claim below is labeled with the scope it was
validated on.

---

## Validated results

| Area | Validated status |
|---|---|
| C++ matching engine | **0 allocations/op proven** on the hot path; real-hardware throughput/latency benchmark pending |
| Real-tape microstructure research | E1–E6 validated across 15 full-day sessions (2018–2025, AAPL/QQQ); see multi-day batch results in `docs/results/multi_day/` |
| Fair RL evaluation | Original "+50.4% vs VWAP" PPO headline **retired**; meaningful PPO advantage observed only under liquidity-shock conditions |
| CUDA risk engine | CPU/reference implementation verified; CPU↔NumPy parity bit-for-bit; GPU benchmark pending a CUDA environment |

> **Scope.** Real-tape results are reported across 15 validated NASDAQ ITCH
> sessions (AAPL/QQQ, see `docs/results/multi_day/`). Each day is evaluated
> under strict walk-forward splits with block-bootstrap CIs, without cross-day pooling.

---

## What the research found

Validated on the single real-tape session where noted, on synthetic control
regimes otherwise:

- **L1 imbalance predicts near-term mid moves** on the validated session. Rank IC
  rises with the forecast horizon: ≈0.14 (h=1) → ≈0.23 (h=25) on AAPL and
  ≈0.16 → ≈0.46 on QQQ, with tight block-bootstrap CIs (~±0.01).
- **Microprice does not consistently add information over simpler features**
  (E2). On 1-tick books, microprice ≈ L1 imbalance and does not beat it.
- **Passive fills are strongly adversely selected** on the validated session:
  96–99% of filled passive orders are run over by price shortly after fill.
- **Fill models** are well-calibrated on the real tape (calibration slope
  1.03–1.09), but on the *synthetic trending* tape a logistic fill model ties the
  base-rate baseline — limited explanatory power on synthetic flow. Queue-position
  signal is degenerate on synthetic flow (the generator always fills at the queue
  front); real order-level data is the right substrate.
- **PPO's advantage is regime-specific, not general** (see the RL fairness study).

> **Statistical ≠ tradable.** A predictive relationship does not imply executable
> alpha. These are statistical findings; economic significance depends on spread,
> fees, latency, queue position, and market impact.

---

## Architecture

Three test-verified subsystems layered on one frozen cross-language state contract
(see [The architecture seam](#the-architecture-seam)).

### 1 · C++ matching engine (`cpp_engine/`)
Header-only C++20, zero-allocation on the hot path, prices as integer ticks.

| Piece | File | State |
|---|---|---|
| Frozen state contract | `include/nexus/book_state.hpp` | ABI-locked v1, `sizeof == 448`, `alignof == 8` |
| Engine value types | `include/nexus/types.hpp` | `OrderId/Price/Qty`, `Fill`, `Status`, `ExecResult` |
| Zero-alloc order pool | `include/nexus/order_pool.hpp` | fixed slab + intrusive free-list |
| Matching engine | `include/nexus/limit_order_book.hpp` | Limit/Market/FOK/IOC/Cancel/Modify, FIFO, O(1) level lookup |
| Zero-alloc id→order map | `include/nexus/limit_order_book.hpp` (`IdMap`) | replaces `std::unordered_map`; benchmark proves 0 allocations/op |
| Shared-memory SPSC ring | `include/nexus/shm_ring.hpp` | POSIX + Windows shmem, drop-new-on-full, 448-B slots |
| Synthetic flow generator | `include/nexus/flow_gen.hpp` | seeded, deterministic pre-ITCH flow |
| Cross-process demo | `demos/ring_producer.cpp`, `ring_probe.cpp` | book → ring → reader, verified 0 dropped |
| Benchmark | `bench/bench.cpp` | throughput + latency harness; 0 allocations/op |

**Verified:** `lob_test` **86/86** · `id_map_test` **4,676,294** checks ·
`ring_test` **30,011** checks · `abi_check` 448-B ABI lock · `risk_test` 9 checks
(CTest **5/5** across the C++ suites).

> **Performance targets are not yet measured.** The benchmark design targets
> >500k orders/sec and sub-microsecond latency; these have **not been re-measured
> on real hardware**. The only hardware-verified claim is 0 allocations/op on the
> hot path.

### 2 · Quant research + execution (`python_quant/`)
Pure-NumPy research and execution stack (no torch), byte-reproducible.

| Piece | File | State |
|---|---|---|
| NumPy dtype mirror + oracle | `nexus_quant/book_state.py` | `StubOrderBook` = diff-test oracle |
| ITCH 5.0 streaming parser | `nexus_quant/itch_parser.py` | Add/MPID/Exec/ExecPx/Cancel/Delete/Replace/Trade |
| ITCH→L2 replay | `nexus_quant/replay.py` | `ReplayEngine` + `check_integrity` |
| Injectable stub↔engine adapter | `nexus_quant/book_port.py` | swap the book, not the env |
| Gymnasium execution env | `nexus_quant/envs/order_book_env.py` | 44-dim obs, IS reward + inv/time/adv penalties, named regimes + null arm, FIFO / empirical-hazard queue models |
| Execution baselines | `nexus_quant/baselines.py` | TWAP / VWAP / POV / Passive + fair `schedule_twap` / `adaptive_pov` / `is_aware` (symmetric regime info) |
| PPO / GRPO agents (pure NumPy) | `nexus_quant/agents/` | MLP + Adam, GAE, clipped surrogate; GRPO on the PPO actor interface |
| Eval harness | `nexus_quant/agents/evaluate.py` | per-regime multi-seed `evaluate_regime_ci`, paired `paired_difference_ci`, fill rate / MDD |
| Research spine | `nexus_quant/research/{features,labels,dataset,experiments,models}.py` | leak-locked features, walk-forward splits, rank IC / block bootstrap / DM |
| Execution realism | `nexus_quant/execution/{cost_model,metrics,backtest,volume_profile}.py` | fees/rebates/impact, IS vs market VWAP, fill rate / MDD, empirical volume profiler |
| Order-level queue + fill models (E5) | `nexus_quant/research/queue_dynamics.py` | `OrderLevelTracker`, Kaplan–Meier P(fill), logistic fill model, `EmpiricalQueueHazard` + RL seam |
| Adverse selection (E6) | `nexus_quant/research/adverse_selection.py` | post-fill drift, Newey–West t, pre-fill matched control |
| Multi-day aggregation tooling | `nexus_quant/research/multi_day_aggregation.py` | cross-day means, between-day variance, bootstrap CIs (one validated day so far) |

**Test coverage:** **355 passed / 2 skipped** with the compiled engine (347 in
`python_quant/`, 8 in `bindings/`); **335 passed / 14 skipped** without it.
Covers the parser, replay, env and queue models, baselines, PPO/GRPO agents, risk
parity (3× bit-for-bit), dashboard, research spine, execution realism, adverse
selection, and the real-tape slicer (which runs when the local ITCH dataset is
present).

### 3 · GPU risk engine (`cuda_risk/`)
Monte-Carlo VaR/CVaR where the per-path RNG is a **pure function of
(seed, path, step)** via counter-based splitmix64, so the CPU reference, the CUDA
kernel, and a NumPy oracle draw *identical* paths — **bit-for-bit parity**
(not Monte-Carlo tolerance).

| Piece | File | State |
|---|---|---|
| Model + deterministic RNG (GBM / jump-diff) | `risk_common.hpp` | same RNG across CPU / GPU / NumPy |
| CPU reference (serial VaR/CVaR) | `risk_cpu.hpp` | 200k×252 ≈ 1.0 s here |
| CUDA kernel (1 thread/path) | `risk_cuda.{cu,h}` | authored; compiles only with a CUDA toolkit |
| CPU↔GPU bench + parity | `risk_bench.cpp` | CPU path runs; GPU path measured on a CUDA box |
| pybind `compute_var_cvar` (CPU) | `bindings/pybind_wrapper.cpp` | CPU path exposed to Python |

**Verified:** CPU/reference implementation tested (CTest 5/5), NumPy parity oracle
**3/3 bit-for-bit**, pybind CPU path wired.

> **The ~40× figure is a target, not a result.** GPU performance has **not been
> measured** — there is no CUDA toolkit/GPU environment available. The CUDA kernel
> is authored but uncompiled.

---

## RL fairness study: what changed

An initial exploratory run (2026-09-07) reported a PPO agent with **+50.4% lower
shortfall than a 2-line VWAP heuristic** on a Markov high-volatility regime. That
result was **retired** after the methodology was audited and corrected.

What the fair re-verification changed:

1. **Symmetric information** — baselines now observe the same regime signal as the agent.
2. **Stronger baselines** — volume-aware schedules (`schedule_twap`) and adaptive participation (`adaptive_pov`).
3. **Realistic execution** — transaction fees and queue priority enabled.
4. **Statistical rigor** — 5 training seeds × 5 evaluation families across 6 regimes (including a random-walk null arm), with paired-difference block-bootstrap confidence intervals.

Findings (`docs/results/rl_fairness.md`):

- **High-volatility / trending / null arm** — PPO shows **no significant edge** over `adaptive_pov` (0/5 seeds significant, |Δ| ≲ 0.3 bps).
- **Calm & low-vol hold-outs** — PPO is **worse** than `adaptive_pov` in 5/5 seeds (−0.05 … −0.16 bps), from unnecessary aggressiveness.
- **Liquidity shock** — PPO shows a **genuine, significant advantage** in 5/5 seeds (+0.4 … +1.4 bps) by executing before liquidity vanishes.

**Conclusion:** PPO does **not** have a general advantage over the best fair
baseline. Its measurable advantage is concentrated in liquidity-shock conditions.

---

## Dashboard — execution replay desk

A single self-contained page that overlays live order-book state and execution
trajectories on the verification console. **It is a replay/synthetic/SHM
execution desk — it is not connected to a live exchange and trades no money.**

| Piece | File | What it does |
|---|---|---|
| Desk page | `python_quant/nexus_quant/dashboard_page.html` | L2 ladder, sparklines, latency histogram, VaR/CVaR tiles, trade ticker, execution timeline |
| Snapshot hub / slot codec | `python_quant/nexus_quant/dashboard.py` | decodes the frozen 448-B `BookStateView`, dedup-by-seq rolling 200-sample history, latency p50/p95 |
| Dashboard server | `python_quant/scripts/serve_dashboard.py` | stdlib HTTP server feeding the page |
| Dashboard tests | `python_quant/tests/test_dashboard.py` | slot round-trip, file-ring latest, hub JSON, latency |

The page **polls `/api/state`** (pull every 400 ms) — no WebSockets. Each request
produces one fresh snapshot from the attached feed:

- `--synthetic` (default) — a deterministic seeded mid random-walk emitting views in the exact frozen-contract shape; no engine build needed.
- `--shm <name>` — newest slot from a live C++ `ShmRing` in `/dev/shm` (POSIX → WSL/Linux).
- `--ring <file>` — last complete 448-B slot from a raw file ring.
- `--live-exec` — overlays a real seeded `OrderBookEnv` episode (TWAP/FIFO) as the execution timeline + inventory chart, animating against any feed above.

```bash
python python_quant/scripts/serve_dashboard.py --synthetic                 # → http://127.0.0.1:8765
python python_quant/scripts/serve_dashboard.py --synthetic --live-exec    # + live execution panel
```

---

## The architecture seam

The two halves of the system build against **one frozen boundary** — the state
contract `BookStateView` (C++) ↔ `BOOK_STATE_DTYPE` (NumPy):

- **Prices are integer ticks** (`int64`), never floats; `0` = empty level.
- **Fixed depth** `DEPTH = 10` per side; bids descend, asks ascend.
- **ABI-frozen:** `sizeof(BookStateView) == 40*10 + 48 == 448` bytes. Any layout
  change is an ABI break (bump the version, update the Python mirror, re-run parity).

The Python environment is built against a pure-Python `StubOrderBook` that emits
the same `view()`/`snapshot()` interface as the real engine — so environment and
agent code run with **no C++ build required**. `StubOrderBook` then becomes the
**reference oracle** the real engine is diff-tested against
(`bindings/tests/test_diff_engine_stub.py`). Full spec: `bindings/CONTRACT.md`.

---

## Repo layout

```
Finance Project-1/
├── cpp_engine/            # C++ matching engine (header-only)
│   ├── include/nexus/     # book_state, types, order_pool, limit_order_book, shm_ring, flow_gen
│   ├── tests/             # lob_test, id_map_test, ring_test, abi_check, risk_test
│   ├── demos/             # ring_producer / ring_probe (cross-process shmem demo)
│   └── bench/             # bench.cpp — throughput + latency + 0 allocs/op
├── cuda_risk/             # Monte-Carlo VaR/CVaR (CPU verified; CUDA kernel authored)
├── python_quant/          # quant research + execution
│   ├── nexus_quant/       # book_state, itch_parser, replay, book_port, envs/, baselines, agents/, execution/, research/, dashboard
│   ├── scripts/           # train_eval_agent.py, serve_dashboard.py, run_all.py, run_research.py, fetch_itch.py
│   ├── tests/             # pytest suite (355 passed / 2 skipped with engine)
│   └── artifacts/         # saved policies (policy_ppo*.npz)
├── bindings/              # pybind_wrapper.cpp, CONTRACT.md, tests/ (+ compiled module)
├── dashboard/             # static verification console
├── CMakeLists.txt         # engine lib + pybind + CTest + CUDA hooks
├── pyproject.toml         # scikit-build-core packaging
├── docs/                  # results & research reports (see below)
├── CLAUDE.md              # session handoff (always current)
├── PROGRESS.md            # plain-language status
├── plan_2.md              # research-layer plan of record
└── HIGHVOL_PLAN.md        # high-vol regime design (historical; headline retired)
```

Key docs: `docs/RESEARCH.md` (full report incl. negative results) ·
`docs/results/real_tape_12302019.md` (single-session tape tables) ·
`docs/results/rl_fairness.md` (fair RL study).

---

## Build & test

```bash
# Tier 1 — Python suite (no C++ build needed for most tests)
python -m pip install numpy gymnasium pytest
python -m pytest python_quant/tests -v

# Tier 2 — build the real engine + parity/diff tests
python -m pip install pybind11 cmake
cmake -S . -B build -DNEXUS_BUILD_PYBIND=ON
cmake --build build --config Release -j
python -m pytest python_quant/tests bindings/tests -v

# C++ engine checks (any box with g++/MSVC)
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include \
    cpp_engine/tests/lob_test.cpp -o lob_test && ./lob_test   # 86 checks, ALL PASS

# Cross-process shared-memory demo
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_producer.cpp -o ring_producer
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_probe.cpp -o ring_probe
./ring_producer nex_aapl 4000 16384 0xC0FFEE 1 &   # terminal 1
./ring_probe nex_aapl 4000 5                        # terminal 2

# Dashboard (replay/synthetic desk) — no engine build needed
python python_quant/scripts/serve_dashboard.py --synthetic   # → http://127.0.0.1:8765
```

**Real-tape research** requires fetching the NASDAQ ITCH sample day first. The raw
market data is **not part of this repository** — it is downloaded to gitignored
`data/` (~3.5 GB gz streamed), and only then can E1–E6 be reproduced:

```bash
python python_quant/scripts/fetch_itch.py --day 12302019 --symbols AAPL,QQQ   # downloads to data/ (gitignored)
python python_quant/scripts/run_research.py --day 12302019 --symbols AAPL,QQQ   # E1–E6 → docs/results/
python python_quant/scripts/rl_fairness_study.py --quick                        # fair RL re-verification (smoke)
python python_quant/scripts/run_all.py                                           # smoke of every stage
```

**Environment notes.** On Windows, CMake uses the Visual Studio generator and the
compiled module is dropped into `bindings/`; CUDA work needs a real toolkit
(WSL/Linux or a Windows CUDA toolkit) — Python + pybind do not. If the checkout
lives on a synced folder (e.g. OneDrive), keep `build/`, `data/`, and virtualenvs
out of the synced tree.

---

## Status

**Core research layer complete; single-session real-data validation complete;
multi-day robustness campaign pending.**

- ✅ C++ matching engine — built, self-tested (0 allocs/op), pybind seam green
- ✅ ITCH parser + replay + execution environment + baselines
- ✅ Quant research layer — leak-locked feature spine, execution realism, queue/fill + adverse-selection studies, fair RL study
- ✅ Dashboard — replay/synthetic/SHM execution desk with live execution timeline
- ✅ CPU risk engine + exact NumPy parity
- ⏳ CUDA GPU benchmark — pending a CUDA/GPU environment
- ⏳ Real-hardware latency/throughput — pending measurement
- ⏳ Multi-day real-tape campaign — one session (12/30/2019) validated; catalogued days not yet run

---

## Known limitations

- **Real-data scope:** validation currently demonstrated on one session
  (12/30/2019, AAPL/QQQ). Multi-day robustness is not yet run.
- **Hardware benchmarks:** the C++ throughput/latency targets and the CUDA
  speedup are **not yet measured** — the targets are benchmark design goals.
- **Synthetic environments are controls, not evidence:** null/negative synthetic
  results (e.g. fill-model ties, spurious imbalance drift) are recorded honestly
  and are not treated as real-market findings.
- **Statistical ≠ executable alpha:** signal results shown here are findings on
  data, not validated trading strategies.
```