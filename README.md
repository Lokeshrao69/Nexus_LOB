# Nexus-LOB

A limit order book and market-microstructure research platform in C++20 and Python.

Nexus-LOB combines:

- A price-time priority matching engine in C++20
- NASDAQ TotalView-ITCH 5.0 parsing and replay
- Leakage-checked market-microstructure research
- A Gymnasium execution environment with PPO/GRPO and strong rule-based baselines
- Queue and passive-fill modelling
- Execution-cost and adverse-selection analysis
- Monte-Carlo VaR/CVaR with a CPU reference, NumPy oracle, and CUDA implementation
- A shared-memory SPSC data path and interactive dashboard

The project emphasizes reproducibility and statistical honesty: results are tied to explicit datasets and scripts, and negative or inconclusive results are reported rather than hidden.

---

## Results at a Glance

| Area | Result | Scope |
|---|---|---|
| L1 order-book imbalance | Positive short-horizon predictive relationship; rank IC 0.11–0.19 at h=1 for AAPL and 0.15–0.24 for QQQ | 15 full-day NASDAQ ITCH sessions, 2018–2025 |
| Passive fills | Negative average post-fill markouts | Same 15 sessions |
| RL execution | Initial +50.4% vs VWAP result retracted after identifying an unfair comparison. Under the corrected evaluation, PPO has no general edge and shows an advantage primarily during liquidity shocks | Synthetic regimes, 5 seeds × 6 regimes |
| Matching engine | 4–7M orders/s, p50 170–260 ns, p99 420–540 ns, zero heap allocations; matches a naive reference book on 51.9M differential checks | 2-vCPU cloud VM, GCC 13.3 |
| Risk engine | CPU and NumPy implementations agree bit-for-bit | CUDA kernel implemented but not yet compiled/run |

> **Statistical is not tradable.** A predictive relationship is not executable alpha. Real profitability depends on spread, fees, latency, queue position, market impact, and other execution costs that are not fully captured by these experiments.

---

# Research Findings

All real-tape results use 15 full trading days of NASDAQ TotalView-ITCH for AAPL and QQQ.

Each trading day is analysed independently using a time-ordered walk-forward split and block-bootstrap confidence intervals. Days are not pooled into a single fitted model.

Detailed per-day results are available under `docs/results/multi_day/`.

### E1 — L1 Imbalance Predicts Short-Horizon Mid Moves

Test-split rank IC across the 15 sessions:

| Symbol | h=1 | h=5 | h=10 | h=25 |
|---|---:|---:|---:|---:|
| AAPL | 0.11 / 0.16 / 0.19 | 0.17 / 0.24 / 0.34 | 0.15 / 0.24 / 0.40 | 0.13 / 0.21 / 0.43 |
| QQQ | 0.15 / 0.18 / 0.24 | 0.26 / 0.31 / 0.36 | 0.30 / 0.37 / 0.39 | 0.29 / 0.40 / 0.48 |

Values are `min / median / max` across individual trading days.

The sign is positive across every tested day, symbol, and horizon.

Because AAPL and QQQ frequently trade with a one-tick spread, many short-horizon mid-price changes are zero. Rank IC is therefore interpreted alongside hit rate, tie rate, and bootstrap confidence intervals rather than treated as standalone trading performance.

### E2 — Microprice Adds Little Over L1 Imbalance

On one-tick books, microprice is close to a linear transformation of L1 imbalance and does not materially outperform it out of sample.

### E5 — Queue and Fill Models

Kaplan–Meier estimates indicate that the probability of a passive order filling within 50 book events is approximately 1–7%, depending on day and symbol.

The logistic fill model is well calibrated on the tested real data. Synthetic-flow results are treated as controls rather than empirical evidence because the synthetic generator places fills at the front of the queue.

### E6 — Passive-Fill Markouts

Passive fills are followed by negative average markouts across the real-tape sessions.

The analysis uses matched pre-fill controls to separate post-fill drift from broader market trends.

An event-time analysis showed a high proportion of fills followed by adverse mid-price movement. Because the event that fills a passive order can also mechanically deplete the corresponding price level, this statistic is **not treated as a direct estimate of informed adverse selection**.

The next research iteration measures:

- 100 ms markouts
- 1 s markouts
- 10 s markouts
- Markouts measured from the post-trade mid
- Markouts normalized by half-spread
- Conditioning by order-flow imbalance, spread, queue position, volatility, and liquidity regime

---

# RL Execution

An early experiment reported a PPO agent with substantially lower implementation shortfall than a VWAP heuristic.

A subsequent audit found that the comparison was not information-symmetric, so the result was **retracted** rather than presented as evidence of alpha.

The corrected evaluation uses:

1. **Symmetric information** — baselines receive the same regime information as the agent.
2. **Stronger baselines** — including volume-aware scheduling and adaptive participation.
3. **Realistic execution** — fees and queue priority are included.
4. **Multiple seeds and statistical testing** — 5 training seeds × 5 evaluation families × 6 regimes, including a random-walk null.

The corrected result is:

| Regime | PPO vs. Best Baseline |
|---|---|
| High volatility / trending | No significant difference |
| Random-walk null | No significant difference |
| Calm / low volatility | PPO underperforms |
| Liquidity shock | PPO outperforms the best baseline across the tested seeds |

The conclusion is deliberately narrow:

> PPO does not demonstrate a general execution advantage over strong adaptive baselines. Its observed advantage is concentrated in liquidity-shock regimes.

Full methodology and results: `docs/results/rl_fairness.md`.

---

# Architecture

```text
NASDAQ TotalView-ITCH 5.0
            │
            ▼
      ITCH Parser
            │
            ▼
          Replay
            │
       ┌────┴────┐
       ▼         ▼
   Research     LOB
       │         │
       │    C++ Matching Engine
       │         │
       │         ▼
       │    BookStateView
       │      (448 bytes)
       │         │
       │         ▼
       │    SPSC Shared Memory
       │         │
       │         ▼
       │      Dashboard
       │
       ▼
 Features / Labels
       │
       ▼
 Walk-Forward Research
       │
       ├── Rank IC
       ├── Bootstrap
       ├── Newey-West
       ├── Diebold-Mariano
       ├── Queue Models
       └── Markouts
       
C++ Engine
    │
    ▼
 Python / pybind11
    │
    ▼
 Gymnasium Execution Environment
    │
    ├── TWAP
    ├── VWAP
    ├── POV
    ├── Adaptive POV
    ├── PPO
    └── GRPO
```

---

# Matching Engine

The matching engine is implemented as a header-oriented C++20 system.

### Design

- Integer tick prices; no floating-point prices in the matching path
- Strict price-time FIFO priority
- Limit, market, IOC and FOK orders
- Cancel and modify operations
- Intrusive doubly linked FIFO queues per price level
- Fixed-size slab allocator with a free list
- Hierarchical occupancy bitmap for best-price and depth lookup
- Open-addressing order-ID map
- Backward-shift deletion
- No dynamic allocation after engine construction
- FOK liquidity validation before mutation
- Exchange-style modify semantics:
  - Same-price size reduction preserves queue priority
  - Size increase or repricing loses priority
- Frozen 448-byte `BookStateView` ABI
- C++/Python differential testing against a reference implementation

---

# Price-Level Representation

The engine uses a flat price-level representation indexed by:

```text
price - min_price
```

This provides direct level addressing.

This provides direct level addressing, but finding the *next occupied* level is a
separate problem. The first version stepped through the array one level at a time.
When a side had fewer than 10 occupied levels, publishing the top 10 walked to the
edge of the price band on every order, so runtime grew with the width of the price
range rather than the number of active levels: about 700 orders per second on a
1,000,000-tick band.

The engine now keeps a hierarchical occupancy bitmap per side
(`cpp_engine/include/nexus/level_bitmap.hpp`). Level 0 has one bit per price; each
level above has one bit per 64-bit word below it, until the top fits in one word.
Finding the next or previous occupied level takes a few count-leading/trailing-zero
instructions per level (four levels for a 1M-tick band), however many empty prices
lie in between. It is used for best-price refresh, the top-10 publish walk and the
fill-or-kill liquidity check, and it is allocated once at construction.

---

# Shared-Memory Data Path

A single-producer, single-consumer ring carries `BookStateView` snapshots from the
engine to another process (the dashboard or `ring_probe`) through POSIX or Windows
shared memory, with no serialization. When the consumer falls behind, new
snapshots are dropped and counted instead of blocking the engine.

- **Separate cache lines.** The producer's index (`write_seq`) and the consumer's
  index (`read_seq`) sit on different 64-byte cache lines, so neither side's
  writes invalidate the other's line. Slots start cache-line aligned.
- **Cached indices.** Each side keeps a private copy of the other side's index and
  re-reads the shared one only when its copy says the ring is full (producer) or
  empty (consumer).
- **Acquire/release ordering** on the two indices; lock-freedom of the shared
  atomics is checked at compile time.
- **Layout stamp.** A magic number and layout version in the header make a reader
  built for a different layout fail loudly instead of misreading offsets. The
  Python dashboard mirrors the layout byte for byte.

| Ring capacity | Before | Padding only | Padding + cached indices |
|---:|---:|---:|---:|
| 64 | 2.0M frames/s | 2.9M | **6.6M** |
| 1,024 | 2.7M | 4.2M | **29.7M** |
| 16,384 | 3.1M | 5.1M | **35.8M** |

Median of 5 runs of 3M 448-byte frames, producer and consumer threads on a 2-vCPU
cloud VM. The ring tests and benchmark run clean under ThreadSanitizer. Details:
[`docs/results/ring_bench.md`](docs/results/ring_bench.md).

---

# Python Research Stack

The Python side is implemented primarily with NumPy and provides:

| Component | Purpose |
|---|---|
| `itch_parser.py` | Streaming NASDAQ ITCH 5.0 parser |
| `replay.py` | ITCH → L2 replay with integrity checks |
| `research/features.py` | Leakage-checked features |
| `research/labels.py` | Forward-looking labels |
| `research/dataset.py` | Walk-forward datasets |
| `research/experiments.py` | IC, bootstrap, DM and Newey-West analysis |
| `research/queue_dynamics.py` | Queue tracking and fill models |
| `research/adverse_selection.py` | Passive-fill markouts |
| `envs/order_book_env.py` | Gymnasium execution environment |
| `baselines.py` | TWAP, VWAP, POV and adaptive baselines |
| `agents/` | PPO and GRPO implementations |
| `execution/` | Fees, rebates, impact and implementation shortfall |

The execution environment can run against a pure-Python `StubOrderBook`, allowing research and agent development without requiring the C++ build. The stub is then differential-tested against the real engine.

---

# Risk Engine

`cuda_risk/` contains a Monte-Carlo VaR/CVaR implementation supporting:

- Geometric Brownian Motion
- Jump-diffusion
- Deterministic counter-based random-number generation
- CPU reference implementation
- NumPy oracle
- CUDA kernel

The CPU and NumPy implementations currently agree bit-for-bit.

The CUDA implementation is present but **has not yet been compiled or benchmarked**, so no GPU speedup is claimed.

---

# Performance

Measured on a 2-vCPU cloud VM, GCC 13.3, `-O3 -march=native`, with
`cpp_engine/bench/bench.cpp`: 2M orders per run on a steady-state book of about
10,000 live orders. These are not tuned bare-metal HFT benchmarks.

| Workload | Throughput | p50 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| Passive-heavy, 2k-tick band | 5.60M ops/s | 167 ns | 419 ns | 520 ns |
| Matching-heavy, 2k-tick band | 6.98M ops/s | 168 ns | 433 ns | 568 ns |
| Passive-heavy, 1M-tick band | 6.48M ops/s | 174 ns | 441 ns | 623 ns |
| Orders spread over 10k ticks, 1M-tick band | 3.98M ops/s | 264 ns | 541 ns | 882 ns |

Zero heap allocations inside the timed loop in every run.

**Before and after the occupancy bitmap**, same harness and workload:

| Workload | Before | After |
|---|---|---|
| Passive-heavy, 2k-tick band | 0.85M ops/s, p50 1,122 ns | 5.60M ops/s, p50 167 ns |
| Passive-heavy, 1M-tick band | 694 ops/s, p50 1.46 ms | 6.48M ops/s, p50 174 ns |

A differential test against a naive reference book passes on both the old and new
engine, so the change affects speed, not matching behaviour.

### How the benchmark measures

- Liquidity is seeded before timing starts, and the book is held at a steady state
  (cancels take over when it grows past ~10,000 orders), so the pool never fills
  and no run produces rejected orders.
- Cancels and modifies target a random known order.
- Latency is measured per engine call with `rdtsc` (including ~20–40 ns of timer
  overhead), with order generation outside the timed window. Throughput is wall
  clock and includes order generation.
- A global `operator new` counter covers only the timed loop; the harness's own
  bookkeeping is reserved beforehand.
- Each workload runs on both a narrow and a 1M-tick band, so band-width scaling is
  visible directly.

Full write-up: [`docs/results/engine_bench.md`](docs/results/engine_bench.md).

---

# Correctness and Testing

The C++ test suite includes:

- `lob_test` — 86 functional checks
- `lob_diff_test` — 51.9M checks against a naive `std::map` reference book (results, fills and the full published ladder) on dense, sparse and pool-exhaustion workloads
- `bitmap_test` — 6.1M checks of the occupancy bitmap against a `std::set` oracle
- `id_map_test` — 4.6M+ checks
- `ring_test` — 30k+ checks, including rejection of a mismatched ring layout
- `abi_check` — 448-byte layout contract
- `risk_test` — CPU/reference validation

Python research and parity tests additionally cover:

- ITCH parsing
- Replay integrity
- Feature/label leakage
- C++ vs. Python engine behaviour
- Execution environment behaviour
- Research statistics
- Risk calculations

CI runs across supported Linux and Windows configurations with Python 3.10–3.12.

---

# Reproducing the Project

## Install

```bash
python -m pip install -r python_quant/requirements.txt -r requirements-dev.txt
```

## Python tests

```bash
python -m pytest python_quant/tests
```

## Build C++

```bash
python -m pip install pybind11 cmake

cmake -S . -B build \
  -DNEXUS_BUILD_PYBIND=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
```

## Python + binding tests

```bash
python -m pytest python_quant/tests bindings/tests
```

## Benchmark

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG \
    -I cpp_engine/include \
    cpp_engine/bench/bench.cpp \
    -o bench

./bench
```

## Shared-memory demo

Terminal 1:

```bash
g++ -std=c++20 -O2 -I cpp_engine/include \
    cpp_engine/demos/ring_producer.cpp \
    -o ring_producer

./ring_producer nex_aapl 4000 16384 0xC0FFEE 1
```

Terminal 2:

```bash
g++ -std=c++20 -O2 -I cpp_engine/include \
    cpp_engine/demos/ring_probe.cpp \
    -o ring_probe

./ring_probe nex_aapl 4000 5
```

## Dashboard

```bash
python python_quant/scripts/serve_dashboard.py --synthetic
```

The dashboard is a local development/demo interface and is not connected to a live exchange.

---

# Reproducing the Real-Tape Research

Market data is intentionally not stored in the repository.

Each ITCH session is downloaded into the gitignored `data/` directory.

```bash
python python_quant/scripts/fetch_itch.py \
    --day 12302019 \
    --symbols AAPL,QQQ

python python_quant/scripts/run_research.py \
    --day 12302019 \
    --symbols AAPL,QQQ

python python_quant/scripts/batch_research_itch.py \
    --days all
```

The research pipeline verifies the expected file size and SHA-256 hash before processing.

---

# Limitations

- Current benchmark numbers are not bare-metal HFT measurements.
- The ring drops the *newest* snapshot when full. That suits a consumer that must see every event in order, but a dashboard wants the *latest* state, so when it polls slowly it shows a snapshot up to one ring-capacity of events old.
- Adverse-selection analysis is being extended from event-time measurements to post-trade clock-time markouts.
- The current real-tape study covers only AAPL and QQQ.
- Both instruments are highly liquid and results may not generalize to thinner securities.
- RL results are from controlled synthetic regimes rather than live trading.
- The CUDA kernel has not yet been compiled or benchmarked.
- Predictive relationships are not profitability claims.
- The project does not claim live-trading readiness.

---

# Design Philosophy

Nexus-LOB deliberately prioritizes:

**Correctness over headline metrics.**

**Reproducibility over cherry-picked experiments.**

**Statistical significance over isolated backtest results.**

**Realistic execution assumptions over frictionless simulation.**

**Negative results over unsupported claims.**

The project is intended as a research and systems engineering platform for studying limit-order-book mechanics, execution, and market microstructure — not as a claim of profitable trading alpha.

---

# Repository Layout

```text
Nexus_LOB/
├── cpp_engine/
│   ├── include/nexus/
│   ├── tests/
│   ├── demos/
│   └── bench/
├── cuda_risk/
├── bindings/
├── python_quant/
│   ├── nexus_quant/
│   ├── scripts/
│   └── tests/
├── docs/
│   └── results/
├── CMakeLists.txt
└── pyproject.toml
```

---

# Authors

**Lokesh Rao** — [@Lokeshrao69](https://github.com/Lokeshrao69)

**ShrikarT**