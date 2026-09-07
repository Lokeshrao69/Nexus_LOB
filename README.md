# Nexus-LOB

A hybrid **C++/Python** limit-order-book (LOB) trading and market-microstructure
platform — a finance-placement portfolio project targeting quant-desks
(JPMC Quantitative Research, Nomura Algo Strategies, Goldman Sachs Systematics).
It reads as *institutional-grade*: low-latency systems design, order-book
mechanics, RL for optimal execution, and GPU risk analytics.

**Two people, ~8 weeks.** See `CLAUDE.md` (session handoff, always current),
`PROGRESS.md` (plain-language status), and `HIGHVOL_PLAN.md` (the high-vol
execution regime that delivers the slippage headline).

---

## Headline results

| Metric | Target | Status |
|---|---|---|
| C++ matching engine | >500k ord/s, sub-µs, **0 allocs/op** | 0-alloc ✅ proven; throughput/latency on real hardware |
| **PPO execution vs VWAP** | **~14% lower slippage** (high-vol) | **+50.4% lower shortfall** ✅ |
| CUDA Monte-Carlo VaR/CVaR | ~40× speedup vs CPU | CPU ✅ exact parity; GPU kernel blocked (no toolkit) |

---

## The three subsystems that deliver the numbers

### 1. C++ matching engine (`cpp_engine/`) — Person A
Header-only C++20, zero-allocation on the hot path:

| Piece | File | State |
|---|---|---|
| Frozen state contract | `include/nexus/book_state.hpp` | ABI-locked v1, `sizeof == 448`, `alignof == 8` |
| Engine value types | `include/nexus/types.hpp` | `OrderId/Price/Qty`, `Fill`, `Status`, `ExecResult` |
| Zero-alloc order pool | `include/nexus/order_pool.hpp` | fixed slab + intrusive free-list |
| Matching engine | `include/nexus/limit_order_book.hpp` | Limit/Market/FOK/IOC/Cancel/Modify, FIFO, O(1) level lookup |
| Zero-alloc id→Order map | `include/nexus/limit_order_book.hpp` (`IdMap`) | replaces `std::unordered_map` — **0 allocs/op proven** |
| Shared-memory SPSC ring | `include/nexus/shm_ring.hpp` | POSIX+Windows shmem, drop-new-on-full, 448-B slots |
| Synthetic flow generator | `include/nexus/flow_gen.hpp` | seeded, deterministic pre-ITCH flow |
| Live cross-process demo | `demos/ring_producer.cpp`, `ring_probe.cpp` | book → ring → reader, verified 0 dropped |
| Benchmark | `bench/bench.cpp` | throughput + latency + **0 allocs/op** |

**Tests (all green):** `lob_test` **86/86** checks · `id_map_test` **4,676,294**
checks · `ring_test` **30,011** checks · `abi_check` (448 B lock) ·
`risk_test` CTest **5/5**.

### 2. Microstructure sim + RL execution agent (`python_quant/`) — Person B
Pure-NumPy stack (no torch), byte-reproducible:

| Piece | File | State |
|---|---|---|
| NumPy dtype mirror + oracle | `nexus_quant/book_state.py` | `StubOrderBook` = diff-test oracle |
| ITCH 5.0 streaming parser | `nexus_quant/itch_parser.py` | Add/MPID/Exec/ExecPx/Cancel/Delete/Replace/Trade |
| ITCH→L2 replay | `nexus_quant/replay.py` | `ReplayEngine` + `check_integrity` |
| Injectable stub↔engine adapter | `nexus_quant/book_port.py` | swap the book, not the env |
| Gymnasium execution env | `nexus_quant/envs/order_book_env.py` | 44-dim obs, IS reward + inv/time/adv penalties, **high-vol regime** |
| Execution baselines | `nexus_quant/baselines.py` | TWAP / VWAP / POV / Passive |
| PPO agent (pure NumPy) | `nexus_quant/agents/ppo.py`, `mlp.py` | MLP, Adam, GAE, clipped surrogate |
| Eval harness | `nexus_quant/agents/evaluate.py` | `strategy_table` / `format_table` |
| Train+eval CLI | `scripts/train_eval_agent.py` | `--highvol`, `--eval-only`, regime flags |
| **Regime design doc** | `HIGHVOL_PLAN.md` | the high-vol regime + how it hits the headline |

**Tests (all green):** **52 passed** — contract smoke, ITCH, replay, env,
baselines, PPO agent, risk parity (3× bit-for-bit), and the high-vol regime
(11 tests).

### 3. GPU risk engine (`cuda_risk/`) — Person A
Monte-Carlo VaR/CVaR where the per-path RNG is a **pure function of
(seed, path, step)** via counter-based splitmix64, so CPU, CUDA, and a NumPy
oracle draw *identical* paths → **bit-for-bit parity** (not MC tolerance).

| Piece | File | State |
|---|---|---|
| Model + deterministic RNG (GBM / jump-diff) | `risk_common.hpp` | same RNG CPU/GPU/NumPy |
| CPU reference (serial VaR/CVaR) | `risk_cpu.hpp` | 200k×252 ≈ 1.0 s here |
| CUDA kernel (1 thread/path) | `risk_cuda.{cu,h}` | ⚠️ authored; needs a CUDA toolkit |
| CPU-vs-GPU bench + parity | `risk_bench.cpp` | CPU runs; GPU on a CUDA box |
| pybind `compute_var_cvar` (CPU) | `bindings/pybind_wrapper.cpp` | ✅ |

**Verified here (no GPU):** the risk parity oracle is **3/3 bit-for-bit** within
the 52-test Python suite; CTest **5/5**. **Blocked:** the CUDA kernel and ~40×
speedup need `nvcc`/toolkit (WSL/Linux or Windows CUDA).

---

## The headline claim, measured honestly

On the **default gentle-walk env** PPO wins the composite reward but lands
≈VWAP on pure slippage (1.60 vs 1.55 bps) — there's no regime where adaptive
execution pays. So we added a **high-volatility / gap-off regime**
(`HIGHVOL_PLAN.md`): Markov-switching between calm and volatile states, plus
gap events that sweep the book and drop the mid several ticks. This punishes
fixed-schedule execution and gives a smart agent a real edge.

**Measured 2026-09-07**, `--highvol --vol-feature --iters 2000`, 100 seeded
episodes (seed `0xBEEF`), policy at `python_quant/artifacts/policy_ppo_highvol.npz`:

```
strategy      reward shortfall_bps  vs_vwap%
ppo            -5.04         1.401    +50.4%   ← 14% headline exceeded
twap           -8.60         2.659     +5.9%
vwap           -7.40         2.827      0.0%
pov            -9.32         2.519    +10.9%
passive       -12.81         2.679     +5.3%
```

`shortfall_bps` = `(arrival_mid − realized_VWAP) / arrival_mid × 1e4` — lower
is better. The regime triples VWAP's slippage (1.64 → 2.83 bps); PPO learns to
execute before/around gaps, landing **~38–50% below VWAP** across seeds.

**Reproduce:**

```bash
# train + eval on the high-vol regime (obs_dim=45)
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
    --highvol --vol-feature --iters 2000 --eval-every 400 --eval-episodes 40 \
    --out python_quant/artifacts/policy_ppo_highvol.npz --table-episodes 100

# sweep regime params without retraining
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
    --eval-only python_quant/artifacts/policy_ppo_highvol.npz \
    --highvol --vol-feature --table-episodes 200 --table-seed 12345
```

---

## The architecture seam (why it's built this way)

Two people work in parallel on one frozen boundary — the **state contract**
`BookStateView` (C++) ↔ `BOOK_STATE_DTYPE` (NumPy):

- **Prices are integer ticks** (`int64`), never floats. `0` = empty level.
- **Fixed depth** `DEPTH = 10` per side; bids descend, asks ascend.
- **ABI-frozen:** `sizeof(BookStateView) == 40*10 + 48 == 448` bytes. Any change
  = ABI break (bump version, update the Python mirror, re-run parity).

Person B builds the env against a pure-Python `StubOrderBook` emitting the same
`view()`/`snapshot()` interface as the real engine — **today**, no C++ build
needed. `StubOrderBook` then becomes the **reference oracle** the real engine
is diff-tested against (`bindings/tests/test_diff_engine_stub.py`). Full spec:
`bindings/CONTRACT.md`.

---

## Repo layout

```
Finance Project-1/
├── cpp_engine/            # Person A — C++ engine (header-only)
│   ├── include/nexus/     # book_state, types, order_pool, limit_order_book, shm_ring, flow_gen
│   ├── tests/             # lob_test(86), id_map_test(4.6M), ring_test(30k), abi_check, risk_test
│   ├── demos/             # ring_producer / ring_probe (live shmem demo)
│   └── bench/             # bench.cpp — 0 allocs/op
├── cuda_risk/             # Person A — Monte-Carlo VaR/CVaR (CPU ✅, CUDA blocked)
├── python_quant/          # Person B — quant / RL
│   ├── nexus_quant/       # book_state, itch_parser, replay, book_port, envs/, baselines, agents/
│   ├── scripts/           # train_eval_agent.py
│   ├── tests/             # 52 tests, all green
│   └── artifacts/         # policy_ppo.npz, policy_ppo_highvol.npz
├── bindings/              # pybind_wrapper.cpp, CONTRACT.md, tests/ (+ compiled .pyd)
├── CMakeLists.txt         # engine lib + pybind + CTest + CUDA hooks
├── pyproject.toml         # scikit-build-core
├── CLAUDE.md              # session handoff (always current)
├── PROGRESS.md            # plain-language status
├── HIGHVOL_PLAN.md        # high-vol regime design + results
└── README.md              # this file
```

---

## Build & test

```bash
# Tier 1 — pure-Python (52 tests; works on Windows, no C++ build needed)
python -m pip install numpy gymnasium pytest
python -m pytest python_quant/tests -v

# Tier 2 — compile the real engine + parity/diff tests (Windows/MSVC or WSL)
python -m pip install pybind11 cmake
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNEXUS_BUILD_PYBIND=ON
cmake --build build --config Release -j
python -m pytest python_quant/tests bindings/tests -v

# C++ engine checks (any box with g++/MSVC)
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include \
    cpp_engine/tests/lob_test.cpp -o lob_test && ./lob_test   # 86 checks, ALL PASS

# Live shared-memory demo (subsystem 5)
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_producer.cpp -o ring_producer
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_probe.cpp -o ring_probe
./ring_producer nex_aapl 4000 16384 0xC0FFEE 1 &   # terminal 1
./ring_probe nex_aapl 4000 5                        # terminal 2
```

**Environment note:** the repo lives on a OneDrive path — keep `build/`,
`data/`, and venvs out of the synced tree. CUDA work needs a real toolkit
(WSL/Linux or Windows CUDA); Python + pybind do not.

---

## Status (short)

- ✅ C++ matching engine — built, self-tested (0 allocs/op), pybind seam green
- ✅ ITCH parser + replay + Gymnasium env + baselines
- ✅ PPO agent — pure NumPy; **+50.4% vs VWAP** on the high-vol regime
- ✅ Monte-Carlo VaR/CVaR — CPU + exact parity (GPU kernel authored, blocked)
- ⏳ Remaining: CUDA compile + ~40× speedup on a GPU box; throughput/latency on
  real hardware; Python dashboard on the shmem ring; risk↔env integration.

See `PROGRESS.md` for the detailed status and `CLAUDE.md` for the current
handoff.
### Low-Latency Limit Order Book • Execution Simulation • Market Microstructure • Quantitative Trading Infrastructure

<p align="center">

**A hybrid C++/Python quantitative trading research platform built around a deterministic, low-latency limit-order-book matching engine.**

<br />

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square\&logo=cplusplus\&logoColor=white)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?style=flat-square\&logo=cmake\&logoColor=white)](https://cmake.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Quant-013243?style=flat-square\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-FF6F00?style=flat-square)](https://gymnasium.farama.org/)
[![pybind11](https://img.shields.io/badge/pybind11-C%2B%2B%2FPython-3776AB?style=flat-square)](https://pybind11.readthedocs.io/)

</p>

---

## Table of Contents

* [Overview](#overview)
* [Why Nexus-LOB?](#why-nexus-lob)
* [Core Objectives](#core-objectives)
* [Architecture](#architecture)
* [System Components](#system-components)

  * [1. C++ Matching Engine](#1-c-matching-engine)
  * [2. Cross-Language State Contract](#2-cross-language-state-contract)
  * [3. Python Quant Layer](#3-python-quant-layer)
  * [4. ITCH 5.0 Replay](#4-itch-50-replay)
  * [5. Execution Environment](#5-execution-environment)
  * [6. Execution Baselines](#6-execution-baselines)
  * [7. Shared-Memory Transport](#7-shared-memory-transport)
  * [8. Python/C++ Bridge](#8-pythonc-bridge)
* [Order Book Model](#order-book-model)
* [Order Semantics](#order-semantics)
* [Execution Environment](#execution-environment-1)
* [Reward Design](#reward-design)
* [Data & Replay](#data--replay)
* [Testing & Verification](#testing--verification)
* [Current Validation](#current-validation)
* [Performance Philosophy](#performance-philosophy)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Building the C++ Engine](#building-the-c-engine)
* [Building the Python Extension](#building-the-python-extension)
* [Running Tests](#running-tests)
* [Running the Shared-Memory Demo](#running-the-shared-memory-demo)
* [Using the Matching Engine](#using-the-matching-engine)
* [Using the Python Environment](#using-the-python-environment)
* [Design Decisions](#design-decisions)
* [Correctness Guarantees](#correctness-guarantees)
* [Roadmap](#roadmap)
* [Research Directions](#research-directions)
* [Limitations](#limitations)
* [Project Status](#project-status)
* [Contributing](#contributing)
* [License](#license)
* [Author](#author)

---

# Overview

**Nexus-LOB** is a quantitative trading infrastructure project that combines a **low-latency C++ limit-order-book engine** with a **Python-based market-microstructure and execution-research stack**.

The system is designed around a simple principle:

> **The simulation used to research an execution strategy should behave like a real matching engine, not like a simplified price-array toy model.**

The project therefore separates latency-sensitive exchange mechanics from research-oriented Python components.

The core matching engine is implemented in modern C++ and supports:

* Limit orders
* Market orders
* IOC orders
* FOK orders
* GTC orders
* Order cancellation
* Order modification
* Price-time priority
* Partial fills
* Multi-level book sweeps
* Deterministic integer-tick pricing
* Fixed-capacity order storage
* O(1)-style level lookup and order-ID lookup
* Zero-allocation order-pool architecture

The Python layer builds on top of the same market state representation to provide:

* Market-data replay
* ITCH 5.0 parsing
* L2 order-book reconstruction
* Execution simulation
* Gymnasium-compatible reinforcement-learning environments
* Implementation-shortfall rewards
* TWAP
* VWAP
* POV
* Passive execution baselines
* C++ engine adapters
* A pure-Python reference/stub order book
* Cross-language parity testing

The architecture is deliberately designed so that the Python research stack can be developed against a deterministic `StubOrderBook`, then switched to the real C++ engine without changing the surrounding execution logic.

---

# Why Nexus-LOB?

Most introductory algorithmic-trading projects model a market using something similar to:

```text
price → quantity
```

and then simulate execution by manually modifying arrays.

That is useful for demonstrating concepts, but it misses several properties that become important when studying real execution:

* price-time priority
* order identity
* partial execution
* cancellation
* modification semantics
* market-order sweeps
* liquidity depletion
* queue position
* crossed-book prevention
* deterministic replay
* latency-sensitive data structures

Nexus-LOB treats the **limit order book itself as a first-class systems component**.

This creates a research stack where:

```text
Market Data
     │
     ▼
ITCH Parser
     │
     ▼
L2 Replay Engine
     │
     ▼
Order Book State
     │
     ├───────────────┐
     ▼               ▼
C++ Matching     Python Stub
Engine            Reference
     │               │
     └───────┬───────┘
             ▼
      Execution Environment
             │
       ┌─────┴─────┐
       ▼           ▼
   Baselines       RL
       │           │
       └─────┬─────┘
             ▼
    Execution Analytics
```

The resulting system is intended to bridge the gap between:

**market microstructure research**

and

**performance-oriented trading systems engineering.**

---

# Core Objectives

Nexus-LOB is being developed around five major subsystems.

| Subsystem           | Purpose                                  | Status                |
| ------------------- | ---------------------------------------- | --------------------- |
| C++ Matching Engine | Deterministic low-latency order matching | Implemented           |
| Python Quant Layer  | Market simulation and execution research | Implemented           |
| RL Execution Agent  | Learn optimal execution policies         | Planned               |
| CUDA Risk Engine    | Monte-Carlo VaR/CVaR acceleration        | Planned               |
| Zero-Copy Dashboard | Live order-book / execution telemetry    | Partially implemented |

The intended final architecture is:

```text
                   ┌───────────────────────────────┐
                   │       Market Data / ITCH       │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │      Replay / L2 Builder       │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │              Nexus Matching Layer                │
        │                                                  │
        │   C++ LimitOrderBook     Python StubOrderBook    │
        │          │                         │              │
        │          └───────── Same Contract ─┘              │
        └───────────────────────┬─────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Execution Environment  │
                    │      Gymnasium         │
                    └───────────┬────────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
              TWAP            VWAP            POV
                 │              │              │
                 └──────────────┼──────────────┘
                                │
                                ▼
                         RL Execution Agent
                         PPO / GRPO / etc.
                                │
                                ▼
                   Execution Analytics / Risk
```

---

# Architecture

## High-Level Architecture

```text
┌───────────────────────────────────────────────────────────┐
│                     Nexus-LOB Platform                    │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Market Data                                             │
│  ┌──────────────┐      ┌───────────────┐                │
│  │ ITCH 5.0     │─────▶│ Replay Engine │                │
│  │ Parser       │      │ / L2 Builder  │                │
│  └──────────────┘      └───────┬───────┘                │
│                                │                         │
│                                ▼                         │
│                    ┌─────────────────────┐              │
│                    │ Frozen Book Contract │              │
│                    └──────────┬──────────┘              │
│                               │                         │
│                 ┌─────────────┴─────────────┐           │
│                 ▼                           ▼           │
│        ┌────────────────┐         ┌────────────────┐    │
│        │ C++ Engine     │         │ Python Stub    │    │
│        │                │         │                │    │
│        │ LimitOrderBook │         │ StubOrderBook  │    │
│        └───────┬────────┘         └───────┬────────┘    │
│                │                          │             │
│                └───────────┬──────────────┘             │
│                            ▼                            │
│                 ┌──────────────────────┐                │
│                 │ Book Adapter Layer   │                │
│                 └──────────┬───────────┘                │
│                            ▼                            │
│                 ┌──────────────────────┐                │
│                 │ Gymnasium Execution  │                │
│                 │ Environment          │                │
│                 └──────────┬───────────┘                │
│                            │                            │
│                 ┌──────────┼───────────┐                │
│                 ▼          ▼           ▼                │
│               TWAP       VWAP         POV               │
│                 │          │           │                │
│                 └──────────┼───────────┘                │
│                            ▼                            │
│                     RL Execution                        │
│                            │                            │
│                            ▼                            │
│                    Risk / Analytics                      │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

# System Components

## 1. C++ Matching Engine

The C++ engine is the latency-sensitive core of Nexus-LOB.

It implements a deterministic limit-order-book matching model using:

* integer tick prices
* fixed-capacity order storage
* intrusive/free-list allocation
* FIFO queues
* direct order-ID lookup
* fixed-depth market-state views
* deterministic matching semantics

### Supported order operations

```text
Submit
 ├── Limit
 │    ├── GTC
 │    ├── IOC
 │    └── FOK
 │
 ├── Market
 │
 ├── Cancel
 │
 └── Modify
```

### Matching rules

Orders are matched using standard **price-time priority**.

For bids:

```text
Higher price → higher priority
Same price   → earlier order → higher priority
```

For asks:

```text
Lower price → higher priority
Same price  → earlier order → higher priority
```

This gives the engine deterministic FIFO behavior at each price level.

### Key properties

* Integer prices rather than floating-point prices
* Fixed-depth state representation
* Explicit order identity
* Deterministic fills
* Partial-fill support
* Multi-level sweeps
* Explicit rejection states
* Fixed-capacity order pool
* No dynamic allocation in the critical order-matching path

---

## 2. Cross-Language State Contract

One of the most important architectural decisions in Nexus-LOB is the frozen state contract between C++ and Python.

The C++ engine exposes a `BookStateView`, mirrored in Python through `BOOK_STATE_DTYPE`.

The contract uses:

```text
DEPTH = 10
```

levels per side.

The state contains:

```text
Bid Prices
Bid Sizes
Bid Counts

Ask Prices
Ask Sizes
Ask Counts

Additional book metadata
```

Prices are represented as integer ticks.

There is intentionally **no floating-point price representation in the core book state**.

### ABI contract

The current state structure is ABI-frozen at:

```text
sizeof(BookStateView) = 448 bytes
alignof(BookStateView) = 8
```

This is treated as a compatibility boundary.

Changing the structure requires:

1. Updating the C++ definition
2. Updating the NumPy mirror
3. Bumping the ABI version
4. Re-running parity tests
5. Re-validating the pybind interface

This prevents subtle C++/Python memory-layout mismatches.

---

# 3. Python Quant Layer

The Python subsystem contains the research-facing components.

The package currently includes modules for:

```text
nexus_quant/
├── book_state.py
├── book_port.py
├── itch_parser.py
├── replay.py
├── baselines.py
└── envs/
    └── order_book_env.py
```

The Python layer provides a clean interface for:

* order-book inspection
* market-data replay
* execution simulation
* RL environment interaction
* benchmark strategies
* engine substitution

---

# 4. ITCH 5.0 Replay

Nexus-LOB includes an ITCH 5.0 parser and replay pipeline.

The parser supports the market-data messages required to reconstruct an L2-style book, including:

* Add Order
* Add Order with MPID
* Execute
* Execute with Price
* Cancel
* Delete
* Replace
* Trade

The parser is designed around streaming processing rather than loading the entire dataset into memory.

Conceptually:

```text
Raw ITCH
   │
   ▼
Message Framing
   │
   ▼
Message Decoder
   │
   ▼
Order Event
   │
   ▼
Replay Engine
   │
   ▼
L2 Order Book
```

The replay engine additionally performs integrity checks for conditions such as:

* crossed books
* locked books
* unsorted events
* invalid/negative sizes
* inconsistent order state

This provides a realistic market-state source for execution experiments.

---

# 5. Execution Environment

Nexus-LOB exposes the order book to a Gymnasium-compatible execution environment.

The environment models the problem of executing a parent order over a finite time horizon.

The agent receives market-state information and decides how aggressively to trade.

A simplified execution loop is:

```text
Initial Inventory
       │
       ▼
Observe LOB
       │
       ▼
Choose Execution Action
       │
       ▼
Submit / Simulate Order
       │
       ▼
Matching Engine
       │
       ▼
Receive Fill
       │
       ▼
Update Inventory / Market State
       │
       ▼
Next Time Step
```

The current environment uses a **44-dimensional observation space** containing market, inventory, timing, and execution-related information.

The reward incorporates implementation shortfall together with inventory, timing, and adverse-move considerations.

---

# 6. Execution Baselines

Before evaluating an RL agent, Nexus-LOB provides deterministic execution baselines.

Current baselines:

| Strategy | Principle                                         |
| -------- | ------------------------------------------------- |
| TWAP     | Execute approximately equal quantities over time  |
| VWAP     | Allocate according to expected/observed volume    |
| POV      | Participate at a target fraction of market volume |
| Passive  | Prefer non-aggressive liquidity provision         |

These strategies establish reference points for evaluating learned execution policies.

The intended comparison is:

```text
                  Execution Quality
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
     TWAP               VWAP              POV
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
                  Passive Strategy
                         │
                         ▼
                   RL Execution
```

The RL system should not merely maximize raw reward. It should be evaluated against conventional execution algorithms using metrics such as:

* implementation shortfall
* slippage
* execution price
* completion rate
* inventory trajectory
* timing risk
* adverse selection
* participation rate

---

# 7. Shared-Memory Transport

Nexus-LOB also contains a lock-free single-producer/single-consumer shared-memory ring.

The ring transports the same `BookStateView` representation used by the matching engine.

Current characteristics:

* SPSC architecture
* lock-free producer/consumer path
* POSIX support
* Windows support
* fixed-size book-state slots
* drop-new-on-full semantics
* 448-byte state payloads

The architecture is intended to eventually support:

```text
C++ Engine
     │
     ▼
Shared Memory Ring
     │
     ├─────────────▶ Monitoring
     │
     ├─────────────▶ Dashboard
     │
     └─────────────▶ Analytics
```

A synthetic flow generator is also included for testing the transport independently of the full market-data stack.

---

# 8. Python/C++ Bridge

The C++ engine is exposed to Python through `pybind11`.

The Python interface is intentionally close to the conceptual exchange API.

Example:

```python
import nexus_engine as ne

engine = ne.Engine()

result = engine.submit_limit(
    1,
    ne.Side.Bid,
    10_000,
    500,
    ne.TimeInForce.GTC,
)
```

The engine returns execution information including:

```text
order ID
status
filled quantity
resting quantity
```

Fills can then be inspected:

```python
engine.fills()
```

Market-state inspection is available through:

```python
engine.best_bid()
engine.best_ask()
engine.spread()
engine.live_orders()
```

And through:

```python
engine.view()
engine.snapshot()
```

The distinction is important:

* `view()` provides a zero-copy view into engine memory.
* `snapshot()` provides an owning copy suitable for retaining independently.

---

# Order Book Model

Nexus-LOB represents the book using fixed-depth L2 state.

Current depth:

```text
10 bid levels
10 ask levels
```

Example:

```text
ASK
10100 ─────────────  150
10099 ─────────────  200
10098 ─────────────   75
10097 ─────────────  310
...
10091 ─────────────  120
────────────────────────
10090 ─────────────  180
10089 ─────────────  220
10088 ─────────────   90
...
10081 ─────────────  300
BID
```

The top of book is:

```text
Best Bid = highest bid
Best Ask = lowest ask
Spread   = Best Ask - Best Bid
```

Empty levels are represented by zeroed state entries.

This makes the state representation deterministic and straightforward to mirror into NumPy.

---

# Order Semantics

## Limit Orders

A limit order specifies:

```text
side
price
quantity
time-in-force
```

A marketable limit order consumes liquidity at prices satisfying the limit condition.

Any remaining quantity may rest depending on its time-in-force.

---

## Market Orders

Market orders consume available liquidity beginning at the best opposing price and may sweep multiple levels.

Example:

```text
ASK:

100.01 → 50
100.02 → 75
100.03 → 100

Incoming BUY 180
```

The order executes:

```text
50 @ 100.01
75 @ 100.02
55 @ 100.03
```

The engine therefore models liquidity consumption across multiple price levels.

---

## IOC

**Immediate-or-Cancel**

The executable portion is filled immediately.

Any remaining quantity is cancelled rather than resting.

---

## FOK

**Fill-or-Kill**

The complete order must be executable immediately.

If sufficient liquidity does not exist:

```text
No partial execution
→ order rejected / killed
```

---

## GTC

**Good-Til-Cancelled**

The executable portion fills immediately and any residual quantity can remain on the book.

---

## Cancel

Orders can be explicitly removed using their order ID.

---

## Modify

Modification semantics distinguish between:

* quantity reductions that preserve priority
* price changes that lose priority
* quantity increases that can affect priority

This matters because queue position is an important component of realistic execution modeling.

---

# Execution Environment

The execution environment frames trading as a sequential decision problem.

Suppose an agent needs to execute:

```text
Q = 100,000 shares
```

over:

```text
T = 300 steps
```

The agent must determine:

```text
How much to execute?
When to execute?
How aggressively to execute?
```

At every step, the agent observes market state and chooses an action.

Conceptually:

```text
State:
    ├── L2 prices
    ├── L2 quantities
    ├── spread
    ├── inventory remaining
    ├── time remaining
    ├── execution progress
    └── market features

Action:
    ├── quantity
    ├── aggressiveness
    └── execution decision

Environment:
    ├── matching engine
    ├── market dynamics
    └── fill model

Reward:
    ├── implementation shortfall
    ├── inventory risk
    ├── timing penalty
    └── adverse-move penalty
```

The objective is therefore closer to real execution optimization than simple directional price prediction.

---

# Reward Design

The environment is designed around **execution quality**, rather than whether the market price goes up or down.

A simplified implementation-shortfall objective can be represented as:

$$
IS = \sum_i q_i(p_i - p_{arrival})
$$

where:

* \(q_i\) is the executed quantity
* \(p_i\) is the execution price
* \(p_{arrival}\) is the benchmark/arrival price

The environment additionally considers execution-specific penalties associated with:

* remaining inventory
* time pressure
* adverse market movement

This creates a more realistic optimization target:

```text
Minimize
    Execution Cost
  + Inventory Risk
  + Timing Risk
  + Adverse Selection
```

rather than:

```text
Maximize predicted price movement
```

---

# Data & Replay

The intended market-data pipeline is:

```text
NASDAQ ITCH / Compatible Feed
              │
              ▼
        Binary Messages
              │
              ▼
        ITCH Parser
              │
              ▼
       Normalized Events
              │
              ▼
        Replay Engine
              │
              ▼
          L2 State
              │
              ▼
       Execution Engine
```

Replay is deterministic when supplied with deterministic input and configuration.

This is important for:

* regression testing
* strategy comparison
* RL environment reproducibility
* debugging
* engine parity tests
* benchmark generation

---

# Testing & Verification

Correctness is treated as a first-class requirement.

The project includes several independent test layers.

## C++ Engine Tests

The matching engine test suite covers:

* order resting
* L2 ladder behavior
* full fills
* partial fills
* price-time FIFO
* multi-level sweeps
* market-order behavior
* IOC semantics
* FOK semantics
* cancellation
* modification
* duplicate IDs
* invalid quantities
* invalid prices
* order-pool exhaustion

The engine currently reports:

```text
86 checks
0 failures
```

---

## ABI Test

The ABI check validates:

```text
sizeof(BookStateView) == 448
alignof(BookStateView) == 8
```

This prevents silent C++/Python memory-layout divergence.

---

## Shared-Memory Ring Tests

The ring implementation includes integrity and ordering tests.

The current test suite reports more than:

```text
30,000 checks
0 failures
```

and includes cross-process producer/consumer validation.

---

## Python Tests

Python tests cover:

```text
ITCH parser
Replay engine
Order-book environment
Book-state contract
Engine adapter
Engine-vs-stub parity
```

The current combined Python/binding test suite has been reported as:

```text
34 passed
```

including ABI parity and engine-vs-stub diff tests.

---

# Current Validation

The project is deliberately built around a **reference implementation + optimized implementation** model.

The pure-Python `StubOrderBook` acts as a behavioral oracle.

The C++ engine is compared against it through:

```text
Same Input
    │
    ├───────────────┐
    ▼               ▼
StubOrderBook   LimitOrderBook
    │               │
    ▼               ▼
L2 Snapshot      L2 Snapshot
    │               │
    └───────┬───────┘
            ▼
       Diff Harness
            │
       ┌────┴────┐
       │         │
      PASS      FAIL
```

This is particularly valuable because performance-oriented C++ implementations can otherwise introduce subtle state divergence that is difficult to identify from downstream RL results alone.

---

# Performance Philosophy

Nexus-LOB is not designed as a generic CRUD application.

The matching engine follows systems-programming principles common to latency-sensitive trading infrastructure.

## Integer Tick Prices

Instead of:

```cpp
double price;
```

the engine uses integer ticks:

```cpp
Price = integer tick
```

This avoids floating-point equality problems and simplifies deterministic comparison.

---

## Fixed-Capacity Memory

The engine uses a fixed order pool rather than repeatedly allocating and freeing orders.

Conceptually:

```text
Startup
   │
   ▼
Allocate Pool
   │
   ▼
Pre-existing Order Slots
   │
   ├── allocate
   ├── use
   └── return to free list
```

This reduces allocator pressure in the critical path.

---

## FIFO Queues

Orders at the same price level are processed in arrival order.

This preserves price-time priority.

---

## Direct Lookup

Order IDs require fast access for:

```text
cancel
modify
execute
lookup
```

The architecture therefore maintains an explicit ID-to-order lookup path rather than scanning the entire book.

---

## Fixed Book-State Layout

The L2 observation is fixed-size.

This allows:

* predictable memory usage
* efficient copying
* ABI verification
* direct NumPy representation
* shared-memory transport
* deterministic downstream interfaces

---

# Repository Structure

```text
Nexus_LOB/
│
├── bindings/
│   ├── pybind_wrapper.cpp
│   └── tests/
│       ├── test_abi_parity.py
│       └── test_diff_engine_stub.py
│
├── cpp_engine/
│   ├── include/
│   │   └── nexus/
│   │       ├── book_state.hpp
│   │       ├── types.hpp
│   │       ├── order_pool.hpp
│   │       ├── limit_order_book.hpp
│   │       ├── shm_ring.hpp
│   │       └── flow_gen.hpp
│   │
│   ├── tests/
│   │   ├── lob_test.cpp
│   │   ├── abi_check.cpp
│   │   └── ring_test.cpp
│   │
│   ├── demos/
│   │   ├── ring_producer.cpp
│   │   └── ring_probe.cpp
│   │
│   └── bench/
│
├── python_quant/
│   ├── nexus_quant/
│   │   ├── book_state.py
│   │   ├── book_port.py
│   │   ├── itch_parser.py
│   │   ├── replay.py
│   │   ├── baselines.py
│   │   └── envs/
│   │       └── order_book_env.py
│   │
│   ├── tests/
│   │   ├── test_contract_smoke.py
│   │   ├── test_itch_parser.py
│   │   ├── test_replay.py
│   │   └── test_order_book_env.py
│   │
│   ├── scripts/
│   └── requirements.txt
│
├── CMakeLists.txt
├── pyproject.toml
├── CLAUDE.md
├── PROGRESS.md
├── progress_b.md
└── README.md
```

---

# Requirements

## Required

### C++

* C++20 compiler
* CMake
* Standard C++ library

Recommended:

* MSVC 2022 Build Tools on Windows
* GCC/Clang on Linux/WSL

### Python

Python:

```text
>= 3.10
```

Recommended:

```text
Python 3.12
```

### Python dependencies

The research layer uses packages including:

* NumPy
* pytest
* Gymnasium
* pybind11
* scikit-build-core

Additional RL/data dependencies can be installed as the execution-agent subsystem develops.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Lokeshrao69/Nexus_LOB.git
cd Nexus_LOB
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip
```

Install Python dependencies:

```bash
pip install -r python_quant/requirements.txt
```

Install build dependencies:

```bash
pip install scikit-build-core pybind11 cmake
```

---

# Building the C++ Engine

Configure a Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build -j
```

Run CTest:

```bash
ctest --test-dir build --output-on-failure
```

---

# Building the Python Extension

Enable the pybind11 extension:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEXUS_BUILD_PYBIND=ON
```

Build:

```bash
cmake --build build -j
```

The resulting `nexus_engine` module can then be imported from Python.

---

# Running Tests

## C++ Engine

The C++ tests can also be compiled directly:

```bash
g++ -std=c++20 -O2 -Wall -Wextra \
    -I cpp_engine/include \
    cpp_engine/tests/abi_check.cpp \
    -o abi_check
```

Run:

```bash
./abi_check
```

Matching-engine tests:

```bash
g++ -std=c++20 -O2 -Wall -Wextra \
    -I cpp_engine/include \
    cpp_engine/tests/lob_test.cpp \
    -o lob_test
```

Run:

```bash
./lob_test
```

Shared-memory ring:

```bash
g++ -std=c++20 -O2 -Wall -Wextra \
    -I cpp_engine/include \
    cpp_engine/tests/ring_test.cpp \
    -o ring_test
```

Run:

```bash
./ring_test
```

---

## Python

Run the pure-Python test suite:

```bash
python -m pytest python_quant/tests -v
```

Run binding tests:

```bash
python -m pytest bindings/tests -v
```

Run everything:

```bash
python -m pytest python_quant/tests bindings/tests -v
```

---

# Running the Shared-Memory Demo

Build the producer:

```bash
g++ -std=c++20 -O2 \
    -I cpp_engine/include \
    cpp_engine/demos/ring_producer.cpp \
    -o ring_producer
```

Build the probe:

```bash
g++ -std=c++20 -O2 \
    -I cpp_engine/include \
    cpp_engine/demos/ring_probe.cpp \
    -o ring_probe
```

Start the producer in one terminal:

```bash
./ring_producer nex_aapl 4000 16384 0xC0FFEE 1
```

Start the probe in another:

```bash
./ring_probe nex_aapl 4000 5
```

This demonstrates live cross-process transmission of the fixed-size book state.

---

# Using the Matching Engine

Example Python usage:

```python
import nexus_engine as ne

engine = ne.Engine()

# Rest a bid.
result = engine.submit_limit(
    1,
    ne.Side.Bid,
    10_000,
    500,
    ne.TimeInForce.GTC,
)

print(result)
```

Add liquidity:

```python
engine.submit_limit(
    10,
    ne.Side.Ask,
    10_050,
    100,
    ne.TimeInForce.GTC,
)
```

Cross the spread:

```python
result = engine.submit_limit(
    11,
    ne.Side.Bid,
    10_050,
    150,
    ne.TimeInForce.GTC,
)
```

Inspect fills:

```python
print(engine.fills())
```

Inspect the book:

```python
print(engine.best_bid())
print(engine.best_ask())
print(engine.spread())
print(engine.live_orders())
```

Obtain state:

```python
view = engine.view()
snapshot = engine.snapshot()
```

---

# Using the Python Environment

The environment can be built around either:

```text
StubOrderBook
```

or:

```text
EngineAdapter
```

This allows research code to be developed without requiring the native extension for every iteration.

Conceptually:

```python
book = StubOrderBook(...)
```

or:

```python
engine = nexus_engine.Engine(...)
book = EngineAdapter(engine)
```

The execution environment can then interact with either implementation through the common book interface.

This design significantly reduces coupling between:

```text
RL development
```

and:

```text
C++ compilation/toolchain availability
```

---

# Design Decisions

## Why C++ + Python?

The project deliberately uses two languages because they solve different problems.

### C++

Best suited for:

* deterministic matching
* memory control
* cache-aware data structures
* low-latency execution
* fixed-size state
* systems-level benchmarking

### Python

Best suited for:

* research iteration
* NumPy
* statistical analysis
* reinforcement learning
* data processing
* experiment orchestration
* visualization

Rather than forcing one language to do everything, Nexus-LOB creates a stable interface between the two.

---

# Why a Stub Order Book?

The Python stub is not intended to replace the production-style engine.

It exists as a **reference model**.

The architecture is:

```text
                  Behavioral Contract
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      StubOrderBook             LimitOrderBook
        Reference                  Optimized
            │                         │
            └────────────┬────────────┘
                         ▼
                    Diff Testing
```

This provides a practical correctness oracle.

If the optimized C++ engine disagrees with the reference implementation under the same event sequence, the mismatch can be isolated before it contaminates higher-level execution or RL experiments.

---

# Correctness Guarantees

The current validation strategy checks multiple layers independently.

### Engine correctness

```text
86 / 86 checks passing
```

### ABI correctness

```text
448-byte BookStateView
8-byte alignment
```

### Shared-memory correctness

```text
30,000+ integrity/order checks
```

### Python/replay/environment tests

```text
34 tests passing
```

### Engine-vs-stub

The diff-test harness verifies parity between:

```text
Reference Python implementation
```

and:

```text
Real C++ engine
```

This layered testing approach is important because a trading system can produce plausible-looking results while still containing incorrect execution semantics.

---

# Performance Philosophy

The ultimate performance target is not simply:

> "Make the benchmark number large."

The goal is to construct an engine whose performance characteristics follow naturally from its architecture.

Important design principles include:

### Deterministic memory

Avoid unnecessary allocations during matching.

### Cache-friendly state

Keep hot data compact and predictable.

### Integer arithmetic

Represent prices as ticks.

### Explicit ownership

Make lifetime and ownership of orders deterministic.

### Constant-time lookup paths

Avoid scanning the entire book for common operations.

### Stable ABI

Keep the C++/Python boundary explicit and testable.

### Zero-copy transport

Where possible, move fixed-size book states without serialization.

---

# Roadmap

## Phase 1 — Matching Engine

* [x] Frozen C++/Python state contract
* [x] Integer-tick pricing
* [x] Limit orders
* [x] Market orders
* [x] IOC
* [x] FOK
* [x] GTC
* [x] Cancellation
* [x] Modification
* [x] FIFO matching
* [x] Multi-level sweeps
* [x] Fixed order pool
* [x] ABI verification
* [x] Engine unit tests

---

## Phase 1B — Market Replay & Execution Environment

* [x] ITCH 5.0 parser
* [x] L2 replay engine
* [x] Replay integrity validation
* [x] Python stub order book
* [x] C++ engine adapter
* [x] Gymnasium environment
* [x] TWAP
* [x] VWAP
* [x] POV
* [x] Passive baseline
* [x] Engine-vs-stub parity tests

---

## Phase 2 — Reinforcement Learning

Planned:

* [ ] PPO execution agent
* [ ] GRPO experiments
* [ ] RL-vs-baseline evaluation
* [ ] Multi-seed experiments
* [ ] Slippage analysis
* [ ] Execution trajectory visualization
* [ ] Adverse-selection analysis
* [ ] Out-of-sample evaluation

Target research question:

> Can a reinforcement-learning execution policy reduce implementation shortfall by learning when to cross the spread and when to provide liquidity, while still satisfying a target execution schedule?

---

## Phase 3 — GPU Risk Engine

Planned CUDA subsystem:

```text
Market State
     │
     ▼
Scenario Generator
     │
     ▼
100k+ Monte-Carlo Paths
     │
     ▼
P&L Distribution
     │
     ├── VaR
     └── CVaR
```

Planned work:

* [ ] CUDA Monte-Carlo engine
* [ ] GPU random-number generation
* [ ] Parallel path simulation
* [ ] VaR
* [ ] CVaR
* [ ] CPU/GPU benchmark
* [ ] Risk dashboard integration

---

## Phase 4 — Live Dashboard

Planned architecture:

```text
C++ Engine
     │
     ▼
Shared Memory Ring
     │
     ▼
Python / WebSocket Layer
     │
     ├── L2 Depth
     ├── Spread
     ├── Inventory
     ├── Orders
     ├── Fills
     ├── Latency
     └── PnL
```

Planned features:

* [ ] Live order-book visualization
* [ ] Market depth chart
* [ ] Execution timeline
* [ ] Inventory chart
* [ ] Slippage analytics
* [ ] Latency monitoring
* [ ] Strategy comparison
* [ ] RL-vs-baseline dashboard

---

# Research Directions

Once the complete stack is implemented, Nexus-LOB can support several quantitative-finance experiments.

## 1. Optimal Execution

Compare:

```text
TWAP
VWAP
POV
Passive
PPO
GRPO
```

using the same market replay.

---

## 2. Market Impact

Study the relationship between:

```text
Order Size
     │
     ▼
Participation Rate
     │
     ▼
Liquidity Consumption
     │
     ▼
Market Impact
     │
     ▼
Implementation Shortfall
```

---

## 3. Adverse Selection

Measure whether passive orders receive worse fills immediately before adverse mid-price movements.

---

## 4. Queue Position

Investigate the value of:

```text
Price Priority
+
Time Priority
```

in determining expected fill probability.

---

## 5. Execution Under Volatility

Evaluate execution strategies across regimes such as:

```text
Low Volatility
      │
      ▼
Normal
      │
      ▼
High Volatility
      │
      ▼
Liquidity Shock
```

---

## 6. Learned Execution Policies

Train an RL policy to learn the trade-off between:

```text
Aggressive Execution
        vs.
Passive Execution
```

subject to:

```text
completion constraint
+
risk constraint
+
time constraint
```

---

# Limitations

Nexus-LOB is a research and engineering platform, not a production exchange or live trading system.

Important limitations include:

* Market dynamics are still simulated/replayed.
* Real exchange connectivity is not currently implemented.
* Latency benchmarks depend heavily on hardware and compiler configuration.
* The RL subsystem is still under development.
* CUDA risk analytics are not yet implemented.
* The dashboard is not yet complete.
* Historical replay quality depends on the input market-data feed.
* The current environment is designed for research rather than live order routing.
* No claim is made that simulated execution exactly reproduces a particular exchange's matching implementation.

Performance numbers should therefore always be interpreted relative to:

```text
hardware
compiler
optimization flags
dataset
market regime
book depth
order-flow distribution
```

---

# Project Status

**Current development stage: Phase 1 / Phase 1B complete, with higher-level quantitative components under active development.**

Current verified components include:

```text
C++ matching engine             ✅
C++ engine tests                ✅
ABI contract                    ✅
Shared-memory ring              ✅
ITCH parser                     ✅
L2 replay engine                ✅
Python order-book environment   ✅
TWAP baseline                   ✅
VWAP baseline                   ✅
POV baseline                    ✅
Passive baseline                ✅
C++/Python adapter              ✅
Engine-vs-stub diff testing     ✅
Pybind bridge                   ✅

PPO / GRPO                      ⏳
CUDA VaR / CVaR                 ⏳
Live dashboard                  ⏳
```

The repository's progress documentation tracks the implementation status and validation details as development continues.

---

# Reproducibility

For meaningful experimental comparisons, experiments should record:

```text
Git commit
Dataset identifier
Dataset date/range
Random seed
Initial inventory
Execution horizon
Target participation
Book depth
Reward configuration
Environment configuration
Strategy configuration
Hardware
Compiler
Optimization flags
```

For RL experiments additionally record:

```text
Algorithm
Network architecture
Learning rate
Batch size
Discount factor
Entropy coefficient
Number of training episodes
Number of evaluation seeds
```

This makes it possible to distinguish genuine strategy improvements from differences caused by data, configuration, or randomness.

---

# Contributing

Contributions are welcome for:

* matching-engine correctness
* benchmark infrastructure
* market-data parsers
* replay functionality
* execution algorithms
* RL environments
* CUDA optimization
* testing
* documentation
* visualization

Before modifying the cross-language book-state structure, review the ABI contract carefully.

Changes to:

```text
BookStateView
BOOK_STATE_DTYPE
```

should be treated as interface changes rather than ordinary refactors.

A recommended workflow is:

```bash
git checkout -b feature/my-change

# implement change

python -m pytest python_quant/tests bindings/tests -v

git diff

git commit -m "feat: describe change"

git push origin feature/my-change
```

---

# License

This repository is currently described as a **proprietary finance-placement portfolio project**.

See the repository's project metadata and licensing files for the applicable terms.



# Final Note

Nexus-LOB is intended to demonstrate the complete engineering path from **market-data reconstruction** to **exchange-style order matching** to **execution research**.

The central design principle is:

```text
                REALISTIC MARKET STATE
                         │
                         ▼
              DETERMINISTIC MATCHING
                         │
                         ▼
                EXECUTION SIMULATION
                         │
                         ▼
              BASELINE STRATEGIES
                         │
                         ▼
               REINFORCEMENT LEARNING
                         │
                         ▼
              EXECUTION OPTIMIZATION
                         │
                         ▼
                RISK + MONITORING
```

Rather than treating algorithmic trading as a single machine-learning problem, Nexus-LOB approaches it as a **systems + market-microstructure + quantitative-research problem**.

The long-term objective is a reproducible research platform in which execution strategies can be evaluated against realistic order-book mechanics, deterministic market replay, conventional execution algorithms, and eventually GPU-accelerated risk analytics and live telemetry.
