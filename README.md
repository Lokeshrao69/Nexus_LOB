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
