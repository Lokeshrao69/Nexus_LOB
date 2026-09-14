# Nexus-LOB — Progress Report

**Status date:** 2026-09-14 · **Branch:** `feature/part2-phase3-rl-fairness` · **Milestone:** Part 1 (systems half) complete; Part 2 (quant research layer) — Phases 0–3 landed, **Phase 3 complete** (queue dynamics E5, adverse selection E6, fair RL rework — per-regime ≥5-seed eval, adaptive baselines, honest vignette). This file is a plain-language
snapshot for anyone (Person A or Person B) picking the project up; the authoritative,
constantly-updated handoff is `CLAUDE.md`, the research roadmap is `plan_2.md`, and the
**approved Phase-3 working plan is `plan.md`**.

> TL;DR: the cross-language state contract is frozen, the C++ matching engine is
> built and passing its own tests (86/86), the Python bridge drives that real engine,
> and the **shared-memory ring** that will feed the dashboard (subsystem 5's C++ core)
> is built and demoed live. **Person B has landed** the ITCH 5.0 parser, an
> ITCH→L2 replay engine, the injectable stub↔engine adapter, the Gymnasium
> `OrderBookEnv`, TWAP/VWAP/POV/Passive baselines, PPO/GRPO agents, the combined
> interactive dashboard, and the risk↔env integration seam. **Phase 3 is complete:**
> E5 fill models (logistic + KM survival), E6 adverse selection (post-fill drift +
> P_adverse), fair per-regime RL eval with bootstrap CIs, adaptive baselines (apov,
> stwap, isaware), and an honest E5/E6 vignette. Full test suite: **157 passed, 1
> skipped**. Next: push this branch, open the PR, then Phase 4 (real NASDAQ ITCH tape).

---

## 0. Part 2 (research layer) — status as of 2026-09-14

The systems half (Part 1) is done. The quant research layer turns Nexus-LOB into
"systems + real market-microstructure research" (`plan_2.md` is the roadmap of record;
`plan.md` was the approved Phase-3 build plan):

| Phase | What | State |
|---|---|---|
| 0–1 | Repo hygiene, CI, research spine (features/labels/dataset/experiments/models), E1–E4 synthetic IC vignette | ✅ on `main` (PR #15); honest null IC on the random-walk tape |
| 2 | Execution realism — cost model, market-VWAP metrics (the "self-VWAP" flaw fix), env cost/queue knobs, backtest harness | ✅ on `main` (PRs #18/#20); 110 passed / 1 skipped |
| 3 | **Queue dynamics (E5) + adverse selection (E6) + fair RL rework** — order-level tracker + P(fill) KM/logistic models; post-fill drift; per-regime ≥5-seed eval with symmetric info; adaptive baselines | ✅ **COMPLETE** on `feature/part2-phase3-rl-fairness` (WS-1/2 via PRs #18/#20; WS-3/4/5 as 3 commits, 157 passed / 1 skipped) |
| 4–5 | Real NASDAQ ITCH tape, research report, honest README | ⏭️ next |

**Phase 3 delivered (2026-09-14, `feature/part2-phase3-rl-fairness`):**

- **WS-1 (E5 fill models):** `queue_dynamics.py` — `fill_dataset`, `standing_order_lifetimes`, KM `fill_prob_survival` with cancel as competing risk, NumPy-IRLS `LogisticFillModel` + calibration/Brier. Merged via PR #20.
- **WS-2 (E6 adverse selection):** `adverse.py` — `post_fill_drift`, `P_adverse`, `adverse_groups`. Merged via PR #20.
- **WS-3 (fair RL eval):** `agents/evaluate.py` — `evaluate_regime_ci(policy, regimes, ...)` with symmetric `vol_feature` toggle, same-seed episodes per strategy, market-VWAP slippage, iid-bootstrap CIs, seeds ≥ 5 enforced.
- **WS-4 (adaptive baselines):** `baselines.py` — `apov` (adaptive-POV), `stwap` (schedule-TWAP on a (1−t)^0.8 curve), `isaware` (IS-aware). All stateless, reading only the live book + t/inventory.
- **WS-5 (vignette + docs):** `scripts/queue_adverse_vignette.py` runs E5 (logistic calibration slope + Brier vs base rate) and E6 (adverse drift + CI gated) on rw vs drift; `docs/RESEARCH.md` carries the E5/E6 tables.

**Honest results (documented, not tuned):**
- E6 null holds: on the random-walk arm every horizon's CI straddles 0.
- E6 structural: on the drift arm, ask fills are adversely drift-chased (p_adverse=1.00), bid fills were early (0.00).
- E5 negative: on the trending arm the fill model ties the base-rate baseline (Brier 0.034 vs 0.034) — calibrated but not informative beyond base rate on a trend.
- Queue-position degenerate on synthetic: ahead_at_fill==0 for every fill (the generator takes the queue front), so adverse_groups' queue terciles collapse — logged as needing Phase-4 tape.

**No `OrderBookEnv` changes in Phase 3.** Working tree is on the feature branch;
updates land via feature PR with a real merge commit (never squash).

---

## 1. What the project is (30 seconds)

Nexus-LOB is a hybrid **C++/Python** limit-order-book (LOB) trading & market-
microstructure platform built as a **finance-placement portfolio project** (targets:
JPMC Quant Research, Nomura Algo, Goldman Systematics). Two people, ~8 weeks.

Five subsystems (from CLAUDE.md §2):
1. **Ultra-low-latency C++ matching engine** (Person A) — Limit / Market / FOK / IOC /
   Cancel / Modify, zero-allocation, integer-tick prices.
2. **Microstructure sim + RL execution agent** (Person B) — Gymnasium env; PPO/GRPO vs
   TWAP/VWAP/Avellaneda–Stoikov baselines.
3. **GPU risk engine (CUDA)** — Monte-Carlo VaR/CVaR over 100k+ paths (stubbed, not started).
4. **Zero-copy pipeline + dashboard** — shmem/WebSocket → live depth, latency, PnL (not started).
5. *(Subsystem 5 in comments — the shmem ring → dashboard.)*

Headline resume targets (not yet met): >500k orders/sec sub-µs matching; ~14% lower
slippage vs VWAP; ~40× CUDA VaR speedup.

---

## 2. The architecture seam (why things are built in this order)

The two people work in **parallel**, so we froze the *one data structure* that crosses
the C++↔Python boundary before writing the engine — the **state contract**
`BookStateView` (C++) ↔ `BOOK_STATE_DTYPE` (NumPy). Both sides build against that frozen
shape:

- **Person B** can build/train the RL env against a pure-Python `StubOrderBook` that
  emits the exact same `view()`/`snapshot()` interface as the real engine — **today**, no
  C++ build needed.
- **Person A** drops the real `LimitOrderBook` behind the same seam. `StubOrderBook`
  then becomes the **reference oracle** the real engine is diff-tested against.

Rules that must never be broken (they keep the two halves compatible):
- **Prices are integer ticks** (`int64`), never floats. `0` in a price slot = empty level.
- **Fixed depth** `kDepth = DEPTH = 10` per side; index 0 = best level; bids descend,
  asks ascend; empty levels zero-padded.
- **Layout is ABI-frozen:** `sizeof(BookStateView) == 40*10 + 48 == 448` bytes.
  Changing it = ABI break (bump the version, update the Python mirror, re-run parity tests).

See `bindings/CONTRACT.md` for the full spec.

---

## 3. What is DONE (and verified)

### 3a. C++ matching engine — **built & self-tested** ✅
| Piece | File | Notes |
|---|---|---|
| Frozen state contract | `cpp_engine/include/nexus/book_state.hpp` | ABI v1, `sizeof == 448`, `alignof == 8` |
| Value types | `cpp_engine/include/nexus/types.hpp` | `OrderId/Price/Qty`, `Fill`, `Status`, `ExecResult` |
| Zero-alloc order pool | `cpp_engine/include/nexus/order_pool.hpp` | fixed slab + intrusive free-list |
| Matching engine | `cpp_engine/include/nexus/limit_order_book.hpp` | Limit/Market/FOK/IOC/Cancel/Modify, FIFO, O(1) level lookup & id map |
| **Engine tests** | `cpp_engine/tests/lob_test.cpp` | **86/86 checks pass** |
| ABI lock | `cpp_engine/tests/abi_check.cpp` | prints 448 / alignof 8 ✅ |

Verified 2026-08-30 with MSYS2 g++: `lob_test` → `86 checks, 0 failed / ALL PASS`;
`abi_check` → `sizeof = 448 (expected 448)`. Engine behavior covered: resting & L2
ladder order, full/partial crosses, price-time FIFO, multi-level sweeps, IOC / FOK /
market semantics, cancel, modify (priority-keeping reduce vs. priority-losing reprice),
and every reject path (bad qty/price, dup id, pool-full).

### 3b. Build system ✅
- `CMakeLists.txt` — engine lib (header-only today → static lib when `.cpp` land), the
  `nexus_engine` pybind module, `abi_check` + `lob_test` under CTest, and CUDA hooks
  (on-but-stubbed).
- `pyproject.toml` — scikit-build-core packaging; pytest `pythonpath` includes
  `python_quant` and `bindings` so tests import both `nexus_quant` and the compiled
  module without a pip install.

### 3c. Python side — **authored, not yet run** ⚠️
- `python_quant/nexus_quant/book_state.py` — NumPy dtype mirror + `StubOrderBook`
  (the oracle).
- `python_quant/tests/test_contract_smoke.py` — pure-NumPy smoke test.
- `python_quant/nexus_quant/__init__.py` — package exports.

### 3d. Pybind bridge — **wired to the real engine, compiled & parity-tested** ✅
`bindings/pybind_wrapper.cpp` no longer a placeholder. `Engine` owns a real
`LimitOrderBook` and exposes order entry, fills, and the two view flavors (see §5).
`bindings/tests/test_abi_parity.py` drives the real engine.

> ✅ **Built and verified 2026-09-04 (Windows/MSVC):** Tier 2 passed — `nexus_engine`
> compiled, `test_abi_parity.py` (6) and the Engine-vs-Stub diff-test (2) green (see §3f).

### 3e. Shared-memory ring + flow (subsystem 5, C++) — **built & demoed** ✅
| Piece | File | Notes |
|---|---|---|
| Shared-memory SPSC ring | `cpp_engine/include/nexus/shm_ring.hpp` | OS shmem (POSIX + Windows), lock-free SPSC, drop-new-on-full, 448-B slots |
| Synthetic flow generator | `cpp_engine/include/nexus/flow_gen.hpp` | seeded LCG, mid random-walk + passive/aggressive mix |
| Ring tests | `cpp_engine/tests/ring_test.cpp` | **30,011 checks pass** (order + integrity; drop semantics) |
| Publisher + probe demos | `cpp_engine/demos/ring_producer.cpp`, `ring_probe.cpp` | live cross-process book demo — **verified 200 frames, 0 dropped** |

This is the transport the future dashboard consumes: the engine (real or synthetic flow)
publishes its `BookStateView` into the ring after every order; a reader process follows
the live book. Same frozen 448-byte payload end to end.

### 3f. Person B — ITCH replay + execution env ✅
Merged in PR #2 (`feature/env-and-itch`). Runs against `StubOrderBook` (no C++ build
needed) or the real engine via `book_port.adapt(...)`.

| Piece | File | Notes |
|---|---|---|
| ITCH 5.0 parser | `python_quant/nexus_quant/itch_parser.py` | streaming, framed+raw, Add/MPID/Exec/ExecPx/Cancel/Delete/Replace/Trade |
| ITCH→L2 replay | `python_quant/nexus_quant/replay.py` | `ReplayEngine` + `check_integrity` (crossed/locked/unsorted/neg-size) |
| Injectable adapter | `python_quant/nexus_quant/book_port.py` | `StubBookAdapter` + `EngineAdapter`; **`cancel_id`/`lookup` added 2026-09-04** so replay works against the real engine |
| Gymnasium env | `python_quant/nexus_quant/envs/order_book_env.py` | 44-dim obs, IS reward + inv/time/adv penalities, terminal dump |
| Baselines | `python_quant/nexus_quant/baselines.py` | TWAP / VWAP / POV / Passive |
| Tests | `python_quant/tests/test_{itch_parser,order_book_env,replay}.py` | ✅ **PASSING** (2026-09-04) |
| Diff-test harness | `bindings/tests/test_diff_engine_stub.py` | ✅ **PASSING** — Engine-vs-Stub L2-ladder parity |

> ✅ **All of §3f is now RUN and PASSING** (2026-09-04, Windows/MSVC). `pytest
> python_quant/tests bindings/tests -v` → **34 passed**, incl. `test_abi_parity.py` (6) and
> `test_diff_engine_stub.py` (2) — the real engine and the stub oracle agree.

---

### 3g. Person A — Monte-Carlo VaR/CVaR risk engine (subsystem 3) ✅ (CPU; GPU blocked)
Authored on `feature/risk-engine` (2026-09-06). Key idea: the per-path RNG is a
**pure function of (seed, path, step)** via counter-based splitmix64, so CPU,
CUDA, and a NumPy oracle all draw *identical* paths → **bit-for-bit** parity
(not Monte-Carlo tolerance).

| Piece | File | State |
|---|---|---|
| Model + RNG (GBM / Merton jump-diffusion) | `cuda_risk/risk_common.hpp` | ✅ |
| CPU serial reference | `cuda_risk/risk_cpu.hpp` | ✅ (200k×252 ≈ 1.0 s here) |
| CUDA kernel + launcher (1 thread/path) | `cuda_risk/risk_cuda.{cu,h}` | ⚠️ authored, needs toolkit |
| CPU-vs-GPU bench + parity | `cuda_risk/risk_bench.cpp` | ✅ CPU; GPU on a CUDA box |
| pybind `compute_var_cvar` (CPU) | `bindings/pybind_wrapper.cpp` | ✅ |
| NumPy exact-parity oracle | `python_quant/tests/test_risk_parity.py` | ✅ 3/3 |
| C++ statistical tests | `cpp_engine/tests/risk_test.cpp` | ✅ CTest 5/5 |
| CMake wiring | `CMakeLists.txt` | ✅ `nexus_risk`/`risk_bench` under toolkit |

**Verified here (no GPU): pytest 49 passed, CTest 5/5.** The CUDA kernel and
~40× speedup cannot be compiled/measured on this machine — that's the blocker.

### 3h. Phase 3 — Queue dynamics (E5) + adverse selection (E6) + fair RL rework ✅
Built on `feature/part2-phase3-rl-fairness` (2026-09-14). WS-1/2 landed via PRs #18/#20;
WS-3/4/5 as 3 branch commits (`ca4b0bd`, `6bb9905`, `b6cb67a`).

| Piece | File | Notes |
|---|---|---|
| E5 fill-model substrate | `python_quant/nexus_quant/queue_dynamics.py` | `fill_dataset`, `standing_order_lifetimes`, KM `fill_prob_survival` (cancel as competing risk), NumPy-IRLS `LogisticFillModel` + calibration/Brier |
| E6 adverse selection | `python_quant/nexus_quant/adverse.py` | `post_fill_drift`, `P_adverse`, `adverse_groups` |
| Fair per-regime RL eval | `python_quant/nexus_quant/agents/evaluate.py` | `evaluate_regime_ci(...)` — symmetric info, ≥5-seed (enforced), same-seed episodes, market-VWAP slippage, iid-bootstrap CIs |
| Adaptive baselines | `python_quant/nexus_quant/baselines.py` | `apov` (adaptive-POV), `stwap` (schedule-TWAP, (1−t)^0.8), `isaware`; stateless, live-book only; original baselines untouched |
| E5/E6 vignette | `python_quant/scripts/queue_adverse_vignette.py` | honest results on rw vs drift |
| Research doc | `docs/RESEARCH.md` | E5/E6 tables + negative results |
| Test suite | full `pytest python_quant/tests` | ✅ **157 passed / 1 skipped**; CI lint scope (nexus_quant + bindings) clean |

**Honest findings:** E6 null holds on rw (all CIs straddle 0); on drift, ask fills
adversely chase drift (p_adverse=1.00) while bid fills were early (0.00). E5's
logistic model ties base rate on a trend (Brier 0.034 vs 0.034) — calibrated but not
informative. Queue-position signal is degenerate on synthetic flow (ahead_at_fill==0);
on the agenda for the Phase-4 NASDAQ tape.

## 4. What is NOT done yet

- ✅ ~~Pybind module built + parity tests green~~ — **DONE 2026-09-04** (Windows/MSVC).
- ✅ ~~Run the authored Python (parser/replay/env/baselines/diff-test)~~ — **DONE 2026-09-04**.
- ✅ ~~RL execution agent (PPO) vs the baselines~~ — **DONE 2026-09-05** (beats all baselines on the
  env's reward; ≈ VWAP on shortfall — see `python_quant/nexus_quant/agents/README.md` for the
  honest numbers and the high-vol path to the slippage headline).
- ✅ ~~**High-volatility regime → ~14% below VWAP headline**~~ — **ACHIEVED 2026-09-07**. Added a
  Markov regime-switching + gap-off flow to `OrderBookEnv` (defaults preserve calm behavior).
  PPO shortfall **1.401 bps vs VWAP 2.827 = +50.4%** on 100 seeded episodes; robust across seeds
  (+38.2% on a 200-episode re-check). Policy saved at `python_quant/artifacts/policy_ppo_highvol.npz`.
  See `HIGHVOL_PLAN.md` at the repo root.
- ✅ ~~**GRPO + Python dashboard on the shmem ring + risk↔env integration seam**~~ — **DONE 2026-09-09**
  (combined desk: `dashboard_page.html` + `SnapshotHub` + `serve_dashboard.py`; GRPO trainer; `lambda_risk`
  CVaR inventory penalty wired into `OrderBookEnv`).
- ✅ ~~**Phase 3: queue dynamics E5 + adverse selection E6 + fair RL rework**~~ — **COMPLETE 2026-09-14**
  (see §3h; 157 passed / 1 skipped; `feature/part2-phase3-rl-fairness`).
- ⚠️ **CUDA VaR/CVaR risk engine (subsystem 3)** — CPU reference + exact NumPy parity + CTest
  **DONE & green** (2026-09-06, branch `feature/risk-engine`); the **GPU kernel + ~40× speedup
  are BLOCKED** here (no CUDA toolkit) — compile `nexus_risk`/`risk_bench` on a CUDA machine.
- ⏭️ **Phase 4/5:** real NASDAQ ITCH tape ingestion (`fetch_itch.py` + order-level parser), research
  report, honest README — next milestone.

---

## 5. For Person B — the Python API you code against

Once `nexus_engine` is built, the seam surface is exactly the same shape as
`StubOrderBook`, so your env code can target either. The real engine adds order entry:

```python
import nexus_engine as ne

e = ne.Engine()                      # default price band 1..100_000 ticks
# e = ne.Engine(min_price=1, max_price=500_000, pool_capacity=1 << 18)

# Rest a GTC limit: place a bid at price 10_000 for 500 shares.
r = e.submit_limit(1, ne.Side.Bid, 10_000, 500, ne.TimeInForce.GTC)
r  # -> {"id": 1, "status": ne.Status.Accepted, "filled": 0, "resting": 500}

# Aggressive buy lifting the best ask(es) — fills come back per call.
e.submit_limit(10, ne.Side.Ask, 10_050, 100, ne.TimeInForce.GTC)
r2 = e.submit_limit(11, ne.Side.Bid, 10_050, 150, ne.TimeInForce.GTC)
r2  # -> {"id": 11, "status": ne.Status.Filled, "filled": 100, "resting": 0}
e.fills()  # -> [(10, 11, 10_050, 100, ne.Side.Bid)]  # (maker, taker, px, qty, aggressor)

# Quote introspection
e.best_bid(), e.best_ask(), e.spread(), e.live_orders()

# Observation (contract arrays): view() is ZERO-COPY (aliases engine memory —
# normalize/copy it now); snapshot() is a safe owning copy.
obs = e.view()          # {"bid_px","bid_sz","bid_ct","ask_px","ask_sz","ask_ct", ...}
snap = e.snapshot()
```

Key enum values:
- `Side`: `Bid`, `Ask`, `None_` (Python `None` is a keyword, hence `None_`).
- `TimeInForce`: `GTC` (rest residual), `IOC` (fill-then-kill), `FOK` (all-or-nothing).
- `Status`: `Accepted`, `Filled`, `PartiallyFilledResting`, `Canceled`,
  `Rejected_DupId`, `Rejected_BadPrice`, `Rejected_BadQty`, `Rejected_PoolFull`,
  `Rejected_FOK`, `NoOp`.

**You can start NOW against `StubOrderBook`** (no engine build needed) — build the ITCH
parser and `OrderBookEnv` on the frozen contract, then swap `StubOrderBook` for
`Engine` when the WSL build lands.

---

## 6. How to see everything work

```bash
# C++ only (works on Windows + MSYS2 g++, no build system needed):
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/abi_check.cpp -o abi_check.exe && ./abi_check.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/lob_test.cpp -o lob_test.exe && ./lob_test.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/ring_test.cpp -o ring_test.exe && ./ring_test.exe

# Live shared-memory demo (subsystem 5) — run the producer in one terminal, the probe in another:
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_producer.cpp -o ring_producer
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_probe.cpp -o ring_probe
./ring_producer nex_aapl 4000 16384 0xC0FFEE 1     # terminal 1: book -> ring
./ring_probe nex_aapl 4000 5                          # terminal 2: watch it live

# Full build + Python tests (WSL / Ubuntu + real Python required):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEXUS_BUILD_PYBIND=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
pytest bindings/tests/test_abi_parity.py -v      # contract parity + engine seam
pytest python_quant/tests/test_contract_smoke.py -v
```

---

## 7. Environment notes

This session runs on **Windows 11 + Git Bash / MSYS2**, repo on a **OneDrive** path.
Current toolchain (updated 2026-09-04/09-14): `g++` (C++20 ✅), **real Python 3.12**
(Tier 1 pure-Python tests **PASS**), `cmake`, and **MSVC Build Tools** (Tier 2
compile + parity **PASSED 2026-09-04**). CUDA still needs Linux or a Windows CUDA toolkit.

- ✅ C++-only compile/run checks work here (engine tests above).
- ✅ Pure-Python tests work here — **157 passed / 1 skipped** as of 2026-09-14.
- ✅ Tier 2 (compile `nexus_engine`) **passed on Windows/MSVC** (parity + diff-test).
- ⚠️ Keep `build/`, `data/`, venvs **out of the OneDrive-synced tree** — sync + build
  artifacts is a known breakage source (copy the repo off OneDrive if the build is slow/flaky).

See `CLAUDE.md` §8 for the full table and the exact Windows build steps.

---

## 8. Suggested next steps

1. ✅ ~~Tier 2 on Windows~~ — **PASSED 2026-09-04** (MSVC build + parity + diff-test).
2. ✅ ~~**Person B: ITCH parser + `OrderBookEnv` + baselines**~~ — done (PR #2, merged).
3. ✅ ~~**Diff-test harness**~~ — done + passing.
4. ✅ ~~**PPO/GRPO agent vs baselines**~~ — done 2026-09-05 (reward beats all baselines;
   shortfall ≈ VWAP; high-vol regime identified as the path to the ~14% headline).
5. ✅ ~~**High-volatility regime → ~14% below VWAP**~~ — **ACHIEVED 2026-09-07** (+50.4%; see §4
   and `HIGHVOL_PLAN.md`).
6. ✅ ~~**Dashboard + GRPO + risk↔env seam**~~ — done 2026-09-09 (combined desk on `main`;
   GRPO trainer; `lambda_risk` CVaR penalty).
7. ✅ ~~**Phase 3: E5 queue dynamics + E6 adverse selection + fair RL rework**~~ — **COMPLETE
   2026-09-14** (157 passed / 1 skipped; §3h).
8. 🔜 **Open the PR for `feature/part2-phase3-rl-fairness`** and merge with a real merge commit
   (per the git workflow).
9. 🔜 **Phase 4: real NASDAQ ITCH tape** — `fetch_itch.py` + order-level parser events; the
   Phase-4 tape unblocks the degenerate queue-position signal found in Phase 3.
10. ⏳ **Verify the GPU risk engine on a CUDA machine** — compile `nexus_risk` + `risk_bench`
    (needs `nvcc`/toolkit: WSL/Linux or Windows CUDA), capture the ~40× speedup and the
    bit-for-bit CPU-vs-GPU parity.
