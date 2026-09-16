# Nexus-LOB — Final Plan (to a placement-ready 10)

> **Status:** APPROVED 2026-09-16. Environment locked: **WSL2 + Ubuntu 24.04 +
> CUDA 12.x**. This file is the single plan of record for the final mile.
> Precursors: `plan_2.md` (research-half history), `plan_10.md` (first draft of
> this plan). Owner: Person A (systems), with Person B touch-points flagged.

---

## Context — why this plan exists

Nexus-LOB reads as a strong **7.5/10** for quant placement. The gap to a 10 is
twofold:

1. **Every headline performance claim is unmeasured.** ">500k orders/s,
   sub-µs latency, ~40× CUDA" are all still labeled *pending* in the README,
   because `cpp_engine/bench/bench.cpp` and `cuda_risk/risk_cuda.cu` are complete
   but have never run on real hardware. These are the first three numbers an
   interviewer reads on a resume — asserted, not proven.
2. **Nothing is tradeable.** E1–E6 establish that L1 imbalance *predicts* mid
   moves (rank IC 0.14–0.46, CIs ±≤0.014) and passive fills are adversely
   selected 96–99% — but the E7 execution study runs only on synthetic regimes,
   never on the real NASDAQ tape that was already downloaded.

The plan makes every claim **measured-with-a-CI** and produces the project's first
**real-tape execution outcome**.

### The honest framing of "win at any cost"

A quant interviewer's first move is to pull a resume number until it snaps or
holds. A fabricated headline survives exactly one interview. The thing that makes
this project distinctive — retiring the +50.4% headline rather than defending
it — is the moat, and this plan is built on it. "At any cost" means maximum
effort and a sustained timeline; it never means publishing a number before it is
measured. Guardrail throughout: **every headline ships with its CI and scope
label; a null result is written up as null.**

---

## Environment reality (verified on this machine 2026-09-16)

| Fact | State |
|---|---|
| GPU | **NVIDIA GeForce RTX 3050 Laptop** (Ampere, CC 8.6) present, driver in Windows |
| WSL | not installed → **installing Ubuntu now** (the driver serves WSL via passthrough) |
| `nvcc` | absent → install CUDA **toolkit-only** (never a driver) inside Ubuntu |
| `bench.cpp` | **complete & measurement-ready** (rdtsc p50/90/99/99.9/max, throughput, 0-alloc counters); no CMake target → build standalone |
| `risk_cuda.cu` / `risk_bench.cpp` | **authored, never compiled**; splitmix64 counter RNG ⇒ CPU=GPU path-identical by design |
| Real tape | NOT on this box (`data/` gitignored, empty here); research ran on the partner's Linux env. 8 catalogued days confirmed 404 (retired), 6 live-but-bandwidth-bound (~4.5 h/day @238 KB/s), 1 fully analyzed (12302019 AAPL+QQQ) |
| CI | `.github/workflows/ci.yml` green-ready (3 jobs, Linux+Windows); no benchmark jobs |
| RL fairness (established) | PPO **sig-better** than `adaptive_pov` under `highvol_null` (5/5), `trending` (5/5 volsym), `liquidity_shock` (5/5 both); worse on `calm`/`lowvol` — a specific, defensible edge story |

---

## Phase A — Environment unlock (running now)

Runbook, exact order. Once green, **everything below measures here.**

**Windows**
1. `wsl --install -d Ubuntu` (admin PowerShell) → reboot → `wsl -l -v` shows Ubuntu at **version 2**.

**Ubuntu**
2. `sudo apt update && sudo apt upgrade -y`
3. `sudo apt install -y build-essential cmake python3 python3-pip python3-venv`
   (24.04 ships g++ 13.x / C++20 ✓, cmake 3.28+ ✓, Python 3.12 ✓).
4. CUDA toolkit (no driver):
   ```
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt -y install cuda-toolkit
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   ```
   (If the keyring URL 404s, fetch the current file from the same `wsl-ubuntu/x86_64/` dir.)
5. Verify: `nvcc --version` green; `nvidia-smi` shows the RTX 3050.
6. **Copy repo to native fs (never measure/9p on `/mnt/c`):**
   ```
   mkdir -p ~/dev && cp -r "/mnt/c/Users/pekka/OneDrive/Documents/Finance Project-1" ~/dev/nexus-lob
   cd ~/dev/nexus-lob && rm -rf build
   ```
   Confirm `df -T ~/dev/nexus-lob` → `ext4` (not `9p`). Windows copy stays the docs/commit surface.
7. **Gate — whole C++ suite green before measuring.** Standalone `abi_check`,
   `lob_test`, `ring_test`, `id_map_test` (CLAUDE.md §6 commands → 448-B lock,
   86 · 30011 · 4,676,294 checks, ALL PASS), then
   `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j &&
   ctest --test-dir build --output-on-failure` → **5/5** (incl. `risk_test`).
8. **Write `docs/setup_wsl_cuda.md`** — exact commands, versions, git SHA
   ("anybody can rebuild my env" — itself interview material).

---

## Phase B — Measure the two performance claims (Person A)

Command-level detail designed and verified against the code. Files touched:
`bench.cpp`, `risk_bench.cpp`, two new driver scripts, two result docs, README cells.

### B1 — engine perf (throughput + latency + 0-allocs)

- Build standalone: `g++ -std=c++20 -O3 -march=native -I cpp_engine/include
  cpp_engine/bench/bench.cpp -o bench`. **Smoke-run first:** the zero-alloc line
  must be **0**. The op stream is fixed-seed deterministic (`bench.cpp:146,289`),
  so Linux must reproduce the Windows count exactly; a nonzero count is a REAL
  finding (candidate: the bench's own `placed` vector reallocating past its
  8,000-slot reserve). Fix-or-explain before any README flip.
- **One minimal additive change to `bench.cpp`:** a `--repetitions N` flag; wrap
  the both-configs tail in an N-loop printing `===== repetition i/N =====` only
  when N>1 (default output byte-identical). Each run already does fresh book +
  `cycles_per_ns()` calibration + alloc reset + both passes ⇒ independent sample.
- **New `scripts/bench_engine_driver.py`** (stdlib-only): `taskset -c <pinned>`
  pin; matrix = both configs (`passive`, `crossing`) × `--ops 2000000 200000` ×
  31 reps (rep 1 = warmup, discard); aggregate **30 runs**; report per config
  median / IQR / 95% bootstrap-CI of throughput and of each latency stat
  (mean/p50/p90/p99/p99.9/max); allocs = "all zeros across all runs →
  `ZERO-ALLOC PROVEN (30 × 2.2M ops)`".
- **WSL2 has no hardware PMU** — do not chase `perf` cache counters. Self-timing
  is robust **iff** `/proc/cpuinfo` has `constant_tsc`/`nonstop_tsc` (verify;
  if absent report cycles/op only). Set Windows power plan to High Performance
  (frequency is host-controlled; guest `scaling_governor` is cosmetic). Label
  WSL2 numbers as a virtualized floor.
- **`docs/results/perf_linux.md` + `.json`:** method table (CPU model, nproc, TSC
  flags, power plan, `uname -r`, ext4 path, git SHA, compiler+flags, warmup,
  N=30, pinning), results per config, WSL2 caveats.

### B2 — CUDA speedup + CPU/GPU/NumPy parity

- Build: `cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
  -DNEXUS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DNEXUS_BUILD_PYBIND=OFF
  -DNEXUS_BUILD_TESTS=OFF -DNEXUS_BUILD_DEMOS=OFF`, then
  `cmake --build build-cuda --target risk_bench -j`. `86` = explicit `sm_86`
  (kills `native` misdetect / PTX JIT).
- **Two additive edits to `risk_bench.cpp`:**
  1. **End-to-end GPU time** (`now_ms()` around `compute_var_cvar_gpu`); print
     `GPU end-to-end : … ms` and BOTH speedups (kernel and end-to-end). The
     legacy line compares CPU(sim+sort) vs GPU(kernel-only) — apples-to-oranges;
     at 2M paths the host sort (~100–300 ms) exceeds the kernel, so the e2e
     number is the honest real-time-VaR headline.
  2. **Parity exit gate:** `return parity_ok ? 0 : 1` on |var|,|cvar|,|mean_loss|
     diffs (identical sorted-loss arrays ⇒ `==0.0` practical gate).
- **Parity reality-check (record honestly):** libdevice (GPU) vs glibc (CPU)
  `cos/log/exp` can differ at the last ULP even with `--use_fast_math` OFF —
  expect `dvar=0` or ~1e-15; record whichever; NumPy↔CPU leg already closed
  (abs=1e-12, `test_risk_parity.py:106-108`).
- **New `scripts/risk_bench_driver.py`** (stdlib-only): warmup discarded; sweep
  `n_paths ∈ {50k, 200k, 800k, 2M} × steps ∈ {252, 504}`; **add**
  `--lambda-jump/--jump-mu/--jump-sigma` pass-throughs (~9 additive lines) so the
  jump-diffusion branch is parity-covered; vary seed once; 3 fresh-process runs
  each; report min kernel-ms, median e2e-ms, parity deltas + exit codes.
- **`docs/results/risk_cuda_bench.md` + `.json`:** GPU (RTX 3050 Laptop, driver,
  CUDA version, VRAM), CPU, nvcc/g++, cmake invocation incl. arch=86, git SHA;
  kernel vs e2e speedup tables; parity table + gate codes; caveats (host-sort
  dominance at 2M, WSL2 floor, launch overhead at small n).

### README cells to flip (after B1+B2)

`README.md:18,21,78-81,126-128,290-291,300-301` convert from "pending" to
measured, each linking to its result doc; the virtualized-floor and parity-delta
caveats live in the result docs, not hidden. Same two flips in `CLAUDE.md` and
`PROGRESS.md` for consistency.

---

## Phase C — Research robustness: one session → N (Person B + A seam, background)

- **C1 · Within-day hour-blocked walk-forward** on the existing 12/30/2019 tape —
  cheap, immediate, zero new data; the right discipline anyway.
- **C2 · Download the 6 live-but-bandwidth-bound dates** with resumable
  `batch_research_itch.py` across nights (~4.5 h/day) →
  `docs/results/multi_day_aggregation.md` becomes **N sessions**, not 1.
- **C3 · Recover/fill the 8 retired 2019 dates** from an alternate public source
  (licence documented) or substitute live 2020–22 sample dates for regime breadth.
- **C4 · Cross-day E1–E6 rerun**; README scope line becomes "N sessions, CIs
  across calendar years".

## Phase D — The tradeable layer (the "10")

- **D1 · E7-on-real-tape:** TWAP / schedule-TWAP / adaptive-POV /
  imbalance-conditioned rules on the real book, **empirical queue hazard** +
  fees/rebates/impact on, IS vs **market-VWAP**, fill %, completion, MDD,
  block-bootstrap CIs. Reuses `execution/backtest.py`, `evaluate.py`,
  `queue_model="empirical_hazard"` — all already built.
- **D2 · Signal-conditioned execution (the honest experiment):** does conditioning
  on L1 imbalance (E1) reduce realized slippage vs a volume-scheduled baseline on
  the same tape under costs? Measured edge (even +1–3 bps) → first real execution
  finding; null → documented negative result.
- **D3 · Full-day engine diff-oracle (Person A):** replay the whole 12/30/2019 day
  through the C++ engine vs `StubOrderBook`, ladder parity at full scale (7,037-
  frame prefix already verified) → every real-tape number inherits "engine-exact".
- **D4 ·** fold real-tape E7 into `run_all.py` so the results tree reproduces from
  raw tape.

## Phase E — Institutional finish

- Optional **CI benchmark smoke** job (bench runs + asserts 0 allocs on public
  runners; heavy numbers stay in `docs/results/`).
- Reproduction audit: `run_all.py --full` regenerates every committed number.
- **Write-up package:** `docs/DESK_MEMO.md` (one page per headline: claim, method,
  CI, caveat, next step); resume bullets each citing a measured doc; optional
  3–5 min recorded walkthrough.
- README final state: numbers-first, all-measured-and-scoped, one-button reproduce.

## Phase F — Stretch (only after B–E)

F1 latency-personalized execution (calibrate env latency/queue to real-tape
empirical hazard); F2 one engine micro-upgrade for the "tell me about" moment
(batched/vectorized matching, cache-aligned pool, or a documented ABI-v2
migration path — measuring what exists is higher leverage than new features, so
this is optional and last); F3 PnL/Sharpe attribution if D2 survives.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| RTX 3050 gives <15× CUDA (laptop memory bandwidth) | B2 headline smaller | Report kernel vs e2e honestly; bigger path counts; **bit-for-bit parity is the differentiator, not the multiple** |
| Parity lands at ~1e-15, not 0 (libdevice vs glibc) | B2 parity claim | Record the actual ULP delta; gate CI on exact-0 *or* documented 1e-12; keep the README claim honest |
| D2 shows no edge after costs | D headline null | First-class negative result with CIs — rigor is the product (the project's brand) |
| 2019 dates irrecoverable; 6 dates bandwidth-bound | C stays small | C1 within-day folds immediately; substitute live 2020–22 dates |
| WSL2 vCPU jitter / host co-residency | B1 tails inflated | taskset pin, high-perf power plan, 30-run median+IQR+CI, label as virtualized floor |
| Windows GPU driver vs CUDA 12.x skew | B2 kernel won't launch | `nvidia-smi` before sweeping; toolkit 12.x vs current Windows driver |
| Zero-alloc breaks on Linux | B1 headline dies | Smoke-run first; fix-or-explain before any README flip |
| Partner schedule (Person B items) | C/D slow | Person A absorbs C1/C3 and D3; D1/D2 buildable by Person A (Python works) |

---

## Verification gateway / Definition of Done

- **After A:** `ctest --test-dir build --output-on-failure` → **5/5**.
- **After B:** `docs/results/perf_linux.md` + `risk_cuda_bench.md` exist with
  method + numbers + git SHA; zero-alloc and parity gates asserted; README has
  no "pending".
- **After D:** real-tape E7 table with block-bootstrap CIs; D2 is either a
  measured edge or a written-up null; `run_all.py --full` regenerates
  `docs/results/*`.
- **Final:** README entirely measured-and-scoped; `docs/DESK_MEMO.md` + resume
  bullets that each cite a measured doc; CI green incl. benchmark smoke.

With all boxes closed the project reads: **systems + microstructure + execution,
every number measured and reproducible, one disciplined negative-result culture.
That is the 10.**

---

## Calendar

| Window | Focus | Person A owns | Person B / shared |
|---|---|---|---|
| Now (you logging into Ubuntu) | A | finish runbook; gate 5/5 green; `setup_wsl_cuda.md` | `fetch dataset licence note` |
| Week 1–2 | B | bench.cpp `--repetitions`; B1 driver + report; CUDA build + B2 driver + report; README flip | C1 within-day folds (absorable by A) |
| Week 2–4 | C + D | D3 full-day engine parity; C3 source hunt | C2 overnight downloads; D1/D2 real-tape E7 |
| Week 4–5 | E + F | CI smoke job; README/progress flips; F2 optional | DESK_MEMO, resume bullets, reproduction audit |

Each phase closes as a standalone resume line — slippage in one never strands
the others. The long-tail waits are C2 (bandwidth, resumable) and D2 (research
risk); the rest is bounded weeks.