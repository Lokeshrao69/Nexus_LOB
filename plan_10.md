# Nexus-LOB — Plan to a 10 (quant-placement win)

> **Author:** Person A, with Person B items flagged. **Status:** draft for approval.
> **Read this after `plan_2.md` and `CLAUDE.md`.** This file is the *final-mile*
> plan: what stands between the current state and a placement-competitive 10,
> how long each piece takes, and what could null out.

## 0. The honest framing of "win at any cost"

A quant interviewer's first move on a resume number is to pull the thread until
it snaps or holds. A fabricated or over-claimed headline survives exactly one
interview. The thing that makes this project different from 95% of portfolio
repos — the fact that we **retired a +50.4% headline** rather than defended it —
is the moat. So "at any cost" means: maximum effort, sustained timeline, no
feature for show — *never* a number published before it is measured and has a CI.
Everything below is built to that rule.

**One sentence of diagnosis.** The gap between 7.5 and 10 is that every remaining
claim is **unmeasured** (perf, CUDA, multi-day robustness) and the project has
**no tradeable result on real data** — prediction exists, execution does not.

---

## 1. The five gaps → the five target headlines

| # | Gap | Target headline (measured, CI'd) |
|---|---|---|
| G1 | Perf asserted, not measured | "C++20 engine: **X** k orders/s, **Y** ns p50 / **Z** ns p99 latency, 0 allocs/op (Linux, Raptor-Lake-class, rdtsc, ≥30 runs)" |
| G2 | CUDA written, never measured | "Monte-Carlo VaR: **40×? ×?** GPU vs CPU on RTX 3050, bit-for-bit parity" |
| G3 | Research = one session | "E1–E6 cross-day robustness on **N** sessions (2019-12-30 + 6 more), CIs across days" |
| G4 | Nothing tradeable | "Imbalance-aware execution reduces IS vs TWAP by **X** bps (CI) on real tape under fees+queue" |
| G5 | No one-button reproduction / finish | "`run_all.py --full` regenerates every committed number; CI green; README all-measured" |

G1 + G2 are **Person A's**, are the fastest path to credibility, and are blocked
only on environment ($2). G3 + G4 are the long-tail quantity-research items.
G5 is the polish that makes the rest *believable* to a reviewer.

---

## 2. Environment reality (verified 2026-09-16, this machine)

- GPU **present**: NVIDIA GeForce RTX 3050 Laptop GPU → CUDA-capable.
- **nvcc absent**, **WSL not installed** → both installable (below).
- `cpp_engine/bench/bench.cpp` is **complete and measurement-ready** (rdtsc
  p50/90/99/99.9/max, throughput both configs, global alloc counters). It just
  needs real hardware — the Windows sandbox throttles memory workloads.
- `cuda_risk/risk_cuda.cu` + `risk_bench.cpp` are **authored, never compiled**.
  The splitmix64 counter-based RNG guarantees bit-for-bit CPU=GPU parity by
  design — it needs proving, not writing.
- **Real tape is NOT on this box** (`data/` gitignored & empty here; research
  ran on the partner's Linux/Python env). `fetch_itch.py`/`batch_research_itch.py`
  resume cleanly. Of the 15 catalogued dates: **8 confirmed 404 (retired on
  emi.nasdaq.com)**, **6 downloadable but bandwidth-bound (~4.5 h/day @238 KB/s)**,
  **1 fully analyzed** (12302019, AAPL+QQQ).
- CI is already green-ready: `.github/workflows/ci.yml` = 3 jobs (Python, C++
  build + CTest + parity, lint) on Linux **and** Windows. No benchmark jobs.
- RL fairness, honestly established: PPO **sig-better** than `adaptive_pov` under
  `highvol_null` (5/5, novol), `trending` (5/5, volsym), `liquidity_shock` (5/5,
  both modes); **worse** on `calm`/`lowvol` hold-outs. A real, defensible
  regime story — not "no edge," a *specific* edge.

---

## 3. The plan — six phases

Dependency rule: **A → B**, then **C/D** in parallel, then **E**, **F** optional.
Each phase is independently resume-bankable.

### Phase A — Environment unlock (Person A · ~0.5–2 days · START HERE)

> **LOCKED 2026-09-16:** WSL2 + Ubuntu 24.04 + CUDA 12.x. The RTX 3050 talks to
> WSL through the **Windows-side NVIDIA driver** (already installed — `nvidia-smi`
> works on Windows). CUDA inside Ubuntu installs **toolkit only, never a driver.**

Nothing measured happens until this box can compile and run the real toolchain.

**Windows side**
1. `wsl --install -d Ubuntu` (admin PowerShell) → reboot when prompted.
2. `wsl -l -v` — confirm Ubuntu at **version 2** (if 1: `wsl --set-version Ubuntu 2`).

**Ubuntu side (runbook, in order)**
4. `sudo apt update && sudo apt upgrade -y`
5. Toolchain: `sudo apt install -y build-essential cmake python3 python3-pip python3-venv`
   — 24.04 ships g++ 13.x (C++20 ✓), cmake 3.28+ ✓, Python 3.12 ✓ (matches the
   project's verified Linux env).
6. CUDA toolkit (no driver — the Windows driver serves WSL):
   ```
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt -y install cuda-toolkit
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   ```
   (If the keyring URL 404s, fetch the current one from the same
   `wsl-ubuntu/x86_64/` directory.)
7. Verify: `nvcc --version` and `nvidia-smi` (must show the RTX 3050).

**Repo location — the OneDrive rule (do not skip)**
8. Build **on the Ubuntu native filesystem**, never on `/mnt/c` (OneDrive + 9p =
   slow and high-noise timing — the bench numbers would be garbage):
   ```
   mkdir -p ~/nexus && cp -r "/mnt/c/Users/pekka/OneDrive/Documents/Finance Project-1" ~/nexus/nexus-lob
   ```
   Work and measure from `~/nexus/nexus-lob`; push code/doc changes back to
   GitHub from there (git remote already points at `Lokeshrao69/Nexus_LOB`).

**Gate — the whole C++ suite green before any measuring**
9. Standalone (as in CLAUDE.md §6): compile+run `abi_check`, `lob_test`,
   `ring_test`, `id_map_test` in `~/nexus/nexus-lob` → 448-B ABI lock, 86 · 30011 ·
   4676294 checks, ALL PASS.
10. Full build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` then
    `cmake --build build -j` then `ctest --test-dir build --output-on-failure` → **5/5**
    (incl. `risk_test`).
11. **Write `docs/setup_wsl_cuda.md`** — exact commands, versions, git SHA.
    This doc is itself interview material ("anybody can rebuild my env").

Once steps 9–10 are green, **Phase B starts immediately** (`bench.cpp` + `risk_bench`).

### Phase B — Measure the three performance claims (Person A · 2–4 days)

The highest-leverage credibility jump in the whole project.

1. **B1 · Engine perf on real hardware.** `bench.cpp` in WSL, pinned (`taskset`),
   `perf stat` for L2/LLC misses, ≥30 runs, report CI/IQR. One wrinkle to add:
   a `--repetitions N` flag or a tiny wrapper loop so the report is reproducible
   (the harness is close; keep any change additive). Target: >500 k ord/s, sub-µs
   p50, single-digit-µs p99, **0 allocs/op re-confirmed on Linux**.
2. **B2 · CUDA speedup.** Compile `risk_bench.cpp` against `risk_cuda.cu`, run
   CPU baseline + GPU kernel at the same (seed, path, step) and *prove* parity,
   then report speedup. Report the kernel time *and* end-to-end (incl. H2D/D2H),
   at 100 k–1 M paths. RTX 3050 laptop is memory-bandwidth-bound: a realistic
   target is **15–40×**, and even the low end is a wall of text in an interview
   if the parity is bit-for-bit.
3. **Write `docs/results/perf_linux.md` + `docs/results/risk_cuda_bench.md`** —
   method (CPU, RAM, compiler flags, kernel, warmup, runs, git SHA, CI), results,
   caveats. Everything in the README currently marked "pending" is now measured.
4. **Flip the README rows.** The table cells change from "pending" to measured
   numbers — the single edit a reviewer notices first.

### Phase C — Research robustness: from 1 session to N (Person B + Person A seam · runs in background)

1. **C1 · Within-day walk-forward on the existing day (cheap, immediate).**
   The 12/30/2019 tape supports hour-blocked walk-forward (train on block k,
   test on k+1). Zero new data, immediate robustness evidence, and it's the
   *right* discipline anyway. Deliverable: within-day folds added to
   `docs/results/real_tape_12302019.md`.
2. **C2 · Pull the 6 live-but-bandwidth-bound dates.** `batch_research_itch.py`
   already resumes. Just let it run across nights (~4.5 h/day). Deliverable:
   `docs/results/multi_day_aggregation.md` becomes **N sessions**, not 1.
3. **C3 · Recover 2019 dates that 404'd, from an alternate source.** Verify
   whether the retired days are mirrored anywhere public (e.g., hosted ITCH
   copies on Kaggle/GitHub). **License + provenance must be documented** in the
   fetch header before use — the current 12302019 provenance line is the model.
   If not found: sub in **live 2020–2022 sample dates** from emi.nasdaq.com
   that still exist, so the corpus spans more than one calendar regime.
4. **C4 · Cross-day rerun.** `run_research.py` E1–E6 across all available days,
   regenerate the README scope line: "single session" → "N sessions across
   M calendar years, CIs across days."

### Phase D — The tradeable layer (Person B core, Person A seam · 1–2 weeks)

**This is the "10."** Prediction (E1–E6) becomes an execution outcome (E7) *on
real tape*, through the real stack.

1. **D1 · Real-tape execution study (E7-on-real).** Drive execution rules against
   the real L2 book on the tape's own event stream, with the **empirical queue
   hazard** (`queue_model="empirical_hazard"`  — already built), calibrated fill
   model, fees + rebates + impact on. Rules: TWAP / schedule-TWAP /
   adaptive-POV / imbalance-/microprice-conditioned. Report **IS vs market-VWAP**,
   fill rate, completion, MDD, block-bootstrap CIs — via the existing
   `execution/backtest.py` + `evaluate.py`. This is E7 as the plan always
   intended it, finally on real data instead of synthetic regimes.
2. **D2 · Signal-conditioned execution — the honest experiment.** Question:
   *does conditioning the child-order schedule on L1 imbalance (the E1 signal)
   reduce realized slippage vs a volume-scheduled baseline on the SAME real
   tape, under fees+queue?* If yes: the project's first genuine, defensible
   execution finding (even +1–3 bps with tight CIs is a win; national-desk
   quality). If null: a documented negative result — equally valuable, and the
   report writes it exactly that way.
3. **D3 · Full-day engine diff-oracle (Person A).** The real 12/30/2019 day
   replayed through the **C++ engine** and through `StubOrderBook`; ladder parity
   confirmed at **full scale** (the 7,037-frame prefix was verified; do the whole
   day). Every downstream real-tape number then inherits an "engine-exact"
   guarantee — a genuinely rare thing to be able to claim.
4. **D4 · One-command regeneration.** Fold the real-tape E7 into `run_all.py`
   so the whole results tree reproduces from raw tape.

### Phase E — Institutional finish (shared · 2–4 days)

1. **CI completeness.** Add an optional, labeled **benchmark job** to CI
   (a short smoke — asserts the bench *runs* and reports 0 allocs, not the
   full perf number) so the zero-alloc claim is continuously re-proven on
   public runners. Keep the heavy numbers in `docs/results/`.
2. **Reproduction audit.** `run_all.py --full` must regenerate every committed
   number (`docs/results/*`); fix any drift; document "how to check my claims"
   at the top of the README.
3. **The write-up package (the part that actually lands interviews):**
   - `docs/DESK_MEMO.md` — one page per headline: claim, method, CI, caveat,
     next step, in desk-note register. This is the document an interviewer
     reads before meeting you.
   - Resume bullets that each cite a measured doc (not a boast). One line per
     subsystem; every number links to the doc that proves it.
   - Optional: a 3–5 min recorded walkthrough (desk + `run_all --full` + the
     perf report) to attach to the application, or a link in the README.
4. **README final state** (you're already on `docs/final-readme`): lean,
   numbers-first, every headline measured + scoped, one-button reproduce,
   scope caveats explicit. After B–D it lists *measured* perf, *measured* CUDA,
   *multi-session* research, and a *tradable* execution result.

### Phase F — Optional stretch (only after B–E land)

- **F1 · Latency-personalized execution.** Calibrate the env's latency/queue
  simulation to the real tape's empirical hazard (the seam for this exists),
  so the D2 numbers get closer to broker reality.
- **F2 · One engine micro-upgrade (Person A, for the "tell me about" moment).**
  A single substantive C++ addition an interviewer can dig into — batched/
  vectorized matching, cache-line-aligned order pool, or a second ABI version
  bump with a migration path. New features are *lower* leverage than measuring
  what exists (B), so this is explicitly optional and last.
- **F3 · If D2 survives:** a small PnL / Sharpe attribution study across the
  remaining tape days. Turns "statistical finding" into "strategy with performance."

---

## 4. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| RTX 3050 gives <15× CUDA (memory bandwidth) | G2 headline smaller | Report kernel vs end-to-end honestly; try a bigger path count; the *bit-for-bit parity* is the differentiator, not the multiple |
| D2 signal-conditioned execution shows no edge | G4 headline null | It's a first-class negative result with CIs — rigor is the product; reframe as "imbalance signals do not survive into naive execution after costs" (still a finding) |
| 2019 dates irrecoverable; 6 dates bandwidth-bound | G3 stays small | C1 within-day folds immediately; sub in live 2020–2022 sample dates for regime breadth |
| WSL2 CUDA install fights us | A blocks B | Fall back: Windows CUDA toolkit (RTX 3050 is Windows-CUDA-supported); bench on WSL without CUDA, risk on Windows |
| Partner bandwidth/schedule (Person B items) | C/D slow | Person A can absorb C1/C3 and the D3 seam; D1/D2 need the Python research layer but are buildable by Person A too (Python works on this box) |

**Guardrail (repeated deliberately):** every headline above ships with its CI and
its scope label. If a number comes back null or small, it is *written up as null*.
A 10 is achieved with one measured edge plus three honest nulls; it is lost with
one invented number.

---

## 5. Ordering into a calendar

| Window | Focus | Person A owns | Person B / shared |
|---|---|---|---|
| Days 1–2 | A (env) | WSL2 + CUDA + C++ gate green | — |
| Week 1–2 | B (measure) | Engine perf + CUDA speedup + both reports | C1 within-day folds (can be absorbed) |
| Week 2–4 | C + D (data + tradeable) | D3 full-day engine parity; C3 source hunt | C2 overnight downloads; D1/D2 real-tape E7 |
| Week 4–5 | E + F (finish) | README flip; CI bench job; F2 optional | DESK_MEMO, resume bullets, reproduction audit |

Each phase closes as a standalone resume line, so slippage in one never strands
the others. The "however long it takes" long tail is concentrated in C2 (waiting
on bandwidth, resumable) and D2 (research risk) — the rest is bounded weeks.

## 6. Definition of Done — this project reads as a 10

- [ ] `docs/results/perf_linux.md`: measured orders/s, p50/p90/p99/p99.9, LLC
      misses, 0 allocs/op — real hardware, full method, CI. README cell flipped.
- [ ] `docs/results/risk_cuda_bench.md`: CPU-vs-GPU speedup **and bit-for-bit
      parity** on real hardware. README cell flipped.
- [ ] Research corpus = **N sessions** (≥3 comfort) with cross-day CIs.
- [ ] A real-tape E7 with fees+queue+empirical-hazard and block-bootstrap CIs;
      D2 either a measured edge or a written-up null.
- [ ] `run_all.py --full` regenerates `docs/results/*`; CI green incl. a
      benchmark smoke; `docs/DESK_MEMO.md` + resume bullets that each cite a doc;
      README entirely measured-and-scoped, scope caveats explicit.

With all six boxes closed the project reads: **systems + microstructure +
execution, every number measured and reproducible, one disciplined negative-result
culture.** That is the 10.