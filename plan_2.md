# Nexus-LOB — Part 2: Quant Research Layer (Plan of Record)

**Last updated:** 2026-09-12 · **Status:** Phase 0 (repo hygiene) + Phase 1 (research spine) landed on `main`; next is Phase 2 (execution realism).
**Read this FIRST each session, then `CLAUDE.md`.** This is the single source of truth for the research-half roadmap. Keep updating it as work progresses (§ Running log at the bottom).

---

## 0. Why this file exists

The project is now cleanly split in two:

- **Part 1 (DONE, verified):** C++ matching engine, frozen 448-byte ABI contract, pybind seam, shmem ring, ITCH parser + replay, `OrderBookEnv`, PPO/GRPO, risk engine (CPU + exact parity). All 12 original plan items complete.
- **Part 2 (NOW, this file):** the **quant research layer** — turning Nexus-LOB from "systems project with a toy RL agent" into *"systems + real market-microstructure research."* This matches what Jane Street / Citadel / Optiver / DRW interviewers actually evaluate.

The full 16-part audit (repository audit → design → experiment → rigor → execution → queue → RL → C++ perf → CUDA → architecture → file plan → prioritization → 6-week roadmap → resume → definition of done) was delivered in the 2026-09-12 session. **This file is the distilled plan of record derived from it.**

### Ground rules (a violation of any = progressive failure)
1. **No feature is used before its leakage is proven absent.** Features are pure functions of state at `t`; labels are strictly `t → t+h`; splits are walk-forward by wall-clock time. A test enforces it.
2. **No headline tuned to a target.** The Part-1 "+50.4% vs VWAP" was produced by tuning the sim until it hit the target. We do not extend that pattern — we fix it (§6).
3. **Every number gets a CI.** IC, ICIR, IS, slippage: block-bootstrap / multiple seeds, t-stats, Newey–West where horizons overlap.
4. **Preserve the architecture.** Frozen `BookStateView` contract, `StubOrderBook`↔engine parity oracle, deterministic PPO, `cuda_risk` bit-for-bit parity design — all stay as-is.
5. **Real data is the end goal.** Synthetic flow establishes the harness; research *claims* are made on a real NASDAQ ITCH tape.
6. **Negative results are reported.** A project with zero negative results is research theater.

---

## 1. Audit snapshot — where the repo genuinely stands (2026-09-12)

| Side | Verdict |
|---|---|
| C++ LOB / matching / ABI / pybind / shmem | Genuinely strong, well-tested, production-shaped. **8/10 systems.** |
| ITCH parser / replay | Correct but **never exercised on real bytes** (synthetic fixtures only). |
| Python research layer | **Essentially absent** — no features, no signals, no IC, no backtest, synthetic-only price process. |
| RL/PPO | Engineering solid (deterministic, seedable). **Research validity weak** — see §6. |
| VaR/CVaR | Correct, exactly-parity. Parametric-MC only; CUDA never compiled. |
| Dashboard / branches / CI / docs | **Broken state** — see §1a. |
| **Overall quant signal** | **≈3.5/10.** Systems ≥ interview-ready; research half ≈ six weeks away. |

### 1a. Repo-state problems that must be fixed at the start (Phase 0)
- Working tree is on **stale `feature/risk-engine`** (missing `grpo.py`, `risk.py`, `dashboard.py`, `serve_dashboard.py`); GitHub default **`main`** has them. `dashboard_page.html` is stranded on **`feature/dashboard-file-ring`**; `dashboard/index.html` only exists on **`feature/risk-engine`**.
- Local origin still points at **`Finance-Project-1`** (repo was renamed to **`Nexus_LOB`** — GitHub redirects the URL, but update it: `git remote set-url origin https://github.com/Lokeshrao69/Nexus_LOB.git`).
- **No CI** (no `.github/` anywhere).
- `README.md` contains **two concatenated full READMEs** and describes the wrong branch state.
- `%TEMP%\pr7-fix` worktree is leftover; `feature/person-b-polish` is fully merged into `main`.

> **Decision pending:** reconcile so `main` == docs == checkout, then build the research layer on `main`. (Recommend: `git checkout main`, drop the stale `feature/risk-engine` / `feature/dashboard-file-ring` unmerged dashboard commits into one reconciled dashboard or drop them; keep `policy_ppo_highvol.npz` artifact.)

### 1b. The one credibility issue fixed before anything else
The **"+50.4% lower slippage than VWAP"** headline. Ground truth from the code:
- PPO was trained with an explicit `vol_feature` **regime indicator** (`obs[44]`) that TWAP/VWAP/POV/Passive **could not observe** → asymmetric information in the comparison.
- Eval ran on the **training distribution** (same env config + seed family), no holdout regimes.
- **No CI / no t-stat** on the vs-VWAP% (two seed checks: +50.4%, +38.2%).
- The high-vol regime was **tuned until the metric hit its target** (`HIGHVOL_PLAN.md §9` lists "tuning levers if it doesn't hit 14%").

We do **not** delete it, and we **do not** claim it. We **re-characterize and fairly re-verify** it (§6). Until then, no resume/README uses the number.

---

## 2. Target architecture (research half)

```
REAL MARKET DATA ─► ITCH/L2 PARSER (reuse itch_parser.py, validate on real bytes)
     ▼
EVENT REPLAY (reuse replay.py) ─► L2 / ORDER-LEVEL BOOK     (stub today; C++ engine = diff oracle)
     ▼
FEATURE ENGINE (NEW research/features.py)          [imbalance, microprice, OFI, deep, spread, vol, flow, momentum]
     ▼
DATASET + LABELS (NEW research/dataset.py)          [event frames, forward labels, leak-free walk-forward split]
     ▼
EXPERIMENTS (NEW research/experiments.py)           [IC/ICIR/hit-rate, bootstraps, E1–E4]
     ▼
EXECUTION SIM (reuse envs/order_book_env.py; add cost+queue knobs)
     ▼
COST/IMPACT/QUEUE (NEW execution/cost_model.py + research/queue_dynamics.py)
     ▼
BACKTEST + METRICS (NEW execution/backtest.py)      [IS, market-VWAP slip, fill %, completion, MDD]
     ▼
RISK (reuse risk.py/cuda_risk; add historical + stress VaR)
     ▼
REPORT (NEW research/report.py + docs/RESEARCH.md)
```

---

## 3. Build order — phases

**Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5.** Each phase has CREATE/MODIFY/TEST/BENCH/DOC entries below.

### Phase 0 — Repo hygiene (day 1, few hours)
- **FIX** branch state so checkout == `main` == docs (§1a decision).
- **CREATE** `.github/workflows/ci.yml` — on PR: CMake build + CTest (Linux, Windows), `pytest python_quant/tests bindings/tests`, lint job. (Zero CI today.)
- **MODIFY** `README.md` — remove the duplicated second copy; status section updated to match `main`.
- **CREATE** `requirements-dev.txt` — pinned `pytest`, `mypy`, `ruff` (runtime deps stay in `python_quant/requirements.txt`).
- **MODIFY** `python_quant/requirements.txt` — pin versions (`numpy>=1.24`, `gymnasium>=0.29` today are deliberately loose — fine, but add nothing heavy).

### Phase 1 — Research spine (P0, weeks 1–2)
- **CREATE `python_quant/nexus_quant/research/__init__.py`** — export the subpackage; wire into `nexus_quant/__init__.py`.
- **CREATE `python_quant/nexus_quant/research/features.py`** — pure, stateless functions over a `View` (dict) or order-level events:
  - `lob_imbalance(view) -> float`  = (V_b − V_a)/(V_b + V_a)
  - `microprice(view) -> float`     = (V_a·P_b + V_b·P_a)/(V_a + V_b)
  - `ofi(prev_events, cur) -> float` (needs order-level deltas — build approximation from ladder first, upgrade when order-level tracker lands in Phase 3/4)
  - `deep_imbalance(view, k=5|10) -> float` (weighted multi-level)
  - `spread_bps(view) -> float`
  - `realized_vol(mid_hist, n) -> float`
  - `momentum(mid_hist, k) -> float`
  - `flow_intensity(events, tau) -> float`
  - **Contract:** no instance state; a test mutes the book after the call and asserts the feature is unchanged (the leakage lock).
- **CREATE `python_quant/nexus_quant/research/labels.py`** — `forward_return(state_t, future_state, h)`, `forward_mid_move(...)`; strictly uses prices at `t+h` or later, never `t`.
- **CREATE `python_quant/nexus_quant/research/dataset.py`** —
  - `event_frame(events, book) -> rows (feature, label, ts, split_tag)`
  - `make_split(rows, mode="walk_forward", train=0.6, val=0.2)` — time-ordered disjoint blocks (k-fold **walk-forward**, not shuffled CV).
- **CREATE `python_quant/nexus_quant/research/experiments.py`** —
  - `rank_ic(y_true, y_pred)`, `icir(ic_series)`, `hit_rate`, `decile_spread`
  - `bootstrap_ci(values, n_boot=2000, kind="block", block=…)`
  - `diebold_mariano(a, b)` for pairwise feature comparisons
  - `run_experiment(feature_fn, label_h, split) -> result` (JSON-serializable)
- **CREATE `python_quant/nexus_quant/research/models.py`** — OLS on z-scored features only (IC benchmark). **No trees/boosters/neural nets until the linear baseline stands.**
- **TEST `python_quant/tests/test_research.py`** — (a) features don't read future state (mutation-after-call); (b) labels are strictly forward; (c) walk-forward blocks are disjoint in time; (d) IC≈1 on a hand-built predictable series; (e) IC≈0 (within CI) on white noise; (f) bootstrap CI covers the true IC on a known distribution.
- **DOC `docs/RESEARCH.md`** — skeleton: scope, methods section, template tables.

### Phase 2 — Execution realism (P0/P1, weeks 2–3)
- **CREATE `python_quant/nexus_quant/execution/cost_model.py`** — `CostParams(fee_bps, rebate_bps, spread_ecn, impact_coef, impact_mode="sqrt")`; `net_pnl(gross, fills, side)`, `impact(qty, part_rate, sigma)` (square-root law). Costs validate: maker rebate sign, taker fee sign, impact monotone in qty.
- **CREATE `python_quant/nexus_quant/execution/metrics.py`** — `implementation_shortfall`, `vwap_slippage(market_vwap, fills)`, `arrival_slippage`, `fill_rate`, `completion_rate`, `inv_risk(sigma, inv, T)`, `max_drawdown(pnl_path)`. **VWAP slippage must be vs market VWAP, not self-executed VWAP** (that was a Part-1 flaw).
- **CREATE `python_quant/nexus_quant/execution/backtest.py`** — run any policy over a tape + book, produce metric set + per-regime breakdown.
- **MODIFY `envs/order_book_env.py`** — accept `fee_bps`, `rebate_bps`, `impact_coef`, `queue_model` (defaults: off / 0 → **byte-identical behavior** to today, same discipline as the regime params). Add `market_vwap` to `info`.
- **TEST `tests/test_cost_model.py`, `tests/test_exec_backtest.py`** — fee/rebate arithmetic, square-root-law monotonicity, MDD on known path, env byte-parity with defaults.

### Phase 3 — RL fairness + queue studies (P0/P1, weeks 3–4)
- **CREATE `python_quant/nexus_quant/research/queue_dynamics.py`** — order-level tracker over an ITCH stream: per resting order `ahead_qty`, `behind_qty`, cancel-cursor, consumed-at-level, fill/delete events; survival: `fill_prob_survival(...)` (Kaplan–Meier, cancel = competing risk), `logistic_fill_model(features) -> P(fill)`.
- **CREATE `python_quant/nexus_quant/research/adverse_selection.py`** — `post_fill_drift(fills, book, h)` over {1,5,25} events, conditioned on (fill side, OFI, queue position); `P(adverse | passive fill)`.
- **MODIFY `agents/evaluate.py`** — `evaluate_regime_ci(policy, regimes, seeds=5)` returns per-regime mean ± bootstrap-95%-CI of IS; add a **fair `volFeat` toggle**: baselines get the same indicator or the feature is disabled — the comparison must be symmetric.
- **MODIFY `agents/baselines.py` (or add `agents/baselines_rl.py`)** — stronger, defensible baselines: adaptive-POV (reacts to spread/vol), schedule-TWAP with a volume curve, IS-aware rule. Retire the 2-line "VWAP" heuristic as the reference.
- **TEST `tests/test_queue_dynamics.py`, `tests/test_adverse_selection.py`** — fill-prob calibration on a synthetic stream with a *known* queue; adverse-drift sign test; E5/E6 wiring.

### Phase 4 — Real data (P0, weeks 4–5 — the credibility unlock)
- **CREATE `scripts/fetch_itch.py`** — download a public NASDAQ ITCH sample (e.g. classic 2020-02-28 `S030220-v50.txt.gz`, or a published SPY/NVDA day) into **`data/`** (already gitignored). Record provenance + license note in the script header.
- **MODIFY `itch_parser.py`** — fix any real-file decode issues that surface; expose order-level events for OFI/queue (this is where the parser earns its keep).
- **CREATE `scripts/run_research.py`** — runs E1–E4 on the real tape; outputs IC/ICIR table + bootstrap CIs + figures.
- **TEST `tests/test_offline_real_tape.py`** — smoke: parser consumes the real tape with 0 truncated, integrity checks clean.

### Phase 5 — Report + write-up (P1/P2, week 5–6)
- **CREATE `docs/RESEARCH.md`** (full) — hypotheses, methods, split discipline, IC tables, per-regime execution results, adverse-selection findings, **negative results section**.
- **CREATE `scripts/run_all.py`** — one-command reproduce: fetch → build → test → research vignettes → exec comparisons → report.
- **MODIFY `README.md`** — final honest rewrite using only measured numbers.

---

## 4. Experiment registry (E1–E7)

| # | Experiment | Primary metric | Split | Tests | Failure criterion |
|---|---|---|---|---|---|
| E1 | LOB imbalance → Δmid | rank IC by horizon {1,5,10,25} | walk-forward | IC≠0 (bootstrap CI 95%), DM vs 0 | IC ≤ 0.01 or unstable sign across folds |
| E2 | Microprice → Δmid | rank IC, vs E1 | walk-forward | pairwise DM, CI on ΔIC | microprice not ≥ bid/ask alone |
| E3 | OFI → Δmid | rank IC (nested vs imbalance) | walk-forward | LR / nested IC | no IC gain |
| E4 | Compare imbalance/microprice/OFI/combined | IC, ICIR, hit rate | identical split | permutation test | best feature changes per fold |
| E5 | Passive fill probability | Brier, survival AUC | time-ordered | calibration slope ≥ 0.8 | slope < 0.8 → queue model is fiction |
| E6 | Adverse selection after passive fills | signed post-fill drift | matched pairs | t-stat (NW) on drift | drift not ≠ 0 once conditioned on OFI/queue |
| E7 | Execution strategies (TWAP/VWAP/POV/Passive/Adaptive/PPO) | IS vs **market** VWAP, fill %, completion, MDD | per-regime × seeds≥5 | bootstrap CIs; fairness symmetric | any edge vanishes under fees+queue+adverse |

---

## 5. Statistical rigor — standing checklist

- Walk-forward time split (never shuffled CV on tape data).
- Event-based sampling (regulate on event cadence; kill volume-hour biases).
- Block / stationary bootstrap for CIs on IC and IS (never i.i.d. bootstrap on autocorrelated series).
- Diebold–Mariano for pairwise comparisons; Newey–West when labels overlap.
- Rank IC + ICIR primary; hit rate + decile spread robustness.
- AUC where appropriate (fill models); multiple seeds ≥5 for RL; per-regime eval.
- Sensitivity analysis: report IC/IS vs (horizon, tick size, λ, participation) — not just point estimates.
- **Leakage points identified in Part-1 code:** (1) `vol_feature` regime indicator asymmetry; (2) eval on training distribution; (3) shortfall vs self VWAP; (4) reward/eval objective mismatch; (5) regime tuned to target; (6) no CIs on headline; (7) "free" passive fills (no queue); (8) fixed arrival mid 15,000 with no drift/regime diversity; (9) `lambda_risk` seam default-off. All must be addressed before any claim leaves the repo.

---

## 6. RL fairness — the standing rework spec (do this before the headline is re-used)

1. Train/eval across regimes: low-vol, normal, high-vol, liquidity-shock, trending, mean-reverting.
2. ≥5 seeds; report mean ± bootstrap-95%-CI of IS, not one seed family.
3. **Symmetric information:** either baselines consume the same `vol_feature` indicator or the feature is removed from the agent.
4. Fair baselines: adaptive-POV, schedule-TWAP with a volume curve, IS-aware rule. Evaluate all strategies under a **shared objective** (same IS metric) and with fees + queue on.
5. Report IS vs **market** VWAP, completion%, MDD per regime.
6. Then, and only then, one line in README: "PPO vs best baseline: [X] bps IS improvement (CI [±Y]) under [regime]." If it's not significantly better, say so.

---

## 7. Prioritization

| Rank | Task |
|---|---|
| P0 | Reconcile branches; checkout == main == docs; one dashboard |
| P0 | CI (GitHub Actions) |
| P0 | Research spine: `features.py` + `labels.py` + `dataset.py` + `experiments.py` + `models.py` |
| P0 | Statistical rigor module (walk-forward, block-bootstrap CI, DM, IC/ICIR) |
| P0 | RL fairness rework (§6) — re-characterize the +50.4% headline |
| P0 | Execution cost/queue model + market-VWAP metrics |
| P0 | Real ITCH tape ingestion + parser validation |
| P0 | Benchmark hardening + real-HW numbers (don't publish before) |
| P1 | Queue dynamics + adverse-selection studies (E5/E6) |
| P1 | Per-regime RL eval + adaptive baselines (E7) |
| P1 | Historical/parametric/stress VaR + scenario bank |
| P1 | `docs/RESEARCH.md` + honest README rewrite |
| P2 | CUDA toolkit + `risk_bench` measurement (fast once decided) |
| P2 | Execution timeline + inventory chart in dashboard |
| P2 | VWAP volume-curve forecast (real baseline) |
| P3 | Notebooks, plotting polish, ruff/mypy config, lockfile, RSS-style writeup |

**DO NOT BUILD:** live market connectivity/FIX; WebSocket fan-out; swapping PPO for a fancier RL to "fix" results; autoML feature stacking; NN predictors before linear IC baseline; cloud deployment; a custom plotting framework; **another tuned regime to chase another headline.**

---

## 8. Six-week calendar (starts Mon 2026-09-14)

| Week | Dates | Focus | Deliverables | Commit names |
|---|---|---|---|---|
| W1 | 09-14 → 09-20 | Phase 0 + Phase 1 start | Green CI; branch reconcile; `features.py`+`labels.py`+`dataset.py` with leak tests | `chore: reconcile branches to main`, `ci: build+test+lint`, `feat: research features + leak-proof datasets`, `bench: workload matrix + cache counters` |
| W2 | 09-21 → 09-27 | Phase 1 + E1/E2 | IC tables with CIs on synthetic; `docs/RESEARCH.md` skeleton | `feat: signal experiments E1/E2 + bootstrap CIs` |
| W3 | 09-28 → 10-04 | Phase 2 + Phase 3 start | Cost/queue model; queue dynamics + adverse selection (E5/E6); fair RL eval harness | `feat: execution cost+queue model`, `feat: queue/adverse-selection studies`, `fix: fair RL eval + regime CIs` |
| W4 | 10-05 → 10-11 | Phase 4 | Real ITCH tape ingested; parser validated; E1–E5 on real tape | `feat: ingest NASDAQ ITCH tape`, `test: real-tape parser smoke` |
| W5 | 10-12 → 10-18 | Phase 4/5 | E7 on real + synthetic with fees/queue/CIs; historical + stress VaR | `feat: E7 exec comparison on real tape`, `feat: historical/stress VaR` |
| W6 | 10-19 → 10-25 | Phase 5 | `docs/RESEARCH.md` (incl. negative results); README rewrite; resume bullets; final benchmarks | `docs: research report + honest README`, `chore: final benchmark numbers` |

---

## 9. Definition of Done — research half

- [ ] Working tree == `main` == README; no stale branches behind the docs.
- [ ] CI green (Linux + Windows): build, CTest, pytest, lint.
- [ ] At least one real NASDAQ ITCH tape parsed + replayed cleanly; asserted in CI.
- [ ] E1–E6 with walk-forward splits, rank IC/ICIR, block-bootstrap CIs; at least one negative/unstable result reported.
- [ ] Execution eval: per-regime, ≥5 seeds, CI-reported, symmetric information, fair baselines, fees+queue on; headline re-verified or explicitly re-characterized.
- [ ] Cost/queue/impact model present; IS reported vs **market** VWAP, with fill %, completion, MDD.
- [ ] Zero-alloc + throughput/latency measured on a documented Linux box (compiler, flags, CPU, ≥30 runs, CI/IQR).
- [ ] CUDA risk measured (or clearly parked with the exact blocker).
- [ ] `docs/RESEARCH.md` exists: hypotheses, methods, tables, and a "what we tried that failed" section.

---

## 10. Running log (append per session)

- **2026-09-12** — Full 16-part repo audit delivered; `plan_2.md` created. Part-1 work (systems half) confirmed complete. Decision flagged: reconcile branches to `main` before building the research half.
- **2026-09-12 (later)** — **Phase 0 + Phase 1 spine landed on `main`** (`e630ad7`; ruff clean; Tier 1 85 passed / 1 skipped).
  - **Phase 0:** checkout `main` == docs; dashboard reconciled (combined desk `dashboard_page.html` + verification console `dashboard/index.html` + enhanced `SnapshotHub`); README deduped (was 2 full READMEs) + status matches main + slippage headline marked re-verification-pending (§1b/§6); CLAUDE.md brought to main's reality + points here; stale `.claude/worktrees/` + `pr7-fix` removed; `.github/workflows/ci.yml` (CMake+CTest+pytest on Linux/Windows, ruff lint); `requirements-dev.txt` (pytest/mypy/ruff pinned); Python layer ruff-clean (67 legacy findings, zero behavior change).
  - **Phase 1:** `research/{features,labels,dataset,experiments,models}.py` + `test_research.py` (17 tests: leak locks, walk-forward disjointness, IC≈1/IC≈0, bootstrap CI coverage, DM, OLS slope recovery) + `docs/RESEARCH.md` skeleton. Wired as `nexus_quant.research`.
  - **Remaining:** Phase 1 E1–E4 on synthetic (needs a small `run_experiment` vignette), then Phase 2 (cost/queue model, market-VWAP metrics), Phase 3 (queue dynamics + RL fairness rework), Phase 4 (real tape).