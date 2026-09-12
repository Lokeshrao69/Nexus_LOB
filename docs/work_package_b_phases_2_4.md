# Work package — Person B: Phases 2–4 (execution realism → RL fairness → real tape)

> **Plan of record:** `plan_2.md` §3–§8. This file is the Person B hand-off for Phases 2–4.
> Every interface named here is **exact as of 2026-09-12** (verified against `main` + the
> Phase 0/1 spine). Where a signature is **TO FREEZE** (bold), Person B freezes it and Person A
> consumes it — no renames without bumping this doc and `plan_2.md`.
>
> Predecessor work already landed: Phase 0 (repo hygiene, CI) + Phase 1 spine
> (`research/{features,labels,dataset,experiments,models}.py`, `test_research.py`, `docs/RESEARCH.md`)
> on `main`; the synthetic E1–E4 IC vignette (`scripts/research_vignette.py`) now runs
> end-to-end and prints the first honest IC table (null result, by design — see §R below).

Owner: **Person B (Quant Research & RL Lead)**. Integration counterpart: **Person A** (C++ engine, shmem, CUDA).

---

## 0. TL;DR

| # | Phase | Goal | Deliverables | A/B seam |
|---|-------|------|--------------|----------|
| WP2 | **Phase 2 — execution realism** | Make the sim's cost/queue realism honest enough that *accuracy* (not rank) means something | `execution/{cost_model,metrics,backtest}.py`; `env` fee/rebate/impact/queue knobs; market-VWAP metrics | Person A reviews the fill/impact realism; B keeps `reset()/step()` contract frozen |
| WP3 | **Phase 3 — RL fairness** | Re-verify (or re-characterize) the +50.4% headline under symmetric information, fair baselines, per-regime CIs | `research/{queue_dynamics,adverse_selection}.py`; `evaluate_regime_ci`; fair `strategy_table` runs | **A provides the random-walk control env** (B's null arm) |
| WP4 | **Phase 4 — real tape** | Real NASDAQ ITCH bytes → replay → features → E1–E7 tables | `scripts/fetch_itch.py`, `scripts/run_research.py`; `itch_parser.py` order-level events; `test_offline_real_tape.py` | A extends `EngineAdapter`/parser for order-level events if needed |

Each phase ends with a **MERGE** (real merge commit, per the git workflow — feature branch, PR, review).

---

## R. What the Phase-1 vignette already establishes (recorded, honest)

`python_quant/scripts/research_vignette.py` (synthetic E1–E4, seeded random-walk flow, h ∈ {1,5,10}):

- The spine works end-to-end: `synthetic_events` → `StubOrderBook` → `features` (single-view **and** pairwise `ofi`) → `event_frame` → `make_split` (walk-forward) → `run_experiment` (rank IC, ICIR, hit rate, decile spread, block-bootstrap CI) → JSON.
- **Honest result:** a random-walk flow gives IC ≈ 0 (CIs straddle 0) for `ofi`, `deep_imbalance`, `microprice_off` across horizons — the null harness behaves as designed. `lob_imbalance` shows a small spurious drift (~0.2 IC at h=5–10, CI excluding 0) — an artifact of random *adds* landing around the walk, not a predictor. Recorded as a **negative/harness result**; real claims wait for Phase 4 tape.
- Vignette CLI: `--steps` `--seed` (int or hex) `--horizons` `--train` `--val` `--out <json>`.
- 18 research tests now cover the spine incl. the `prev_feature_fns` **row-0-drop + exact OFI** contract (`test_event_frame_pairwise_feature_drops_first_row`).

**Do not** tune the synthetic flow to produce nonzero IC — that repeats the Part-1 headline-tuning sin (`plan_2.md §0 rule 2`). The null result is the correct result here.

---

## 1. Exact interfaces (verified 2026-09-12 — freeze these)

### 1.1 Research spine (already on `main`) — Person B owns, runs *as-is*

```python
from nexus_quant.research import (               # __init__ re-exports
    event_frame, make_split,                     # event_frame: see below
    # experiments/rigor
    rank_ic, icir, hit_rate, decile_spread,
    bootstrap_ci, diebold_mariano, run_experiment,
    # features
    lob_imbalance, microprice, deep_imbalance, spread_bps,
    ofi, realized_vol, momentum, flow_intensity,
    # labels
    forward_mid_move, forward_return,
    # models (linear-only, per §3 Phase 1 rule)
    fit_ols_ic, zscore,
)
from nexus_quant.research.dataset import Row     # Row is NOT in package __all__

# event_frame — pairwise prev-features supported (new since spine):
def event_frame(
    events, apply, *,
    feature_fns: Mapping[str, Callable[[View], float]],
    prev_feature_fns: Mapping[str, Callable[[View, View], float]] | None = None,
    h: int = 5,
    label_fn: Callable[[View, View, int], float] = forward_mid_move,
    ts_fn: Callable[[View], int] = lambda v: int(v["seq"]),
) -> list[Row]: ...
#   Row 0 is DROPPED when prev_feature_fns is set (no pairwise predecessor).
#   apply(ev) mutates the book and returns the current view (mirrors ReplayEngine.apply).

# run_experiment — the E1–E4 result bundle:
def run_experiment(y_true, y_pred, *, split_tags=None, n_boot=2000, seed=0x51ED) -> dict: ...
#   {"n": int, "overall": {...}, "per_split": {"train"/"val"/"test": {...}}}
#   per-split fields: rank_ic, icir, hit_rate, decile_spread, ci95{"lo","hi","mean"}
```

### 1.2 Env + baselines + agent harness (from Phase 1c/1e — freeze)

```python
# envs/order_book_env.py
class OrderBookEnv(gym.Env):            # obs dim = 44, Gymnasium reset()/step()
    def __init__(self, *, seed=0, is_coef=0.0, lambda_sched=..., lambda_risk=0.0,
                 regime_prob=0.0, vol_decay=0.0, gap_*=..., vol_*=..., ...): ...
    def reset(self, *, seed=None, options=None) -> (np.ndarray, dict): ...
    def step(self, action) -> (obs, reward, terminated, truncated, info): ...
    #   high-vol regime params default-off (byte-identical to calm env)
    #   lambda_risk>0 -> CVaR inventory penalty from risk.py (exact-parity oracle)

# baselines.py
def run_episode(env, name: AgentId, *, seed=None, policy=None) -> EpisodeResult: ...
def compare(seed=20973, **env_kw) -> list[EpisodeResult]: ...
EpisodeResult: dataclass(name, reward, shortfall_bps, vwap, arrival, leftover, filled, steps)

# agents/evaluate.py
def strategy_table(agent=None, *, agent_name="ppo", n_episodes=50, seed=0,
                   baselines=("twap","vwap","pov","passive"),
                   env_factory=OrderBookEnv) -> list[dict]: ...
def format_table(rows: list[dict]) -> str: ...

# agents/ppo.py
def train_ppo(env_factory=OrderBookEnv, cfg=None, *, tracker=None) -> (PPOPolicy, list[TrainHistory]): ...
class PPOPolicy(obs_dim=44, hidden=(64,32), *, seed=0, std0=0.4): ...
PPOPolicy.save(path) / PPOPolicy.load(path)   # .npz, no torch
# agents/grpo.py — GRPO trainer on the same actor interface
```

### 1.3 Tape/engine seam (from Phase 1b/1e — freeze)

```python
# replay.py
class ReplayEngine: __init__(book, ...); apply(msg, ...) -> view-ish; check_integrity(...) 
# book_port.py — the injectable book seam
StubBookAdapter  (oracle)  <->  EngineAdapter  (real C++ engine, via pybind)
#   EngineAdapter has cancel_id()/lookup() (added for replay) — full cancel via
#   engine.cancel, partial via engine.modify at same price (time priority). [CLAUDE.md §6]
```

---

## 2. WP2 — Phase 2: execution realism (weeks 2–3)

### 2.1 Files
- **CREATE `python_quant/nexus_quant/execution/__init__.py`** + `cost_model.py` + `metrics.py` + `backtest.py`
- **MODIFY `envs/order_book_env.py`** — additive, default-off
- **CREATE `python_quant/tests/test_cost_model.py`**, `python_quant/tests/test_exec_backtest.py`
- **CREATE `docs/RESEARCH.md`** (§ methods + template tables; skeleton exists)

### 2.2 Interfaces (exact, from plan_2.md §3 Phase 2)
```python
# execution/cost_model.py
CostParams(fee_bps=0.0, rebate_bps=0.0, spread_ecn=0.0, impact_coef=0.0, impact_mode="sqrt")
net_pnl(gross, fills, side) -> float
impact(qty, part_rate, sigma) -> float     # square-root law
# execution/metrics.py
implementation_shortfall(...)               # per-trade + aggregate
vwap_slippage(market_vwap, fills)           # vs MARKET vwap, NOT self-executed vwap (Part-1 flaw fix)
arrival_slippage, fill_rate, completion_rate, inv_risk(sigma, inv, T), max_drawdown(pnl_path)
# execution/backtest.py
run_backtest(policy, tape, book, *, cost=CostParams(...), metric="is") -> dict  # + per-regime breakdown
```
- **env additions (defaults keep today byte-identical — same discipline as regime params):** `fee_bps`, `rebate_bps`, `impact_coef`, `queue_model`; add **`market_vwap` to `info`** in every `step`.
- **Contract-freeze rule:** `reset()/step()` shapes, obs dim 44, reward type — **unchanged**. New knobs are keyword-only with defaults that reproduce the 2026-09 numbers from `README`.
- **Acceptance:** `test_<...>.py` assert (a) maker-rebate sign, taker-fee sign, (b) impact monotone in qty, (c) MDD on a hand-built path, (d) **env byte-parity when all new knobs default** (train a few episodes twice — identical reward trace).

---

## 3. WP3 — Phase 3: RL fairness + queue studies (weeks 3–4)

### 3.1 Goal
Re-verify the +50.4% headline under the §6 rework spec **before** it re-enters any README:
symmetric info, fair baselines, per-regime CIs, random-walk null arm. Expected honest outcome
may be "PPO not significantly better than best baseline under fees+queue" — that is an
acceptable, honest result.

### 3.2 Files
- **CREATE `research/queue_dynamics.py`**, `research/adverse_selection.py`
- **MODIFY `agents/evaluate.py`** (`evaluate_regime_ci`), **MODIFY `baselines.py`** (adaptive-POV, schedule-TWAP, IS-aware rule) or **CREATE `agents/baselines_rl.py`**
- **CREATE tests**: `test_queue_dynamics.py`, `test_adverse_selection.py`

### 3.3 Interfaces (exact)
```python
# research/queue_dynamics.py
class OrderLevelTracker:  # per resting order: ahead_qty, behind_qty, cancel-cursor, consumed-at-level
    on_add(msg); on_execute(msg); on_cancel_delete(msg); on_trade(msg)
    fill_prob_survival(levels, *, hazard_fn)   # Kaplan–Meier; cancel = competing risk
    logistic_fill_model(features) -> P(fill)   # calibrated on the synthetic stream w/ KNOWN queue
# research/adverse_selection.py
post_fill_drift(fills, book, h)                # signed drift over {1,5,25} events
P_adverse(fill, book, ofi, queue_pos)          # P(adverse | passive fill), grouped by side/OFI/queue
# agents/evaluate.py  (MODIFY — add, keep strategy_table intact)
evaluate_regime_ci(policy, regimes: dict[str, OrderBookEnv], seeds=5) -> dict[str, {"mean","ci95"}]
```
- **Fairness fixes that must land here (§5 leakage list 1,2,3,5):**
  1. **vol_feat symmetric:** baselines observe the same regime indicator, or the feature is removed — never asymmetric.
  2. **eval ≠ train distribution:** hold out a regime set the agent never trains in.
  3. **VWAP vs market VWAP** (`vwap_slippage`), not self-VWAP.
  4. **fees + queue ON** for every reported number (Phase 2).
  5. **≥5 seeds, per-regime, CI from block bootstrap** — one seed family never reported alone.
- **Random-walk null arm (Person A):** env subclass whose mid follows a seeded random walk with identical fill/cost/penalty mechanics. PPO ≈ baseline on shortfall is the correct result.

---

## 4. WP4 — Phase 4: real tape (weeks 4–5, the credibility unlock)

### 4.1 Files
- **CREATE `scripts/fetch_itch.py`** (public NASDAQ ITCH sample → `data/`, gitignored; provenance + license in header), **CREATE `scripts/run_research.py`**
- **MODIFY `itch_parser.py`** (real-file decode fixes; expose order-level events for OFI/queue)
- **CREATE `python_quant/tests/test_offline_real_tape.py`**
- **MODIFY `research_vignette.py`** if it should accept a real day (`--tape`) — optional; `run_research.py` is the tape runner.

### 4.2 Exact seams
```python
# scripts/run_research.py
def main(day: str) -> None:
    msgs = itch_parser.read_tape(data / day)        # framed+raw, lazy parse
    book = StubBookAdapter()                        # or EngineAdapter (real engine)
    re = ReplayEngine(book); apply all msgs; check_integrity()
    rows = event_frame(msgs, re.apply,
                       feature_fns=_FEATURE_FNS, prev_feature_fns={"ofi": ofi},
                       h=[1,5,10], label_fn=forward_mid_move)
    split = make_split(rows, train=0.6, val=0.2, gap=10)
    per feature: run_experiment(...) -> IC/ICIR/hit/decile + CI  # same fmt as vignette
    E5/E6 run on top (queue + adverse selection from WP3)
    write docs/RESEARCH.md tables + figures
# test_offline_real_tape.py
def test_real_tape_parses_clean():  # 0 truncated, integrity clean, < n sec runtime
```
- **Parser order-level upgrade (A/B):** OFI must move from the L2-ladder *approximation* (`features.ofi(prev_view, cur_view)`) to order-level deltas once the parser emits them — that upgrade is the E5/E6 precondition and is flagged in plan_2.md Phase 2 of `features.py` (`ofi` "approximation first, upgrade when order-level tracker lands").

---

## 5. Merge/workflow rules (both halves)

- Work on `feature/part2-*` branches, small PRs, **real merge commits** (project rule — commit count visible, never squash).
- Each PR: Tier-1 pytest green (`86 passed / 1 skipped` today), ruff clean, test count ticks up.
- End-of-token handoff: update **`plan_2.md` §10 running log** (append one bullet per session) so the next session resumes without re-derivation.

## 6. Definition of done (checklist — copy into the WP4 PR body)

- [ ] `python -m pytest python_quant/tests` green (today: 86 passed / 1 skipped)
- [ ] `scripts/research_vignette.py` runs, prints honest null IC table (≈0, CIs straddling 0)
- [ ] Phase 2: `test_cost_model.py` + `test_exec_backtest.py` green; env byte-parity test with defaults
- [ ] Phase 3: `evaluate_regime_ci` ≥5 seeds per regime; fairness toggles verified symmetric; random-walk null arm result recorded
- [ ] Phase 4: `fetch_itch.py` + `run_research.py` produce an E1–E7 table from a real day; `test_offline_real_tape.py` green
- [ ] Headline re-verified or re-characterized with CI; README only ever uses measured numbers
- [ ] `plan_2.md` running log updated; this doc's interfaces still match the code