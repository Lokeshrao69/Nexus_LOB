# Plan — Part 2 Phase 3: queue dynamics, adverse selection & fair RL

> Detailed build plan for Phase 3 of the Nexus-LOB quant research layer. This is a
> working plan; the roadmap/experiment-registry authority is `plan_2.md` (read that
> FIRST each session), and `PROGRESS.md` carries the plain-language status. Status:
> **phase-approved 2026-09-13** · branch: `feature/part2-phase3-queue-rl`.

## Context

Phase 2 (execution realism) is **verified green** (110 passed / 1 skipped, plan_2
running log 2026-09-13). Phase 3 is the research-half P1 block — `plan_2.md` §3
Phase 3 + experiment registry E5/E6/E7:

1. **Queue dynamics (E5)** — order-level tracker + `P(fill)` models: Kaplan–Meier
   survival (cancel = competing risk) and a logistic fill model, calibrated on a
   synthetic tape whose FIFO queue is **known ground truth**.
2. **Adverse selection (E6)** — signed mid drift after passive fills over {1,5,25}
   events, conditioned on side / OFI / queue position.
3. **RL fairness rework (`plan_2.md` §6)** — per-regime eval, ≥5 seeds, bootstrap-95%-
   CIs on IS, **symmetric information** (baselines see the same signal the agent sees),
   stronger defensible baselines. **Decision (2026-09-13): FULL 6-regime** — this adds
   additive `OrderBookEnv` drift/mean-revert knobs + `REGIME_PRESETS`.

Honesty contract (`plan_2.md` §0): no leakage (features read only state at `t`),
no target-tuned sims, every number gets a CI, negative results reported. Frozen API
names come from `docs/work_package_b_phases_2_4.md §3.3`: `fill_prob_survival`,
`logistic_fill_model`, `post_fill_drift`, `P_adverse`, `evaluate_regime_ci`.

Exploration already produced (untracked): `research/synthetic_flow.py` (seeded
order-level ITCH tape with FIFO ground truth `internal_book()`, `FLOW_PRESETS`
rw/drift/revert/stress) and `research/queue_dynamics.py` (`QueueTracker` with
`on_event`/`fills`/`mid_history`/`reconstruct`; `queue_ahead_walk` hypothetical-fill
counterfactual).

## Workstreams

### WS-0 — Env regimes: drift/mean-revert knobs + REGIME_PRESETS (FULL 6-regime)

`python_quant/nexus_quant/envs/order_book_env.py`:
- Add keyword-only ctor params **`drift_ticks: float = 0.0`**, **`mean_revert: float = 0.0`**,
  **`mrv_anchor: float = 15_000.0`** (defaults byte-identical — the price process has **no
  center state**; the book seeds at mid 15,000 and moves only via `take`/`rest` in
  `_exogenous_flow`).
- In the **calm (non-volatile) branch** of `_exogenous_flow`, when any knob is nonzero,
  bias the taker side by a proportional controller that draws from `self._rng` **only when
  the knob is nonzero** (defaults consume zero RNG → deterministic identity):
  - `drift_ticks`: trend target `T = arrival_mid + drift_ticks · self.t`; push mid up (lift
    asks → `take(Bid)`) with probability `clamp(0.5 + (T−mid)·0.1, 0.1, 0.9)`, down with the
    complement.
  - `mean_revert`: push mid toward `mrv_anchor` with probability
    `clamp(0.5 + (mrv_anchor−mid)·0.1, 0.1, 0.9)`.
  - Volatile branch untouched (presets that need vol set `regime_prob`/`gap_*` themselves).

`python_quant/nexus_quant/__init__.py`:
- Add **`REGIME_PRESETS: dict[str, dict]`** with six keys — `"lowvol"`, `"calm"`, `"highvol"`,
  `"liquidity_shock"`, `"trending"`, `"mean_reverting"` — each passable as
  `OrderBookEnv(**REGIME_PRESETS[k])`. Keep **`HIGHVOL_PRESETS = {"highvol":
  REGIME_PRESETS["highvol"]}`** for backward compat (`train_eval_agent.py --highvol`,
  HIGHVOL_PLAN).
- Presets encode behavioral character, not alpha: `trending` = `drift_ticks>0, regime_prob=0`;
  `mean_reverting` = `mean_revert>0, mrv_anchor=15000, drift=0, regime_prob=0`; `lowvol` =
  tight adds, vol params 0; `liquidity_shock` = stress-like (`gap_prob`, thin book); `highvol`
  = today's dict. Note: `highvol` keeps `vol_feature=True` for back-compat; **fair eval
  builds regime envs with `vol_feature=False`** (see WS-3).

Tests (`tests/test_highvol_env.py`, +8): `test_drift_revert_zero_knobs_identity` (explicit 0s
reproduce the no-knob trace bit-for-bit); `test_trending_mid_has_positive_mean_move`;
`test_mean_revert_autocorr_negative_and_bounded`; `test_revert_pulls_far_mid_back_to_anchor`;
`test_regime_presets_exist_and_load`; `test_regime_presets_behave_distinctly`.

### WS-1 — E5 fill probability (`research/queue_dynamics.py`, extend)

Two honest populations: **controlled drops** (no cancel-selection bias → train logistic,
calibration, Brier = E5 headline) and **real standing orders** (fills/cancels/deletes/
replaces → descriptive KM survival; cancel is a competing risk).

```python
@dataclass(frozen=True, slots=True)
class OrderLife:                       # one standing order's lifetime
    oid: int; side: Side; price: int; born_index: int; birth_size: int
    end_index: int | None; end_kind: str      # "fill"|"cancel"|"delete"|"replace"|"open"
    filled: int; tau: int | None

def standing_order_lifetimes(events) -> list[OrderLife]   # one forward pass; REPLACE ends old oid
def fill_prob_survival(levels: Sequence[OrderLife], *, hazard_fn=None) -> dict
    # KM: S_fill(τ)=∏(1−d(u)/n(u)); cancels/deletes/replaces censor (competing risk) → also
    # S_cancel. Returns {"tau","survival_fill","survival_cancel","at_risk","n_orders",
    #  "n_fills","n_censored"} (JSON-able).

@dataclass(frozen=True, slots=True)
class FillRow:
    decision_index: int; side: Side; price: int; qty: int
    filled: bool; fill_qty: int; tau: int | None
    queue_ahead: int; level_size: int; mid_at_t: float
    features: dict[str, float]          # observable at decision t ONLY

def fill_dataset(flow: SyntheticFlow, *, each_tau_drop=40, qty=100, window=200,
                 sides=(Side.Bid, Side.Ask),
                 features_fn: Callable[[QueueTracker,int],dict] | None = None) -> list[FillRow]
    # incremental QueueTracker; at every each_tau_drop with a live touch:
    #   price=best(side); queue=queue(side,price); queue_ahead=sum sizes; level_size=total
    #   outcome = queue_ahead_walk(queue, events, decision_index=i, side, price, qty, window)
    #   default features (past-only): log_level, spread_bps, lob_imbalance, ofi(prev,cur),
    #       realized_vol(last 10 mids), flow_intensity(events[i-10:i]), momentum, queue_ahead
    #   (queue_ahead_frac dropped — constant 1.0 for a back-of-touch drop; documented)
class LogisticFillModel:
    feature_names; mu; sigma; w
    @classmethod fit(cls, rows, *, l2=1.0, max_iter=200, tol=1e-7, seed=0x51ED)
    def predict_proba(rows) -> np.ndarray; def __call__(features) -> float   # P(fill)
    def brier(rows) -> float; def calibration(rows, *, n_bins=10) -> dict
    # NumPy IRLS: M=[1,X] z-scored; H=Mᵀ diag(p(1−p)) M + l2·R (ridge slope-only);
    #   w ← w + lstsq(H, −g); p clipped to [1e-9, 1−1e-9].
```
- Patch `queue_ahead_walk(..., *, queue_ahead=0, level_size=0)` → `_outcome` (additive,
  defaults unchanged); `fill_dataset` passes real values.
- Add to `research/experiments.py`: `brier_score(y_true, y_pred)`; `calibration_curve(y_true,
  y_pred, *, n_bins=10) -> {"bins":[{lo,hi,n,pred_mean,obs_rate,se}],"slope","intercept"}`;
  `hac_se(values, *, lag=0)` (Newey–West, Bartlett); `normal_cdf(z)`.

### WS-2 — E6 adverse selection (`research/adverse_selection.py`, CREATE)

- `FillRecord` gets two **defaulted** fields `level_size_at_fill: int = 0`, `ofi: float = 0.0`
  (additive, back-compat). `QueueTracker.__init__` adds `_prev_view`; set
  `self._prev_view = self.view()` at the top of `on_event`; in `_execute` capture
  `level_size_at_fill = lvl.total()` before `_reduce` and `ofi = features.ofi(_prev_view,
  self.view())` (the fill's own flow is known to the passive order at fill — no lookahead).

```python
def post_fill_drift(fills, book: QueueTracker, h: int, *, n_boot=2000, seed=0x51ED) -> dict
    # drift = mid_history[event_index+h] − mid_history[event_index];
    # bps = ticks / max(1, mid_history[event_index]) · 1e4
    # DROP rows with event_index+h >= len(mid_history). CI = bootstrap_ci(kind="iid").
    # h>1: t = mean / hac_se(ticks, lag=h); p = normal_cdf(|t|) two-sided.
    # Returns {"h","n","mean_ticks","mean_bps","ci95","t_stat","p_value","hac_se"} (JSON).
def P_adverse(fill, book, ofi, queue_pos, *, h=5) -> dict
    # adverse = drift unfavourable for the passive side (sell fill adverse when mid rises).
    # {"side":"ask"|"bid","ofi","queue_pos","h","drift_ticks"|None,"drift_bps"|None,"adverse"|None}
def adverse_groups(fills, book, *, horizons=(1,5,25), n_queue_terciles=3, n_boot=2000, seed=0x51ED) -> dict
    # bins = side × sign(ofi) × queue_pos tercile (np.quantile, deterministic); per bin per h:
    # mean drift_bps + bootstrap_ci(iid) + frac_adverse (= P(adverse|side,ofi,queue)) + t/p.
    # Fully JSON-serializable; keys str ("bid"/"ask").
```
`queue_pos = ahead_at_fill / max(1, level_size_at_fill)` computed downstream.

### WS-3 — Fair RL eval (`agents/evaluate.py`, extend — `strategy_table` intact)

```python
def evaluate_regime_ci(policy: Policy | None = None,
                       regimes: dict[str, OrderBookEnv | Callable[[], OrderBookEnv]] | None = None,
                       *, seeds: int = 5,
                       baselines: tuple[BaselineId, ...] = ("twap","vwap","pov","passive",
                                                            "apov","stwap","isaware"),
                       fair: bool = True, seed0: int = 0x51ED, n_boot: int = 2000) -> dict[str, dict]
```
- **Primary form = env instances** per frozen API; a zero-arg callable value is duck-typed as
  a factory, freshened per episode. Episode `s` per regime uses seed `seed0 + s`. Enforce
  `seeds >= 5`.
- **Fair toggle** (§6 item 3 — the "remove the feature" branch): `fair=True` (default) asserts
  all regime envs have `vol_feature is False`, else **raises** with an explicit message;
  `fair=False` is the escape hatch for agents trained with the flag. Regime still reaches both
  sides through the book state. Regimes built as `{k: OrderBookEnv(**{**REGIME_PRESETS[k],
  "vol_feature": False})}`.
- **One shared objective loop** (bridges the two policy shapes; does NOT use `run_episode`,
  which exposes neither `fills` nor `market_vwap`): `_run_regime_episode(env, act, seed)` →
  reset(seed), loop `act(env, obs)`, return `{reward, shortfall_bps, vwap_slip_bps:
  vwap_slippage(env.fills, info["market_vwap"]), completion: completion_rate(env.fills, inv0,
  leftover), mdd_ticks: max_drawdown(mtm_path), leftover}` (all from `execution.metrics`,
  which accept `env.fills` `(t,px,sz)` directly — verified). Agent wrapper
  `policy.act(obs, deterministic=True)`; baseline wrapper `baselines.policy_action(name, env)`.
- **Return** per regime: `{"mean","ci95","n","metric":"shortfall_bps","strategies":{"ppo":{...},
  "apov":{...}},...}`, strategy rows carry `mean ± ci95`, `vwap_slip_bps`, `completion`,
  `mdd_ticks`, `leftover_mean`; `"rows"` per-episode. CIs via `bootstrap_ci(kind="iid")`
  (episodes are independent seeds). Reports **both** IS (shortfall_bps, continuity) and **IS vs
  market VWAP** (§6 item 5).
- Docstring: headline-level use requires `seeds >= 20`; at 5 CIs are wide-by-design.

### WS-4 — Stronger baselines (`python_quant/nexus_quant/baselines.py`, extend in place)

`AgentId = Literal["twap","vwap","pov","passive","apov","stwap","isaware"]`. Three new
env-aware, stateless branches reading only `env.book.view()` + `env.t/horizon/inventory`
(no history/lookahead):
- **`apov` (adaptive-POV)** — `urgency = t_frac + 0.15·(1 − min(1, spr/8))`; `−1.0` if
  `urgency>0.85`; `−0.6` if `spr<=1`; else `−0.15 + 0.25·t_frac`.
- **`stwap` (schedule-TWAP, volume curve)** — `curve = (1 − t_frac)**0.8`; `−1.0` if
  `inv_frac > curve + 0.05` else `0.05`.
- **`isaware`** — `−1.0` if `t_frac>0.7 and (spr>=4 or inv_frac>0.5)`; `−0.8` if
  `spr>=4 and inv_frac>0.9`; else `0.0 − 0.3·min(1,spr/6) + 0.4·t_frac`.

`strategy_table`'s **default** baselines tuple stays `("twap","vwap","pov","passive")`
(existing tests green); the new baselines are opt-in there and part of `evaluate_regime_ci`'s
own default.

### WS-5 — Wiring, vignette, docs

- `research/__init__.py`: `__all__` += queue_dynamics names (`QueueTracker`, `RestingOrder`,
  `FillRecord`, `FillOutcome`, `FillRow`, `fill_dataset`, `fill_prob_survival`,
  `standing_order_lifetimes`, `LogisticFillModel`, `queue_ahead_walk`), synthetic_flow names
  (`FlowConfig`, `FLOW_PRESETS`, `SyntheticFlow`), adverse_selection names (`adverse_groups`,
  `post_fill_drift`, `P_adverse`), experiments additions (`brier_score`, `calibration_curve`,
  `hac_se`, `normal_cdf`).
- `agents/__init__.py` + `nexus_quant/__init__.py`: export `evaluate_regime_ci`, `REGIME_PRESETS`.
- CREATE `python_quant/scripts/queue_adverse_vignette.py` (self-inserting `python_quant` on
  `sys.path`, like `research_vignette.py` — which stays byte-identical). Prints two honest
  tables on `FLOW_PRESETS["rw"]` vs `["drift"]`: **E5** — `fill_dataset`→`LogisticFillModel`
  → `calibration_curve` slope + Brier vs base rate; **E6** — `adverse_groups` mean-drift sign
  + iid CI at h∈{1,5,25}. Nulls reported, never tuned. `--out` writes JSON.
- `docs/RESEARCH.md`: fill in the E5/E6 rows + results from the vignette.

## Tests & expected count

Baseline: **110 passed / 1 skipped** (111 collected) today. New:
- `tests/test_queue_dynamics.py` (~11): lockstep `reconstruct()==internal_book()` after every
  event; `fill_dataset` rows have known queue (`price==tracker.best(side)`, queue_ahead ==
  level_size at touch); no cancel-selection bias (drops run at every decision index regardless
  of tape); **leak lock** (recompute row i's features from a fresh tracker over `events[:i]` →
  exact equality); KM hand-case; cancel is competing risk; REPLACE ends old oid as "replace";
  logistic recovers separated means (P decreasing in queue_ahead, Brier<0.25); calibration
  slope≈1 for perfect model; Brier edge cases; `tau>0` forward-only.
- `tests/test_adverse_selection.py` (~6): drift uses `mid_history[event_index+h] −
  mid_history[event_index]` exactly; tape-end rows dropped; bins partition all fills; no-peek
  OFI (recomputed from independent replay); whole structure `json.dumps`-able; honest null —
  on `FLOW_PRESETS["rw"]` the h=5 CI straddles 0.
- `test_ppo_agent.py` (+4): `evaluate_regime_ci` shape + reproducibility; fair toggle raises
  on `vol_feature=True`, runs under `fair=False`; rows carry `vwap_slip_bps`/`completion`/
  `mdd_ticks`; `apov`/`stwap`/`isaware` actions ∈[−1,1] and `strategy_table(baselines=("apov",))`.
- `test_research.py` (+3): `hac_se` hand-rolled; `calibration_curve` slope/bins; `normal_cdf`.
- `test_highvol_env.py` (+8): WS-0 env/preset tests.

≈ **135 collected**, all green, ruff clean.

## Verification

```bash
python -m pytest python_quant/tests -v                            # ~135 pass, 1 skip
ruff check python_quant/nexus_quant python_quant/tests python_quant/scripts
python python_quant/scripts/queue_adverse_vignette.py             # E5 slope + E6 drift tables
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py --eval-only \
    python_quant/artifacts/policy_ppo_highvol.npz --table-episodes 20   # strategy_table intact
# commit WS-by-WS on this branch (feature-only; merge via real merge commit, not squash)
```

## Files touched
CREATE: `research/adverse_selection.py`, `scripts/queue_adverse_vignette.py`,
`tests/test_queue_dynamics.py`, `tests/test_adverse_selection.py`.
MODIFY: `research/queue_dynamics.py`, `research/experiments.py`, `research/__init__.py`,
`envs/order_book_env.py`, `nexus_quant/__init__.py`, `agents/evaluate.py`, `agents/__init__.py`,
`baselines.py`, `tests/test_ppo_agent.py`, `tests/test_research.py`, `tests/test_highvol_env.py`,
`docs/RESEARCH.md`, `plan_2.md` (running log + Phase 3 done).