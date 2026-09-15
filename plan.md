# Plan — Part 2 Phase 3: queue dynamics, adverse selection & fair RL

> Detailed build plan for Phase 3 of the Nexus-LOB quant research layer. This is the
> **single authority for Phase 3 scope** — read it whenever a stale plan file says
> otherwise. Roadmap/experiment-registry authority: `plan_2.md` (read that FIRST each
> session); plain-language status: `PROGRESS.md`; systems handoff: `CLAUDE.md`.
> Status: **COMPLETE 2026-09-14** — WS-1/WS-2 merged to `main` (PRs #18/#20);
> WS-3/WS-4/WS-5 on `feature/part2-phase3-rl-fairness` (open PR against `main`).
> Branch of record below is historical (`feature/part2-phase3-queue-rl` was merged
> and deleted on the remote).
>
> **Previous working-plan text that predates this refresh is SUPERSEDED:**
> a prior plan (and `plan_2.md` running log / `PROGRESS.md` / `progress_b.md` /
> `HIGHVOL_PLAN.md`) included a **WS-0** workstream adding additive
> `OrderBookEnv` `drift_ticks`/`mean_revert`/`mrv_anchor` knobs + a 6-key
> `REGIME_PRESETS`. **WS-0 is REMOVED from scope.** Do not add env knobs; do not
> follow any file that describes them as current scope.

## Scope guard (why there is no WS-0)

No `OrderBookEnv` changes in Phase 3. The env is frozen additive-default-only, and no new
knob is needed: the **Markov vol-regime knobs already exist** (`regime_prob`/`vol_decay`/
`gap_*`/`vol_*`; preset `HIGHVOL_PRESETS["highvol"]` in `nexus_quant/__init__.py`).
`evaluate_regime_ci` receives **ready-built env instances** per regime built from those
existing knobs (defaults and `HIGHVOL_PRESETS`), with `vol_feature=False` in fair mode.
`SyntheticFlow`/`FLOW_PRESETS` (rw/drift/revert/stress) are the **queue-research tapes**
for E5/E6, NOT an RL-env input — do not wire them into `OrderBookEnv`.

## Context

Phase 2 (execution realism) is verified green on this branch (117 passed / 1 skipped).
Phase 3 is the research-half P1 block — `plan_2.md` §3 Phase 3 + experiment registry
E5/E6/E7, honoring the honesty contract (`plan_2.md` §0): no leakage, no target-tuned
sims, every number gets a CI, negative results reported. Frozen API names come from
`docs/work_package_b_phases_2_4.md §3.3`: `fill_prob_survival`, `logistic_fill_model`,
`post_fill_drift`, `P_adverse`, `evaluate_regime_ci`.

Exploration already on the branch (untracked): `research/synthetic_flow.py` (seeded
order-level ITCH tape with FIFO ground truth `internal_book()`, `FLOW_PRESETS`) and
`research/queue_dynamics.py` (`QueueTracker`: `on_event`/`fills`/`mid_history`/
`reconstruct`/`view`/`queue`; `RestingOrder`/`FillRecord`/`FillOutcome`;
`queue_ahead_walk` hypothetical-fill counterfactual — currently returns
`queue_ahead=0, level_size=0` placeholders).

## Workstreams

### WS-1 — E5 fill probability (`research/queue_dynamics.py` extend; `research/experiments.py` extend)

Two honest populations:
- **Controlled drops** (`fill_dataset`) — no cancel-selection bias → trains
  `LogisticFillModel`, feeds calibration/Brier (the E5 headline).
- **Real standing orders** (`standing_order_lifetimes` + `fill_prob_survival` KM) —
  descriptive confirmatory arm; cancel/delete/replace is a competing risk. Reported
  alongside, never fed to the model; the gap vs controlled drops is the honest
  informative-cancel finding.

```python
@dataclass(frozen=True, slots=True)
class OrderLife:                       # one standing order's lifetime (replay of the tape)
    oid: int; side: Side; price: int; born_index: int; birth_size: int
    end_index: int | None              # None = still resting at tape end (right-censored)
    end_kind: str                      # "fill"|"cancel"|"delete"|"replace"|"open"
    filled: int

def standing_order_lifetimes(events: Sequence[NormalizedEvent]) -> list[OrderLife]
    # one forward pass; EXECUTE reduces to 0 => "fill"; CANCEL/DELETE => "cancel"/"delete";
    # REPLACE ends the old oid ("replace") and starts the new; tape-end resting => "open".

def fill_prob_survival(levels: Sequence[OrderLife], *, hazard_fn=None) -> dict
    # KM on the fill event: S(t)=PI_(u<=t)(1-d(u)/n(u)); cancels/deletes/replaces censor
    # (competing risk) and a parallel S_cancel is returned. hazard_fn (optional callable
    # on the risk vector) injects a baseline hazard; None => empirical KM.
    # Returns {"tau","survival_fill","survival_cancel","at_risk","cum_fill","cum_cancel",
    #          "n_orders"} — JSON-able.

@dataclass(frozen=True, slots=True)
class FillRow:
    decision_index: int; side: Side; price: int; qty: int
    filled: bool; fill_qty: int; tau: int | None
    queue_ahead: int; level_size: int; mid_ticks: float
    features: dict[str, float]         # observable at decision t ONLY (the leak lock)

def queue_ahead_walk(queue, events, *, decision_index, side, price, qty, window,
                     queue_ahead: int = 0, level_size: int = 0) -> FillOutcome
    # additive kwargs; defaults keep every existing call identical; _outcome passes through.

def fill_dataset(flow: SyntheticFlow, *, each_tau_drop: int = 40, qty: int = 100,
                 window: int = 200, sides: tuple[Side, ...] = (Side.Bid, Side.Ask),
                 seed: int = 0x51ED) -> list[FillRow]
    # events = flow.events or flow.generate(). Step a fresh QueueTracker incrementally;
    # at every decision index with a live touch: price=best(side); queue=queue(side,price);
    # queue_ahead=sum(sizes) == level_size (back-of-touch join, documented); run
    # queue_ahead_walk(...) over the window. Default features (past-only):
    # log_level, queue_ahead, spread_bps, lob_imbalance, ofi(prev_view,view),
    # realized_vol(last 10 mids), flow_intensity(events[i-10:i]), momentum.
    # queue_ahead_frac is DROPPED — identically 1.0 for a back-of-touch drop.

class LogisticFillModel:               # the frozen "logistic_fill_model(features) -> P(fill)"
    feature_names: list[str]; mean: np.ndarray; std: np.ndarray
    coef: np.ndarray                   # (k+1,) w[0]=intercept
    @classmethod fit(cls, rows: Sequence[FillRow], *, feature_names=None,
                     l2: float = 1.0, max_iter: int = 200, tol: float = 1e-6)
    def predict_proba(self, X: np.ndarray) -> np.ndarray
    def __call__(self, features: dict[str, float] | np.ndarray) -> float  # P(fill) in [0,1]
    def brier(self, X: np.ndarray, y: np.ndarray) -> float
    def calibration(self, X: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> dict
    # Pure-NumPy IRLS with L2 ridge on slopes: z-score X, add intercept;
    # iterate w -= lstsq(M^T diag(p(1-p)) M + l2*R, M^T(p-y) + l2*[0,w[1:]]);
    # p clipped to [1e-9,1-1e-9]; stop on max|dw| < tol. Label = row.filled.
```
**Full-episode flow tape (`compute_metrics`, landed 2026-09-14)** — a context-manager
accumulator dragged across a step loop (`with compute_metrics() as m: for ev in tape:
m(ev)`): steps record `StepFlow` snapshots (mid/spread/touch sizes pre- and post-event,
per-step OFI, lob imbalance) into a per-episode accumulator; `__exit__` freezes a closure
summary computing **queue_decay** (per-event touch-queue attrition/growth via
`mean_attrition`/`mean_growth`/`consumed_fraction`), **latency_attrition** (the same queue
eaten per second of waiting, s⁻¹ rate + half-life), and **CAR** — conditional average
response: mean `h`-event mid move conditioned on event kind + flow (OFI) terciles, with a
rank-IC. No-lookahead by construction (per-step posted book only; forward response is a
closure-time label). Tests: `tests/test_compute_metrics.py` (11).

Add to `research/experiments.py` (statistical toolbox, reused by E1–E4 and Phase 4):
```python
def brier_score(y_true, y_pred) -> float                       # mean((p-y)^2)
def calibration_curve(y_true, y_pred, *, n_bins: int = 10) -> dict
    # {"bins":[{bin_lo,bin_hi,n,pred_mean,obs_rate,se}], "slope", "intercept"}
    # slope/intercept = np.linalg.lstsq over n-weighted bin means (the calibration slope)
def hac_se(values, *, lag: int = 0) -> float
    # Newey-West/Bartlett (same gamma formula as diebold_mariano): se = sqrt(var/n)
def normal_cdf(z: float) -> float
```

### WS-2 — E6 adverse selection (`research/adverse_selection.py` CREATE; tiny additive tracker changes)

`FillRecord` gains two **defaulted** fields `level_size_at_fill: int = 0`, `ofi: float = 0.0`
(additive; grep-verified no external construction). `QueueTracker.__init__` adds
`self._prev_view: View | None = None`, set `self._prev_view = self.view()` as the FIRST line
of `on_event` (view BEFORE mutation); in `_execute` capture `level_size_at_fill = lvl.total()`
before `_reduce` and `ofi = features.ofi(self._prev_view, self.view())` — the fill's own flow
is known to the passive order at fill time; no lookahead.

```python
def post_fill_drift(fills: Sequence[FillRecord], book: QueueTracker, h: int) -> dict
    # drift_ticks = mid_history[event_index+h] - mid_history[event_index]  (post-fill base);
    # bps = ticks/max(1,base)*1e4. DROP rows with event_index+h >= len(mid_history).
    # mean!=0 test: t = mean / hac_se(ticks, lag=max(0,h-1)); p two-sided via normal_cdf.
    # CI = bootstrap_ci(drift_ticks, kind="block", block=max(1,h))  (fills overlap at h>1).
    # Returns {"h","n","mean_ticks","mean_bps","ci95","t_stat","p_value","hac_se","nw_lag"}.

def P_adverse(fill: FillRecord, book: QueueTracker, ofi: float, queue_pos: float, *,
              h: int = 5) -> dict
    # adverse = mid moved against the passive side: sell(ask) fill adverse iff drift>0;
    # buy(bid) adverse iff drift<0. {"side":"ask"|"bid","ofi","queue_pos","h",
    #  "drift_ticks"|None,"drift_bps"|None,"adverse"|None}  (None at tape-end).

def adverse_groups(fills, book, *, horizons=(1,5,25), n_queue_terciles=3, n_boot=2000,
                   seed: int = 0x51ED) -> dict
    # queue_pos = ahead_at_fill / max(1, level_size_at_fill); ofi_sign = sign(fill.ofi);
    # terciles via np.quantile (deterministic). Bins = side x ofi_sign x queue_tercile;
    # per bin per h: mean drift_bps, n, p_adverse (= P(adverse|side,ofi,queue)), ci95,
    # t/p (NW). Fully JSON-able (sides as str "bid"/"ask").
```

### WS-3 — Fair RL eval (`agents/evaluate.py` extend — `strategy_table`/`evaluate_policy` intact)

```python
def evaluate_regime_ci(policy: Policy | None = None,
                       regimes: dict[str, OrderBookEnv | Callable[[], OrderBookEnv]] | None = None,
                       *, seeds: int = 5,
                       baselines: tuple[str, ...] = ("twap","vwap","pov","passive",
                                                     "apov","stwap","isaware"),
                       fair: bool = True, seed0: int = 0x51ED, n_boot: int = 2000) -> dict[str, dict]
```
- **Primary form = env instances** (frozen API); a zero-arg callable value is duck-typed as a
  factory, freshened per episode. Episode s per regime uses seed `seed0 + s`
  (`env.reset(seed=seed0+s)` re-seeds RNG + book — instances reusable and deterministic).
  Enforce `seeds >= 5` (raise otherwise).
- **Fair toggle** (`plan_2.md` §6 item 3 — the "remove the feature" branch): `fair=True`
  (default) **raises** if any regime env has `vol_feature=True`; message directs building
  regimes with `vol_feature=False`. `fair=False` is the escape hatch for flag-trained agents —
  never the headline. Regime still reaches both sides through book state.
- **One shared objective loop** — does NOT use `baselines.run_episode` (exposes neither
  `fills` nor `market_vwap`): private `_regime_episode(env, act, seed)` — reset(seed), loop
  `act(env, obs)`, return `{reward, shortfall_bps, vwap_slip_bps: vwap_slippage(env.fills,
  info["market_vwap"]), completion: completion_rate(...), mdd_ticks: max_drawdown(mtm),
  leftover}`. Agent wrapper `policy.act(obs, deterministic=True)`; baseline wrapper
  `policy_action(name, env)`.
- **Return** per regime: `{"mean": agent_shortfall_mean, "ci95": {...}, "n": seeds,
  "metric": "shortfall_bps", "strategies": {name: {"mean","ci95","n","vwap_slip_bps",
  "completion","mdd_ticks","leftover"}, ...}, "rows": [per (strategy, seed)]}`. CIs via
  `bootstrap_ci(..., kind="iid")` (independent seeds). Reports BOTH arrival-IS shortfall_bps
  (continuity) and vs-MARKET-VWAP slippage (`plan_2.md` §6 item 5).
- Docstring: headline use requires `seeds >= 20`; at 5 the CIs are wide-by-design.

### WS-4 — Stronger baselines (`python_quant/nexus_quant/baselines.py` extend in place)

`AgentId = Literal["twap","vwap","pov","passive","apov","stwap","isaware"]`. Three new
env-aware, stateless branches in `policy_action(name, env)` reading ONLY
`env.book.view()` + `env.t`/`env.horizon`/`env.inventory` (no history, no lookahead):
- `apov` (adaptive-POV): `urgency = t_frac + 0.15*(1 - min(1, spr/8))`; `-1.0` if
  `urgency>0.85`; `-0.6` if `spr<=1`; else `-0.15 + 0.25*t_frac`.
- `stwap` (schedule-TWAP, deterministic volume curve): `curve=(1-t_frac)**0.8`; `-1.0` if
  `inv_frac > curve+0.05` else `0.05`.
- `isaware`: `-1.0` if `t_frac>0.7 and (spr>=4 or inv_frac>0.5)`; `-0.8` if
  `spr>=4 and inv_frac>0.9`; else `0.0 - 0.3*min(1,spr/6) + 0.4*t_frac`.

`strategy_table`'s **default** `baselines` tuple stays `("twap","vwap","pov","passive")`
(existing tests `test_strategy_table_contains_all_and_vwap_delta` and
`test_agent_vs_baselines_runs_with_zero_env_kw` must stay green); new baselines are opt-in
there and part of `evaluate_regime_ci`'s own default.

### WS-5 — Wiring, vignette, docs

- `research/__init__.py`: `__all__` += queue_dynamics (`QueueTracker`, `RestingOrder`,
  `FillRecord`, `FillOutcome`, `FillRow`, `OrderLife`, `fill_dataset`, `fill_prob_survival`,
  `standing_order_lifetimes`, `LogisticFillModel`, `queue_ahead_walk`), synthetic_flow
  (`FlowConfig`, `FLOW_PRESETS`, `SyntheticFlow`), adverse_selection (`adverse_groups`,
  `post_fill_drift`, `P_adverse`), experiments additions (`brier_score`, `calibration_curve`,
  `hac_se`, `normal_cdf`).
- `agents/__init__.py` + `nexus_quant/__init__.py`: export `evaluate_regime_ci`.
- CREATE `python_quant/scripts/queue_adverse_vignette.py` (self-inserts `python_quant` on
  `sys.path` like `research_vignette.py` — which stays byte-identical). Prints two honest
  tables on `FLOW_PRESETS["rw"]` vs `["drift"]`: **E5** — `fill_dataset` →
  `LogisticFillModel` → `calibration_curve` slope + Brier vs base rate; **E6** —
  `adverse_groups` mean-drift sign + CI at h in {1,5,25}. Nulls reported, never tuned.
  `--out` writes JSON.
- `docs/RESEARCH.md`: fill in the E5/E6 rows + results from the vignette.

## Tests & expected count

Baseline: **117 passed / 1 skipped** (118 collected) today. New ≈ 27 (+ 11 from the
`compute_metrics` flow-tape addition — `tests/test_compute_metrics.py`):
- `tests/test_queue_dynamics.py` (~11): lockstep `reconstruct()==internal_book()` after every
  event (+ mid-trace equality once both have a BBO); `fill_dataset` rows carry known queue
  (`price==tracker.best(side)`, `queue_ahead==level_size`); **leak lock** (recompute row i
  features from a fresh tracker over `events[:i]` → exact equality; mutating the future tape
  changes only the label); no cancel-selection bias; `queue_ahead_walk` full + partial fill
  hand case; KM hand-case; cancel is a competing risk; REPLACE ends old oid as "replace";
  logistic recovers a monotone queue hazard (P decreasing in `log_level`, Brier < 0.25);
  predict in [0,1] + deterministic; calibration slope ≈ 1 for a perfect model; Brier edges.
- `tests/test_adverse_selection.py` (~7): drift == `mid_history[i+h] - mid_history[i]`
  exactly; tape-end rows dropped; honest null on `FLOW_PRESETS["rw"]` (CI straddles 0);
  `hac_se` inflates under overlap; `P_adverse` adverse sign (seller + up = adverse);
  `adverse_groups` bins partition all fills + `json.dumps` clean; structural drift on `["drift"]`.
- `test_ppo_agent.py` (+6): `evaluate_regime_ci` shape + reproducibility; fair toggle raises
  on `vol_feature=True`, runs under `fair=False`; `seeds<5` raises; baselines-only
  (`policy=None`); `apov`/`stwap`/`isaware` actions in [-1,1] + deterministic;
  `strategy_table(baselines=("apov",))` opt-in works.
- `test_research.py` (+3): `hac_se` hand-known; `brier_score` hand-known;
  `calibration_curve` bin edges + slope.

Suite grows to ≈ **145 collected**, all green, ruff clean.

## Verification

```bash
python -m pytest python_quant/tests -v            # approx 138 pass, 1 skip
ruff check python_quant/nexus_quant python_quant/tests
python python_quant/scripts/queue_adverse_vignette.py       # E5 slope + E6 drift tables
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py --eval-only \
    python_quant/artifacts/policy_ppo_highvol.npz --table-episodes 20   # strategy_table intact
# commit WS-by-WS on this branch (feature-only; merge via real merge commit, never squash)
```

## Files touched

CREATE: `research/adverse_selection.py`, `scripts/queue_adverse_vignette.py`,
`tests/test_queue_dynamics.py`, `tests/test_adverse_selection.py`.
MODIFY: `research/queue_dynamics.py`, `research/experiments.py`, `research/__init__.py`,
`agents/evaluate.py`, `agents/__init__.py`, `baselines.py`, `tests/test_ppo_agent.py`,
`tests/test_research.py`, `docs/RESEARCH.md`, `plan_2.md` (running log).
NOT touched (frozen): `envs/order_book_env.py`; `nexus_quant/__init__.py` stays as-is
(`evaluate_regime_ci` is reachable via `nexus_quant.agents`).
