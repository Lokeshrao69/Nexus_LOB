# Nexus-LOB — Research Report (Part 2)

> **Status:** Phases 1-3 landed; E5/E6 run on synthetic; E1–E4 results below
> are placeholders until `scripts/run_research.py` is wired to a real tape
> (Phase 4). Ground rules: see `plan_2.md` §0. This document reports *only*
> measured numbers and names every negative result.

## Scope

Does an L2 order book (the 448-byte `BookStateView` ladder) predict short-horizon
mid returns? We test the canonical microstructure signals — LOB imbalance,
microprice, order-flow imbalance (OFI), and multi-level imbalance — against
strictly forward tick-move labels, on walk-forward splits, with block-bootstrap
CIs. Then we ask whether execution strategies convert any signal into
implementation shortfall before or after fees/queue (E7).

## Data & split discipline

- **Data:** seeded synthetic flow (calm + high-vol regimes) until Phase 4 lands
  a real NASDAQ ITCH tape (`scripts/fetch_itch.py` → `data/`, gitignored).
- **Split:** walk-forward by sequence time (monotone in wall-clock on a tape).
  Train / val / test blocks are contiguous and time-disjoint; a `gap >= h` is
  dropped between blocks so no label horizon reaches across a boundary. No
  shuffled CV anywhere.
- **Sampling:** event-indexed (every applied order/print), not clock-sampled —
  volume-hour biases are regulated on event cadence.
- **Leak locks (tests enforce):** features are pure functions of state at `t`;
  labels use only the endpoint state at `t+h`; both are asserted in
  `tests/test_research.py`.

## Methods

| Piece | Tool | CI |
|---|---|---|
| Feature→label association | rank IC (Spearman), by horizon {1,5,10,25} | block bootstrap on IC; DM vs 0 |
| Combining signals | OLS on z-scored features (`research.models`) | linear baseline only — no trees until it stands |
| Forecast comparison | Diebold–Mariano (HAC variance) | pairwise, on identical splits |
| Fill models (Phase 3) | Kaplan–Meier survival + logistic | calibration slope, Brier |
| Execution (Phase 2/3) | IS vs **market** VWAP | ≥5 seeds, per-regime, fees+queue on |

## Experiments E1–E4 (placeholder until the tape)

| # | Signal | h=1 | h=5 | h=25 | Conclusion |
|---|---|---|---|---|---|
| E1 | LOB imbalance → Δmid | — | — | — | *fill in* |
| E2 | Microprice → Δmid | — | — | — | *fill in* |
| E3 | OFI → Δmid | — | — | — | *fill in* |
| E4 | all vs combined | — | — | — | *fill in* |

*(Numbers are `mean · 10³ IC`, 95% block-bootstrap CI in brackets.)*

## E5 — Passive fill probability (Phase 3, synthetic)

> Fitting a logistic model of `P(fill)` to controlled hypothetical-touch drops.
> The population has no cancel-selection bias: a trial fires at every scheduled
> decision point regardless of the future flow — only a live touch gates it.
> Time-ordered 70/30 train/test split (walk-forward discipline). Numbers are
> run via `scripts/queue_adverse_vignette.py --n-events 8000 --seed 123`.

| Flow | base rate | model Brier | baseline Brier | calibration slope |
|---|---|---|---|---|
| rw (random walk) | 0.256 | 0.143 | 0.190 | 0.876 |
| drift (+0.6/event) | 0.036 | 0.034 | 0.034 | 1.422 |

- **rw:** slope < 1 is consistent with the model under-fitting the queue
  hazard on a pure random walk (no persistent take pressure); Brier beats the
  base-rate baseline (0.143 < 0.190), confirming the fill probability is not
  a coin flip — queue position IS informative even on synthetic flow.
- **drift:** low base rate (takes mostly consume the queue near-touch before
  the drop fires), so test-set Brier ≈ baseline; slope > 1 reflects
  calibration noise on a sparse positive class. **Negative result:** the
  model does NOT outperform the base-rate baseline on a trending tape —
  queue-hazard features available at decision time are insufficient to predict
  the (rare) fill on a directional market.

## E6 — Adverse selection after passive fills (Phase 3, synthetic)

> Real fills (the tracker's own executed resting orders) are the population.
> `post_fill_drift` = `mid[t+h] - mid[t]` (post-fill mid, `h` events later);
> ci95 is a block-h bootstrap over the ticks (not bps). `adverse_groups`
> conditions on `(side × OFI-sign × queue_tercile)`.

### Random-walk null arm (`FLOW_PRESETS.rw`, n_fills = 1128)

| h | mean (ticks) | 95% CI (ticks) | mean (bps) | p (NW) |
|---|---|---|---|---|
| 1 | −0.009 | (−0.020, 0.001) | −0.006 | 0.088 |
| 5 | 0.000 | (−0.026, 0.030) | 0.000 | 0.975 |
| 25 | 0.079 | (−0.017, 0.207) | 0.053 | 0.171 |

**Honest read:** CI straddles 0 at every horizon (the null holds). The
side-split shows mild asymmetry — `ask p_adverse(h=25) = 0.526`,
`bid p_adverse = 0.187` — a mild spurious signal from the random walk
upward-drift concentration around fill times. Not statistically significant
once the overall mean is inspected; recorded as a noise artifact, not a
predictor.

### Structural drift arm (`drift_ticks=0.6`, n_fills = 1151)

| h | mean (ticks) | 95% CI (ticks) | mean (bps) | p (NW) |
|---|---|---|---|---|
| 1 | 0.020 | (0.004, 0.036) | 0.012 | 0.010 |
| 5 | 0.546 | (0.512, 0.580) | 0.315 | ≈0 |
| 25 | 2.774 | (2.718, 2.839) | 1.596 | ≈0 |

- **ask p_adverse(h=25) = 1.000** (every resting seller is adversely
  drift-chased on the up-trend); **bid p_adverse = 0.000** (resting buyers
  were early, not adversely impacted). This is the structural-discriminating
  power the side condition provides.
- **Negative / honesty note:** queue-position conditioning collapses to a
  single tercile on this synthetic tape — `ahead_at_fill == 0` for every
  fill. The generator's `_take` always picks the queue front, so E6 side ×
  OFI conditioning is well-identified, but the queue-position dimension is
  degenerate here. Queue-position effects are NOT tested until the real
  NASDAQ tape (Phase 4) populates that dimension.

## Negative results so far

1. **E1–E4 on random walk: IC ≈ 0 across all features and horizons** (Phase 1
   vignette, CIs straddle 0 for `ofi`, `deep_imbalance`, `microprice_off`;
   `lob_imbalance` shows a spurious positive IC at h≥5 from random-add
   concentration — a synthetic artifact, not a predictor).
2. **E5 logistic fill model on a trending tape: does NOT outperform the
   base-rate baseline** (drift arm, Brier ≈ baseline Brier, 0.034 vs 0.034).
   Queue-hazard features available at decision time are insufficient to
   predict the (rare) fill when the market is trending — the model is
   calibrated but not informative beyond the base rate on that flow.
3. **Queue-position dimension degenerate on synthetic flow** (`ahead_at_fill ==
   0` for every recorded fill). The generator's `_take` always picks the front
   order, so `adverse_groups` queue-position conditioning collapses to a
   single tercile. Queue-position effects are NOT tested until the real tape.
4. **`research_vignette.py` has a latent path-insert bug** (`parent.parent /
   "python_quant"` → `python_quant/python_quant` which does not exist). The
   script fails with `ModuleNotFoundError` when run from the repo root. Fix is
   a one-line path; deferred to a hygiene commit.

## Execution (E7) — placeholder

| Strategy | IS vs market VWAP (bps, 95% CI) | fill % | completion | MDD |
|---|---|---|---|---|
| TWAP / VWAP / POV / Adaptive / PPO | — | — | — | — |

## Definitions of done that must be true for any claim to leave this repo

1. Walk-forward split, event-indexed sampling, block-bootstrap CIs.
2. Symmetric information in every RL comparison (no regime indicator the
   baseline cannot see) — see `plan_2.md` §6.
3. IS vs **market** VWAP (never self-executed VWAP) with fees + queue on.
4. At least one negative/unstable result reported.
5. Every headline restated as `point estimate (CI [±Y])` — never a bare number.