# Nexus-LOB — Research Report (Part 2)

> **Status:** skeleton — Phase 1 (research spine) landed 2026-09-12; E1–E4
> results below are placeholders until `scripts/run_research.py` is wired to a
> real tape (Phase 4). Ground rules: see `plan_2.md` §0. This document reports
> *only* measured numbers and names every negative result.

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

## Negative results so far

- *None yet — Phase 1 built the harness; a project with no negative results is
  research theater, so this section is expected to grow.*

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