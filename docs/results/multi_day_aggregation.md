# Multi-Day ITCH Empirical Aggregation Report

> **Policy**: Aggregated 1 calendar sessions across 2 symbol(s). Between-day dispersion reflects genuine session variation; cross-day sample size is small (< 5 days); metrics remain descriptive rather than asymptotic.

## Summary Overview

- **Days analyzed**: 1 (12302019)
- **Symbols**: AAPL, QQQ
- **Total regular-session events**: 3,693,390
- **Total standing orders tracked (E5)**: 1,932,619
- **Total passive fills analyzed (E6)**: 64,019

## Per-Day Session Breakdown

| Day | Symbol | Coverage | Regular Events | Rank IC (h=5) | Rank IC (h=10) | Orders | KM P(fill 50) | Adverse Drift (h=5) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 12302019 | AAPL | full_day | 1,484,259 | 0.2201 | 0.2381 | 791,477 | 0.0335 | -50.04 |
| 12302019 | QQQ | full_day | 2,209,131 | 0.2961 | 0.3751 | 1,141,142 | 0.0120 | -25.72 |

## Cross-Day Signal Metrics (LOB Imbalance Rank IC)

| Horizon (h) | Sessions | Unweighted Mean | Weighted Mean | Between-Day Std | 95% Bootstrap CI |
|---:|---:|---:|---:|---:|---|
| 1 | 2 | 0.1472 | 0.1487 | 0.0109 | [0.1395, 0.1549] |
| 5 | 2 | 0.2581 | 0.2656 | 0.0537 | [0.2201, 0.2961] |
| 10 | 2 | 0.3066 | 0.3201 | 0.0969 | [0.2381, 0.3751] |
| 25 | 2 | 0.3457 | 0.3686 | 0.1651 | [0.2289, 0.4624] |

## Missing / Incomplete Sessions

| Day | Reason / Status |
|---|---|
| 02282019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 04302019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 05312019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 06282019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 09302019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 11292019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 12312019 | HTTP 404: Tape permanently retired on emi.nasdaq.com |
| 01302019 | Bandwidth-bound: 3.8 GB raw tape requires ~4.5h download at current 238 KB/s |
| 01302020 | Bandwidth-bound: 3.5 GB raw tape requires ~4.2h download at current 238 KB/s |
| 03272019 | Bandwidth-bound: 3.6 GB raw tape requires ~4.3h download at current 238 KB/s |
| 07302019 | Bandwidth-bound: 3.7 GB raw tape requires ~4.4h download at current 238 KB/s |
| 08302019 | Bandwidth-bound: 3.5 GB raw tape requires ~4.2h download at current 238 KB/s |
| 10302019 | Bandwidth-bound: 3.9 GB raw tape requires ~4.6h download at current 238 KB/s |
