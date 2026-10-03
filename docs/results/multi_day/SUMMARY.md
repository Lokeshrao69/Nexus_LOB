# Reconciled Multi-Day ITCH Microstructure Empirical Report (15 Sessions)

> **Scope & Sample Provenance.** This empirical synthesis aggregates the **15 fully completed, validated NASDAQ TotalView-ITCH 5.0 sessions** across AAPL and QQQ (2018–2025). These 15 sessions represent catalogued public sample days published on `emi.nasdaq.com/ITCH/`, **not a random sample of trading days**. Seven historical catalogued dates (from 2018) are permanently retired (HTTP 404), and three partial sessions (AAPL only: 12/08/2025, 12/10/2025, 12/11/2025) are strictly excluded. One session (11/28/2025) was an early-close trading day (13:00 ET); all headline metrics are reported both with and without this session.

> **Statistical ≠ Tradable.** A statistically significant predictive relationship or adverse-selection signature does not imply executable trading alpha. Economic significance depends on spread, exchange fees, queue position, latency, and market impact.

> **Methodology & Independence Policy.** No event rows are pooled across calendar dates. Per-session estimates are computed independently by the single-day pipeline on chronological order-book time. All forecast horizons are indexed in **order-book update events** (h = 1, 5, 10, 25 events). All evaluation is out-of-sample on a walk-forward **60% train / 20% validation / 20% test** split with a 25-event embargo gap to prevent lookahead leakage. Cross-session confidence intervals use a date-level cluster bootstrap (2,000 resamples of trading dates).

> **Pipeline Fingerprint & Provenance.** All 15 sessions were generated at commit `69534ab` (pipeline fingerprint `17a5ba28f5d8bdb37aecc26b1499c3d2711b26226df789d6982fd46b018a7090`, computed on raw file bytes, so it depends on line endings). `python_quant/` has changed since (run_research.py gained a clock-time collector (`collect_clock_fills`) that its own docstring says does not change E1–E6; clock-time markouts, dashboard, RL regime fix, validation-harness tolerances). The E1–E6 code paths (itch_parser, replay, features, labels, dataset, experiments, queue_dynamics, adverse_selection) are unchanged, so the current-tree fingerprint differs (`959ac48653d79a2688e506b11aa4d214982936ca15e396a2c4396aaddd5ede49` with CRLF, `691bbcbaa2fe4472ab6348374d9a413eff6a1f04b118cced9c1711fd13cec43e` LF-normalized).

## 1. Overview & Dataset Provenance

- **Completed trading days analyzed**: 15 (01302019, 01302020, 03272019, 05302019, 07132021, 07302019, 08132021, 08302019, 10182019, 10302019, 11282025, 12132018, 12142018, 12302019, 12312018)
- **Symbols**: AAPL, QQQ (30 total single-day symbol sessions)
- **Total regular-session events analyzed**: 65,948,055
- **Total standing orders tracked (E5)**: 35,282,368
- **Total passive fills analyzed (E6)**: 1,295,257
- **Permanently retired 404 dates**: 7 (`01302018`, `03292018`, `05302018`, `07302018`, `08302018`, `10302018`, `12282018`)
- **Excluded partial dates**: 3 (`12082025`, `12102025`, `12112025` — AAPL only, QQQ incomplete)
- **Pending / un-downloaded dates**: 5 (`12092025`, `12122025`, `05152026`, `05182026`, `06122026`)

## 2. Audit of Single-Session Claims vs 15-Day Reality

The earlier repository writeups (based solely on the 12/30/2019 tape) made three specific numerical claims. Here is the honest audit of how those claims generalize across the 15 verified trading sessions:

| Headline Claim | Single-Session Writeup (12/30/2019) | 15-Day Multi-Session Reality | Holds Across Days? | Empirical Commentary |
|---|---|---|:---:|---|
| **E6 Passive Adverse Selection** | *"96–99% of filled passive orders are run over by price shortly after fill"* | **44.42% – 74.67%** unconditional against (mean: AAPL 61.88%, QQQ 50.55%); conditional: 87.84% – 97.36% (mean: 95.00%) | **NO** | The single-session writeup reported the conditional rate as if it applied to all fills: 22.01%–53.98% of fills experience zero mid move at h=5. Unconditionally, the mid moves against the passive fill 50.55% (QQQ) to 61.88% (AAPL) of the time. Conditional on the mid moving, adverse selection is 94.74% (AAPL) and 95.26% (QQQ), ranging from 87.84% to 97.36% across sessions, never reaching the claimed 99%. |
| **E5 Fill Model Calibration** | *"calibration slope 1.03–1.09"* | **0.669 – 1.492** (mean: **1.035**, median: **1.047**, std: 0.160) | **NO** | The 1.03–1.09 range was an artifact of 12/30/2019 (AAPL 1.034, QQQ 1.088). Cross-day dispersion is much wider: QQQ slopes drop as low as 0.669 (08/13/2021) and AAPL reaches 1.492 (12/14/2018). While the pooled mean (1.035) is near 1.0, individual day calibration varies significantly. |
| **E1 Rank IC by Horizon** | *"Rank IC rises with horizon: ≈0.14 (h=1) → ≈0.23 (h=25) on AAPL, ≈0.16 → ≈0.46 on QQQ"* | **0.113 – 0.475** (AAPL: 0.153 → 0.240, QQQ: 0.183 → 0.390) | **PARTIALLY** | The direction is 100% consistent (15/15 sessions positive for both symbols across all horizons). However, on AAPL the IC peaks at h=10 (mean 0.254) and plateaus/softens at h=25 (mean 0.240), rather than monotonically increasing through h=25. |

## 3. E1 LOB Imbalance Rank IC — Horizon & Cross-Day Breakdown

All rank ICs are measured on the **out-of-sample test split** (last 20% of trading day time, walk-forward). Horizons are in **order-book update events**.

### All 15 Sessions (including 11/28/2025 early close)

| Symbol | Horizon (events) | N Sessions | Mean IC | Median | Between-Day SD | 95% Bootstrap CI | Min | Max | Positive Days |
|---|---:|---:|---:|---:|---:|---|---:|---:|:---:|
| AAPL | 1 | 15 | 0.1530 | 0.1552 | 0.0240 | [0.1416, 0.1649] | 0.1125 | 0.1885 | 15/15 (100%) |
| AAPL | 5 | 15 | 0.2353 | 0.2390 | 0.0504 | [0.2115, 0.2594] | 0.1662 | 0.3390 | 15/15 (100%) |
| AAPL | 10 | 15 | 0.2541 | 0.2427 | 0.0667 | [0.2235, 0.2883] | 0.1545 | 0.3985 | 15/15 (100%) |
| AAPL | 25 | 15 | 0.2399 | 0.2141 | 0.0867 | [0.2022, 0.2858] | 0.1267 | 0.4305 | 15/15 (100%) |
| QQQ | 1 | 15 | 0.1833 | 0.1773 | 0.0298 | [0.1697, 0.1978] | 0.1461 | 0.2364 | 15/15 (100%) |
| QQQ | 5 | 15 | 0.3118 | 0.3112 | 0.0235 | [0.3005, 0.3235] | 0.2615 | 0.3575 | 15/15 (100%) |
| QQQ | 10 | 15 | 0.3606 | 0.3686 | 0.0243 | [0.3488, 0.3716] | 0.3023 | 0.3893 | 15/15 (100%) |
| QQQ | 25 | 15 | 0.3902 | 0.3996 | 0.0661 | [0.3591, 0.4223] | 0.2935 | 0.4749 | 15/15 (100%) |

### 14 Sessions (Excluding 11/28/2025 Early-Close Session)

> **Note on 11/28/2025**: The day after US Thanksgiving has a 13:00 ET early close (3.5h regular session vs standard 6.5h). As expected, removing this shorter, thinner session slightly increases mean rank IC and narrows cross-day standard deviation:

| Symbol | Horizon (events) | N Sessions | Mean IC (14-day) | Diff vs 15-day | Between-Day SD | 95% Bootstrap CI | Min | Max | Positive Days |
|---|---:|---:|---:|---:|---:|---|---:|---:|:---:|
| AAPL | 1 | 14 | 0.1559 | +0.0029 | 0.0220 | [0.1450, 0.1667] | 0.1237 | 0.1885 | 14/14 (100%) |
| AAPL | 5 | 14 | 0.2395 | +0.0042 | 0.0495 | [0.2156, 0.2630] | 0.1662 | 0.3390 | 14/14 (100%) |
| AAPL | 10 | 14 | 0.2581 | +0.0040 | 0.0674 | [0.2257, 0.2911] | 0.1545 | 0.3985 | 14/14 (100%) |
| AAPL | 25 | 14 | 0.2428 | +0.0029 | 0.0892 | [0.2012, 0.2893] | 0.1267 | 0.4305 | 14/14 (100%) |
| QQQ | 1 | 14 | 0.1860 | +0.0027 | 0.0290 | [0.1717, 0.2000] | 0.1483 | 0.2364 | 14/14 (100%) |
| QQQ | 5 | 14 | 0.3154 | +0.0036 | 0.0196 | [0.3061, 0.3256] | 0.2901 | 0.3575 | 14/14 (100%) |
| QQQ | 10 | 14 | 0.3648 | +0.0042 | 0.0189 | [0.3547, 0.3739] | 0.3190 | 0.3893 | 14/14 (100%) |
| QQQ | 25 | 14 | 0.3968 | +0.0066 | 0.0633 | [0.3662, 0.4284] | 0.2935 | 0.4749 | 14/14 (100%) |

## 4. E6 Adverse Selection — Three-Way Outcome Split & Post-Fill Drift (15 Sessions)

> **Definition of P(adverse)**: In this research, conditional P(adverse) is defined as the fraction of fills where the mid moved against the passive order (s · Δmid < 0), conditional on the mid moving (Δmid ≠ 0), with zero-move fills strictly excluded from the denominator. Unconditional P(adverse) retains all fills in the denominator, counting zero-move fills as non-adverse.

> **Price Units**: In NASDAQ TotalView-ITCH 5.0, price values are fixed-point integers in units of $0.0001 (1/100th of a cent, or 0.01 standard 1-cent tick). For example, an adverse drift of -46.20 price units ($0.0001) corresponds to -0.462 cents (-$0.00462, or 0.462 of a standard 1-cent tick).

### Three-Way Outcome Split of Passive Fills (h = 1, 5, 25 events)

| Symbol | Horizon (events) | Moved Against (Unconditional P(adv)) [Main] | Did Not Move (Zero Drift Fraction) | Moved In Favor | Conditional P(adv) [Secondary] |
|---|---:|---:|---:|---:|---:|
| AAPL | 1 | **32.70%** (18.14%–44.78%) | 66.08% (53.79%–81.63%) | 1.21% (0.23%–3.74%) | 96.51% (88.93%–98.77%) |
| AAPL | 5 | **61.88%** (46.02%–74.67%) | 34.67% (22.01%–51.77%) | 3.44% (1.69%–7.68%) | 94.74% (87.84%–97.36%) |
| AAPL | 25 | **71.90%** (65.55%–77.49%) | 18.63% (14.49%–26.09%) | 9.46% (4.68%–15.50%) | 88.42% (80.88%–93.76%) |
| QQQ | 1 | **26.04%** (21.64%–41.12%) | 73.36% (57.71%–77.94%) | 0.60% (0.28%–1.24%) | 97.82% (95.47%–98.79%) |
| QQQ | 5 | **50.55%** (44.42%–65.95%) | 46.91% (30.89%–53.98%) | 2.54% (1.24%–3.73%) | 95.26% (92.71%–97.36%) |
| QQQ | 25 | **66.61%** (63.10%–70.40%) | 26.25% (21.14%–32.28%) | 7.13% (4.42%–10.65%) | 90.40% (86.17%–93.71%) |

> *Footnote on Three-Way Counts: Each session JSON stores total valid fills N, unconditional rate P(adv), and conditional rate P(adv | Δmid ≠ 0). Category counts are derived as N_against = round(N · P_uncond), N_zero = round(N · (1 - P_uncond / P_cond)), and N_favor = N - N_against - N_zero.*

### Signed Post-Fill Drift & Newey–West Statistics (15 Sessions)

| Metric | Symbol | Mean | Median | Between-Day SD | 95% Bootstrap CI | Min | Max | Hypothesised Sign Consistency | Individually Significant Sessions (|t| > 1.96) |
|---|---|---:|---:|---:|---|---:|---:|:---:|:---:|
| Signed mid drift at h=5 (price units ($0.0001)) | AAPL | -46.20 | -46.95 | 11.99 | [-51.87, -39.89] | -67.05 | -24.84 | 15/15 | — |
| Signed mid drift at h=5 (price units ($0.0001)) | QQQ | -28.68 | -27.50 | 5.47 | [-31.70, -26.48] | -46.52 | -24.10 | 15/15 | — |
| Post − pre matched drift h=5 (price units ($0.0001)) | AAPL | -45.17 | -45.55 | 11.34 | [-50.72, -39.59] | -64.89 | -25.70 | 15/15 | — |
| Post − pre matched drift h=5 (price units ($0.0001)) | QQQ | -29.10 | -27.57 | 5.65 | [-32.14, -26.81] | -47.47 | -24.01 | 15/15 | — |
| Newey–West t-statistic at h=5 | AAPL | -123.65 | -121.84 | 25.00 | [-135.33, -111.19] | -155.35 | -75.46 | 15/15 | 15/15 (100%) |
| Newey–West t-statistic at h=5 | QQQ | -112.70 | -109.77 | 31.26 | [-129.38, -97.98] | -162.45 | -64.81 | 15/15 | 15/15 (100%) |

## 5. E2, E3, E4 & E5 Cross-Day Microstructure Results

| Experiment | Description | Symbol | Mean | Median | Between-Day SD | 95% Bootstrap CI | Min | Max |
|---|---|---|---:|---:|---:|---|---:|---:|
| **E2** | Microprice rank IC (h=5) | AAPL | 0.2257 | 0.2235 | 0.0599 | [0.1959, 0.2553] | 0.1238 | 0.3538 |
| **E2** | Microprice rank IC (h=5) | QQQ | 0.3098 | 0.3112 | 0.0259 | [0.2956, 0.3214] | 0.2354 | 0.3480 |
| **E3** | Order-level OFI (w=20, h=5) | AAPL | 0.1421 | 0.1382 | 0.0293 | [0.1281, 0.1562] | 0.0856 | 0.1910 |
| **E3** | L2-ladder OFI approximation (h=5) | AAPL | 0.1559 | 0.1601 | 0.0207 | [0.1457, 0.1651] | 0.1134 | 0.1889 |
| **E3** | Order-level OFI (w=20, h=5) | QQQ | 0.1329 | 0.1330 | 0.0148 | [0.1259, 0.1403] | 0.1098 | 0.1577 |
| **E3** | L2-ladder OFI approximation (h=5) | QQQ | 0.1205 | 0.1150 | 0.0277 | [0.1070, 0.1338] | 0.0757 | 0.1697 |
| **E4** | Combined OLS rank IC (h=5) | AAPL | 0.2522 | 0.2596 | 0.0512 | [0.2269, 0.2781] | 0.1852 | 0.3499 |
| **E4** | Combined OLS rank IC (h=5) | QQQ | 0.3236 | 0.3187 | 0.0268 | [0.3099, 0.3364] | 0.2698 | 0.3730 |
| **E5** | Logistic calibration slope | AAPL | 1.122 | 1.100 | 0.159 | [1.048, 1.201] | 0.941 | 1.492 |
| **E5** | Brier skill vs base rate | AAPL | 0.0718 | 0.0704 | 0.0282 | [0.0591, 0.0865] | 0.0311 | 0.1431 |
| **E5** | KM P(fill <= 50 events) | AAPL | 0.0462 | 0.0461 | 0.0108 | [0.0409, 0.0515] | 0.0269 | 0.0655 |
| **E5** | Ever-filled order fraction | AAPL | 6.33% | 5.97% | 1.51% | [5.60%, 7.07%] | 3.72% | 8.88% |
| **E5** | Logistic calibration slope | QQQ | 0.949 | 0.977 | 0.125 | [0.882, 1.009] | 0.669 | 1.133 |
| **E5** | Brier skill vs base rate | QQQ | 0.0297 | 0.0185 | 0.0252 | [0.0189, 0.0437] | 0.0091 | 0.1002 |
| **E5** | KM P(fill <= 50 events) | QQQ | 0.0214 | 0.0178 | 0.0104 | [0.0168, 0.0269] | 0.0120 | 0.0477 |
| **E5** | Ever-filled order fraction | QQQ | 2.43% | 2.09% | 0.99% | [2.01%, 2.95%] | 1.40% | 5.06% |

## 6. Paired Within-Session Comparisons

Evaluated on the exact same test partition per session (a − b):

| Comparison | Description | Pooled Mean Diff | 95% Bootstrap CI | Sessions a > b | Sessions b > a | Conclusion |
|---|---|---:|---|:---:|:---:|---|
| **E2**: Microprice − Imbalance (h=5) | Does microprice add predictive value over simple L1 imbalance? | **-0.0058** | [-0.0115, -0.0009] | 10 | 20 | **No.** Simpler L1 imbalance beats microprice on 20 of 30 symbol-sessions (p < 0.05). On tight 1-tick spreads, microprice adds noise. |
| **E3**: Order OFI − L2 OFI (h=5) | Does order-by-order tracking beat top-of-book L2 approximation? | **-0.0007** | [-0.0129, 0.0109] | 18 | 12 | **Tied.** Order-level OFI is slightly better on QQQ (+0.0123) but worse on AAPL (-0.0138). CI straddles 0. |
| **E4**: Combined − Imbalance (h=5) | Does multi-feature OLS beat single L1 imbalance? | **+0.0143** | [0.0121, 0.0172] | 28 | 2 | **Yes.** Train-fit combination yields modest, robust out-of-sample rank IC gain on 28 of 30 sessions. |
| **E6**: Post-fill − Pre-fill Drift (h=5) | Is post-fill drift worse than matched pre-fill price drift? | **-37.14 price units ($0.0001)** | [-40.79, -33.79] | 0 | 30 | **Yes.** Post-fill drift is worse than matched pre-fill drift on 100% of sessions (all 30 sessions post < pre). |

## 7. Full 15-Session Panel (All 30 Symbol Sessions)

| Day | Sym | Regular Events | Orders (E5) | Passive Fills (E6) | IC h=1 | IC h=5 | IC h=10 | IC h=25 | OLS h=5 | KM P(fill 50) | Ever Filled | Calib Slope | Drift h=5 ($0.0001) | P(adv) h=5 | Post−Pre h=5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12132018 | AAPL | 1,086,473 | 575,145 | 41,268 | 0.124 | 0.169 | 0.177 | 0.158 | 0.185 | 0.0538 | 7.17% | 1.12 | -62.2 | 95.7% | -58.0 |
| 12132018 | QQQ | 3,292,683 | 1,693,367 | 52,772 | 0.236 | 0.358 | 0.371 | 0.333 | 0.373 | 0.0321 | 3.12% | 0.88 | -29.5 | 95.0% | -30.3 |
| 12142018 | AAPL | 1,300,873 | 678,515 | 58,748 | 0.155 | 0.239 | 0.256 | 0.241 | 0.246 | 0.0625 | 8.65% | 1.15 | -49.4 | 94.4% | -47.5 |
| 12142018 | QQQ | 2,343,498 | 1,211,734 | 61,346 | 0.200 | 0.302 | 0.319 | 0.294 | 0.319 | 0.0477 | 5.06% | 0.87 | -32.0 | 94.5% | -32.5 |
| 12312018 | AAPL | 1,267,696 | 660,596 | 39,456 | 0.139 | 0.194 | 0.196 | 0.160 | 0.192 | 0.0453 | 5.97% | 0.97 | -51.3 | 95.6% | -49.8 |
| 12312018 | QQQ | 2,859,411 | 1,480,119 | 57,285 | 0.218 | 0.341 | 0.360 | 0.317 | 0.365 | 0.0377 | 3.87% | 1.02 | -31.0 | 95.4% | -31.2 |
| 01302019 | AAPL | 1,596,503 | 862,485 | 70,104 | 0.171 | 0.240 | 0.243 | 0.214 | 0.260 | 0.0542 | 8.11% | 0.94 | -44.9 | 92.8% | -45.5 |
| 01302019 | QQQ | 3,291,294 | 1,734,462 | 49,171 | 0.177 | 0.326 | 0.386 | 0.426 | 0.330 | 0.0264 | 2.83% | 0.67 | -27.5 | 93.3% | -28.3 |
| 03272019 | AAPL | 1,960,770 | 1,056,435 | 46,715 | 0.163 | 0.273 | 0.302 | 0.287 | 0.285 | 0.0327 | 4.42% | 1.27 | -47.6 | 96.7% | -47.4 |
| 03272019 | QQQ | 4,040,376 | 2,127,129 | 47,507 | 0.165 | 0.313 | 0.389 | 0.469 | 0.329 | 0.0187 | 2.23% | 0.98 | -26.0 | 96.3% | -26.7 |
| 05302019 | AAPL | 1,165,209 | 640,451 | 41,538 | 0.181 | 0.297 | 0.327 | 0.305 | 0.316 | 0.0439 | 6.48% | 0.95 | -43.3 | 97.4% | -42.6 |
| 05302019 | QQQ | 2,608,902 | 1,356,558 | 34,242 | 0.157 | 0.302 | 0.376 | 0.455 | 0.313 | 0.0190 | 2.52% | 0.84 | -24.1 | 96.0% | -24.7 |
| 07302019 | AAPL | 1,214,806 | 640,274 | 39,607 | 0.177 | 0.283 | 0.306 | 0.286 | 0.303 | 0.0461 | 6.18% | 1.09 | -30.9 | 95.1% | -30.4 |
| 07302019 | QQQ | 1,674,517 | 871,970 | 15,602 | 0.148 | 0.290 | 0.369 | 0.475 | 0.302 | 0.0133 | 1.79% | 1.13 | -25.4 | 97.4% | -26.6 |
| 08302019 | AAPL | 1,151,383 | 626,231 | 37,342 | 0.162 | 0.228 | 0.231 | 0.180 | 0.263 | 0.0479 | 5.96% | 1.07 | -45.6 | 92.9% | -43.7 |
| 08302019 | QQQ | 2,276,617 | 1,195,082 | 25,044 | 0.176 | 0.317 | 0.380 | 0.426 | 0.342 | 0.0156 | 2.09% | 1.02 | -27.7 | 96.1% | -27.6 |
| 10182019 | AAPL | 1,636,764 | 868,512 | 44,114 | 0.182 | 0.249 | 0.254 | 0.209 | 0.269 | 0.0463 | 5.08% | 1.49 | -44.7 | 95.3% | -44.5 |
| 10182019 | QQQ | 2,448,505 | 1,262,644 | 23,370 | 0.161 | 0.298 | 0.370 | 0.446 | 0.304 | 0.0138 | 1.85% | 1.10 | -24.5 | 96.5% | -24.0 |
| 10302019 | AAPL | 739,800 | 400,031 | 35,576 | 0.135 | 0.193 | 0.203 | 0.179 | 0.214 | 0.0655 | 8.88% | 1.30 | -47.0 | 87.8% | -44.0 |
| 10302019 | QQQ | 1,864,809 | 978,084 | 20,384 | 0.185 | 0.311 | 0.365 | 0.400 | 0.310 | 0.0157 | 2.08% | 0.84 | -26.3 | 94.1% | -24.6 |
| 12302019 | AAPL | 1,484,259 | 791,477 | 46,357 | 0.140 | 0.220 | 0.238 | 0.229 | 0.255 | 0.0335 | 5.84% | 1.12 | -50.0 | 95.7% | -45.8 |
| 12302019 | QQQ | 2,209,131 | 1,141,142 | 17,662 | 0.155 | 0.296 | 0.375 | 0.462 | 0.298 | 0.0120 | 1.54% | 1.03 | -25.7 | 95.8% | -26.4 |
| 01302020 | AAPL | 1,956,673 | 1,048,496 | 39,131 | 0.131 | 0.166 | 0.155 | 0.127 | 0.185 | 0.0269 | 3.72% | 1.01 | -67.1 | 93.5% | -64.9 |
| 01302020 | QQQ | 4,211,652 | 2,207,984 | 36,004 | 0.206 | 0.316 | 0.340 | 0.332 | 0.331 | 0.0133 | 1.63% | 0.84 | -29.3 | 95.1% | -28.3 |
| 07132021 | AAPL | 2,307,107 | 1,218,116 | 93,502 | 0.189 | 0.339 | 0.398 | 0.431 | 0.350 | 0.0543 | 7.62% | 1.10 | -24.8 | 95.4% | -25.7 |
| 07132021 | QQQ | 4,798,141 | 2,473,392 | 41,609 | 0.234 | 0.341 | 0.362 | 0.338 | 0.350 | 0.0178 | 1.68% | 0.92 | -25.6 | 92.7% | -27.6 |
| 08132021 | AAPL | 1,546,590 | 857,767 | 50,834 | 0.134 | 0.263 | 0.327 | 0.394 | 0.273 | 0.0370 | 5.91% | 1.31 | -25.8 | 96.2% | -26.6 |
| 08132021 | QQQ | 2,297,971 | 1,274,130 | 17,950 | 0.185 | 0.306 | 0.347 | 0.381 | 0.317 | 0.0151 | 1.40% | 1.05 | -29.0 | 95.2% | -30.3 |
| 11282025 | AAPL | 1,212,991 | 793,829 | 39,867 | 0.112 | 0.177 | 0.199 | 0.199 | 0.186 | 0.0432 | 5.01% | 0.97 | -58.4 | 96.5% | -61.3 |
| 11282025 | QQQ | 4,102,651 | 2,556,211 | 71,150 | 0.146 | 0.261 | 0.302 | 0.298 | 0.270 | 0.0225 | 2.78% | 1.04 | -46.5 | 95.4% | -47.5 |
