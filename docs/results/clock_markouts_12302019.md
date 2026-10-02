# Clock-time passive-fill markouts — NASDAQ ITCH 5.0, 12/30/2019

Markouts on the **clock**, from ITCH nanosecond timestamps, over the same passive-fill population as E6 (the first execution of each tracked resting order, regular session only). The terminal mid is the mid of the last book event at or before `fill_ts + H` — never interpolated. Sign is from the passive order's side (`s = +1` resting bid, `s = −1` resting ask), so **negative = adverse**.

**Units.** ITCH prices are integers in **$0.0001**. That is not a tick: AAPL and QQQ both quote on a **$0.01 tick = 100 price units**, so 100 units = 1 tick = 1 cent. Columns headed `$0.0001` are in price units; columns headed `×HS` are multiples of the mean quoted half-spread at the fill.

**Unit of observation.** One aggressive order can fill many resting orders at the same nanosecond; those fills share one aggressor, one price path and one mechanical BBO move, so they are not independent draws. The main tables use one observation per **aggressor event** (same ITCH timestamp, same resting side), and the Newey–West s.e. and bootstrap CIs are computed over events. The per-fill table at the end is the same measurement with the sweep multiplicity left in, which overstates the effective sample size.

**Two reference mids.**

* **before** — the mid of the last book event strictly *before* the event's first message, so the sweep's own BBO move is **included**;
* **after** — the mid once the event's *last* message is applied, so that move is **excluded**. Aggregating to the event is what lets this reference step over the whole sweep rather than only its first message.

`before − after` is the mechanical component of the move: price change caused by the fill itself clearing displayed depth, not by anything the taker knew.

**Realized spread** = `half-spread earned + markout` = `s · (mid[t+H] − P)`, `P` the event's size-weighted execution price. It is the same number under either reference — the reference only moves value between the two terms. Positive means the passive fill made money net of adverse selection.

**Location.** Means over markouts are tail-sensitive, so each horizon reports the mean, the median and the 10 %-each-tail trimmed mean. The mean and the trimmed mean carry 95 % CIs; **the median does not** — mids move on a 50-unit half-tick lattice, so median bootstrap replicates collapse onto a few lattice points and the interval would describe the grid, not the sampling error. The sign-split table is the distribution claim that survives discreteness.

**Dependence.** CIs are moving-block bootstraps whose block is `max(H, 60 s)` of **wall-clock** time, converted to the typical number of observations in such a window — at H = 100 ms a block sized to the horizon would span a second or two and resample moves that are still correlated. The Newey–West lag is the number of later observations inside a typical `H`-window. `effective blocks` is how many such blocks tile the sample, i.e. how much independent information each CI actually rests on; at H = 60 s it is small, and the wide intervals there are honest about that.

`pre_fill` is the same signed move over the `H` *before* the fill, ending strictly before it — the clock analogue of E6's pre-fill control — so `post − pre` nets out a session already trending into the fill.

## AAPL

46,357 passive fills grouped into 29,861 aggressor events (1.55 fills per event). Observations dropped per horizon:

| H | past 16:00 | past tape end | no pre-fill mid | one-sided book | pre-window before 09:30 | pre-window one-sided | usable events |
|---|---|---|---|---|---|---|---|
| 100ms | 0 | 0 | 0 | 0 | 0 | 0 | 29,861 |
| 1s | 20 | 0 | 0 | 0 | 8 | 0 | 29,841 |
| 10s | 142 | 0 | 0 | 0 | 62 | 0 | 29,719 |
| 60s | 467 | 0 | 0 | 0 | 273 | 0 | 29,394 |

### Realized spread — does a passive fill make money?

**This is the headline.** `realized spread = half-spread earned + markout` = `s · (mid[t+H] − P)`, one observation per aggressor event, `P` the event's size-weighted execution price. Reference-invariant, so one row per horizon. Negative means the passive side gave back more to adverse selection than the spread it earned. `×HS` expresses that as a fraction of the half-spread quoted at the fill, which is the scale-free read.

| H | n events | mean ($0.0001) | 95% CI | mean (×HS) | median ($0.0001) † | trimmed mean ($0.0001) | 95% CI | NW t | p | HS earned ($0.0001) | mean markout ($0.0001) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | 29,861 |   -23.32 | [ -27.17,  -19.75] |  -0.22 |     0.00 |   -17.75 | [ -20.50,  -15.23] |   -20.5 |  0.000 |    65.01 |   -88.33 |
| 1s | 29,841 |   -47.81 | [ -55.23,  -40.97] |  -0.45 |   -50.00 |   -43.72 | [ -49.43,  -38.92] |   -20.7 |  0.000 |    64.92 |  -112.73 |
| 10s | 29,719 |   -59.12 | [ -78.29,  -39.44] |  -0.56 |   -50.00 |   -54.98 | [ -69.35,  -39.58] |    -6.9 |  0.000 |    64.29 |  -123.41 |
| 60s | 29,394 |   -90.08 | [-139.53,  -44.44] |  -0.85 |  -100.00 |   -81.28 | [-124.79,  -46.19] |    -3.7 |  0.000 |    64.35 |  -154.43 |

† no CI on the median: medians sit on the 50-unit half-tick lattice, so bootstrap CIs on them degenerate and are not reported — use the sign-split table instead.

### Sign split — the claim that survives the price lattice

Mids move in 50-unit (half-tick) steps, so a median is pinned to a lattice point. How often the outcome lands on each side of zero is not. Shares are over aggressor events, measured from the `after` mid, with block-bootstrap CIs; the three shares in each group sum to 1.

| H | quantity | n | share < 0 | 95% CI | share = 0 | 95% CI | share > 0 | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 100ms | realized spread | 29,861 |  0.462 | [  0.453,   0.470] |  0.171 | [  0.164,   0.178] |  0.367 | [  0.357,   0.375] |
| 100ms | markout | 29,861 |  0.649 | [  0.641,   0.657] |  0.220 | [  0.213,   0.228] |  0.131 | [  0.124,   0.138] |
| 1s | realized spread | 29,841 |  0.516 | [  0.507,   0.525] |  0.106 | [  0.100,   0.112] |  0.377 | [  0.368,   0.386] |
| 1s | markout | 29,841 |  0.635 | [  0.627,   0.642] |  0.120 | [  0.114,   0.125] |  0.246 | [  0.236,   0.254] |
| 10s | realized spread | 29,719 |  0.517 | [  0.508,   0.526] |  0.032 | [  0.029,   0.035] |  0.451 | [  0.441,   0.459] |
| 10s | markout | 29,719 |  0.556 | [  0.546,   0.564] |  0.031 | [  0.029,   0.033] |  0.413 | [  0.405,   0.423] |
| 60s | realized spread | 29,394 |  0.515 | [  0.504,   0.525] |  0.013 | [  0.011,   0.015] |  0.473 | [  0.463,   0.482] |
| 60s | markout | 29,394 |  0.528 | [  0.517,   0.538] |  0.014 | [  0.012,   0.016] |  0.458 | [  0.448,   0.467] |

### Break-even including an assumed maker rebate

**The rebate here is assumed, not measured: maker rebate is an ASSUMED constant per share, not measured from the tape — ITCH carries no fee data.** The values below are illustrative, chosen to bracket a typical 2019 US equity maker rebate, and every figure in this table is conditional on them. A rebate is a constant per share, so it shifts the realized-spread distribution without changing its shape — which is why the share profitable moves as well as the mean.

| H | realized spread ($0.0001) | + 20 rebate | 95% CI | ×HS | share > 0 | + 30 rebate | 95% CI | ×HS | share > 0 |
|---|---|---|---|---|---|---|---|---|---|
| 100ms |   -23.32 |    -3.32 | [  -6.65,    0.50] |  -0.03 |  0.539 |     6.68 | [   2.82,   10.57] |   0.06 |  0.540 |
| 1s |   -47.81 |   -27.81 | [ -34.80,  -21.06] |  -0.26 |  0.485 |   -17.81 | [ -25.01,  -10.72] |  -0.17 |  0.485 |
| 10s |   -59.12 |   -39.12 | [ -56.05,  -18.78] |  -0.37 |  0.483 |   -29.12 | [ -49.06,  -10.49] |  -0.28 |  0.483 |
| 60s |   -90.08 |   -70.08 | [-113.85,  -24.29] |  -0.66 |  0.485 |   -60.08 | [-112.39,  -13.85] |  -0.57 |  0.485 |

### Attribution — where E6's 0.957 goes

Each row changes exactly one thing from the row above, on the **same** fills, so every step is attributable. The statistic is `P(adverse | mid moved)` — E6's own headline — with the unconditional and zero shares beside it.

Row 1 / `after` reproduces E6's published figure, which establishes that **E6 already measured from the post-fill mid** (`mid_history[event_index]`). The walk down the `after` column is therefore the decomposition of E6's number: two steps, the unit of observation and the clock, and they account for the whole change.

The `before` column is a *separate* comparison, not a third step: it shows how much of each row's measured move is the fill's own BBO jump. Both endpoints of the walk are already measured from the same reference, so adding the before/after gap to the two steps would double-count.

| row | unit | clock | ref | n | P(adv\|moved) | P(adv) uncond | share zero | mean markout ($0.0001) |
|---|---|---|---|---|---|---|---|---|
| 1 | per fill | event h=5 | before | 46,357 | ** 0.989** |  0.788 |  0.203 |   -77.47 |
| 1 | per fill | event h=5 | after | 46,357 | ** 0.957** |  0.656 |  0.315 |   -50.04 |
| 2 | per aggressor event | event h=5 | before | 29,861 | ** 0.986** |  0.804 |  0.185 |   -81.96 |
| 2 | per aggressor event | event h=5 | after | 29,861 | ** 0.933** |  0.600 |  0.357 |   -39.38 |
| 3 | per aggressor event | 100ms | before | 29,861 | ** 0.916** |  0.799 |  0.128 |  -130.91 |
| 3 | per aggressor event | 100ms | after | 29,861 | ** 0.832** |  0.649 |  0.220 |   -88.33 |

Steps along the `before` column (these are the decomposition):

| step | what changed | from | to | Δ |
|---|---|---|---|---|
| 1 → 2 | unit of observation |  0.989 |  0.986 |  -0.002 |
| 2 → 3 | clock |  0.986 |  0.916 |  -0.071 |

Sum of steps -0.073 vs total change -0.073 (0.989 → 0.916) — **checks out**.

Steps along the `after` column (these are the decomposition):

| step | what changed | from | to | Δ |
|---|---|---|---|---|
| 1 → 2 | unit of observation |  0.957 |  0.933 |  -0.024 |
| 2 → 3 | clock |  0.933 |  0.832 |  -0.101 |

Sum of steps -0.125 vs total change -0.125 (0.957 → 0.832) — **checks out**.

Within-row `before − after` gap (the fill's own BBO jump, **not** a walk step):

| row | unit | clock | before − after |
|---|---|---|---|
| 1 | per fill | event h=5 |   0.032 |
| 2 | per aggressor event | event h=5 |   0.053 |
| 3 | per aggressor event | 100ms |   0.084 |

### Markout

One observation per aggressor event. † median: no CI (lattice).

| H | ref | n | mean ($0.0001) | 95% CI | median ($0.0001) † | trimmed mean ($0.0001) | 95% CI | mean (×HS) | NW t | p | NW lag | block (obs) | eff. blocks | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | before | 29,861 |  -130.91 | [-135.76, -126.73] |  -100.00 |  -119.22 | [-122.54, -116.26] |  -1.23 |  -113.4 |  0.000 | 1 | 80 | 374 |  0.799 | 0.128 |  0.916 |
| 100ms | after | 29,861 |   -88.33 | [ -92.75,  -84.51] |   -50.00 |   -77.32 | [ -80.35,  -74.79] |  -0.83 |   -77.8 |  0.000 | 1 | 80 | 374 |  0.649 | 0.220 |  0.832 |
| 1s | before | 29,841 |  -155.31 | [-163.16, -147.83] |  -150.00 |  -146.92 | [-152.75, -141.64] |  -1.46 |   -67.2 |  0.000 | 2 | 80 | 374 |  0.724 | 0.088 |  0.794 |
| 1s | after | 29,841 |  -112.73 | [-120.81, -105.12] |  -100.00 |  -104.43 | [-110.01,  -99.28] |  -1.06 |   -48.9 |  0.000 | 2 | 80 | 374 |  0.635 | 0.120 |  0.721 |
| 10s | before | 29,719 |  -165.99 | [-184.86, -147.35] |  -150.00 |  -158.45 | [-172.94, -142.12] |  -1.57 |   -19.5 |  0.000 | 14 | 80 | 372 |  0.582 | 0.031 |  0.600 |
| 10s | after | 29,719 |  -123.41 | [-142.22, -104.72] |  -100.00 |  -116.00 | [-130.10,  -99.90] |  -1.17 |   -14.5 |  0.000 | 14 | 80 | 372 |  0.556 | 0.031 |  0.573 |
| 60s | before | 29,394 |  -197.18 | [-246.62, -151.80] |  -200.00 |  -183.69 | [-227.65, -148.67] |  -1.87 |    -8.1 |  0.000 | 79 | 80 | 368 |  0.540 | 0.012 |  0.547 |
| 60s | after | 29,394 |  -154.43 | [-203.58, -109.24] |  -150.00 |  -141.42 | [-185.55, -106.66] |  -1.46 |    -6.4 |  0.000 | 79 | 80 | 368 |  0.528 | 0.014 |  0.536 |

### Reversion check — mechanical or informational?

This is the test that discriminates. The `before − after` gap cannot: it equals `s · (mid_after − mid_before)`, which involves neither the terminal mid nor `H`, so it is **constant across horizons by construction** and tests nothing. What does discriminate is whether the move comes back.

| H | markout from the `after` mid ($0.0001) | 95% CI |
|---|---|---|
| 100ms |   -88.33 | [ -92.75,  -84.51] |
| 1s |  -112.73 | [-120.81, -105.12] |
| 10s |  -123.41 | [-142.22, -104.72] |
| 60s |  -154.43 | [-203.58, -109.24] |

From 100ms to 60s the markout goes -88.33 → -154.43, so the impact is **permanent**: the markout does not shrink with horizon, so the price does not come back. A purely mechanical dislocation would revert once displaced depth was replenished; this does not.

### Who loses, and by how much

| H | share losing | share winning | mean loss when losing ($0.0001) | mean gain when winning ($0.0001) |
|---|---|---|---|---|
| 100ms |  0.462 |  0.367 |  -145.33 |   119.47 |
| 1s |  0.516 |  0.377 |  -250.70 |   216.18 |
| 10s |  0.517 |  0.451 |  -641.66 |   605.33 |
| 60s |  0.515 |  0.473 | -1516.50 |  1461.59 |

About **half** of aggressor events lose and about half win; the mean is negative because **losers lose more than winners win**. This is not a minority of bad fills dragging an otherwise profitable book — it is a roughly symmetric hit rate with asymmetric magnitudes.

### Bootstrap blocks and dependence

| H | NW lag (events) | events per H-window | block (clock) | block (events) | effective blocks |
|---|---|---|---|---|---|
| 100ms | 1 | 1 | 60s | 80 | 374 |
| 1s | 2 | 3 | 60s | 80 | 374 |
| 10s | 14 | 15 | 60s | 80 | 372 |
| 60s | 79 | 80 | 60s | 80 | 368 |

### Pre-fill control and the bid/ask split

`pre` is the signed mid move over the `H` before the fill, ending strictly before it; `post − pre` is the fill-conditional excess. A session-wide trend shows up in `pre`; adverse selection should not.

| H | ref | group | n | post ($0.0001) | 95% CI | pre ($0.0001) | 95% CI | post − pre ($0.0001) | 95% CI | NW t | adverse | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | before | all | 29,861 |  -130.91 | [-135.76, -126.73] |   -52.09 | [ -56.73,  -47.71] |   -78.82 | [ -82.54,  -75.18] |   -46.1 |  0.799 |  0.916 |
| 100ms | before | side=bid | 14,064 |  -143.59 | [-151.32, -134.87] |   -61.11 | [ -74.57,  -49.69] |   -82.48 | [ -91.05,  -72.78] |   -28.3 |  0.812 |  0.924 |
| 100ms | before | side=ask | 15,797 |  -119.62 | [-125.32, -114.05] |   -44.07 | [ -53.61,  -34.52] |   -75.56 | [ -84.72,  -67.25] |   -29.5 |  0.787 |  0.909 |
| 100ms | after | all | 29,861 |   -88.33 | [ -92.75,  -84.51] |   -52.09 | [ -56.73,  -47.71] |   -36.24 | [ -39.84,  -32.51] |   -21.4 |  0.649 |  0.832 |
| 100ms | after | side=bid | 14,064 |  -100.82 | [-108.68,  -92.18] |   -61.11 | [ -74.57,  -49.69] |   -39.72 | [ -48.60,  -29.58] |   -13.7 |  0.668 |  0.848 |
| 100ms | after | side=ask | 15,797 |   -77.21 | [ -82.61,  -71.74] |   -44.07 | [ -53.61,  -34.52] |   -33.15 | [ -42.28,  -24.89] |   -13.0 |  0.631 |  0.817 |
| 1s | before | all | 29,841 |  -155.31 | [-163.16, -147.83] |   -84.56 | [ -92.38,  -77.64] |   -70.71 | [ -77.87,  -63.67] |   -21.7 |  0.724 |  0.794 |
| 1s | before | side=bid | 14,052 |  -179.54 | [-192.47, -164.61] |   -89.37 | [-109.03,  -69.30] |   -90.08 | [-106.46,  -70.90] |   -16.0 |  0.748 |  0.813 |
| 1s | before | side=ask | 15,789 |  -133.75 | [-149.96, -119.82] |   -80.28 | [ -98.74,  -62.71] |   -53.47 | [ -70.81,  -37.43] |    -9.9 |  0.703 |  0.777 |
| 1s | after | all | 29,841 |  -112.73 | [-120.81, -105.12] |   -84.56 | [ -92.38,  -77.64] |   -28.16 | [ -35.23,  -20.89] |    -8.7 |  0.635 |  0.721 |
| 1s | after | side=bid | 14,052 |  -136.76 | [-149.24, -121.96] |   -89.37 | [-109.03,  -69.30] |   -47.37 | [ -63.72,  -27.96] |    -8.4 |  0.661 |  0.746 |
| 1s | after | side=ask | 15,789 |   -91.34 | [-107.03,  -77.58] |   -80.28 | [ -98.74,  -62.71] |   -11.06 | [ -28.32,    4.94] |    -2.1 |  0.611 |  0.698 |
| 10s | before | all | 29,719 |  -165.99 | [-184.86, -147.35] |  -124.25 | [-144.38, -106.93] |   -40.73 | [ -60.86,  -19.99] |    -3.4 |  0.582 |  0.600 |
| 10s | before | side=bid | 14,000 |  -195.91 | [-248.50, -142.19] |  -122.65 | [-193.39,  -49.25] |   -67.79 | [-127.60,   -7.95] |    -2.3 |  0.589 |  0.607 |
| 10s | before | side=ask | 15,719 |  -139.35 | [-185.69,  -91.19] |  -125.67 | [-196.82,  -65.61] |   -16.66 | [ -69.79,   39.23] |    -0.6 |  0.576 |  0.594 |
| 10s | after | all | 29,719 |  -123.41 | [-142.22, -104.72] |  -124.25 | [-144.38, -106.93] |     1.81 | [ -18.33,   22.89] |     0.1 |  0.556 |  0.573 |
| 10s | after | side=bid | 14,000 |  -153.14 | [-205.66,  -99.90] |  -122.65 | [-193.39,  -49.25] |   -25.11 | [ -85.09,   35.26] |    -0.8 |  0.562 |  0.579 |
| 10s | after | side=ask | 15,719 |   -96.93 | [-143.50,  -48.85] |  -125.67 | [-196.82,  -65.61] |    25.76 | [ -28.32,   81.62] |     0.9 |  0.551 |  0.568 |
| 60s | before | all | 29,394 |  -197.18 | [-246.62, -151.80] |  -170.08 | [-227.25, -112.74] |   -23.12 | [ -93.89,   39.71] |    -0.7 |  0.540 |  0.547 |
| 60s | before | side=bid | 13,843 |  -199.38 | [-439.40,   30.15] |  -188.02 | [-425.81,   87.90] |     9.84 | [-289.92,  293.41] |     0.1 |  0.519 |  0.525 |
| 60s | before | side=ask | 15,551 |  -195.22 | [-425.14,    3.90] |  -154.25 | [-423.31,   90.19] |   -52.20 | [-349.77,  259.04] |    -0.3 |  0.559 |  0.566 |
| 60s | after | all | 29,394 |  -154.43 | [-203.58, -109.24] |  -170.08 | [-227.25, -112.74] |    19.55 | [ -51.00,   82.57] |     0.6 |  0.528 |  0.536 |
| 60s | after | side=bid | 13,843 |  -156.43 | [-396.46,   73.20] |  -188.02 | [-425.81,   87.90] |    52.72 | [-247.29,  336.90] |     0.4 |  0.507 |  0.514 |
| 60s | after | side=ask | 15,551 |  -152.65 | [-382.96,   45.67] |  -154.25 | [-423.31,   90.19] |    -9.72 | [-307.60,  300.82] |    -0.1 |  0.547 |  0.555 |

### Breakdowns

**Breakdowns at H = 100ms**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 14,064 |  -143.59 | [-151.32, -134.87] |  -1.34 |   -35.45 | [ -43.04,  -27.32] |   -73.1 |  0.812 | 0.121 |  0.924 |
| side=bid | after | 14,064 |  -100.82 | [-108.68,  -92.18] |  -0.94 |   -35.45 | [ -43.04,  -27.32] |   -52.2 |  0.668 | 0.212 |  0.848 |
| side=ask | before | 15,797 |  -119.62 | [-125.32, -114.05] |  -1.13 |   -12.53 | [ -17.41,   -7.26] |   -78.7 |  0.787 | 0.133 |  0.909 |
| side=ask | after | 15,797 |   -77.21 | [ -82.61,  -71.74] |  -0.73 |   -12.53 | [ -17.41,   -7.26] |   -51.7 |  0.631 | 0.227 |  0.817 |
| queue=front | before | 29,804 |  -131.08 | [-135.55, -127.00] |  -1.24 |   -23.34 | [ -27.51,  -19.18] |  -113.4 |  0.799 | 0.127 |  0.916 |
| queue=front | after | 29,804 |   -88.42 | [ -92.61,  -84.54] |  -0.83 |   -23.34 | [ -27.51,  -19.18] |   -77.8 |  0.649 | 0.220 |  0.832 |
| queue=mid | before | 34 |   -39.71 | [ -39.71,  -39.71] |  -0.32 |   -10.65 | [ -10.65,  -10.65] |    -3.0 |  0.500 | 0.353 |  0.773 |
| queue=mid | after | 34 |   -39.71 | [ -39.71,  -39.71] |  -0.32 |   -10.65 | [ -10.65,  -10.65] |    -3.0 |  0.500 | 0.353 |  0.773 |
| size=small | before | 29,138 |  -128.98 | [-133.02, -125.36] |  -1.22 |   -24.08 | [ -27.84,  -20.52] |  -111.9 |  0.797 | 0.129 |  0.916 |
| size=small | after | 29,138 |   -87.72 | [ -91.78,  -84.27] |  -0.83 |   -24.08 | [ -27.84,  -20.52] |   -77.1 |  0.647 | 0.223 |  0.833 |
| size=large | before | 723 |  -208.78 | [-235.02, -184.99] |  -1.62 |     7.05 | [ -10.57,   24.96] |   -23.4 |  0.871 | 0.059 |  0.926 |
| size=large | after | 723 |  -112.86 | [-137.98,  -90.17] |  -0.87 |     7.05 | [ -10.57,   24.96] |   -13.4 |  0.700 | 0.115 |  0.791 |
| emptied_touch=yes | before | 18,823 |  -150.44 | [-155.39, -144.90] |  -1.41 |   -40.64 | [ -44.78,  -36.17] |  -104.0 |  0.874 | 0.063 |  0.933 |
| emptied_touch=yes | after | 18,823 |   -82.89 | [ -87.83,  -77.68] |  -0.78 |   -40.64 | [ -44.78,  -36.17] |   -57.7 |  0.635 | 0.210 |  0.804 |
| emptied_touch=no | before | 11,038 |   -97.61 | [-102.68,  -91.95] |  -0.93 |     6.21 | [   1.59,   11.85] |   -57.9 |  0.671 | 0.238 |  0.881 |
| emptied_touch=no | after | 11,038 |   -97.61 | [-102.68,  -91.95] |  -0.93 |     6.21 | [   1.59,   11.85] |   -57.9 |  0.671 | 0.238 |  0.881 |

**Breakdowns at H = 1s**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 14,052 |  -179.54 | [-192.47, -164.61] |  -1.68 |   -71.53 | [ -84.59,  -56.77] |   -45.8 |  0.748 | 0.080 |  0.813 |
| side=bid | after | 14,052 |  -136.76 | [-149.24, -121.96] |  -1.28 |   -71.53 | [ -84.59,  -56.77] |   -35.1 |  0.661 | 0.114 |  0.746 |
| side=ask | before | 15,789 |  -133.75 | [-149.96, -119.82] |  -1.27 |   -26.70 | [ -40.67,  -13.34] |   -37.0 |  0.703 | 0.094 |  0.777 |
| side=ask | after | 15,789 |   -91.34 | [-107.03,  -77.58] |  -0.87 |   -26.70 | [ -40.67,  -13.34] |   -25.4 |  0.611 | 0.125 |  0.698 |
| queue=front | before | 29,784 |  -155.57 | [-163.20, -147.82] |  -1.47 |   -47.91 | [ -54.99,  -41.12] |   -67.2 |  0.725 | 0.088 |  0.794 |
| queue=front | after | 29,784 |  -112.90 | [-120.49, -105.37] |  -1.07 |   -47.91 | [ -54.99,  -41.12] |   -48.9 |  0.635 | 0.120 |  0.721 |
| queue=mid | before | 34 |    29.41 | [  29.41,   29.41] |   0.24 |    58.46 | [  58.46,   58.46] |     0.6 |  0.412 | 0.176 |  0.500 |
| queue=mid | after | 34 |    29.41 | [  29.41,   29.41] |   0.24 |    58.46 | [  58.46,   58.46] |     0.6 |  0.412 | 0.176 |  0.500 |
| size=small | before | 29,119 |  -154.42 | [-162.57, -146.92] |  -1.46 |   -49.60 | [ -58.05,  -42.80] |   -66.4 |  0.724 | 0.088 |  0.794 |
| size=small | after | 29,119 |  -113.16 | [-121.43, -105.59] |  -1.07 |   -49.60 | [ -58.05,  -42.80] |   -48.8 |  0.636 | 0.121 |  0.723 |
| size=large | before | 722 |  -191.14 | [-229.79, -156.92] |  -1.48 |    24.52 | [  -4.57,   50.97] |   -12.4 |  0.733 | 0.072 |  0.790 |
| size=large | after | 722 |   -95.15 | [-130.23,  -65.01] |  -0.74 |    24.52 | [  -4.57,   50.97] |    -6.3 |  0.594 | 0.087 |  0.651 |
| emptied_touch=yes | before | 18,813 |  -171.85 | [-180.95, -162.63] |  -1.61 |   -62.12 | [ -70.67,  -53.66] |   -61.4 |  0.761 | 0.064 |  0.813 |
| emptied_touch=yes | after | 18,813 |  -104.31 | [-113.50,  -95.41] |  -0.98 |   -62.12 | [ -70.67,  -53.66] |   -37.3 |  0.619 | 0.115 |  0.699 |
| emptied_touch=no | before | 11,028 |  -127.10 | [-137.49, -116.69] |  -1.21 |   -23.40 | [ -32.24,  -15.04] |   -36.4 |  0.662 | 0.128 |  0.759 |
| emptied_touch=no | after | 11,028 |  -127.10 | [-137.49, -116.69] |  -1.21 |   -23.40 | [ -32.24,  -15.04] |   -36.4 |  0.662 | 0.128 |  0.759 |

**Breakdowns at H = 10s**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 14,000 |  -195.91 | [-248.50, -142.19] |  -1.85 |   -88.53 | [-140.12,  -35.03] |    -9.1 |  0.589 | 0.031 |  0.607 |
| side=bid | after | 14,000 |  -153.14 | [-205.66,  -99.90] |  -1.44 |   -88.53 | [-140.12,  -35.03] |    -7.1 |  0.562 | 0.031 |  0.579 |
| side=ask | before | 15,719 |  -139.35 | [-185.69,  -91.19] |  -1.33 |   -32.92 | [ -80.22,   14.36] |    -7.0 |  0.576 | 0.031 |  0.594 |
| side=ask | after | 15,719 |   -96.93 | [-143.50,  -48.85] |  -0.92 |   -32.92 | [ -80.22,   14.36] |    -4.8 |  0.551 | 0.031 |  0.568 |
| queue=front | before | 29,662 |  -166.62 | [-184.79, -146.97] |  -1.58 |   -59.59 | [ -77.71,  -39.50] |   -19.5 |  0.582 | 0.031 |  0.601 |
| queue=front | after | 29,662 |  -123.96 | [-142.58, -103.99] |  -1.18 |   -59.59 | [ -77.71,  -39.50] |   -14.5 |  0.556 | 0.031 |  0.574 |
| queue=mid | before | 34 |    97.06 | [  97.06,   97.06] |   0.79 |   126.11 | [ 126.11,  126.11] |     1.0 |  0.500 | 0.029 |  0.515 |
| queue=mid | after | 34 |    97.06 | [  97.06,   97.06] |   0.79 |   126.11 | [ 126.11,  126.11] |     1.0 |  0.500 | 0.029 |  0.515 |
| size=small | before | 29,008 |  -165.41 | [-184.01, -147.42] |  -1.58 |   -61.14 | [ -79.51,  -41.86] |   -19.2 |  0.582 | 0.031 |  0.601 |
| size=small | after | 29,008 |  -124.13 | [-142.79, -105.84] |  -1.18 |   -61.14 | [ -79.51,  -41.86] |   -14.4 |  0.557 | 0.031 |  0.575 |
| size=large | before | 711 |  -189.66 | [-253.75, -129.96] |  -1.50 |    23.51 | [ -31.65,   79.61] |    -5.6 |  0.568 | 0.039 |  0.592 |
| size=large | after | 711 |   -94.02 | [-155.85,  -36.20] |  -0.74 |    23.51 | [ -31.65,   79.61] |    -2.8 |  0.504 | 0.025 |  0.517 |
| emptied_touch=yes | before | 18,744 |  -180.58 | [-202.13, -161.06] |  -1.70 |   -71.44 | [ -92.52,  -52.42] |   -18.5 |  0.590 | 0.031 |  0.609 |
| emptied_touch=yes | after | 18,744 |  -113.06 | [-134.75,  -94.08] |  -1.06 |   -71.44 | [ -92.52,  -52.42] |   -11.6 |  0.549 | 0.031 |  0.566 |
| emptied_touch=no | before | 10,975 |  -141.08 | [-163.22, -120.67] |  -1.36 |   -38.07 | [ -58.43,  -17.31] |   -12.3 |  0.568 | 0.030 |  0.586 |
| emptied_touch=no | after | 10,975 |  -141.08 | [-163.22, -120.67] |  -1.36 |   -38.07 | [ -58.43,  -17.31] |   -12.3 |  0.568 | 0.030 |  0.586 |

**Breakdowns at H = 60s**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 13,843 |  -199.38 | [-439.40,   30.15] |  -1.88 |   -91.86 | [-331.31,  132.34] |    -1.7 |  0.519 | 0.013 |  0.525 |
| side=bid | after | 13,843 |  -156.43 | [-396.46,   73.20] |  -1.47 |   -91.86 | [-331.31,  132.34] |    -1.3 |  0.507 | 0.014 |  0.514 |
| side=ask | before | 15,551 |  -195.22 | [-425.14,    3.90] |  -1.86 |   -88.49 | [-318.82,  110.72] |    -1.8 |  0.559 | 0.012 |  0.566 |
| side=ask | after | 15,551 |  -152.65 | [-382.96,   45.67] |  -1.45 |   -88.49 | [-318.82,  110.72] |    -1.4 |  0.547 | 0.014 |  0.555 |
| queue=front | before | 29,339 |  -197.16 | [-247.55, -155.26] |  -1.87 |   -89.91 | [-139.20,  -48.85] |    -8.1 |  0.540 | 0.012 |  0.547 |
| queue=front | after | 29,339 |  -154.33 | [-204.60, -112.19] |  -1.46 |   -89.91 | [-139.20,  -48.85] |    -6.4 |  0.528 | 0.014 |  0.536 |
| queue=mid | before | 33 |  -372.73 | [-372.73, -372.73] |  -3.00 |  -343.07 | [-343.07, -343.07] |    -3.2 |  0.576 | 0.000 |  0.576 |
| queue=mid | after | 33 |  -372.73 | [-372.73, -372.73] |  -3.00 |  -343.07 | [-343.07, -343.07] |    -3.2 |  0.576 | 0.000 |  0.576 |
| size=small | before | 28,695 |  -197.77 | [-243.25, -154.13] |  -1.88 |   -93.27 | [-138.39,  -51.06] |    -8.2 |  0.540 | 0.012 |  0.547 |
| size=small | after | 28,695 |  -156.30 | [-201.63, -113.07] |  -1.49 |   -93.27 | [-138.39,  -51.06] |    -6.5 |  0.529 | 0.014 |  0.536 |
| size=large | before | 699 |  -172.82 | [-393.72,   30.45] |  -1.36 |    40.98 | [-174.10,  240.27] |    -1.5 |  0.545 | 0.011 |  0.551 |
| size=large | after | 699 |   -77.40 | [-299.00,  126.32] |  -0.61 |    40.98 | [-174.10,  240.27] |    -0.7 |  0.522 | 0.010 |  0.527 |
| emptied_touch=yes | before | 18,594 |  -213.25 | [-267.80, -164.16] |  -2.00 |  -103.96 | [-159.36,  -56.89] |    -7.8 |  0.544 | 0.012 |  0.550 |
| emptied_touch=yes | after | 18,594 |  -145.67 | [-200.38,  -96.54] |  -1.37 |  -103.96 | [-159.36,  -56.89] |    -5.3 |  0.525 | 0.014 |  0.533 |
| emptied_touch=no | before | 10,800 |  -169.51 | [-233.54, -104.29] |  -1.62 |   -66.17 | [-130.45,   -3.12] |    -5.1 |  0.534 | 0.013 |  0.541 |
| emptied_touch=no | after | 10,800 |  -169.51 | [-233.54, -104.29] |  -1.62 |   -66.17 | [-130.45,   -3.12] |    -5.1 |  0.534 | 0.013 |  0.541 |

### Secondary — per individual fill

The same measurement without the aggressor grouping. Sweep multiplicity inflates `n`, so the CIs here are too narrow; kept for comparison with the event-time E6, which is also per fill.

One observation per individual fill. † median: no CI (lattice).

| H | ref | n | mean ($0.0001) | 95% CI | median ($0.0001) † | trimmed mean ($0.0001) | 95% CI | mean (×HS) | NW t | p | NW lag | block (obs) | eff. blocks | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | before | 46,357 |  -144.27 | [-151.89, -137.66] |  -100.00 |  -129.69 | [-134.92, -125.33] |  -1.30 |  -131.0 |  0.000 | 1 | 122 | 380 |  0.809 | 0.115 |  0.914 |
| 100ms | after | 46,357 |  -116.85 | [-124.85, -110.29] |  -100.00 |  -102.14 | [-107.38,  -98.21] |  -1.05 |  -105.4 |  0.000 | 1 | 122 | 380 |  0.712 | 0.174 |  0.863 |
| 1s | before | 46,331 |  -170.74 | [-184.12, -159.53] |  -150.00 |  -158.91 | [-168.01, -150.26] |  -1.54 |   -71.3 |  0.000 | 3 | 122 | 380 |  0.731 | 0.082 |  0.796 |
| 1s | after | 46,331 |  -143.31 | [-157.21, -132.11] |  -100.00 |  -131.38 | [-140.40, -122.49] |  -1.29 |   -59.6 |  0.000 | 3 | 122 | 380 |  0.673 | 0.103 |  0.750 |
| 10s | before | 46,122 |  -178.32 | [-208.30, -154.69] |  -150.00 |  -169.26 | [-195.38, -146.46] |  -1.62 |   -17.0 |  0.000 | 23 | 122 | 379 |  0.585 | 0.031 |  0.604 |
| 10s | after | 46,122 |  -150.88 | [-180.97, -126.78] |  -150.00 |  -141.85 | [-167.96, -118.87] |  -1.37 |   -14.4 |  0.000 | 23 | 122 | 379 |  0.569 | 0.031 |  0.587 |
| 60s | before | 45,612 |  -237.99 | [-313.02, -162.22] |  -200.00 |  -211.77 | [-270.86, -154.93] |  -2.15 |    -6.6 |  0.000 | 121 | 122 | 374 |  0.545 | 0.013 |  0.552 |
| 60s | after | 45,612 |  -210.44 | [-286.02, -133.83] |  -200.00 |  -184.70 | [-243.66, -128.12] |  -1.90 |    -5.8 |  0.000 | 121 | 122 | 374 |  0.537 | 0.014 |  0.545 |

## QQQ

17,662 passive fills grouped into 11,407 aggressor events (1.55 fills per event). Observations dropped per horizon:

| H | past 16:00 | past tape end | no pre-fill mid | one-sided book | pre-window before 09:30 | pre-window one-sided | usable events |
|---|---|---|---|---|---|---|---|
| 100ms | 2 | 0 | 0 | 0 | 0 | 0 | 11,405 |
| 1s | 11 | 0 | 0 | 0 | 3 | 0 | 11,396 |
| 10s | 73 | 0 | 0 | 0 | 39 | 0 | 11,334 |
| 60s | 204 | 0 | 0 | 0 | 132 | 0 | 11,203 |

### Realized spread — does a passive fill make money?

**This is the headline.** `realized spread = half-spread earned + markout` = `s · (mid[t+H] − P)`, one observation per aggressor event, `P` the event's size-weighted execution price. Reference-invariant, so one row per horizon. Negative means the passive side gave back more to adverse selection than the spread it earned. `×HS` expresses that as a fraction of the half-spread quoted at the fill, which is the scale-free read.

| H | n events | mean ($0.0001) | 95% CI | mean (×HS) | median ($0.0001) † | trimmed mean ($0.0001) | 95% CI | NW t | p | HS earned ($0.0001) | mean markout ($0.0001) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | 11,405 |   -20.46 | [ -23.11,  -18.30] |  -0.39 |   -50.00 |   -17.10 | [ -18.29,  -15.95] |   -25.1 |  0.000 |    21.75 |   -42.21 |
| 1s | 11,396 |   -19.96 | [ -23.34,  -16.37] |  -0.38 |   -50.00 |   -18.62 | [ -20.97,  -16.33] |   -15.8 |  0.000 |    21.74 |   -41.70 |
| 10s | 11,334 |   -24.73 | [ -33.59,  -15.65] |  -0.47 |   -50.00 |   -24.44 | [ -32.36,  -16.87] |    -6.0 |  0.000 |    21.69 |   -46.41 |
| 60s | 11,203 |   -34.70 | [ -59.33,   -9.51] |  -0.66 |   -50.00 |   -31.15 | [ -52.06,  -11.97] |    -2.6 |  0.008 |    21.59 |   -56.29 |

† no CI on the median: medians sit on the 50-unit half-tick lattice, so bootstrap CIs on them degenerate and are not reported — use the sign-split table instead.

### Sign split — the claim that survives the price lattice

Mids move in 50-unit (half-tick) steps, so a median is pinned to a lattice point. How often the outcome lands on each side of zero is not. Shares are over aggressor events, measured from the `after` mid, with block-bootstrap CIs; the three shares in each group sum to 1.

| H | quantity | n | share < 0 | 95% CI | share = 0 | 95% CI | share > 0 | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 100ms | realized spread | 11,405 |  0.559 | [  0.546,   0.571] |  0.154 | [  0.145,   0.166] |  0.286 | [  0.278,   0.297] |
| 100ms | markout | 11,405 |  0.603 | [  0.592,   0.614] |  0.295 | [  0.284,   0.305] |  0.102 | [  0.095,   0.109] |
| 1s | realized spread | 11,396 |  0.535 | [  0.523,   0.548] |  0.099 | [  0.091,   0.108] |  0.366 | [  0.356,   0.377] |
| 1s | markout | 11,396 |  0.564 | [  0.553,   0.575] |  0.202 | [  0.192,   0.213] |  0.234 | [  0.221,   0.244] |
| 10s | realized spread | 11,334 |  0.517 | [  0.505,   0.528] |  0.036 | [  0.030,   0.042] |  0.446 | [  0.435,   0.457] |
| 10s | markout | 11,334 |  0.530 | [  0.519,   0.541] |  0.088 | [  0.080,   0.096] |  0.381 | [  0.371,   0.395] |
| 60s | realized spread | 11,203 |  0.512 | [  0.500,   0.525] |  0.010 | [  0.007,   0.012] |  0.478 | [  0.467,   0.489] |
| 60s | markout | 11,203 |  0.514 | [  0.501,   0.527] |  0.037 | [  0.031,   0.042] |  0.449 | [  0.436,   0.459] |

### Break-even including an assumed maker rebate

**The rebate here is assumed, not measured: maker rebate is an ASSUMED constant per share, not measured from the tape — ITCH carries no fee data.** The values below are illustrative, chosen to bracket a typical 2019 US equity maker rebate, and every figure in this table is conditional on them. A rebate is a constant per share, so it shifts the realized-spread distribution without changing its shape — which is why the share profitable moves as well as the mean.

| H | realized spread ($0.0001) | + 20 rebate | 95% CI | ×HS | share > 0 | + 30 rebate | 95% CI | ×HS | share > 0 |
|---|---|---|---|---|---|---|---|---|---|
| 100ms |   -20.46 |    -0.46 | [  -2.73,    1.71] |  -0.01 |  0.441 |     9.54 | [   6.76,   11.75] |   0.18 |  0.441 |
| 1s |   -19.96 |     0.04 | [  -3.10,    3.06] |   0.00 |  0.465 |    10.04 | [   6.84,   13.40] |   0.19 |  0.465 |
| 10s |   -24.73 |    -4.73 | [ -13.73,    4.01] |  -0.09 |  0.483 |     5.27 | [  -3.50,   14.50] |   0.10 |  0.483 |
| 60s |   -34.70 |   -14.70 | [ -43.57,   12.59] |  -0.28 |  0.488 |    -4.70 | [ -30.61,   20.26] |  -0.09 |  0.488 |

### Attribution — where E6's 0.957 goes

Each row changes exactly one thing from the row above, on the **same** fills, so every step is attributable. The statistic is `P(adverse | mid moved)` — E6's own headline — with the unconditional and zero shares beside it.

Row 1 / `after` reproduces E6's published figure, which establishes that **E6 already measured from the post-fill mid** (`mid_history[event_index]`). The walk down the `after` column is therefore the decomposition of E6's number: two steps, the unit of observation and the clock, and they account for the whole change.

The `before` column is a *separate* comparison, not a third step: it shows how much of each row's measured move is the fill's own BBO jump. Both endpoints of the walk are already measured from the same reference, so adding the before/after gap to the two steps would double-count.

| row | unit | clock | ref | n | P(adv\|moved) | P(adv) uncond | share zero | mean markout ($0.0001) |
|---|---|---|---|---|---|---|---|---|
| 1 | per fill | event h=5 | before | 17,662 | ** 0.990** |  0.629 |  0.365 |   -42.08 |
| 1 | per fill | event h=5 | after | 17,662 | ** 0.958** |  0.459 |  0.521 |   -25.72 |
| 2 | per aggressor event | event h=5 | before | 11,407 | ** 0.988** |  0.640 |  0.352 |   -44.69 |
| 2 | per aggressor event | event h=5 | after | 11,407 | ** 0.927** |  0.379 |  0.591 |   -19.35 |
| 3 | per aggressor event | 100ms | before | 11,405 | ** 0.949** |  0.689 |  0.274 |   -67.54 |
| 3 | per aggressor event | 100ms | after | 11,405 | ** 0.855** |  0.603 |  0.295 |   -42.21 |

Steps along the `before` column (these are the decomposition):

| step | what changed | from | to | Δ |
|---|---|---|---|---|
| 1 → 2 | unit of observation |  0.990 |  0.988 |  -0.003 |
| 2 → 3 | clock |  0.988 |  0.949 |  -0.039 |

Sum of steps -0.042 vs total change -0.042 (0.990 → 0.949) — **checks out**.

Steps along the `after` column (these are the decomposition):

| step | what changed | from | to | Δ |
|---|---|---|---|---|
| 1 → 2 | unit of observation |  0.958 |  0.927 |  -0.031 |
| 2 → 3 | clock |  0.927 |  0.855 |  -0.072 |

Sum of steps -0.103 vs total change -0.103 (0.958 → 0.855) — **checks out**.

Within-row `before − after` gap (the fill's own BBO jump, **not** a walk step):

| row | unit | clock | before − after |
|---|---|---|---|
| 1 | per fill | event h=5 |   0.032 |
| 2 | per aggressor event | event h=5 |   0.060 |
| 3 | per aggressor event | 100ms |   0.094 |

### Markout

One observation per aggressor event. † median: no CI (lattice).

| H | ref | n | mean ($0.0001) | 95% CI | median ($0.0001) † | trimmed mean ($0.0001) | 95% CI | mean (×HS) | NW t | p | NW lag | block (obs) | eff. blocks | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | before | 11,405 |   -67.54 | [ -70.45,  -65.06] |  -100.00 |   -63.64 | [ -64.91,  -62.43] |  -1.28 |   -77.8 |  0.000 | 1 | 36 | 317 |  0.689 | 0.274 |  0.949 |
| 100ms | after | 11,405 |   -42.21 | [ -45.03,  -39.89] |   -50.00 |   -40.01 | [ -41.23,  -38.86] |  -0.80 |   -50.8 |  0.000 | 1 | 36 | 317 |  0.603 | 0.295 |  0.855 |
| 1s | before | 11,396 |   -67.04 | [ -70.47,  -63.45] |  -100.00 |   -65.06 | [ -67.56,  -62.68] |  -1.27 |   -51.9 |  0.000 | 1 | 36 | 317 |  0.616 | 0.252 |  0.823 |
| 1s | after | 11,396 |   -41.70 | [ -45.09,  -38.32] |   -50.00 |   -38.79 | [ -41.41,  -36.63] |  -0.79 |   -32.6 |  0.000 | 1 | 36 | 317 |  0.564 | 0.202 |  0.707 |
| 10s | before | 11,334 |   -71.79 | [ -80.76,  -62.45] |  -100.00 |   -71.09 | [ -79.19,  -63.86] |  -1.36 |   -17.5 |  0.000 | 6 | 36 | 315 |  0.549 | 0.137 |  0.636 |
| 10s | after | 11,334 |   -46.41 | [ -55.23,  -36.95] |   -50.00 |   -46.18 | [ -53.91,  -38.74] |  -0.88 |   -11.3 |  0.000 | 6 | 36 | 315 |  0.530 | 0.088 |  0.582 |
| 60s | before | 11,203 |   -81.75 | [-106.14,  -56.53] |  -100.00 |   -77.79 | [ -99.07,  -58.49] |  -1.55 |    -6.2 |  0.000 | 35 | 36 | 312 |  0.520 | 0.060 |  0.554 |
| 60s | after | 11,203 |   -56.29 | [ -80.69,  -30.93] |   -50.00 |   -52.68 | [ -73.79,  -33.41] |  -1.07 |    -4.3 |  0.000 | 35 | 36 | 312 |  0.514 | 0.037 |  0.534 |

### Reversion check — mechanical or informational?

This is the test that discriminates. The `before − after` gap cannot: it equals `s · (mid_after − mid_before)`, which involves neither the terminal mid nor `H`, so it is **constant across horizons by construction** and tests nothing. What does discriminate is whether the move comes back.

| H | markout from the `after` mid ($0.0001) | 95% CI |
|---|---|---|
| 100ms |   -42.21 | [ -45.03,  -39.89] |
| 1s |   -41.70 | [ -45.09,  -38.32] |
| 10s |   -46.41 | [ -55.23,  -36.95] |
| 60s |   -56.29 | [ -80.69,  -30.93] |

From 100ms to 60s the markout goes -42.21 → -56.29, so the impact is **permanent**: the markout does not shrink with horizon, so the price does not come back. A purely mechanical dislocation would revert once displaced depth was replenished; this does not.

### Who loses, and by how much

| H | share losing | share winning | mean loss when losing ($0.0001) | mean gain when winning ($0.0001) |
|---|---|---|---|---|
| 100ms |  0.559 |  0.286 |   -67.62 |    60.75 |
| 1s |  0.535 |  0.366 |  -100.51 |    92.22 |
| 10s |  0.517 |  0.446 |  -260.59 |   246.71 |
| 60s |  0.512 |  0.478 |  -634.66 |   606.54 |

About **half** of aggressor events lose and about half win; the mean is negative because **losers lose more than winners win**. This is not a minority of bad fills dragging an otherwise profitable book — it is a roughly symmetric hit rate with asymmetric magnitudes.

### Bootstrap blocks and dependence

| H | NW lag (events) | events per H-window | block (clock) | block (events) | effective blocks |
|---|---|---|---|---|---|
| 100ms | 1 | 1 | 60s | 36 | 317 |
| 1s | 1 | 2 | 60s | 36 | 317 |
| 10s | 6 | 7 | 60s | 36 | 315 |
| 60s | 35 | 36 | 60s | 36 | 312 |

### Pre-fill control and the bid/ask split

`pre` is the signed mid move over the `H` before the fill, ending strictly before it; `post − pre` is the fill-conditional excess. A session-wide trend shows up in `pre`; adverse selection should not.

| H | ref | group | n | post ($0.0001) | 95% CI | pre ($0.0001) | 95% CI | post − pre ($0.0001) | 95% CI | NW t | adverse | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | before | all | 11,405 |   -67.54 | [ -70.45,  -65.06] |     3.75 | [   0.23,    6.40] |   -71.29 | [ -74.06,  -68.39] |   -60.6 |  0.689 |  0.949 |
| 100ms | before | side=bid | 5,680 |   -67.08 | [ -76.76,  -60.74] |    -3.05 | [ -14.79,    5.73] |   -64.02 | [ -71.38,  -56.45] |   -30.7 |  0.660 |  0.943 |
| 100ms | before | side=ask | 5,725 |   -67.99 | [ -71.19,  -63.53] |    10.51 | [   4.46,   18.55] |   -78.50 | [ -83.51,  -72.92] |   -45.7 |  0.717 |  0.954 |
| 100ms | after | all | 11,405 |   -42.21 | [ -45.03,  -39.89] |     3.75 | [   0.23,    6.40] |   -45.96 | [ -48.61,  -43.11] |   -39.4 |  0.603 |  0.855 |
| 100ms | after | side=bid | 5,680 |   -43.64 | [ -52.72,  -37.89] |    -3.05 | [ -14.79,    5.73] |   -40.59 | [ -47.69,  -32.06] |   -19.5 |  0.580 |  0.846 |
| 100ms | after | side=ask | 5,725 |   -40.78 | [ -43.88,  -36.91] |    10.51 | [   4.46,   18.55] |   -51.28 | [ -56.54,  -46.01] |   -29.9 |  0.626 |  0.864 |
| 1s | before | all | 11,396 |   -67.04 | [ -70.47,  -63.45] |     2.62 | [  -1.73,    6.66] |   -69.68 | [ -73.79,  -65.42] |   -33.0 |  0.616 |  0.823 |
| 1s | before | side=bid | 5,679 |   -68.67 | [ -78.78,  -60.71] |   -12.33 | [ -27.60,    1.39] |   -56.38 | [ -68.82,  -40.25] |   -14.9 |  0.603 |  0.813 |
| 1s | before | side=ask | 5,717 |   -65.42 | [ -70.20,  -60.18] |    17.46 | [   5.90,   32.18] |   -82.89 | [ -96.84,  -71.71] |   -24.6 |  0.629 |  0.833 |
| 1s | after | all | 11,396 |   -41.70 | [ -45.09,  -38.32] |     2.62 | [  -1.73,    6.66] |   -44.34 | [ -48.72,  -39.87] |   -20.9 |  0.564 |  0.707 |
| 1s | after | side=bid | 5,679 |   -45.24 | [ -54.68,  -37.42] |   -12.33 | [ -27.60,    1.39] |   -32.95 | [ -45.46,  -16.83] |    -8.6 |  0.555 |  0.703 |
| 1s | after | side=ask | 5,717 |   -38.19 | [ -42.83,  -33.20] |    17.46 | [   5.90,   32.18] |   -55.66 | [ -69.85,  -44.33] |   -16.5 |  0.573 |  0.712 |
| 10s | before | all | 11,334 |   -71.79 | [ -80.76,  -62.45] |   -12.39 | [ -23.25,   -2.27] |   -60.32 | [ -71.31,  -48.03] |   -10.1 |  0.549 |  0.636 |
| 10s | before | side=bid | 5,645 |   -99.73 | [-129.44,  -66.41] |   -53.51 | [ -86.98,  -12.58] |   -45.75 | [ -83.51,   -9.00] |    -3.0 |  0.572 |  0.656 |
| 10s | before | side=ask | 5,689 |   -44.08 | [ -68.23,  -19.65] |    28.43 | [  -2.55,   65.94] |   -74.79 | [-103.43,  -47.09] |    -5.7 |  0.526 |  0.616 |
| 10s | after | all | 11,334 |   -46.41 | [ -55.23,  -36.95] |   -12.39 | [ -23.25,   -2.27] |   -34.94 | [ -46.50,  -22.52] |    -5.8 |  0.530 |  0.582 |
| 10s | after | side=bid | 5,645 |   -76.24 | [-105.52,  -43.80] |   -53.51 | [ -86.98,  -12.58] |   -22.26 | [ -59.73,   14.75] |    -1.4 |  0.555 |  0.606 |
| 10s | after | side=ask | 5,689 |   -16.82 | [ -40.70,    8.28] |    28.43 | [  -2.55,   65.94] |   -47.52 | [ -75.78,  -19.42] |    -3.6 |  0.506 |  0.558 |
| 60s | before | all | 11,203 |   -81.75 | [-106.14,  -56.53] |   -36.82 | [ -63.88,  -13.54] |   -45.41 | [ -79.28,  -12.90] |    -2.7 |  0.520 |  0.554 |
| 60s | before | side=bid | 5,577 |  -213.94 | [-331.17, -100.70] |  -218.55 | [-334.72,  -79.13] |     8.06 | [-137.64,  143.42] |     0.1 |  0.563 |  0.600 |
| 60s | before | side=ask | 5,626 |    49.28 | [ -41.01,  148.04] |   143.63 | [  31.88,  258.21] |   -98.51 | [-238.34,   21.69] |    -1.6 |  0.478 |  0.508 |
| 60s | after | all | 11,203 |   -56.29 | [ -80.69,  -30.93] |   -36.82 | [ -63.88,  -13.54] |   -19.95 | [ -54.16,   12.57] |    -1.2 |  0.514 |  0.534 |
| 60s | after | side=bid | 5,577 |  -190.32 | [-307.95,  -77.55] |  -218.55 | [-334.72,  -79.13] |    31.68 | [-113.06,  167.30] |     0.4 |  0.559 |  0.580 |
| 60s | after | side=ask | 5,626 |    76.57 | [ -13.75,  174.80] |   143.63 | [  31.88,  258.21] |   -71.22 | [-210.48,   48.06] |    -1.2 |  0.470 |  0.488 |

### Breakdowns

**Breakdowns at H = 100ms**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 5,680 |   -67.08 | [ -76.76,  -60.74] |  -1.26 |   -18.63 | [ -27.40,  -12.82] |   -42.6 |  0.660 | 0.300 |  0.943 |
| side=bid | after | 5,680 |   -43.64 | [ -52.72,  -37.89] |  -0.82 |   -18.63 | [ -27.40,  -12.82] |   -29.2 |  0.580 | 0.314 |  0.846 |
| side=ask | before | 5,725 |   -67.99 | [ -71.19,  -63.53] |  -1.30 |   -22.27 | [ -25.34,  -17.82] |   -59.3 |  0.717 | 0.248 |  0.954 |
| side=ask | after | 5,725 |   -40.78 | [ -43.88,  -36.91] |  -0.78 |   -22.27 | [ -25.34,  -17.82] |   -36.8 |  0.626 | 0.275 |  0.864 |
| queue=front | before | 10,746 |   -71.98 | [ -75.56,  -69.39] |  -1.40 |   -20.72 | [ -23.98,  -18.26] |   -78.3 |  0.716 | 0.259 |  0.965 |
| queue=front | after | 10,746 |   -45.13 | [ -48.44,  -42.64] |  -0.88 |   -20.72 | [ -23.98,  -18.26] |   -50.9 |  0.625 | 0.280 |  0.869 |
| queue=mid | before | 120 |     8.75 | [  -3.75,   22.51] |   0.12 |   -14.36 | [ -28.34,    0.76] |     1.5 |  0.225 | 0.483 |  0.435 |
| queue=mid | after | 120 |    10.00 | [  -2.08,   23.34] |   0.14 |   -14.36 | [ -28.34,    0.76] |     1.7 |  0.217 | 0.483 |  0.419 |
| queue=back | before | 539 |     4.08 | [  -4.09,   16.53] |   0.05 |   -16.60 | [ -24.58,   -4.90] |     0.9 |  0.254 | 0.532 |  0.544 |
| queue=back | after | 539 |     4.55 | [  -3.63,   16.54] |   0.06 |   -16.60 | [ -24.58,   -4.90] |     1.0 |  0.252 | 0.532 |  0.540 |
| size=small | before | 5,705 |   -45.92 | [ -48.46,  -43.29] |  -0.84 |    -3.16 | [  -5.64,   -0.83] |   -38.6 |  0.522 | 0.419 |  0.898 |
| size=small | after | 5,705 |   -45.82 | [ -48.38,  -43.19] |  -0.84 |    -3.16 | [  -5.64,   -0.83] |   -38.6 |  0.521 | 0.419 |  0.897 |
| size=large | before | 5,700 |   -89.17 | [ -94.71,  -85.03] |  -1.75 |   -37.77 | [ -43.43,  -33.96] |   -69.9 |  0.856 | 0.129 |  0.983 |
| size=large | after | 5,700 |   -38.59 | [ -44.26,  -34.72] |  -0.76 |   -37.77 | [ -43.43,  -33.96] |   -31.3 |  0.685 | 0.170 |  0.826 |
| emptied_touch=yes | before | 5,704 |   -89.23 | [ -94.60,  -85.09] |  -1.75 |   -37.77 | [ -42.90,  -34.02] |   -69.7 |  0.856 | 0.129 |  0.982 |
| emptied_touch=yes | after | 5,704 |   -38.59 | [ -43.83,  -34.73] |  -0.76 |   -37.77 | [ -42.90,  -34.02] |   -31.2 |  0.685 | 0.170 |  0.825 |
| emptied_touch=no | before | 5,701 |   -45.83 | [ -48.50,  -43.29] |  -0.84 |    -3.13 | [  -5.70,   -0.88] |   -38.7 |  0.522 | 0.419 |  0.898 |
| emptied_touch=no | after | 5,701 |   -45.83 | [ -48.49,  -43.27] |  -0.84 |    -3.13 | [  -5.70,   -0.88] |   -38.7 |  0.522 | 0.419 |  0.898 |

**Breakdowns at H = 1s**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 5,679 |   -68.67 | [ -78.78,  -60.71] |  -1.29 |   -20.22 | [ -29.93,  -12.44] |   -30.0 |  0.603 | 0.258 |  0.813 |
| side=bid | after | 5,679 |   -45.24 | [ -54.68,  -37.42] |  -0.85 |   -20.22 | [ -29.93,  -12.44] |   -20.1 |  0.555 | 0.210 |  0.703 |
| side=ask | before | 5,717 |   -65.42 | [ -70.20,  -60.18] |  -1.25 |   -19.70 | [ -24.52,  -14.76] |   -34.6 |  0.629 | 0.245 |  0.833 |
| side=ask | after | 5,717 |   -38.19 | [ -42.83,  -33.20] |  -0.73 |   -19.70 | [ -24.52,  -14.76] |   -20.4 |  0.573 | 0.195 |  0.712 |
| queue=front | before | 10,737 |   -72.04 | [ -75.94,  -68.54] |  -1.40 |   -20.78 | [ -24.54,  -17.06] |   -52.8 |  0.636 | 0.247 |  0.845 |
| queue=front | after | 10,737 |   -45.19 | [ -48.90,  -41.68] |  -0.88 |   -20.78 | [ -24.54,  -17.06] |   -33.3 |  0.581 | 0.195 |  0.722 |
| queue=mid | before | 120 |    24.17 | [   7.07,   48.33] |   0.34 |     1.06 | [ -18.59,   26.71] |     1.5 |  0.267 | 0.292 |  0.376 |
| queue=mid | after | 120 |    25.42 | [   8.32,   48.76] |   0.35 |     1.06 | [ -18.59,   26.71] |     1.6 |  0.258 | 0.300 |  0.369 |
| queue=back | before | 539 |    12.34 | [   1.29,   26.72] |   0.16 |    -8.35 | [ -19.39,    5.48] |     2.2 |  0.299 | 0.330 |  0.446 |
| queue=back | after | 539 |    12.80 | [   1.85,   27.18] |   0.16 |    -8.35 | [ -19.39,    5.48] |     2.3 |  0.297 | 0.328 |  0.442 |
| size=small | before | 5,700 |   -49.21 | [ -53.34,  -45.11] |  -0.90 |    -6.45 | [ -10.56,   -2.02] |   -27.4 |  0.532 | 0.301 |  0.761 |
| size=small | after | 5,700 |   -49.11 | [ -53.26,  -44.99] |  -0.90 |    -6.45 | [ -10.56,   -2.02] |   -27.3 |  0.532 | 0.301 |  0.760 |
| size=large | before | 5,696 |   -84.88 | [ -89.58,  -79.67] |  -1.67 |   -33.48 | [ -38.03,  -28.58] |   -46.2 |  0.700 | 0.202 |  0.878 |
| size=large | after | 5,696 |   -34.30 | [ -38.99,  -29.23] |  -0.67 |   -33.48 | [ -38.03,  -28.58] |   -18.8 |  0.597 | 0.104 |  0.666 |
| emptied_touch=yes | before | 5,700 |   -85.00 | [ -89.61,  -80.09] |  -1.67 |   -33.54 | [ -37.92,  -28.66] |   -46.3 |  0.701 | 0.202 |  0.878 |
| emptied_touch=yes | after | 5,700 |   -34.36 | [ -38.97,  -29.49] |  -0.67 |   -33.54 | [ -37.92,  -28.66] |   -18.9 |  0.596 | 0.104 |  0.665 |
| emptied_touch=no | before | 5,696 |   -49.06 | [ -52.99,  -44.96] |  -0.90 |    -6.37 | [ -10.34,   -2.03] |   -27.3 |  0.532 | 0.301 |  0.761 |
| emptied_touch=no | after | 5,696 |   -49.05 | [ -52.98,  -44.95] |  -0.90 |    -6.37 | [ -10.34,   -2.03] |   -27.3 |  0.532 | 0.301 |  0.761 |

**Breakdowns at H = 10s**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 5,645 |   -99.73 | [-129.44,  -66.41] |  -1.88 |   -51.28 | [ -81.17,  -18.49] |    -9.4 |  0.572 | 0.128 |  0.656 |
| side=bid | after | 5,645 |   -76.24 | [-105.52,  -43.80] |  -1.43 |   -51.28 | [ -81.17,  -18.49] |    -7.2 |  0.555 | 0.084 |  0.606 |
| side=ask | before | 5,689 |   -44.08 | [ -68.23,  -19.65] |  -0.84 |     1.62 | [ -22.03,   26.07] |    -4.7 |  0.526 | 0.146 |  0.616 |
| side=ask | after | 5,689 |   -16.82 | [ -40.70,    8.28] |  -0.32 |     1.62 | [ -22.03,   26.07] |    -1.8 |  0.506 | 0.092 |  0.558 |
| queue=front | before | 10,677 |   -76.16 | [ -85.43,  -66.52] |  -1.48 |   -24.90 | [ -34.34,  -15.59] |   -17.6 |  0.557 | 0.139 |  0.647 |
| queue=front | after | 10,677 |   -49.26 | [ -58.52,  -39.59] |  -0.96 |   -24.90 | [ -34.34,  -15.59] |   -11.4 |  0.537 | 0.087 |  0.589 |
| queue=mid | before | 119 |   -53.36 | [-147.93,   20.18] |  -0.75 |   -76.66 | [-167.61,   -5.82] |    -1.1 |  0.437 | 0.134 |  0.505 |
| queue=mid | after | 119 |   -52.10 | [-147.93,   22.27] |  -0.73 |   -76.66 | [-167.61,   -5.82] |    -1.0 |  0.437 | 0.126 |  0.500 |
| queue=back | before | 538 |    10.87 | [ -21.84,   43.61] |   0.14 |    -9.85 | [ -41.46,   21.94] |     0.7 |  0.413 | 0.102 |  0.460 |
| queue=back | after | 538 |    11.34 | [ -21.29,   43.89] |   0.14 |    -9.85 | [ -41.46,   21.94] |     0.8 |  0.413 | 0.100 |  0.459 |
| size=small | before | 5,659 |   -56.82 | [ -68.93,  -44.45] |  -1.04 |   -14.10 | [ -25.72,   -1.74] |   -10.1 |  0.522 | 0.140 |  0.607 |
| size=small | after | 5,659 |   -56.71 | [ -68.84,  -44.39] |  -1.03 |   -14.10 | [ -25.72,   -1.74] |   -10.0 |  0.522 | 0.140 |  0.607 |
| size=large | before | 5,675 |   -86.72 | [ -97.82,  -75.69] |  -1.71 |   -35.32 | [ -46.40,  -24.42] |   -15.3 |  0.576 | 0.134 |  0.664 |
| size=large | after | 5,675 |   -36.14 | [ -47.20,  -25.22] |  -0.71 |   -35.32 | [ -46.40,  -24.42] |    -6.4 |  0.539 | 0.036 |  0.559 |
| emptied_touch=yes | before | 5,679 |   -86.46 | [ -97.40,  -75.04] |  -1.70 |   -35.00 | [ -46.09,  -23.46] |   -15.2 |  0.575 | 0.134 |  0.664 |
| emptied_touch=yes | after | 5,679 |   -35.82 | [ -46.74,  -24.18] |  -0.70 |   -35.00 | [ -46.09,  -23.46] |    -6.3 |  0.539 | 0.036 |  0.559 |
| emptied_touch=no | before | 5,655 |   -57.06 | [ -68.70,  -44.28] |  -1.04 |   -14.41 | [ -25.87,   -1.81] |   -10.1 |  0.522 | 0.140 |  0.607 |
| emptied_touch=no | after | 5,655 |   -57.06 | [ -68.70,  -44.27] |  -1.04 |   -14.41 | [ -25.87,   -1.81] |   -10.1 |  0.522 | 0.140 |  0.607 |

**Breakdowns at H = 60s**, one observation per aggressor event.

`emptied_touch=yes` means the aggressive order was large enough to clear the displayed level. Losses concentrating there, and persisting across horizons, is **size-dependent permanent impact** — the Kyle / Glosten–Milgrom signature of informed flow, not evidence against adverse selection. `queue=` buckets are near-degenerate on a FIFO book, since the order that gets hit is by definition at the front of its level.

| group | ref | n | markout ($0.0001) | 95% CI | mean (×HS) | realized spread ($0.0001) | 95% CI | NW t | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| side=bid | before | 5,577 |  -213.94 | [-331.17, -100.70] |  -4.02 |  -165.50 | [-282.76,  -52.91] |    -3.6 |  0.563 | 0.061 |  0.600 |
| side=bid | after | 5,577 |  -190.32 | [-307.95,  -77.55] |  -3.58 |  -165.50 | [-282.76,  -52.91] |    -3.2 |  0.559 | 0.036 |  0.580 |
| side=ask | before | 5,626 |    49.28 | [ -41.01,  148.04] |   0.94 |    94.96 | [   4.66,  193.21] |     0.9 |  0.478 | 0.059 |  0.508 |
| side=ask | after | 5,626 |    76.57 | [ -13.75,  174.80] |   1.46 |    94.96 | [   4.66,  193.21] |     1.5 |  0.470 | 0.037 |  0.488 |
| queue=front | before | 10,551 |   -85.62 | [-114.08,  -62.16] |  -1.67 |   -34.34 | [ -63.11,  -10.75] |    -6.4 |  0.524 | 0.062 |  0.559 |
| queue=front | after | 10,551 |   -58.62 | [ -87.45,  -35.11] |  -1.14 |   -34.34 | [ -63.11,  -10.75] |    -4.3 |  0.518 | 0.037 |  0.538 |
| queue=mid | before | 117 |   -74.36 | [-364.31,  172.83] |  -1.05 |   -98.06 | [-384.37,  149.84] |    -0.4 |  0.496 | 0.034 |  0.513 |
| queue=mid | after | 117 |   -73.08 | [-363.46,  174.12] |  -1.03 |   -98.06 | [-384.37,  149.84] |    -0.4 |  0.496 | 0.034 |  0.513 |
| queue=back | before | 535 |    -7.10 | [-100.69,   91.46] |  -0.09 |   -27.94 | [-121.25,   71.32] |    -0.1 |  0.450 | 0.030 |  0.464 |
| queue=back | after | 535 |    -6.64 | [-100.41,   91.83] |  -0.08 |   -27.94 | [-121.25,   71.32] |    -0.1 |  0.450 | 0.028 |  0.463 |
| size=small | before | 5,575 |   -79.77 | [-119.70,  -35.72] |  -1.45 |   -37.11 | [ -77.60,    6.66] |    -3.8 |  0.514 | 0.062 |  0.548 |
| size=small | after | 5,575 |   -79.66 | [-119.64,  -35.64] |  -1.45 |   -37.11 | [ -77.60,    6.66] |    -3.8 |  0.514 | 0.062 |  0.548 |
| size=large | before | 5,628 |   -83.72 | [-115.72,  -53.80] |  -1.65 |   -32.31 | [ -64.56,   -2.36] |    -5.2 |  0.527 | 0.058 |  0.560 |
| size=large | after | 5,628 |   -33.14 | [ -65.05,   -3.25] |  -0.65 |   -32.31 | [ -64.56,   -2.36] |    -2.1 |  0.514 | 0.012 |  0.521 |
| emptied_touch=yes | before | 5,632 |   -83.13 | [-113.09,  -53.31] |  -1.63 |   -31.66 | [ -60.94,   -1.74] |    -5.2 |  0.526 | 0.059 |  0.559 |
| emptied_touch=yes | after | 5,632 |   -32.48 | [ -61.99,   -2.69] |  -0.64 |   -31.66 | [ -60.94,   -1.74] |    -2.0 |  0.514 | 0.012 |  0.520 |
| emptied_touch=no | before | 5,571 |   -80.36 | [-120.94,  -33.97] |  -1.47 |   -37.78 | [ -79.20,    8.69] |    -3.9 |  0.515 | 0.062 |  0.549 |
| emptied_touch=no | after | 5,571 |   -80.35 | [-120.93,  -33.96] |  -1.47 |   -37.78 | [ -79.20,    8.69] |    -3.9 |  0.515 | 0.062 |  0.549 |

### Secondary — per individual fill

The same measurement without the aggressor grouping. Sweep multiplicity inflates `n`, so the CIs here are too narrow; kept for comparison with the event-time E6, which is also per fill.

One observation per individual fill. † median: no CI (lattice).

| H | ref | n | mean ($0.0001) | 95% CI | median ($0.0001) † | trimmed mean ($0.0001) | 95% CI | mean (×HS) | NW t | p | NW lag | block (obs) | eff. blocks | adverse | zero | adverse\|Δ≠0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100ms | before | 17,660 |   -73.37 | [ -80.41,  -68.34] |  -100.00 |   -63.99 | [ -65.99,  -62.54] |  -1.34 |   -72.1 |  0.000 | 1 | 58 | 305 |  0.691 | 0.270 |  0.947 |
| 100ms | after | 17,660 |   -57.02 | [ -64.26,  -51.79] |   -50.00 |   -48.76 | [ -50.41,  -47.15] |  -1.04 |   -56.1 |  0.000 | 1 | 58 | 305 |  0.636 | 0.283 |  0.887 |
| 1s | before | 17,649 |   -70.94 | [ -77.34,  -65.72] |  -100.00 |   -66.61 | [ -69.84,  -63.46] |  -1.29 |   -49.6 |  0.000 | 2 | 58 | 305 |  0.619 | 0.246 |  0.820 |
| 1s | after | 17,649 |   -54.59 | [ -60.74,  -49.26] |   -50.00 |   -49.67 | [ -53.07,  -46.40] |  -1.00 |   -38.1 |  0.000 | 2 | 58 | 305 |  0.585 | 0.214 |  0.745 |
| 10s | before | 17,559 |   -80.13 | [ -97.55,  -64.70] |  -100.00 |   -75.27 | [ -84.83,  -65.48] |  -1.46 |   -14.5 |  0.000 | 11 | 58 | 303 |  0.555 | 0.135 |  0.642 |
| 10s | after | 17,559 |   -63.75 | [ -81.09,  -48.48] |   -50.00 |   -59.26 | [ -68.75,  -49.60] |  -1.16 |   -11.5 |  0.000 | 11 | 58 | 303 |  0.543 | 0.104 |  0.606 |
| 60s | before | 17,355 |   -76.19 | [-107.75,  -40.57] |  -100.00 |   -73.05 | [ -99.10,  -47.09] |  -1.39 |    -4.4 |  0.000 | 57 | 58 | 300 |  0.518 | 0.057 |  0.549 |
| 60s | after | 17,355 |   -59.76 | [ -91.41,  -23.96] |   -50.00 |   -56.60 | [ -82.82,  -30.93] |  -1.09 |    -3.4 |  0.001 | 57 | 58 | 300 |  0.514 | 0.042 |  0.536 |

## Bottom line

The original 96 % figure was a **conditional, event-time share** that overstated how consistent adverse selection is. Measured as realized spread, passive fills on 2019-12-30 lost 0.22–0.85 (AAPL), 0.39–0.66 (QQQ) half-spreads from 100ms to 60s, CIs excluding zero. The loss is concentrated in fills hit by level-clearing aggressive orders and does not revert within 60 s. Single day; shape over time differs by symbol. The path between those endpoints is not monotone for QQQ, so the range should not be read as a straight line.

Three things separate that from the headline figure, in order of size: the statistic was conditional on the mid having moved (a fifth to a third of fills see no move at all); five book events is a far shorter window than it sounds in a busy name; and same-nanosecond fills from one aggressor were counted as independent observations. The attribution table in each symbol section quantifies the last two and checks that they sum.

**Not established here.** Why the horizon profile differs between the two symbols is an open question — that information decays faster in the more liquid name is a *hypothesis* this study does not test. Nor is one session enough to generalise the horizon shape; the multi-day aggregation is the place for that.

