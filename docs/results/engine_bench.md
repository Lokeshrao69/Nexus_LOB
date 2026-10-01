# Matching-engine benchmark: occupancy bitmap fix

## Problem

After every order, the engine republishes the top 10 price levels per side and,
when a level empties, moves its best-price cursor to the next occupied level.
Both were done by stepping through the price array one level at a time and
skipping empty levels.

That is cheap when the book is dense. It is not when a side has fewer than 10
occupied levels, or when the next occupied level is far away: the top-10 walk
then runs all the way to the edge of the configured price band. Cost grew with
**band width**, not with the number of orders. On a 1,000,000-tick band with a
few levels quoted near the touch, every order touched about 30 MB of mostly
empty level structs.

## Fix

`cpp_engine/include/nexus/level_bitmap.hpp` adds a hierarchical occupancy
bitmap per side. Level 0 has one bit per price; each higher level has one bit
per 64-bit word below it, until the top fits in one word (4 levels for a
1M-tick band, 5 for the maximum 2^30). `next(i)` and `prev(i)` find the nearest
occupied level with a handful of count-trailing/leading-zero instructions,
however many empty prices lie in between. Setting or clearing a bit touches at
most one word per level, and stops early as soon as a word was already
non-empty (set) or stays non-empty (clear).

The engine uses it in three places:

- `refresh_best_bid_` / `refresh_best_ask_` (best price after a level empties),
- `publish_` (the top-10 ladder walk after every mutation),
- `can_fill_` (the fill-or-kill liquidity pre-check).

Memory is allocated once in the constructor; the order path still performs zero
heap allocations.

## Correctness

| Test | What it checks | Result |
|---|---|---|
| `bitmap_test` | `next`/`prev`/`test` vs a `std::set` oracle at 14 sizes (1 through 1,000,000, including 63/64/65 and 4095/4096/4097 word boundaries), sparse/balanced/dense | 6,120,000 checks pass |
| `lob_diff_test` | Engine vs a naive `std::map<price, deque>` reference book on random limit (GTC/IOC/FOK), market, cancel and modify orders, including duplicate ids, zero qty, out-of-band prices and a tiny pool. Compares every result, every fill, and the full published ladder after every call | 51,879,356 checks pass across dense, wide, 2^26-tick and pool-exhaustion scenarios |
| Existing suites | `lob_test`, `id_map_test`, `ring_test`, `abi_check`, `risk_test` | Pass |
| Python | Full suite plus pybind parity and stub-vs-engine diff tests | 401 passed, 1 skipped |

Two checks on the differential test itself: it also passes against the
**original** engine (so the fix changes performance, not behaviour), and it
fails at step 109 when a one-character bug is planted in the ask-side ladder
walk. All C++ tests were also run under AddressSanitizer and UBSan.

## Benchmark changes

The harness had problems of its own, fixed alongside:

- **Quadratic bookkeeping.** Cancels targeted the newest order but removed it
  with a linear scan from the front of the list. Now an O(1) swap-remove.
- **Pool exhaustion.** The 90%-passive workload barely cancelled, so the book
  grew until the 1M-slot pool filled and later orders became cheap rejects. The
  workload now holds a steady-state book of about 10,000 live orders and reports
  rejects (0 in every run).
- **Unrealistic cancels.** Cancels and modifies now target a random known order,
  not always the newest.
- **Allocation counter.** The harness's own bookkeeping vector could grow inside
  the timed loop and count as an "engine" allocation. It is now reserved up front.
- **Band coverage.** Each workload runs on a 2,000-tick band and on a
  1,000,000-tick band, plus a sparse configuration with orders spread over
  10,000 ticks.

## Results

2-vCPU cloud VM, GCC 13.3, `-O3 -march=native -DNDEBUG`. Same harness and same
random workload for both engines. 2,000,000 ops per throughput run and 200,000
per latency run, except where marked. Latency is per engine call measured with
`rdtsc` and includes ~20–40 ns of timer overhead. Throughput is wall clock and
includes op generation.

| Config | Before: ops/s | Before: p50 / p99 | After: ops/s | After: p50 / p99 / p99.9 |
|---|---:|---|---:|---|
| `passive` (2k-tick band) | 0.85 M | 1,122 / 1,636 ns | **5.60 M** | **167 / 419 / 520 ns** |
| `crossing` (2k-tick band) | 1.19 M | 1,176 / 2,327 ns | **6.98 M** | **168 / 433 / 568 ns** |
| `passive-wide` (1M-tick band) | 694 \* | 1.46 / 2.26 ms | **6.48 M** | **174 / 441 / 623 ns** |
| `crossing-wide` (1M-tick band) | 754 \* | 1.47 / 2.27 ms | **6.45 M** | **191 / 446 / 600 ns** |
| `sparse-wide` (orders over 10k ticks) | 1.82 M | 550 / 1,090 ns | **3.98 M** | **264 / 541 / 882 ns** |

\* 20,000 / 5,000 ops; the full run takes too long on the original engine.

Zero heap allocations inside the timed loop in every run.

Notes:

- Throughput no longer depends on band width: `passive` and `passive-wide` run
  the identical order stream on bands 500× apart and land within ~15%.
- `sparse-wide` was the original engine's best case (each side has more than 10
  occupied levels, so the top-10 walk stops early) and still roughly doubles.
- Max latencies of 70 µs–900 µs are single outliers on a shared cloud VM
  (scheduler preemption), not engine behaviour; p99.9 is the meaningful tail.
- The `crossing` workloads drain the book to a handful of orders, so many
  aggressive orders find little liquidity. A future workload should replenish
  liquidity at the rate it is taken.

## Reproduce

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG -I cpp_engine/include \
    cpp_engine/bench/bench.cpp -o bench && ./bench
```
