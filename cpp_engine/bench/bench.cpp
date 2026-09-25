//
// Nexus-LOB :: matching-engine benchmark + zero-alloc proof (cpp_engine)
// Owner: Person A.
//
// Measures the headline engine claims on the REAL engine:
//   1. throughput  — order-entry ops/sec (submit/cancel/modify)
//   2. latency     — mean / p50 / p90 / p99 / p99.9 / max per op (cycle-accurate
//                    via rdtsc, calibrated against steady_clock)
//   3. zero-alloc  — global operator new/delete counters around the timed loop:
//                    a nonzero count is a REAL finding (the claim is false).
//
// Standalone, portable C++20:
//   g++ -std=c++20 -O3 -I cpp_engine/include cpp_engine/bench/bench.cpp -o bench
//   ./bench                                   # both configs, default sizes
//   ./bench passive | crossing                # one config at default sizes
//   ./bench --ops 2000000 200000 crossing     # explicit throughput/latency sizes
//
// Methodology notes (read before trusting the numbers):
//   * A fresh book is seeded with resting liquidity OUTSIDE the timed window.
//   * Each config gets two passes with the SAME generator: a large pure-throughput
//     pass (wall clock only) and a smaller per-op latency pass (rdtsc). The op
//     itself is generated first; only the engine call is timed.
//   * The global alloc counter is reset AFTER construction + reservation, so it
//     counts ONLY engine-internal allocations during the timed loop.
//   * The executor's results are folded into an accumulator to defeat DCE.
//
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <algorithm>
#include <vector>

#include "nexus/limit_order_book.hpp"

using namespace nexus;

// ============================================================================
// Global allocation counters (this TU only): wrap operator new/delete.
// ============================================================================
namespace {
std::atomic<std::uint64_t> g_allocs{0};
std::atomic<std::uint64_t> g_frees{0};
} // namespace

void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
    g_frees.fetch_add(1, std::memory_order_relaxed);
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void* operator new[](std::size_t n)                   { return ::operator new(n); }
void operator delete[](void* p) noexcept              { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void operator delete(void* p, const std::nothrow_t&) noexcept { ::operator delete(p); }

// ============================================================================
// Cycle-accurate timer (x86-64): lfence-delimited rdtsc + ns calibration.
// ============================================================================
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
inline std::uint64_t rdtsc() noexcept {
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}
#else
inline std::uint64_t rdtsc() noexcept { return 0; }
#endif

double cycles_per_ns() {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t c0 = rdtsc();
    const auto spin = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - spin < std::chrono::milliseconds(50)) {}
    const auto t1 = std::chrono::steady_clock::now();
    const std::uint64_t c1 = rdtsc();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns > 0 ? double(c1 - c0) / ns : 1.0;
}

// Deterministic, allocation-free RNG for op generation (not part of the engine).
struct Lcg {
    std::uint64_t s;
    std::uint64_t next() noexcept { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 33; }
};

// One order op as the workload sees it (what the engine call will be).
struct Op {
    OrderId id;
    Side    side;
    Price   price;        // 0 => market (is_market)
    Qty     qty;
    TimeInForce tif;
    bool    is_market;
    bool    is_cancel;
    bool    is_modify;
};

// The thinner possible variadic seam: exactly the engine calls the bench times.
ExecResult exec_op(LimitOrderBook& book, const Op& op) {
    if (op.is_cancel) return book.cancel(op.id, 0);
    if (op.is_modify) return book.modify(op.id, op.price, op.qty, nullptr, 0);
    if (op.is_market) return book.submit_market(op.id, op.side, op.qty, nullptr, 0);
    return book.submit_limit(op.id, op.side, op.price, op.qty, op.tif, nullptr, 0);
}

struct Config {
    const char* name;
    double      aggressor_ratio;   // share of ops that cross liquidity
};

constexpr std::uint64_t kSeedOrders = 2'000;    // resting liquidity placed before timing
constexpr Price         kMid         = 50'000;

// Placers we can still cancel/modify: {id, price} of orders we know are resting.
struct Placed { OrderId id; Price price; };

// ============================================================================
// One run of `ops` order ops at the given aggressor ratio.
// `lat_ns` (pre-reserved) collects per-op engine-call latency when non-null.
// ============================================================================
struct RunStats {
    std::uint64_t allocs;
    std::uint64_t ops;
    std::uint64_t filled;
    double        wall_sec;
};

RunStats run_ops(const Config& cfg, std::uint64_t ops,
                 std::vector<std::uint32_t>* lat_ns) {
    LimitOrderBook book(1, 1'000'000, std::size_t(1) << 20);

    // Seed passive liquidity at a few levels either side of mid (outside timing).
    Lcg rng{0x5EED5EEDULL};
    for (OrderId id = 1; id <= kSeedOrders; ++id) {
        const bool  bid = (rng.next() & 1u) == 0;
        const Price off = 1 + (Price)(rng.next() % 6);
        book.submit_limit(id, bid ? Side::Bid : Side::Ask,
                          bid ? kMid - off : kMid + off,
                          10 + (Qty)(rng.next() % 90), TimeInForce::GTC);
    }

    // Bookkeeping for cancel/modify targets (pre-reserved; alloc-free in the loop).
    std::vector<Placed> placed;
    placed.reserve(kSeedOrders * 4);

    double cyc_per_ns = 1.0;
    if (lat_ns) {                                   // latency pass needs calibration
        cyc_per_ns = cycles_per_ns();
        lat_ns->resize(ops);                        // reserve BEFORE the counter reset
    }

    g_allocs.store(0, std::memory_order_relaxed);   // from here, only engine allocs count

    const auto t0 = std::chrono::steady_clock::now();
    OrderId id = kSeedOrders + 1;
    Price   mid = kMid;
    std::uint64_t filled_acc = 0;
    std::size_t   lat_idx = 0;

    const double wp = 1.0 - cfg.aggressor_ratio;    // passive rest
    const double wa = cfg.aggressor_ratio;           // aggressor cross
    const double wc = 0.05;                          // cancel
    const double wm = 0.05;                          // modify
    const double wTotal = wp + wa + wc + wm;

    for (std::uint64_t i = 0; i < ops; ++i) {
        // ---- generate the op (outside the timed window of the call) ----------
        const double u = double(rng.next() % 1000000u) / 1e6 * wTotal;
        const bool   bid = (rng.next() & 1u) == 0;

        Op op{};
        op.tif = TimeInForce::GTC;

        if (u < wp) {                               // rest a passive limit
            const Price off = 1 + (Price)(rng.next() % 4);
            op.id = id++; op.side = bid ? Side::Bid : Side::Ask;
            op.price = bid ? mid - off : mid + off;
            op.qty = 10 + (Qty)(rng.next() % 190);
        } else if (u < wp + wa) {                   // cross into the opposite side
            op.id = id++;
            op.side = bid ? Side::Bid : Side::Ask;
            if ((rng.next() & 1u) == 0) {           // half: sweep the whole side
                op.is_market = true;
                op.qty = 10 + (Qty)(rng.next() % 120);
            } else {                                // half: lift exactly the best quote
                const Price lvl = bid ? book.best_ask() : book.best_bid();
                op.price = lvl != 0 ? lvl : mid + (bid ? 1 : -1);
                op.qty = 10 + (Qty)(rng.next() % 120);
                op.tif = TimeInForce::IOC;
            }
        } else if (u < wp + wa + wc) {              // cancel a recent order
            if (!placed.empty()) { op.is_cancel = true; op.id = placed.back().id; }
        } else {                                    // modify: in-place size-down
            if (!placed.empty()) {
                op.is_modify = true;
                const Placed p = placed.back();
                op.id = p.id; op.price = p.price;
                op.qty = 1;                          // shrink by one
            }
        }

        // ---- the timed engine call ------------------------------------------
        ExecResult res{};
        if (lat_ns) {
            const std::uint64_t c0 = rdtsc();
            res = exec_op(book, op);
            const std::uint64_t c1 = rdtsc();
            // F24 fix: report actual latency without artificial 999,999 ns ceiling.
            const double ns = c1 > c0 ? double(c1 - c0) / cyc_per_ns : 0.0;
            (*lat_ns)[lat_idx++] = (std::uint32_t)(ns > 4294967295.0 ? 4294967295u : (std::uint32_t)ns);
        } else {
            res = exec_op(book, op);
        }

        filled_acc += res.filled;

        // F24 fix: maintain accurate active order IDs — evict the specific target on
        // cancel/modify rather than blindly popping the back of the vector.
        if (res.status == Status::Accepted || res.status == Status::PartiallyFilledResting) {
            placed.push_back({op.id, op.price});
        } else if (op.is_cancel || op.is_modify) {
            for (auto it = placed.begin(); it != placed.end(); ++it) {
                if (it->id == op.id) { placed.erase(it); break; }
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    RunStats s;
    s.allocs   = g_allocs.load(std::memory_order_relaxed);
    s.ops      = ops;
    s.filled   = filled_acc;
    s.wall_sec = std::chrono::duration<double>(t1 - t0).count();
    return s;
}

// ============================================================================
// Report one config: throughput pass (large) + latency pass (small).
// ============================================================================
void bench(const Config& cfg, std::uint64_t tp_ops, std::uint64_t lat_ops) {
    std::printf("--- config: %-9s aggressor_ratio=%.2f ---\n", cfg.name, cfg.aggressor_ratio);

    const RunStats tp = run_ops(cfg, tp_ops, /*lat_ns=*/nullptr);
    const std::uint64_t per_second = (std::uint64_t)(tp.ops / tp.wall_sec);

    std::vector<std::uint32_t> lat;
    const RunStats lt = run_ops(cfg, lat_ops, &lat);
    std::sort(lat.begin(), lat.end());
    const auto q   = [&](double p) { return lat[(std::size_t)(p * (double)lat.size() - 1)]; };
    double mean = 0;
    for (const auto x : lat) mean += x;
    mean /= (double)lat.size();

    std::printf("  ops          : %12llu  in %.2f s\n",
                (unsigned long long)tp.ops, tp.wall_sec);
    std::printf("  throughput   : %12llu ops/s\n", (unsigned long long)per_second);
    std::printf("  latency      : mean %8.1f  p50 %6.0f  p90 %6.0f  p99 %6.0f  p99.9 %6.0f  max %6.0f ns\n",
                mean, (double)q(0.50), (double)q(0.90), (double)q(0.99),
                (double)q(0.999), (double)lat.back());
    std::printf("  heap allocs  : %12llu in the timed loop  ", (unsigned long long)tp.allocs);
    if (tp.allocs == 0) std::printf("=> ZERO-ALLOC PROVEN\n");
    else                std::printf("<= engine ALLOCATED on the hot path (%.3f/op)\n",
                                    tp.ops ? (double)tp.allocs / (double)tp.ops : 0.0);
    // Latency pass re-runs the same workload at smaller n (per-op rdtsc dominant).
    std::printf("  latency-pass : %12llu ops, %12llu allocs (%.3f/op)\n",
                (unsigned long long)lt.ops, (unsigned long long)lt.allocs,
                lt.ops ? (double)lt.allocs / (double)lt.ops : 0.0);
    std::printf("  fills        : %12llu (sum over all order calls)\n\n",
                (unsigned long long)tp.filled);
}

int main(int argc, char** argv) {
    std::printf("=== Nexus-LOB :: matching-engine benchmark ===\n");
    std::printf("compiler   : GCC %d.%d.%d (C++20)\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#if defined(__x86_64__) || defined(_M_X64)
    std::printf("target     : x86-64\n");
#endif
    std::printf("\n");

    const Config configs[] = {
        {"passive",  0.10},   // resting-liquidity-heavy workload
        {"crossing", 0.55},   // heavy matching workload (fills + erases)
    };

    // Optional --ops override: `bench --ops <tp_ops> <lat_ops> [config]`.
    // Omit it to use the defaults (kDefThroughput / kDefLatency):
    //   ./bench                          # both configs, default sizes
    //   ./bench --ops 100000 30000       # quick smoke, both configs
    //   ./bench passive | crossing       # one config at default sizes
    //   ./bench --ops 100000 30000 crossing
    constexpr std::uint64_t kDefThroughput = 2'000'000;
    constexpr std::uint64_t kDefLatency    =   200'000;
    std::uint64_t tp_ops = kDefThroughput, lat_ops = kDefLatency;
    int argi = 1;
    if (argc >= 4 && 0 == std::strcmp(argv[1], "--ops")) {
        tp_ops = std::strtoull(argv[2], nullptr, 10);
        lat_ops = std::strtoull(argv[3], nullptr, 10);
        argi = 4;
    }

    if (argi < argc) {                              // optional single-config selection
        const char* want = argv[argi];
        for (const auto& c : configs)
            if (0 == std::strcmp(c.name, want)) { bench(c, tp_ops, lat_ops); return 0; }
        std::fprintf(stderr, "unknown config '%s' (try 'passive' or 'crossing')\n", want);
        return 2;
    }
    for (const auto& c : configs) bench(c, tp_ops, lat_ops);
    return 0;
}
