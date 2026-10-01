//
// Nexus-LOB :: LevelBitmap unit test (cpp_engine)
//
// Drives LevelBitmap with random set/clear operations at awkward sizes (1, 63,
// 64, 65, 4096, 4097, ...) and checks next()/prev() against a std::set oracle at
// random query points, including -1, 0, the last index and past-the-end.
//
//   g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include
//       cpp_engine/tests/bitmap_test.cpp -o bitmap_test && ./bitmap_test
//
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <set>

#include "nexus/level_bitmap.hpp"

using namespace nexus;

static long g_checks = 0;

static int fail(int line, const char* expr, std::size_t n, long step) {
    std::printf("FAIL line %d (n=%zu step=%ld): %s\n", line, n, step, expr);
    return 1;
}
#define CHECK(cond)                                         \
    do {                                                    \
        if (!(cond)) return fail(__LINE__, #cond, n, step); \
        ++g_checks;                                         \
    } while (0)

struct Lcg {
    std::uint64_t s;
    std::uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 33; }
};

static std::ptrdiff_t oracle_next(const std::set<std::ptrdiff_t>& s, std::ptrdiff_t i) {
    auto it = s.lower_bound(i < 0 ? 0 : i);
    return it == s.end() ? -1 : *it;
}
static std::ptrdiff_t oracle_prev(const std::set<std::ptrdiff_t>& s, std::ptrdiff_t i) {
    if (i < 0) return -1;
    auto it = s.upper_bound(i);
    return it == s.begin() ? -1 : *std::prev(it);
}

static int run(std::size_t n, long steps, std::uint64_t seed, unsigned density_pct) {
    LevelBitmap bm(n);
    std::set<std::ptrdiff_t> ref;
    Lcg rng{seed};
    const auto N = static_cast<std::ptrdiff_t>(n);

    for (long step = 0; step < steps; ++step) {
        const auto i = static_cast<std::ptrdiff_t>(rng.next() % n);
        if (rng.next() % 100 < density_pct) { bm.set(i); ref.insert(i); }
        else                                { bm.clear(i); ref.erase(i); }

        CHECK(bm.test(i) == (ref.count(i) == 1));

        // Random probes plus the edges.
        const std::ptrdiff_t probes[] = {
            static_cast<std::ptrdiff_t>(rng.next() % n), i, i - 1, i + 1, -1, 0, N - 1, N};
        for (std::ptrdiff_t q : probes) {
            if (q < N) CHECK(bm.next(q) == oracle_next(ref, q));
            if (q >= N) CHECK(bm.next(q) == -1);
            CHECK(bm.prev(q) == oracle_prev(ref, q < N ? q : N - 1));
        }
    }
    return 0;
}

int main() {
    const std::size_t sizes[] = {1, 2, 63, 64, 65, 127, 128, 4095, 4096, 4097,
                                 262143, 262144, 262145, 1000000};
    std::uint64_t seed = 0xB17B17ULL;
    for (std::size_t n : sizes) {
        // Sparse, balanced and dense regimes exercise empty words, partial words
        // and summary-bit maintenance on clear().
        for (unsigned density : {2u, 50u, 95u}) {
            const long steps = n < 5000 ? 4000 : 20000;
            if (int rc = run(n, steps, seed++, density)) return rc;
        }
    }
    std::printf("%ld checks, 0 failed\nALL PASS\n", g_checks);
    return 0;
}
