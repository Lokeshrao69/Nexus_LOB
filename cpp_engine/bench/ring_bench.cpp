//
// Nexus-LOB :: ShmRing throughput benchmark
//
// One producer thread publishes N frames into a real OS shared-memory ring while
// the main thread consumes them. The producer retries when the ring is full (so
// every frame is delivered) and the consumer spins when it is empty. Reports
// frames per second and the number of full-ring retries, for several capacities.
// Results depend heavily on core placement; run on an otherwise idle machine and
// take the median of several runs.
//
//   g++ -std=c++20 -O3 -march=native -DNDEBUG -pthread -I cpp_engine/include
//       cpp_engine/bench/ring_bench.cpp -o ring_bench -lrt && ./ring_bench
//
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "nexus/shm_ring.hpp"

using namespace nexus;

struct Result { double mfps; std::uint64_t retries; bool ok; };

static Result run_once(std::size_t capacity, std::uint64_t frames) {
    const char* kName = "nexus_ring_bench";
    ShmRing::destroy(kName);
    ShmRing ring(kName, capacity, ShmRing::Mode::Create);

    std::uint64_t retries = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::thread producer([&] {
        BookStateView v{};
        for (std::uint64_t i = 0; i < frames; ++i) {
            v.seq = i;
            v.bid_sz[0] = i;
            while (!ring.publish(v)) ++retries;     // full: retry until delivered
        }
    });

    bool ok = true;
    BookStateView v{};
    for (std::uint64_t got = 0; got < frames;) {
        if (ring.try_read(v)) {
            ok &= (v.seq == got && v.bid_sz[0] == got);
            ++got;
        }
    }
    producer.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ShmRing::destroy(kName);
    return {frames / s / 1e6, retries, ok};
}

int main(int argc, char** argv) {
    const std::uint64_t frames = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 5'000'000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    std::printf("=== ShmRing throughput: %llu frames x %d runs, 448-byte slots ===\n",
                (unsigned long long)frames, reps);
    for (std::size_t cap : {64u, 1024u, 16384u}) {
        std::vector<double> m;
        std::uint64_t retries = 0;
        bool ok = true;
        for (int r = 0; r < reps; ++r) {
            const Result res = run_once(cap, frames);
            m.push_back(res.mfps); retries += res.retries; ok &= res.ok;
        }
        std::sort(m.begin(), m.end());
        std::printf("  capacity %6zu : median %6.2f M frames/s  (min %6.2f, max %6.2f)"
                    "  full-ring retries/run %llu  %s\n",
                    cap, m[m.size() / 2], m.front(), m.back(),
                    (unsigned long long)(retries / reps), ok ? "order OK" : "ORDER BROKEN");
    }
    return 0;
}
