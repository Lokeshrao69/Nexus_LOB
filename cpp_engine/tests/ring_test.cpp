// Standalone correctness tests for nexus::ShmRing (subsystem 5).
// No framework / Python / CMake required:
//   g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include
//       cpp_engine/tests/ring_test.cpp -o ring_test && ./ring_test
//
// Exercises the REAL OS shared-memory path in-process: a producer thread streams
// BookStateView frames into the ring while this thread consumes them, then we
// verify order-preservation and drop-on-full semantics.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "nexus/shm_ring.hpp"

using namespace nexus;

namespace {

int g_failed = 0;
int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failed;                                                        \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond);\
        }                                                                      \
    } while (0)

// A recognisable synthetic frame: seq + a tilted L2 ladder + a trade print.
BookStateView make_frame(std::uint64_t i) {
    BookStateView v{};
    v.seq           = i;
    v.ts_ns         = static_cast<std::int64_t>(i * 1000);
    v.cum_volume    = i * 7;
    v.last_trade_px = 100 * static_cast<std::int64_t>(i) + 1;
    v.last_trade_sz = 10;
    v.last_trade_side = Side::None;
    v.bid_px[0] = 1'000 + static_cast<std::int64_t>(i);
    v.bid_sz[0] = i * 2 + 5;
    v.ask_px[0] = v.bid_px[0] + 5;
    v.version    = 2 * static_cast<std::uint32_t>(i) + 2;   // even == stable
    return v;
}

// Producer thread: publish `n` frames (returns count of successful publishes).
std::uint64_t publish_all(ShmRing& ring, std::uint64_t n) {
    std::uint64_t ok = 0;
    for (std::uint64_t i = 0; i < n; ++i) {
        const BookStateView v = make_frame(i);
        if (ring.publish(v)) ++ok;
    }
    return ok;
}

// Large ring, fast consumer: every frame must arrive, in order, uncorrupted.
void test_order_and_integrity() {
    std::printf("test_order_and_integrity\n");
    const char* kName = "nexus_test_order";
    ShmRing::destroy(kName);
    // Capacity above the frame count => even a fully-ahead producer can never
    // drop, so the zero-loss / strict-order assertions are deterministic.
    ShmRing ring(kName, /*capacity=*/16'384, ShmRing::Mode::Create);

    constexpr std::uint64_t kFrames = 10'000;
    std::thread pub([&] { publish_all(ring, kFrames); });

    std::uint64_t got = 0;
    BookStateView v;
    while (got < kFrames) {
        if (ring.try_read(v)) {
            CHECK(v.seq == got);                      // strict order, zero loss
            CHECK(v.version == 2 * static_cast<std::uint32_t>(got) + 2);  // copy intact
            CHECK(v.bid_sz[0] == got * 2 + 5);
            ++got;
        } else {
            std::this_thread::yield();
        }
    }
    pub.join();
    CHECK(got == kFrames);
    CHECK(ring.dropped() == 0);                       // consumer kept up
    std::printf("  read %llu frames, dropped %llu\n",
                (unsigned long long)got, (unsigned long long)ring.dropped());
    ShmRing::destroy(kName);
}

// Tiny ring, idle consumer: the producer must DROP the NEW overflow (never block /
// never corrupt). Only `capacity` frames fit; the consumer then drains exactly
// those earliest frames, strictly in order. A second handle ATTACHED to the same
// segment must see identical control state.
void test_drop_on_full() {
    std::printf("test_drop_on_full\n");
    const char* kName = "nexus_test_drop";
    ShmRing::destroy(kName);
    constexpr std::uint64_t kCap  = 4;
    constexpr std::uint64_t kSent = 100;
    ShmRing ring(kName, kCap, ShmRing::Mode::Create);

    // Publish everything before starting to read. Also prove a second ring that
    // ATTACHES the same segment sees identical state (the all-important seams).
    const std::uint64_t published = publish_all(ring, kSent);

    ShmRing viewer(kName, kCap, ShmRing::Mode::Attach);   // cross-handle view
    std::uint64_t got = 0;
    BookStateView v;
    while (ring.try_read(v)) {                           // drain the original handle
        ++got;
    }
    // Drop-new semantics: the capacity frames that FIT are seqs 0..kCap-1; the
    // rest were dropped at the producer. Earliest survive, strict order.
    CHECK(published == kCap);
    CHECK(got == kCap);
    CHECK(ring.dropped() == kSent - kCap);
    CHECK(v.seq == kCap - 1);                            // last drained == newest survivor
    // The attach-handle agrees on the same control state.
    CHECK(viewer.write_seq() == ring.write_seq());
    CHECK(viewer.read_seq() == ring.read_seq());
    CHECK(viewer.dropped()  == ring.dropped());
    ShmRing::destroy(kName);
}

#if !defined(_WIN32)
// Attaching to a segment whose control block carries a different magic/layout
// version must fail loudly instead of misreading offsets.
void test_layout_mismatch_rejected() {
    std::printf("test_layout_mismatch_rejected\n");
    const char* kName = "nexus_test_layout";
    ShmRing::destroy(kName);
    {
        ShmRing ring(kName, 4, ShmRing::Mode::Create);
        // Corrupt the layout-version stamp (offset 4) in the live mapping.
        int fd = shm_open(kName, O_RDWR, 0600);
        CHECK(fd >= 0);
        std::uint32_t bad = ShmRing::kLayoutVersion + 1;
        CHECK(pwrite(fd, &bad, sizeof bad, 4) == (ssize_t)sizeof bad);
        close(fd);
        bool threw = false;
        try { ShmRing viewer(kName, 4, ShmRing::Mode::Attach); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    ShmRing::destroy(kName);
}
#endif

// The slot payload is exactly the frozen 448-byte contract state.
void test_slot_abi() {
    std::printf("test_slot_abi\n");
    CHECK(ShmRing::kSlotBytes == sizeof(BookStateView));
    CHECK(sizeof(BookStateView) == 448);                // contract v1 lock
}

} // namespace

int main() {
    std::printf("=== nexus::ShmRing tests ===\n\n");
    test_slot_abi();
    test_order_and_integrity();
    test_drop_on_full();
#if !defined(_WIN32)
    test_layout_mismatch_rejected();
#endif

    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    if (g_failed == 0) std::printf("ALL PASS\n");
    return g_failed == 0 ? 0 : 1;
}