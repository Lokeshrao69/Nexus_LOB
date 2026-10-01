#pragma once
//
// Nexus-LOB :: cross-domain order-book state contract (C++ side)
// Owner: Person A (cpp_engine) — LAYOUT IS FROZEN BY CONTRACT.
//
// Any change to this struct is an ABI break. When you must change it:
//   1. bump kBookStateContractVersion,
//   2. update python_quant/nexus_quant/book_state.py (BOOK_STATE_DTYPE + DEPTH),
//   3. run tests/test_abi_parity.py.
// See bindings/CONTRACT.md for the full specification and rationale.
//
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nexus {

// Bump whenever the BookStateView layout or field semantics change.
inline constexpr std::uint32_t kBookStateContractVersion = 1;

// L2 ladder depth exposed across the seam, per side (index 0 == best level).
// TUNABLE KNOB: this is the primary modeling dial for the RL observation.
// Changing it is an ABI change — keep it in lock-step with DEPTH in
// python_quant/nexus_quant/book_state.py.
inline constexpr std::size_t kDepth = 10;

enum class Side : std::uint8_t { Bid = 0, Ask = 1, None = 2 };

// Fixed-size, trivially-copyable snapshot of the top-of-book ladder plus the
// last trade and bookkeeping. This POD is the SINGLE ABI that crosses both the
// Pybind11 seam (zero-copy views) and, later, the shared-memory ring to the
// dashboard (subsystem 5). Prices are INTEGER TICKS — never floating point —
// to stay bit-exact with the matching engine and the ITCH 5.0 feed.
//
// Field order is chosen so all 8-byte members precede all 4-byte members, which
// precede the 1-byte member. This yields deterministic, interior-padding-free
// layout on LP64 (MSVC/GCC/Clang). Do NOT reorder without re-locking the
// static_asserts below and regenerating the Python mirror.
struct BookStateView {
    // -- bookkeeping (8-byte) --
    std::uint64_t seq;             // monotonic engine / ITCH event sequence number
    std::int64_t  ts_ns;           // event timestamp, ns since epoch (ITCH clock)
    std::uint64_t cum_volume;      // cumulative shares traded this session
    std::uint64_t last_trade_sz;   // size of the most recent trade (shares)
    std::int64_t  last_trade_px;   // price of the most recent trade (ticks)

    // -- L2 ladder, 8-byte members (index 0 == best) --
    std::int64_t  bid_px[kDepth];  // bid prices in ticks; 0 == empty level
    std::uint64_t bid_sz[kDepth];  // aggregate resting size per bid level (shares)
    std::int64_t  ask_px[kDepth];  // ask prices in ticks; 0 == empty level
    std::uint64_t ask_sz[kDepth];  // aggregate resting size per ask level (shares)

    // -- 4-byte members --
    std::uint32_t version;         // publish counter: odd while publish_() is writing, even after.
                                   // A same-thread consistency marker, NOT a cross-thread seqlock:
                                   // fields are plain (non-atomic) memory. Share the state across
                                   // threads/processes through ShmRing, which copies whole slots.
    std::uint32_t bid_ct[kDepth];  // resting order count per bid level
    std::uint32_t ask_ct[kDepth];  // resting order count per ask level

    // -- 1-byte members --
    Side          last_trade_side; // aggressor side of the most recent trade

    // trailing padding to the struct's 8-byte alignment is implicit & deterministic
};

// --- ABI locks: these must hold for the zero-copy views and the Python mirror ---
static_assert(std::is_trivially_copyable_v<BookStateView>,
              "BookStateView must be memcpy/shmem-safe");
static_assert(std::is_standard_layout_v<BookStateView>,
              "BookStateView must have standard layout to mirror in NumPy");
static_assert(alignof(BookStateView) == 8, "unexpected alignment");
static_assert(sizeof(BookStateView) == 40 * kDepth + 48,
              "BookStateView size drift — update the Python mirror & parity test");

} // namespace nexus
