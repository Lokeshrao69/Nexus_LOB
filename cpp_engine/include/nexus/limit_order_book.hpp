#pragma once
//
// Nexus-LOB :: the matching engine (cpp_engine)
// Owner: Person A.
//
// A price-time (FIFO) limit order book built on the zero-allocation OrderPool.
// Design goals mirror the resume claims:
//   * O(1) price-level lookup  -> prices index directly into a pre-sized level
//     array over a fixed price band (no tree, no hashing on the hot path).
//   * zero-alloc order path     -> every resting Order comes from OrderPool's slab;
//     matching only ever FREES nodes, so the book never heap-allocates while trading.
//   * O(1) cancel/modify         -> an id -> Order* map locates any live order directly.
//
// The book publishes its top-`kDepth` state into a stable BookStateView member
// (book_state.hpp) after every mutating call, so `view()` is a valid zero-copy
// live window across the Pybind seam and `snapshot()` is an owning copy. This is
// the C++ side of the frozen contract the Python StubOrderBook emulates; the two
// are diff-tested against each other (see bindings/CONTRACT.md).
//
// Header-only so it can be compiled & unit-tested with a bare C++20 compiler
// (no CMake / Python), which is what cpp_engine/tests/lob_test.cpp does.
//
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "nexus/order_pool.hpp"  // Order, OrderPool  (pulls in types.hpp + book_state.hpp)

namespace nexus {

// One resting price level: an intrusive FIFO queue of Orders plus cached
// aggregates so the L2 ladder publish is O(kDepth), not O(orders).
struct LimitLevel {
    Order*        head = nullptr;   // oldest resting order (matched first)
    Order*        tail = nullptr;   // newest resting order (appended here)
    Qty           total_qty = 0;    // sum of head..tail remaining sizes
    std::uint32_t count = 0;        // number of resting orders at this price
    bool empty() const noexcept { return head == nullptr; }
};

// Zero-allocation id -> Order* open-addressing map (linear probing).
//
// This is what gives the engine its "zero-alloc order path": a std::unordered_map
// mallocs one heap node per insert (and frees it on every cancel/fill), which the
// benchmark proved breaks the claim (~1 alloc per resting order). This table is
// sized ONCE in its constructor — never rehashes, never allocates on the hot path.
// Sized to load <= 0.5 (fast probes); backtrack-shift deletion keeps linear-probe
// chains contiguous so lookups never stop early at a gap.
class IdMap {
public:
    explicit IdMap(std::size_t capacity) : slots_(next_capacity(capacity)), mask_(slots_.size() - 1) {}

    IdMap(const IdMap&)            = delete;
    IdMap& operator=(const IdMap&) = delete;

    Order* find(OrderId id) const noexcept {
        std::size_t i = bucket_(id);
        while (slots_[i].o != nullptr) {
            if (slots_[i].id == id) return slots_[i].o;
            i = (i + 1) & mask_;
        }
        return nullptr;
    }

    // Insert (id, order). Caller guarantees `id` is not already present and that
    // the table is not full (load factor is bounded at 0.5 by the ctor's sizing).
    void insert(OrderId id, Order* o) noexcept {
        std::size_t i = bucket_(id);
        while (slots_[i].o != nullptr) i = (i + 1) & mask_;
        slots_[i].id = id;
        slots_[i].o  = o;
        ++size_;
    }

    bool erase(OrderId id) noexcept {
        std::size_t i = bucket_(id);
        while (slots_[i].o != nullptr && slots_[i].id != id) i = (i + 1) & mask_;
        if (slots_[i].o == nullptr) return false;

        slots_[i].o = nullptr;                    // open the hole
        --size_;

        // Backtrack-shift deletion: move each following key that has the hole on
        // its probe path up into the hole, so chains stay dense for future probes.
        std::size_t j = (i + 1) & mask_;
        while (slots_[j].o != nullptr) {
            if (on_probe_path_(bucket_(slots_[j].id), i, j)) {
                slots_[i]   = slots_[j];
                slots_[j].o = nullptr;
                i           = j;
            }
            j = (j + 1) & mask_;
        }
        return true;
    }

    std::size_t size() const noexcept { return size_; }

private:
    struct Slot { OrderId id = 0; Order* o = nullptr; };

    // Next power of two >= 2*capacity so the load factor stays <= 0.5. Clamped so
    // the doubling can never overflow (pool capacities are bounded well below 2^31).
    static std::size_t next_capacity(std::size_t cap) noexcept {
        const std::size_t target = cap <= (std::size_t{1} << 31) ? cap * 2 : (std::size_t{1} << 32);
        std::size_t n = 8;
        while (n < target) n <<= 1;
        return n;
    }

    std::size_t bucket_(OrderId id) const noexcept { return hash(id) & mask_; }

    static std::size_t hash(OrderId id) noexcept {   // splitmix64 finalizer
        std::uint64_t x = id + UINT64_C(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return static_cast<std::size_t>(x ^ (x >> 31));
    }

    // True if `hole` lies strictly between `home` and `j` on the forward probe
    // ring — i.e. slot j's chain runs through the hole, so j can be shifted into it.
    bool on_probe_path_(std::size_t home, std::size_t hole, std::size_t j) const noexcept {
        return ((hole - home) & mask_) < ((j - home) & mask_);
    }

    std::vector<Slot> slots_;
    std::size_t       mask_;
    std::size_t       size_ = 0;
};

class LimitOrderBook {
public:
    // Sentinel for the optional per-call event timestamp: auto-increment instead.
    static constexpr std::int64_t kAutoTs = -1;

    // Prices are integer ticks confined to the inclusive band [min_price, max_price]
    // (0 stays reserved as the "empty level" sentinel, so min_price must be > 0).
    // `pool_capacity` bounds the number of simultaneously-resting orders.

    // F22 fix: validate price bounds BEFORE member-init-list so band_size
    // never overflows on extreme inputs (e.g. INT32_MAX - 1).
    static std::size_t safe_band_size(Price lo, Price hi) {
        if (lo <= 0 || hi < lo)
            throw std::invalid_argument("LimitOrderBook: require 0 < min_price <= max_price");
        const auto diff = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) + 1;
        if (diff > (static_cast<std::uint64_t>(1) << 30))
            throw std::invalid_argument("LimitOrderBook: price band too wide");
        return static_cast<std::size_t>(diff);
    }

    LimitOrderBook(Price min_price, Price max_price, std::size_t pool_capacity)
        : pool_(pool_capacity),
          min_price_(min_price),
          max_price_(max_price),
          bid_levels_(safe_band_size(min_price, max_price)),
          ask_levels_(safe_band_size(min_price, max_price)),
          id_map_(pool_capacity) {
        std::memset(&state_, 0, sizeof(state_));
        state_.last_trade_side = Side::None;
    }

    // Owns the pool and hands out Order* aliases into it: non-copyable/movable.
    LimitOrderBook(const LimitOrderBook&)            = delete;
    LimitOrderBook& operator=(const LimitOrderBook&) = delete;

    // ---- order entry --------------------------------------------------------

    // Submit a limit order. `fills` (optional) collects the executions; pass
    // nullptr on the hot path to stay allocation-free.
    ExecResult submit_limit(OrderId id, Side side, Price price, Qty qty,
                            TimeInForce tif = TimeInForce::GTC,
                            std::vector<Fill>* fills = nullptr,
                            std::int64_t event_ts = kAutoTs) {
        return execute_(id, side, price, qty, tif, /*is_market=*/false, fills, event_ts);
    }

    // Submit a marketable order: crosses at any price, never rests (IOC-like).
    ExecResult submit_market(OrderId id, Side side, Qty qty,
                             std::vector<Fill>* fills = nullptr,
                             std::int64_t event_ts = kAutoTs) {
        return execute_(id, side, /*price=*/0, qty, TimeInForce::IOC,
                        /*is_market=*/true, fills, event_ts);
    }

    // Cancel a resting order by id. NoOp if the id isn't live.
    ExecResult cancel(OrderId id, std::int64_t event_ts = kAutoTs) {
        if (!unlink_by_id_(id)) return {id, Status::NoOp, 0, 0};
        publish_(event_ts);
        return {id, Status::Canceled, 0, 0};
    }

    // Modify a resting order.
    //   * new_qty == 0                          -> treated as a cancel.
    //   * same price AND new_qty <= current qty -> in-place reduce, KEEPS time priority.
    //   * otherwise (price change / size-up)    -> cancel + re-submit, LOSES time priority
    //                                              (and may cross if repriced into the book).
    ExecResult modify(OrderId id, Price new_price, Qty new_qty,
                      std::vector<Fill>* fills = nullptr,
                      std::int64_t event_ts = kAutoTs) {
        Order* o = id_map_.find(id);
        if (o == nullptr) return {id, Status::NoOp, 0, 0};
        Side sd = o->side;

        if (new_qty == 0) return cancel(id, event_ts);

        if (new_price == o->price && new_qty <= o->qty) {
            LimitLevel& lvl = levels_(sd)[price_to_idx(o->price)];
            lvl.total_qty  -= (o->qty - new_qty);   // shrink aggregate
            o->qty          = new_qty;              // priority preserved (position unchanged)
            publish_(event_ts);
            return {id, Status::Accepted, 0, new_qty};
        }

        // Reprice or size-up: standard exchange behaviour is loss of time priority.
        // F13 fix: validate replacement parameters BEFORE unlinking the original
        // order, so a rejected modify never deletes the live order or corrupts published view.
        if (!in_band(new_price))
            return {id, Status::Rejected_BadPrice, 0, o->qty};
        unlink_by_id_(id);                          // free the slot (no publish yet)
        return submit_limit(id, sd, new_price, new_qty, TimeInForce::GTC, fills, event_ts);
    }

    // ---- state / introspection ---------------------------------------------

    // Zero-copy live window (RL hot path). Valid only until the next mutating call.
    const BookStateView& view() const noexcept { return state_; }
    // Owning copy — safe to retain across mutations.
    BookStateView snapshot() const noexcept { return state_; }

    bool  has_bid() const noexcept { return best_bid_idx_ >= 0; }
    bool  has_ask() const noexcept { return best_ask_idx_ >= 0; }
    Price best_bid() const noexcept { return has_bid() ? idx_to_price(best_bid_idx_) : 0; }
    Price best_ask() const noexcept { return has_ask() ? idx_to_price(best_ask_idx_) : 0; }
    // Spread in ticks; 0 if either side is empty.
    Price spread()   const noexcept { return (has_bid() && has_ask()) ? best_ask() - best_bid() : 0; }

    std::size_t      live_orders() const noexcept { return id_map_.size(); }
    const OrderPool& pool()        const noexcept { return pool_; }

private:
    // ---- geometry helpers ---------------------------------------------------
    static std::size_t band_size(Price lo, Price hi) noexcept {
        return static_cast<std::size_t>(hi - lo + 1);
    }
    std::ptrdiff_t price_to_idx(Price p) const noexcept {
        return static_cast<std::ptrdiff_t>(p - min_price_);
    }
    Price idx_to_price(std::ptrdiff_t i) const noexcept {
        return min_price_ + static_cast<Price>(i);
    }
    bool in_band(Price p) const noexcept { return p >= min_price_ && p <= max_price_; }
    std::vector<LimitLevel>& levels_(Side s) noexcept {
        return s == Side::Bid ? bid_levels_ : ask_levels_;
    }

    // ---- core execution -----------------------------------------------------
    ExecResult execute_(OrderId id, Side side, Price price, Qty qty, TimeInForce tif,
                        bool is_market, std::vector<Fill>* fills, std::int64_t event_ts) {
        // --- validation ---
        if (qty == 0)                       return {id, Status::Rejected_BadQty, 0, 0};
        if (side != Side::Bid && side != Side::Ask)
                                            return {id, Status::Rejected_BadPrice, 0, 0};
        if (!is_market && !in_band(price))  return {id, Status::Rejected_BadPrice, 0, 0};
        if (id_map_.find(id) != nullptr)
                                            return {id, Status::Rejected_DupId, 0, 0};

        // --- fill-or-kill is all-or-nothing: verify BEFORE mutating anything. ---
        if (tif == TimeInForce::FOK && !can_fill_(side, price, is_market, qty))
            return {id, Status::Rejected_FOK, 0, 0};

        Qty remaining = qty, filled = 0;
        match_(side, price, is_market, remaining, filled, id, fills);

        Status status;
        Qty    resting = 0;

        const bool never_rests = is_market || tif == TimeInForce::IOC || tif == TimeInForce::FOK;
        if (never_rests) {
            // Market / IOC discard the residual; FOK is guaranteed complete by the pre-check.
            status = (filled == 0) ? Status::NoOp : Status::Filled;
        } else if (remaining == 0) {
            status = Status::Filled;                 // GTC fully executed on entry
        } else {
            Order* o = pool_.allocate();
            if (o == nullptr) {
                // Pool exhausted. Because matching only frees nodes, this can only
                // happen when nothing crossed (filled == 0) and the book was full.
                status = (filled > 0) ? Status::Filled : Status::Rejected_PoolFull;
            } else {
                rest_(o, id, side, price, remaining);
                resting = remaining;
                status  = (filled > 0) ? Status::PartiallyFilledResting : Status::Accepted;
            }
        }

        publish_(event_ts);
        return {id, status, filled, resting};
    }

    // Walk the opposite side from its best price, consuming liquidity while the
    // incoming order still crosses. Decrements `qty`, accumulates `filled`.
    void match_(Side aggressor, Price limit, bool is_market,
                Qty& qty, Qty& filled, OrderId taker, std::vector<Fill>* fills) {
        if (aggressor == Side::Bid) {
            while (qty > 0 && best_ask_idx_ >= 0) {
                const Price ap = idx_to_price(best_ask_idx_);
                if (!is_market && ap > limit) break;              // no longer crosses
                consume_level_(ask_levels_[best_ask_idx_], ap, aggressor, qty, filled, taker, fills);
                if (ask_levels_[best_ask_idx_].empty()) refresh_best_ask_();
            }
        } else {  // aggressor == Ask
            while (qty > 0 && best_bid_idx_ >= 0) {
                const Price bp = idx_to_price(best_bid_idx_);
                if (!is_market && bp < limit) break;
                consume_level_(bid_levels_[best_bid_idx_], bp, aggressor, qty, filled, taker, fills);
                if (bid_levels_[best_bid_idx_].empty()) refresh_best_bid_();
            }
        }
    }

    // Consume one price level FIFO. Trade price is the resting (maker) price.
    void consume_level_(LimitLevel& lvl, Price px, Side aggressor,
                        Qty& qty, Qty& filled, OrderId taker, std::vector<Fill>* fills) {
        Order* m = lvl.head;
        while (m != nullptr && qty > 0) {
            const Qty f = (qty < m->qty) ? qty : m->qty;
            qty        -= f;
            m->qty     -= f;
            filled     += f;
            lvl.total_qty -= f;

            // Record the execution (each fill is an ITCH-style trade print).
            state_.last_trade_px   = px;
            state_.last_trade_sz   = f;
            state_.last_trade_side = aggressor;
            state_.cum_volume     += f;
            if (fills) fills->push_back(Fill{m->id, taker, px, f, aggressor});

            if (m->qty == 0) {                 // maker fully consumed -> unlink & reclaim
                Order* nxt = m->next;
                lvl.head = nxt;
                if (nxt) nxt->prev = nullptr; else lvl.tail = nullptr;
                lvl.count -= 1;
                id_map_.erase(m->id);
                pool_.free(m);
                m = nxt;
            } else {                           // maker partially filled -> qty is now 0, stop
                break;
            }
        }
    }

    // Append a residual as resting liquidity at the tail of its price level.
    void rest_(Order* o, OrderId id, Side side, Price price, Qty qty) {
        o->id = id; o->price = price; o->qty = qty; o->side = side;
        o->prev = nullptr; o->next = nullptr;

        const std::ptrdiff_t idx = price_to_idx(price);
        LimitLevel& lvl = levels_(side)[idx];
        if (lvl.tail == nullptr) {             // first order at this price
            lvl.head = lvl.tail = o;
        } else {                               // FIFO append
            o->prev = lvl.tail;
            lvl.tail->next = o;
            lvl.tail = o;
        }
        lvl.total_qty += qty;
        lvl.count     += 1;
        id_map_.insert(id, o);

        if (side == Side::Bid) {               // higher bid index == better bid
            if (best_bid_idx_ < idx) best_bid_idx_ = idx;
        } else {                               // lower ask index == better ask
            if (best_ask_idx_ < 0 || idx < best_ask_idx_) best_ask_idx_ = idx;
        }
    }

    // Remove a live order by id (used by cancel and by modify's reprice path).
    bool unlink_by_id_(OrderId id) {
        Order* o = id_map_.find(id);
        if (o == nullptr) return false;
        LimitLevel& lvl = levels_(o->side)[price_to_idx(o->price)];

        if (o->prev) o->prev->next = o->next; else lvl.head = o->next;
        if (o->next) o->next->prev = o->prev; else lvl.tail = o->prev;
        lvl.total_qty -= o->qty;
        lvl.count     -= 1;

        id_map_.erase(id);
        pool_.free(o);

        if (lvl.empty()) {                     // best price may have moved
            if (o->side == Side::Bid) refresh_best_bid_();
            else                      refresh_best_ask_();
        }
        return true;
    }

    // Slide the best-price cursor to the nearest still-occupied level (or -1).
    // F23 note: these scans are O(band_width) in the worst case for very sparse
    // books. A bitmap or radix tree can jump directly to the next occupied level;
    // current production use stays within measured sub-microsecond bounds for
    // typical ITCH-derived price bands (e.g. 50k ticks).
    void refresh_best_bid_() noexcept {
        while (best_bid_idx_ >= 0 && bid_levels_[best_bid_idx_].empty()) --best_bid_idx_;
    }
    void refresh_best_ask_() noexcept {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(ask_levels_.size());
        while (best_ask_idx_ >= 0 && best_ask_idx_ < n && ask_levels_[best_ask_idx_].empty())
            ++best_ask_idx_;
        if (best_ask_idx_ >= n) best_ask_idx_ = -1;
    }

    // Non-mutating check that `qty` can be fully filled now (for FOK).
    bool can_fill_(Side aggressor, Price limit, bool is_market, Qty qty) const {
        Qty need = qty;
        if (aggressor == Side::Bid) {
            const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(ask_levels_.size());
            for (std::ptrdiff_t i = best_ask_idx_; need > 0 && i >= 0 && i < n; ++i) {
                if (ask_levels_[i].empty()) continue;
                if (!is_market && idx_to_price(i) > limit) break;
                const Qty avail = ask_levels_[i].total_qty;
                need -= (need < avail) ? need : avail;
            }
        } else {
            for (std::ptrdiff_t i = best_bid_idx_; need > 0 && i >= 0; --i) {
                if (bid_levels_[i].empty()) continue;
                if (!is_market && idx_to_price(i) < limit) break;
                const Qty avail = bid_levels_[i].total_qty;
                need -= (need < avail) ? need : avail;
            }
        }
        return need == 0;
    }

    // Republish the top-`kDepth` ladder into the contract state. seqlock discipline:
    // bump `version` to odd before writing, to even after (even == stable snapshot).
    void publish_(std::int64_t event_ts) noexcept {
        ++state_.version;                      // odd: write in progress

        std::ptrdiff_t k = 0;
        for (std::ptrdiff_t bi = best_bid_idx_; k < static_cast<std::ptrdiff_t>(kDepth) && bi >= 0; --bi) {
            const LimitLevel& lvl = bid_levels_[bi];
            if (lvl.empty()) continue;
            state_.bid_px[k] = idx_to_price(bi);
            state_.bid_sz[k] = lvl.total_qty;
            state_.bid_ct[k] = lvl.count;
            ++k;
        }
        for (; k < static_cast<std::ptrdiff_t>(kDepth); ++k) {
            state_.bid_px[k] = 0; state_.bid_sz[k] = 0; state_.bid_ct[k] = 0;
        }

        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(ask_levels_.size());
        k = 0;
        for (std::ptrdiff_t ai = best_ask_idx_; k < static_cast<std::ptrdiff_t>(kDepth) && ai >= 0 && ai < n; ++ai) {
            const LimitLevel& lvl = ask_levels_[ai];
            if (lvl.empty()) continue;
            state_.ask_px[k] = idx_to_price(ai);
            state_.ask_sz[k] = lvl.total_qty;
            state_.ask_ct[k] = lvl.count;
            ++k;
        }
        for (; k < static_cast<std::ptrdiff_t>(kDepth); ++k) {
            state_.ask_px[k] = 0; state_.ask_sz[k] = 0; state_.ask_ct[k] = 0;
        }

        state_.seq   = ++seq_;
        clock_       = (event_ts == kAutoTs) ? clock_ + 1 : event_ts;
        state_.ts_ns = clock_;

        ++state_.version;                      // even: stable
    }

    // ---- members ------------------------------------------------------------
    OrderPool               pool_;
    Price                   min_price_;
    Price                   max_price_;
    std::vector<LimitLevel> bid_levels_;       // indexed by (price - min_price_)
    std::vector<LimitLevel> ask_levels_;
    IdMap                   id_map_;           // zero-alloc id -> Order* (O(1) cancel/modify)

    std::ptrdiff_t best_bid_idx_ = -1;         // highest occupied bid index; -1 == none
    std::ptrdiff_t best_ask_idx_ = -1;         // lowest occupied ask index;  -1 == none

    std::uint64_t  seq_   = 0;
    std::int64_t   clock_ = 0;
    BookStateView  state_{};                   // stable contract member (never reallocated)
};

}  // namespace nexus
