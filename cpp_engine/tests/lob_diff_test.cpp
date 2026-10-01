//
// Nexus-LOB :: differential test, LimitOrderBook vs a naive reference book
//
// The reference book is deliberately simple: std::map<price, std::deque<order>>
// per side, with every rule written out the obvious way. Both books receive the
// same random stream of limit (GTC/IOC/FOK), market, cancel and modify orders,
// and after every call we compare:
//   * the ExecResult (status, filled, resting),
//   * every Fill (maker, taker, price, qty, aggressor side), in order,
//   * the full published top-10 ladder on both sides, last trade and volume.
//
// Runs use both a narrow band and a very wide, sparse band (where the occupancy
// bitmap is doing all the work), plus a tiny pool so Rejected_PoolFull fires.
//
//   g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include
//       cpp_engine/tests/lob_diff_test.cpp -o lob_diff_test && ./lob_diff_test
//
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "nexus/limit_order_book.hpp"

using namespace nexus;

static long g_checks = 0;
static long g_step   = 0;

static int fail(int line, const char* expr) {
    std::printf("FAIL line %d (step %ld): %s\n", line, g_step, expr);
    return 1;
}
#define CHECK(cond)                                \
    do {                                           \
        if (!(cond)) return fail(__LINE__, #cond); \
        ++g_checks;                                \
    } while (0)

// ---------------------------------------------------------------------------
// Naive reference book
// ---------------------------------------------------------------------------
class RefBook {
public:
    RefBook(Price lo, Price hi, std::size_t cap) : lo_(lo), hi_(hi), cap_(cap) {}

    ExecResult limit(OrderId id, Side side, Price px, Qty qty, TimeInForce tif,
                     std::vector<Fill>* fills) {
        return exec(id, side, px, qty, tif, false, fills);
    }
    ExecResult market(OrderId id, Side side, Qty qty, std::vector<Fill>* fills) {
        return exec(id, side, 0, qty, TimeInForce::IOC, true, fills);
    }
    ExecResult cancel(OrderId id) {
        if (!erase(id)) return {id, Status::NoOp, 0, 0};
        return {id, Status::Canceled, 0, 0};
    }
    ExecResult modify(OrderId id, Price px, Qty qty, std::vector<Fill>* fills) {
        auto it = where_.find(id);
        if (it == where_.end()) return {id, Status::NoOp, 0, 0};
        if (qty == 0) return cancel(id);
        const Side  side = it->second.side;
        const Price cur  = it->second.price;
        Live& o = find_live(id);
        if (px == cur && qty <= o.qty) {
            o.qty = qty;
            return {id, Status::Accepted, 0, qty};
        }
        if (px < lo_ || px > hi_) return {id, Status::Rejected_BadPrice, 0, o.qty};
        erase(id);
        return exec(id, side, px, qty, TimeInForce::GTC, false, fills);
    }

    // Top-of-book in the same shape the engine publishes.
    void ladder(Side side, Price* px, Qty* sz, std::uint32_t* ct) const {
        std::size_t k = 0;
        auto emit = [&](Price p, const std::deque<Live>& q) {
            if (k >= kDepth) return;
            Qty t = 0;
            for (const auto& o : q) t += o.qty;
            px[k] = p; sz[k] = t; ct[k] = static_cast<std::uint32_t>(q.size()); ++k;
        };
        if (side == Side::Bid) for (auto it = bids_.rbegin(); it != bids_.rend(); ++it) emit(it->first, it->second);
        else                   for (auto it = asks_.begin();  it != asks_.end();  ++it) emit(it->first, it->second);
        for (; k < kDepth; ++k) { px[k] = 0; sz[k] = 0; ct[k] = 0; }
    }

    Price last_px = 0; Qty last_sz = 0; Side last_side = Side::None; Qty cum = 0;

private:
    struct Live { OrderId id; Qty qty; };
    struct Loc  { Side side; Price price; };

    ExecResult exec(OrderId id, Side side, Price px, Qty qty, TimeInForce tif,
                    bool is_market, std::vector<Fill>* fills) {
        if (qty == 0) return {id, Status::Rejected_BadQty, 0, 0};
        if (!is_market && (px < lo_ || px > hi_)) return {id, Status::Rejected_BadPrice, 0, 0};
        if (where_.count(id)) return {id, Status::Rejected_DupId, 0, 0};

        auto crosses = [&](Price lvl) {
            return is_market || (side == Side::Bid ? lvl <= px : lvl >= px);
        };
        if (tif == TimeInForce::FOK) {
            Qty avail = 0;
            if (side == Side::Bid) {
                for (auto& [p, q] : asks_) {
                    if (!crosses(p)) break;
                    for (auto& o : q) avail += o.qty;
                }
            } else {
                for (auto it = bids_.rbegin(); it != bids_.rend(); ++it) {
                    if (!crosses(it->first)) break;
                    for (auto& o : it->second) avail += o.qty;
                }
            }
            if (avail < qty) return {id, Status::Rejected_FOK, 0, 0};
        }

        Qty remaining = qty, filled = 0;
        auto& opp = (side == Side::Bid) ? asks_ : bids_;
        while (remaining > 0 && !opp.empty()) {
            auto lvl = (side == Side::Bid) ? opp.begin() : std::prev(opp.end());
            if (!crosses(lvl->first)) break;
            auto& q = lvl->second;
            while (remaining > 0 && !q.empty()) {
                Live& m = q.front();
                const Qty f = remaining < m.qty ? remaining : m.qty;
                remaining -= f; m.qty -= f; filled += f;
                last_px = lvl->first; last_sz = f; last_side = side; cum += f;
                if (fills) fills->push_back(Fill{m.id, id, lvl->first, f, side});
                if (m.qty == 0) { where_.erase(m.id); q.pop_front(); }
            }
            if (q.empty()) opp.erase(lvl);
        }

        const bool never_rests = is_market || tif != TimeInForce::GTC;
        if (never_rests) return {id, filled ? Status::Filled : Status::NoOp, filled, 0};
        if (remaining == 0) return {id, Status::Filled, filled, 0};
        if (where_.size() >= cap_)
            return {id, filled ? Status::Filled : Status::Rejected_PoolFull, filled, 0};
        auto& own = (side == Side::Bid) ? bids_ : asks_;
        own[px].push_back(Live{id, remaining});
        where_[id] = Loc{side, px};
        return {id, filled ? Status::PartiallyFilledResting : Status::Accepted, filled, remaining};
    }

    Live& find_live(OrderId id) {
        const Loc& l = where_.at(id);
        auto& q = (l.side == Side::Bid ? bids_ : asks_).at(l.price);
        for (auto& o : q) if (o.id == id) return o;
        std::abort();
    }

    bool erase(OrderId id) {
        auto it = where_.find(id);
        if (it == where_.end()) return false;
        auto& book = (it->second.side == Side::Bid) ? bids_ : asks_;
        auto lvl = book.find(it->second.price);
        auto& q = lvl->second;
        for (auto o = q.begin(); o != q.end(); ++o) if (o->id == id) { q.erase(o); break; }
        if (q.empty()) book.erase(lvl);
        where_.erase(it);
        return true;
    }

    Price lo_, hi_;
    std::size_t cap_;
    std::map<Price, std::deque<Live>> bids_, asks_;
    std::unordered_map<OrderId, Loc> where_;
};

struct Lcg {
    std::uint64_t s;
    std::uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 33; }
};

static int compare(const LimitOrderBook& eng, const RefBook& ref) {
    const BookStateView& v = eng.view();
    Price px[kDepth]; Qty sz[kDepth]; std::uint32_t ct[kDepth];
    ref.ladder(Side::Bid, px, sz, ct);
    for (std::size_t k = 0; k < kDepth; ++k) {
        CHECK(v.bid_px[k] == px[k]); CHECK(v.bid_sz[k] == sz[k]); CHECK(v.bid_ct[k] == ct[k]);
    }
    ref.ladder(Side::Ask, px, sz, ct);
    for (std::size_t k = 0; k < kDepth; ++k) {
        CHECK(v.ask_px[k] == px[k]); CHECK(v.ask_sz[k] == sz[k]); CHECK(v.ask_ct[k] == ct[k]);
    }
    CHECK(v.last_trade_px == ref.last_px);
    CHECK(v.last_trade_sz == ref.last_sz);
    CHECK(v.last_trade_side == ref.last_side);
    CHECK(v.cum_volume == ref.cum);
    CHECK(v.version % 2 == 0);
    CHECK(eng.best_bid() == (v.bid_px[0]));
    CHECK(eng.best_ask() == (v.ask_px[0]));
    return 0;
}

// `spread` controls how far from mid orders land: small => dense book,
// large => sparse book with long runs of empty price levels.
static int run(Price lo, Price hi, std::size_t cap, Price mid, Price spread,
               long steps, std::uint64_t seed) {
    LimitOrderBook eng(lo, hi, cap);
    RefBook        ref(lo, hi, cap);
    Lcg rng{seed};
    std::vector<OrderId> ids;          // ids we have submitted (some may be gone)
    OrderId next_id = 1;
    std::vector<Fill> fe, fr;

    for (long step = 0; step < steps; ++step) {
        g_step = step;
        fe.clear(); fr.clear();
        const unsigned u    = static_cast<unsigned>(rng.next() % 100);
        const Side     side = (rng.next() & 1) ? Side::Bid : Side::Ask;
        const Qty      qty  = static_cast<Qty>(rng.next() % 50);   // includes 0 -> BadQty
        Price px = mid + static_cast<Price>(rng.next() % (2 * spread + 1)) - spread;
        if (rng.next() % 50 == 0) px = hi + 1 + static_cast<Price>(rng.next() % 3);   // out of band
        if (rng.next() % 50 == 0) px = lo;                                            // band edges
        if (rng.next() % 50 == 0) px = hi;

        ExecResult a{}, b{};
        if (u < 50) {                                  // limit, mostly GTC
            const unsigned t = static_cast<unsigned>(rng.next() % 10);
            const TimeInForce tif = t < 7 ? TimeInForce::GTC : t < 9 ? TimeInForce::IOC : TimeInForce::FOK;
            OrderId id = next_id++;
            if (!ids.empty() && rng.next() % 40 == 0) id = ids[rng.next() % ids.size()];   // dup id
            else ids.push_back(id);
            a = eng.submit_limit(id, side, px, qty, tif, &fe);
            b = ref.limit(id, side, px, qty, tif, &fr);
        } else if (u < 60) {                           // market
            const OrderId id = next_id++; ids.push_back(id);
            a = eng.submit_market(id, side, qty, &fe);
            b = ref.market(id, side, qty, &fr);
        } else if (u < 82) {                           // cancel (live, dead or unknown)
            const OrderId id = ids.empty() || rng.next() % 10 == 0 ? next_id + 1000
                                                                     : ids[rng.next() % ids.size()];
            a = eng.cancel(id);
            b = ref.cancel(id);
        } else {                                       // modify: shrink, reprice, grow, zero
            if (ids.empty()) continue;
            const OrderId id = ids[rng.next() % ids.size()];
            a = eng.modify(id, px, qty, &fe);
            b = ref.modify(id, px, qty, &fr);
        }

        CHECK(a.id == b.id);
        CHECK(a.status == b.status);
        CHECK(a.filled == b.filled);
        CHECK(a.resting == b.resting);
        CHECK(fe.size() == fr.size());
        for (std::size_t i = 0; i < fe.size(); ++i) {
            CHECK(fe[i].maker_id == fr[i].maker_id);
            CHECK(fe[i].taker_id == fr[i].taker_id);
            CHECK(fe[i].px == fr[i].px);
            CHECK(fe[i].qty == fr[i].qty);
            CHECK(fe[i].aggressor == fr[i].aggressor);
        }
        if (int rc = compare(eng, ref)) return rc;
    }
    return 0;
}

int main() {
    struct Case { const char* name; Price lo, hi; std::size_t cap; Price mid, spread; long steps; };
    const Case cases[] = {
        {"dense, narrow band",          1,       200,       4096,  100,     8,      200000},
        {"sparse, wide band",           1,       1'000'000, 4096,  500'000, 20'000, 200000},
        {"very sparse, huge band",      1,       1 << 26,   2048,  1 << 25, 1 << 22, 100000},
        {"tiny pool (PoolFull path)",   1,       5'000,     16,    2'500,   40,     100000},
        {"orders near band edges",      10,      600,       1024,  12,      30,     100000},
    };
    std::uint64_t seed = 0xD1FFULL;
    for (const Case& c : cases) {
        if (int rc = run(c.lo, c.hi, c.cap, c.mid, c.spread, c.steps, seed++)) {
            std::printf("  in case: %s\n", c.name);
            return rc;
        }
        std::printf("  ok: %s\n", c.name);
    }
    std::printf("%ld checks, 0 failed\nALL PASS\n", g_checks);
    return 0;
}
