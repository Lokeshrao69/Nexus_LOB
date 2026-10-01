#pragma once
//
// Nexus-LOB :: hierarchical occupancy bitmap over price levels (cpp_engine)
//
// One bit per price level: set == at least one resting order at that price.
// Level 0 holds the raw bits; bit j of level k+1 is set iff word j of level k is
// non-zero. Levels are added until the top fits in a single 64-bit word, so a
// 2^30-tick band needs 5 levels and a 1M-tick band needs 4.
//
// next(i) / prev(i) find the nearest occupied level at or after / at or before i
// in O(levels) word operations using count-trailing / count-leading zeros,
// independent of how many empty levels lie in between. This replaces the linear
// scans over the price array that made the engine O(band width) on sparse books.
//
// Memory: ~1/64 of a bit-array per level beyond the first; allocated once in the
// constructor, never on the order path.
//
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexus {

class LevelBitmap {
public:
    explicit LevelBitmap(std::size_t nbits) : nbits_(nbits) {
        std::size_t words = (nbits + 63) / 64;
        if (words == 0) words = 1;
        for (;;) {
            lv_.emplace_back(words, 0);
            if (words == 1) break;
            words = (words + 63) / 64;
        }
    }

    std::size_t size() const noexcept { return nbits_; }

    bool test(std::ptrdiff_t i) const noexcept {
        const auto u = static_cast<std::size_t>(i);
        return (lv_[0][u >> 6] >> (u & 63)) & 1u;
    }

    void set(std::ptrdiff_t i) noexcept {
        auto u = static_cast<std::size_t>(i);
        for (auto& words : lv_) {
            std::uint64_t& w = words[u >> 6];
            const bool was_empty = (w == 0);
            w |= std::uint64_t{1} << (u & 63);
            if (!was_empty) return;          // parent bits already set
            u >>= 6;
        }
    }

    void clear(std::ptrdiff_t i) noexcept {
        auto u = static_cast<std::size_t>(i);
        for (auto& words : lv_) {
            std::uint64_t& w = words[u >> 6];
            w &= ~(std::uint64_t{1} << (u & 63));
            if (w != 0) return;              // word still occupied: parents stay set
            u >>= 6;
        }
    }

    // Smallest set index >= i, or -1 if none.
    std::ptrdiff_t next(std::ptrdiff_t i) const noexcept {
        if (i < 0) i = 0;
        auto pos = static_cast<std::size_t>(i);
        const std::size_t top = lv_.size() - 1;
        std::size_t k = 0;
        for (;;) {
            const std::size_t w = pos >> 6;
            if (w >= lv_[k].size()) return -1;
            const std::uint64_t m = lv_[k][w] & (~std::uint64_t{0} << (pos & 63));
            if (m != 0) {
                std::size_t p = (w << 6) | static_cast<std::size_t>(std::countr_zero(m));
                while (k > 0) {              // descend to the first set bit below
                    --k;
                    p = (p << 6) | static_cast<std::size_t>(std::countr_zero(lv_[k][p]));
                }
                return static_cast<std::ptrdiff_t>(p);
            }
            if (k == top) return -1;
            pos = w + 1;                     // first word after this one, one level up
            ++k;
        }
    }

    // Largest set index <= i, or -1 if none.
    std::ptrdiff_t prev(std::ptrdiff_t i) const noexcept {
        if (i < 0) return -1;
        auto pos = static_cast<std::size_t>(i);
        if (pos >= nbits_) pos = nbits_ - 1;
        const std::size_t top = lv_.size() - 1;
        std::size_t k = 0;
        for (;;) {
            const std::size_t w = pos >> 6;
            const unsigned b = static_cast<unsigned>(pos & 63);
            const std::uint64_t mask = (b == 63) ? ~std::uint64_t{0}
                                                 : ((std::uint64_t{1} << (b + 1)) - 1);
            const std::uint64_t m = lv_[k][w] & mask;
            if (m != 0) {
                std::size_t p = (w << 6) | static_cast<std::size_t>(63 - std::countl_zero(m));
                while (k > 0) {              // descend to the last set bit below
                    --k;
                    p = (p << 6) | static_cast<std::size_t>(63 - std::countl_zero(lv_[k][p]));
                }
                return static_cast<std::ptrdiff_t>(p);
            }
            if (w == 0 || k == top) return -1;
            pos = w - 1;                     // last word before this one, one level up
            ++k;
        }
    }

private:
    std::size_t                             nbits_;
    std::vector<std::vector<std::uint64_t>> lv_;   // lv_[0] = raw bits, lv_.back() = 1 word
};

}  // namespace nexus
