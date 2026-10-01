#pragma once
//
// Nexus-LOB :: subsystem 5 — shared-memory SPSC ring for BookStateView
// Owner: Person A.
//
// The publishing end of the "zero-copy pipeline": a fixed-capacity, lock-free
// SINGLE-producer / SINGLE-consumer ring of BookStateView (448-byte contract
// state) living in an OS shared-memory segment. The matching engine's live
// state (or a synthetic flow) is memcpy'd into a slot each event; a reader
// process — the future dashboard, telemetry, or a probe — follows the live book
// across the process boundary with no serialization.
//
// Correctness core (SPSC, drop-new-on-full):
//   * The producer only writes slots outside [read_seq, write_seq), i.e. slots the
//     consumer has already released, so a reader never sees a slot mid-write.
//   * publish(): copy the view into slot[write_seq % N], then advance write_seq
//     with a RELEASE store. The consumer's ACQUIRE load of write_seq orders the copy.
//   * try_read(): copy slot[read_seq % N], then advance read_seq with a RELEASE
//     store, which the producer ACQUIRE-loads before reusing that slot.
//   * While the consumer is behind, NEW events are dropped (counted in `dropped`)
//     rather than overwriting unread slots: bounded backpressure that never blocks
//     matching. The consumer still drains what survived, strictly in order.
//   * Exactly ONE consumer per ring. A second reader advancing read_seq (e.g. the
//     dashboard and ring_probe at once) breaks the SPSC protocol.
//
// Shared-memory layout (layout v2; mirrored in python_quant/nexus_quant/dashboard.py):
//   offset   0  metadata line : magic, layout version, capacity, slot bytes, state
//   offset  64  producer line : write_seq, dropped        (written only by producer)
//   offset 128  consumer line : read_seq                  (written only by consumer)
//   offset 192  slot array    : capacity x 448-byte BookStateView
// Keeping the two indices on separate 64-byte cache lines stops every publish from
// invalidating the consumer's line and vice versa (false sharing). Each side also
// caches the other side's index in process-local memory and only re-reads the
// shared one when its cached copy says the ring is full (producer) or empty
// (consumer). 192 is a multiple of 64 and 448 = 7 x 64, so every slot starts on a
// cache-line boundary.
//
// Cross-platform: POSIX shm_open + mmap (WSL/Linux) and Windows MapViewOfFile.
// The ring copies whole slots under the protocol above, so readers never need
// BookStateView::version for consistency.
//
// Lifecycle:
//   Producer:  ShmRing::create(name, cap)  ... publish() ...  (own call to destroy(name))
//   Consumer:  ShmRing::attach(name)       ... try_read() ...
//   Create re-initializes the control block in place (strictly sequence your
//   runs: let the previous consumer exit before re-creating the segment).
//
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

#include "nexus/book_state.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nexus {

class ShmRing {
public:
    static constexpr std::size_t   kSlotBytes     = sizeof(BookStateView);
    static constexpr std::uint32_t kMagic         = 0x4E58524Eu;   // "NRXN"
    static constexpr std::uint32_t kLayoutVersion = 2;

    enum class Mode : std::uint8_t { Create, Attach };

    // Open (or create) the named ring.
    //   Mode::Create  -> (re)initialize the control block. Caller should have
    //                    callers of the previous generation closed first.
    //   Mode::Attach  -> open an existing segment; spin until a creator marks it
    //                    ready, then validate capacity/slot-size stamps.
    ShmRing(const char* name, std::size_t capacity, Mode mode) : name_(name) {
        if (capacity == 0 || capacity > kMaxSlots)
            throw std::invalid_argument("ShmRing: capacity out of [1, kMaxSlots]");
        mapped_size_ = control_bytes() + capacity * kSlotBytes;
        base_        = map_segment_(name, mapped_size_, mode);
        ctrl_        = reinterpret_cast<ControlBlock*>(base_);

        if (mode == Mode::Create) {
            // Construct the control block (and its atomics) in the fresh mapping.
            ctrl_ = ::new (base_) ControlBlock{};
            ctrl_->magic          = kMagic;
            ctrl_->layout_version = kLayoutVersion;
            ctrl_->capacity       = capacity;
            ctrl_->slot_bytes     = kSlotBytes;
            ctrl_->state.store(kReady, std::memory_order_release);
        } else {
            // Wait for the producer to publish the control block.
            // F28 fix: bounded timeout instead of indefinite spin to detect
            // a dead or absent producer.
            constexpr int kMaxAttachWaitMs = 5000;
            int waited_ms = 0;
            while (ctrl_->state.load(std::memory_order_acquire) != kReady) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                waited_ms += 10;
                if (waited_ms >= kMaxAttachWaitMs)
                    throw std::runtime_error(
                        "ShmRing: timed out waiting for producer initialization");
            }
            if (ctrl_->magic != kMagic || ctrl_->layout_version != kLayoutVersion)
                throw std::runtime_error(
                    "ShmRing: segment was created by a different ring layout version");
            if (ctrl_->capacity != capacity || ctrl_->slot_bytes != kSlotBytes)
                throw std::runtime_error(
                    "ShmRing: existing segment capacity/ABI mismatch");
        }
        capacity_     = capacity;
        cached_read_  = ctrl_->read_seq.load(std::memory_order_acquire);
        cached_write_ = ctrl_->write_seq.load(std::memory_order_acquire);
    }

    ShmRing(const ShmRing&)            = delete;
    ShmRing& operator=(const ShmRing&) = delete;

    ~ShmRing() {
        if (base_) unmap_segment_(base_, mapped_size_);
        close_handle_();
    }

    // ---- producer side ------------------------------------------------------
    // Copy `v` into the next slot and advance. Returns false (and counts a drop)
    // if the consumer hasn't drained the ring — the publisher never blocks.
    bool publish(const BookStateView& v) noexcept {
        // Only the producer writes write_seq, so a relaxed load of our own index is enough.
        const std::uint64_t w = ctrl_->write_seq.load(std::memory_order_relaxed);
        if (w - cached_read_ >= capacity_) {
            // Looks full from our cached copy: refresh it from the consumer's line.
            // ACQUIRE pairs with the consumer's RELEASE store of read_seq, so the
            // consumer has finished copying any slot we are about to reuse.
            cached_read_ = ctrl_->read_seq.load(std::memory_order_acquire);
            if (w - cached_read_ >= capacity_) {  // really full -> drop the NEW event
                ctrl_->dropped.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
        }
        std::memcpy(slot_(w), &v, kSlotBytes);
        ctrl_->write_seq.store(w + 1, std::memory_order_release);
        return true;
    }

    // ---- consumer side ------------------------------------------------------
    // Copy the next available snapshot into `out`; false when the ring is empty.
    bool try_read(BookStateView& out) noexcept {
        // Only the consumer writes read_seq, so a relaxed load of our own index is enough.
        const std::uint64_t r = ctrl_->read_seq.load(std::memory_order_relaxed);
        if (r == cached_write_) {
            // Looks empty from our cached copy: refresh it from the producer's line.
            // ACQUIRE pairs with the producer's RELEASE store, making the slot copy visible.
            cached_write_ = ctrl_->write_seq.load(std::memory_order_acquire);
            if (r == cached_write_) return false;  // really empty
        }
        std::memcpy(&out, slot_(r), kSlotBytes);
        ctrl_->read_seq.store(r + 1, std::memory_order_release);
        return true;
    }

    // ---- introspection ------------------------------------------------------
    std::uint64_t write_seq() const noexcept { return ctrl_->write_seq.load(std::memory_order_relaxed); }
    std::uint64_t read_seq()  const noexcept { return ctrl_->read_seq.load(std::memory_order_relaxed); }
    std::uint64_t dropped()   const noexcept { return ctrl_->dropped.load(std::memory_order_relaxed); }
    std::uint64_t capacity()  const noexcept { return ctrl_->capacity; }
    std::size_t   slot_bytes() const noexcept { return kSlotBytes; }

    // Remove a POSIX segment (Linux/WSL). No-op on Windows: named file mappings
    // die with their last open handle, so just let both sides close.
    static void destroy(const char* name) noexcept {
#if !defined(_WIN32)
        shm_unlink(name);
#else
        (void)name;
#endif
    }

private:
    static constexpr std::size_t kMaxSlots    = 1u << 20;
    static constexpr std::uint32_t kReady     = 1;

    // Shared control block (lives at offset 0 of the segment); layout described
    // in the header comment and mirrored byte-for-byte by the Python dashboard.
    static constexpr std::size_t kCacheLine = 64;
    struct ControlBlock {
        // -- metadata line: written once by the creator, read-only afterwards --
        alignas(kCacheLine) std::uint32_t magic = 0;
        std::uint32_t              layout_version = 0;
        std::uint64_t              capacity = 0;
        std::uint64_t              slot_bytes = 0;
        std::atomic<std::uint32_t> state{0};      // 0 uninit, 1 ready (release on create)
        // -- producer line --
        alignas(kCacheLine) std::atomic<std::uint64_t> write_seq{0};
        std::atomic<std::uint64_t> dropped{0};
        // -- consumer line --
        alignas(kCacheLine) std::atomic<std::uint64_t> read_seq{0};
    };
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free &&
                  std::atomic<std::uint32_t>::is_always_lock_free,
                  "shared-memory atomics must be lock-free: a lock would live in "
                  "one process and not protect the other");
    static_assert(offsetof(ControlBlock, magic)          ==   0, "ring layout v2");
    static_assert(offsetof(ControlBlock, layout_version) ==   4, "ring layout v2");
    static_assert(offsetof(ControlBlock, capacity)       ==   8, "ring layout v2");
    static_assert(offsetof(ControlBlock, slot_bytes)     ==  16, "ring layout v2");
    static_assert(offsetof(ControlBlock, state)          ==  24, "ring layout v2");
    static_assert(offsetof(ControlBlock, write_seq)      ==  64, "ring layout v2");
    static_assert(offsetof(ControlBlock, dropped)        ==  72, "ring layout v2");
    static_assert(offsetof(ControlBlock, read_seq)       == 128, "ring layout v2");
    static_assert(sizeof(ControlBlock)                   == 192, "ring layout v2");
    static_assert(sizeof(ControlBlock) % alignof(BookStateView) == 0,
                  "control block must not misalign the slot array");

    static std::size_t control_bytes() noexcept { return sizeof(ControlBlock); }

    BookStateView* slot_(std::uint64_t pos) noexcept {
        std::size_t off = control_bytes() + (pos % capacity_) * kSlotBytes;
        return reinterpret_cast<BookStateView*>(reinterpret_cast<char*>(base_) + off);
    }

    // ---- platform shared-memory backend --------------------------------------
    // Maps the named segment into this process. On Windows the section HANDLE
    // MUST be kept open for the view's lifetime: closing it deregisters the name
    // from the object namespace, so a later Attach by name would fail — yet the
    // view keeps the memory alive. The handle is stored in os_handle_ and closed
    // in close_handle_(). On POSIX the fd can be closed immediately after mmap;
    // os_handle_ stays null.
    void* map_segment_(const char* name, std::size_t bytes, Mode mode) {
#if defined(_WIN32)
        const std::wstring wname = widen_("Local\\" + std::string(name));
        HANDLE h = (mode == Mode::Create)
                       ? CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr,
                                            PAGE_READWRITE,
                                            (DWORD)(bytes >> 32), (DWORD)bytes,
                                            wname.c_str())
                       : OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());
        if (!h)
            throw std::runtime_error("ShmRing: cannot open/create mapping (GetLastError="
                                     + std::to_string(GetLastError()) + ")");
        os_handle_ = h;
        void* p = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
        if (!p) { close_handle_(); throw std::runtime_error("ShmRing: MapViewOfFile failed"); }
        return p;
#else
        const bool create   = (mode == Mode::Create);
        const int  oflags   = create ? (O_CREAT | O_RDWR) : O_RDWR;
        int fd = shm_open(name, oflags, 0600);
        if (fd < 0)
            throw std::runtime_error(create ? "ShmRing: shm_open(O_CREAT) failed"
                                            : "ShmRing: shm_open(attach) failed");
        if (create && ftruncate(fd, (off_t)bytes) != 0) { close(fd); throw std::runtime_error("ShmRing: ftruncate failed"); }
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("ShmRing: mmap failed");
        return p;
#endif
    }

    static void unmap_segment_(void* p, std::size_t bytes) noexcept {
#if defined(_WIN32)
        (void)bytes;
        UnmapViewOfFile(p);
#else
        munmap(p, bytes);
#endif
    }

    void close_handle_() noexcept {
#if defined(_WIN32)
        if (os_handle_) { CloseHandle(static_cast<HANDLE>(os_handle_)); os_handle_ = nullptr; }
#else
        (void)os_handle_;
#endif
    }

#if defined(_WIN32)
    static std::wstring widen_(const std::string& s) {
        if (s.empty()) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
        std::wstring w(n, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
        return w;
    }
#endif

    std::string       name_;
    void*             base_ = nullptr;
    std::size_t       mapped_size_ = 0;
    ControlBlock*     ctrl_ = nullptr;
    void*             os_handle_ = nullptr;   // Windows section HANDLE (kept for name lookup)
    // Process-local copies (not in shared memory): capacity, and each side's last
    // view of the OTHER side's index. Only the producer uses cached_read_ and only
    // the consumer uses cached_write_.
    std::uint64_t     capacity_     = 0;
    std::uint64_t     cached_read_  = 0;
    std::uint64_t     cached_write_ = 0;
};

} // namespace nexus