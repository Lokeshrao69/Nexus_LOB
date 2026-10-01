# Shared-memory ring: cache-line layout and cached indices

## What was wrong

**False sharing.** The control block packed `write_seq`, `read_seq`, `dropped`,
`capacity` and `state` into one 48-byte struct, which sits on a single 64-byte
cache line. The producer writes `write_seq` on every publish and the consumer
writes `read_seq` on every read, so each write invalidated the line the other
core was reading. The line bounced between cores on every frame.

**Every call touched the other side's index.** `publish()` loaded `read_seq`
and `try_read()` loaded `write_seq` on every call, so even without false
sharing, each operation pulled the other core's cache line.

**The Python dashboard could rewind the producer.** `read_shm_ring_latest`
acknowledged a read by re-packing the whole control block from the copy it read
at the start, including `write_seq` and `dropped`. If the C++ producer
published in between, the dashboard wrote the old `write_seq` back. The producer
then reused slot numbers it had already published, and frames were silently
lost or duplicated. It also copied the entire shared segment (up to ~7 MB) on
every 400 ms poll to read one 448-byte slot.

**The "seqlock" was not a seqlock.** `BookStateView::version` goes odd during
`publish_()` and even after, and was documented as a seqlock. The field and the
data are plain, non-atomic memory with no fences, so another thread cannot use it
to detect a torn read. Nothing actually read it concurrently (the ring copies
whole slots), so the fix is to document it accurately rather than add atomics.

## Fix

Shared-memory layout v2 (`cpp_engine/include/nexus/shm_ring.hpp`):

| Offset | Cache line | Fields | Written by |
|---:|---|---|---|
| 0 | metadata | magic, layout version, capacity, slot bytes, state | creator, once |
| 64 | producer | `write_seq`, `dropped` | producer only |
| 128 | consumer | `read_seq` | consumer only |
| 192 | slots | capacity × 448-byte `BookStateView` | producer |

- Each index has its own cache line. 192 and 448 are multiples of 64, so every
  slot is cache-line aligned too.
- Each side keeps a process-local copy of the other side's index
  (`cached_read_` in the producer, `cached_write_` in the consumer) and only
  re-reads the shared one, with acquire ordering, when its copy says the ring is
  full or empty.
- The control block is constructed with placement `new` in the fresh mapping, and
  a `static_assert` requires the shared atomics to be lock-free (a lock-based
  atomic would keep its lock in one process's memory).
- `static_assert`s pin every field offset. A magic number and layout version are
  checked on attach, in C++ and in Python, so a mismatched reader fails instead
  of misreading.
- The Python reader writes only the 8-byte `read_seq` field, and reads only the
  header and the one slot it needs.

## Results

`cpp_engine/bench/ring_bench.cpp`: one producer thread publishes 3M frames
(retrying when full, so every frame is delivered) while the consumer thread reads
and checks order. Median of 5 runs, 2-vCPU cloud VM, GCC 13.3,
`-O3 -march=native`. All runs delivered every frame in order.

| Capacity | Before | Padding only | Padding + cached indices | Speed-up |
|---:|---:|---:|---:|---:|
| 64 | 2.00M frames/s | 2.94M | 6.55M | 3.3× |
| 1,024 | 2.71M | 4.18M | 29.69M | 11.0× |
| 16,384 | 3.09M | 5.11M | 35.81M | 11.6× |

The "padding only" column is an ablation: same layout, but both sides re-read
the other's index on every call. Separating the cache lines gives about 1.5×.
Caching the remote index gives the rest, because once the consumer is a few
slots behind, most publishes and reads never touch the other core's line.
Small rings benefit less because they fill or empty constantly, which forces
the reload.

At 35.8M frames/s × 448 bytes the ring moves about 16 GB/s. Each frame is
copied twice (into the slot and out of it), so slot copying is now likely a large
share of the cost; profiling that split is a possible next step.

## Verification

- `ring_test` (ordering, drop-on-full, cross-handle state, and a new check that
  attaching to a segment with the wrong layout version throws) passes under
  ThreadSanitizer, AddressSanitizer and UBSan.
- `ring_bench` runs clean under ThreadSanitizer.
- New Python tests: the dashboard changes only bytes 128–135 (`read_seq`) of
  the control block, leaves `write_seq` and `dropped` intact, and ignores a
  segment with a different layout version.
- Live cross-language check: `ring_producer` (C++) publishing while
  `read_shm_ring_latest` (Python) polled it. Sequence numbers were strictly
  increasing, bids were below asks, and `write_seq` never went backwards.

## Known limitation

The ring drops the newest snapshot when full. That is right for a consumer that
must see events in order, but a dashboard wants the latest state. When the
dashboard polls slower than the engine publishes, it shows a snapshot up to one
ring-capacity of events old. A separate latest-value slot (a real seqlock with
atomic sequence and fences, or a double buffer) would fit the dashboard better.

## Reproduce

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG -pthread -I cpp_engine/include \
    cpp_engine/bench/ring_bench.cpp -o ring_bench -lrt && ./ring_bench
```
