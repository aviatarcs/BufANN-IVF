// IVF-PQ raw-vector heap: fixed-size slots in fixed-size pages, addressed by
// flat slot index (see RawVectorRID). Plain pread/pwrite, no buffer pool.
// open() creates a fresh heap; open_existing() resumes one from the geometry
// and allocation cursor the index file recorded. Nothing else is persisted
// (in particular not the free list), so the heap file is only meaningful
// together with the index file that describes it.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_raw_vector_heap_layout.h"

namespace diskann {
namespace inplace {

constexpr uint32_t RAW_VECTOR_BITMAP_LOCK_STRIPES = 64;

// allocate_slot/write_vector/free_slot may be called concurrently: allocation
// is serialized on _grow_mtx, bitmap read-modify-writes on a per-page stripe.
// A freed slot is reusable immediately, with no grace period, so read_vector
// must not race free_slot on the same slot.
class RawVectorHeap {
public:
    RawVectorHeap() = default;
    ~RawVectorHeap();
    RawVectorHeap(const RawVectorHeap&)            = delete;
    RawVectorHeap& operator=(const RawVectorHeap&) = delete;

    void open(const std::string& path, RawVectorHeapLayout layout);

    // Resumes a heap that open()/bulk load wrote earlier. The file must be
    // exactly `allocated_pages` pages and its first and last page headers must
    // carry their own ids; every other page is verified when it is read.
    void open_existing(const std::string& path, RawVectorHeapLayout layout,
                       uint32_t next_flat_slot, uint32_t allocated_pages);
    void close();

    // Pops from free_list if possible, else grows the heap by a page as needed.
    uint32_t allocate_slot(RawVectorFreeList& free_list);
    void write_vector(uint32_t flat_slot, const void* data);  // sets the occupancy bit
    void read_vector(uint32_t flat_slot, void* out) const;    // throws if the page header is not the slot's page
    bool is_slot_occupied(uint32_t flat_slot) const;
    void free_slot(uint32_t flat_slot, RawVectorFreeList& free_list);  // clears the bit

    // Writes `num_pages` complete pages starting at `first_page_id` with one
    // pwrite; each page must already carry its header. Does not touch the
    // allocation cursor; see RawVectorHeapBulkWriter.
    void write_pages(uint32_t first_page_id, const void* pages, uint32_t num_pages);

    // Not safe to call concurrently with allocate_slot.
    void restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages);
    uint32_t next_flat_slot() const { return _next_flat_slot.load(std::memory_order_relaxed); }
    uint32_t allocated_pages() const { return _allocated_pages.load(std::memory_order_relaxed); }

    const RawVectorHeapLayout& layout() const { return _layout; }

private:
    void set_layout(RawVectorHeapLayout layout);
    void read_at(uint64_t offset, void* buf, size_t bytes, const char* what) const;
    void write_at(uint64_t offset, const void* buf, size_t bytes, const char* what);
    void require_page_header_on_disk(uint32_t page_id) const;
    void set_occupancy_bit(uint32_t flat_slot, bool occupied);
    void require_allocated(uint32_t flat_slot) const;
    std::mutex& bitmap_mutex(uint32_t flat_slot) const {
        return _bitmap_mtx[_layout.page_of(flat_slot) % RAW_VECTOR_BITMAP_LOCK_STRIPES];
    }

    int _fd = -1;
    RawVectorHeapLayout _layout;
    std::mutex _grow_mtx;
    mutable std::array<std::mutex, RAW_VECTOR_BITMAP_LOCK_STRIPES> _bitmap_mtx;
    std::atomic<uint32_t> _next_flat_slot{0};
    std::atomic<uint32_t> _allocated_pages{0};
};

// Build-time loader for an empty heap: vector k of the load lands in flat
// slot k. Pages are assembled in memory and written `pages_per_flush` at a
// time. finish() flushes the tail and sets the heap's allocation cursor.
// Not thread-safe.
class RawVectorHeapBulkWriter {
public:
    RawVectorHeapBulkWriter(RawVectorHeap& heap, uint32_t pages_per_flush);

    uint32_t append(const void* data);  // returns the flat slot
    void finish();

private:
    void flush();
    uint32_t slots_in_buffer() const { return _next_flat_slot - _first_page_in_buf * _layout.slots_per_page; }

    RawVectorHeap& _heap;
    RawVectorHeapLayout _layout;
    std::vector<char> _buf;  // pages_per_flush pages, always starting at _first_page_in_buf
    uint32_t _buf_capacity_slots = 0;
    uint32_t _first_page_in_buf = 0;
    uint32_t _next_flat_slot = 0;
    bool _finished = false;
};

}  // namespace inplace
}  // namespace diskann
