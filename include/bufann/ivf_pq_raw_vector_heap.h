// IVF-PQ raw-vector heap: fixed-size slots in fixed-size pages, addressed by
// flat slot index (see RawVectorRID). Plain pread/pwrite, no buffer pool.
// open() always creates a fresh heap; reopen/recovery is not implemented.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

constexpr uint32_t RAW_VECTOR_PAGE_HEADER_BYTES = 8;
constexpr uint32_t RAW_VECTOR_BITMAP_LOCK_STRIPES = 64;

// Page layout: [header][occupancy bitmap, 1 bit per slot, LSB first][slot 0..N-1][pad].
struct RawVectorHeapLayout {
    uint32_t page_size      = 0;
    uint32_t elem_size      = 0;
    uint32_t slots_per_page = 0;
    uint32_t bitmap_bytes   = 0;
    uint32_t slots_offset   = 0;  // byte offset of slot 0 within a page

    uint32_t page_of(uint32_t flat_slot) const { return flat_slot / slots_per_page; }
    uint32_t index_in_page(uint32_t flat_slot) const { return flat_slot % slots_per_page; }

    uint64_t page_offset(uint32_t page_id) const { return uint64_t(page_id) * page_size; }
    uint32_t slot_offset_in_page(uint32_t index) const { return slots_offset + index * elem_size; }
    uint32_t bitmap_byte_in_page(uint32_t index) const { return RAW_VECTOR_PAGE_HEADER_BYTES + index / 8; }
    static uint8_t bitmap_mask(uint32_t index) { return uint8_t(1u << (index % 8)); }

    uint64_t slot_offset(uint32_t flat_slot) const {
        return page_offset(page_of(flat_slot)) + slot_offset_in_page(index_in_page(flat_slot));
    }
    uint64_t bitmap_byte_offset(uint32_t flat_slot) const {
        return page_offset(page_of(flat_slot)) + bitmap_byte_in_page(index_in_page(flat_slot));
    }
};

// Largest slots_per_page whose header, bitmap and slots fit in page_size.
// Throws if not even one slot fits.
RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size);

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
    void close();

    // Pops from free_list if possible, else grows the heap by a page as needed.
    uint32_t allocate_slot(RawVectorFreeList& free_list);
    void write_vector(uint32_t flat_slot, const void* data);  // sets the occupancy bit
    void read_vector(uint32_t flat_slot, void* out) const;
    bool is_slot_occupied(uint32_t flat_slot) const;
    void free_slot(uint32_t flat_slot, RawVectorFreeList& free_list);  // clears the bit

    // Writes `num_pages` complete pages starting at `first_page_id` with one
    // pwrite. Does not touch the allocation cursor; see RawVectorHeapBulkWriter.
    void write_pages(uint32_t first_page_id, const void* pages, uint32_t num_pages);

    // Not safe to call concurrently with allocate_slot.
    void restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages);
    uint32_t next_flat_slot() const { return _next_flat_slot.load(std::memory_order_relaxed); }
    uint32_t allocated_pages() const { return _allocated_pages.load(std::memory_order_relaxed); }

    const RawVectorHeapLayout& layout() const { return _layout; }

private:
    void read_at(uint64_t offset, void* buf, size_t bytes, const char* what) const;
    void write_at(uint64_t offset, const void* buf, size_t bytes, const char* what);
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
