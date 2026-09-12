// IVF-PQ raw-vector heap: fixed-size slots in fixed-size pages, addressed by
// RawVectorRID. Uses plain pread/pwrite, bypassing the buffer pool for now.
// open() always creates a fresh heap; reopen/recovery is not implemented.

#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

constexpr uint32_t RAW_VECTOR_PAGE_HEADER_BYTES = 8;

// Occupancy-bitmap updates lock a mutex striped by page_id.
constexpr uint32_t RAW_VECTOR_BITMAP_LOCK_STRIPES = 64;

// ---------------------------------------------------------------------------
// RawVectorHeapLayout -- page/slot geometry for a given (page_size, elem_size)
// ---------------------------------------------------------------------------
// Page layout: [header][occupancy bitmap][slot 0]...[slot N-1][pad].
// slots_per_page is derived from (page_size, elem_size), not fixed.
struct RawVectorHeapLayout {
    uint32_t page_size      = 0;
    uint32_t elem_size      = 0;
    uint32_t bitmap_bytes   = 0;  // occupancy bitmap size, 1 bit per slot
    uint32_t slots_per_page = 0;
    uint32_t slots_offset   = 0;  // byte offset of slot 0 within a page

    uint64_t page_offset(uint32_t page_id) const {
        return static_cast<uint64_t>(page_id) * page_size;
    }
    uint64_t slot_offset(uint32_t page_id, uint32_t slot_idx) const {
        return page_offset(page_id) + slots_offset +
               static_cast<uint64_t>(slot_idx) * elem_size;
    }
    uint64_t bitmap_offset(uint32_t page_id) const {
        return page_offset(page_id) + RAW_VECTOR_PAGE_HEADER_BYTES;
    }
};

// Throws if elem_size does not fit in page_size.
RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size);

// ---------------------------------------------------------------------------
// RawVectorHeap
// ---------------------------------------------------------------------------
// allocate_slot/write_vector/free_slot may be called concurrently: allocation
// is serialized on _grow_mtx, bitmap updates on a per-page stripe (slots in a
// page share bitmap bytes). A freed slot is reusable immediately, with no
// grace period, so read_vector must not race free_slot on the same slot.
class RawVectorHeap {
public:
    RawVectorHeap() = default;
    ~RawVectorHeap();

    RawVectorHeap(const RawVectorHeap&)            = delete;
    RawVectorHeap& operator=(const RawVectorHeap&) = delete;

    void open(const std::string& path, RawVectorHeapLayout layout);
    void close();

    // Pops from free_list if possible, else grows the heap (extending the file
    // by a page when needed). Returns the flat slot index.
    uint32_t allocate_slot(RawVectorFreeList& free_list);

    // Writes elem_size bytes into `flat_slot` and sets its occupancy bit.
    void write_vector(uint32_t flat_slot, const void* data);

    // Reads elem_size bytes from `flat_slot` into `out`.
    void read_vector(uint32_t flat_slot, void* out) const;

    // Resets the allocation cursor, e.g. after recovery reads persisted state.
    // Not safe to call concurrently with allocate_slot.
    void restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages);

    bool is_slot_occupied(uint32_t flat_slot) const;

    // Clears the occupancy bit and pushes `flat_slot` onto `free_list`.
    void free_slot(uint32_t flat_slot, RawVectorFreeList& free_list);

    const RawVectorHeapLayout& layout() const { return _layout; }

private:
    void set_occupancy_bit(uint32_t page_id, uint32_t slot_idx, bool occupied);

    int _fd = -1;
    RawVectorHeapLayout _layout;
    std::mutex _grow_mtx;
    mutable std::array<std::mutex, RAW_VECTOR_BITMAP_LOCK_STRIPES> _bitmap_mtx;
    uint32_t _next_flat_slot = 0;
    uint32_t _allocated_pages = 0;
};

}  // namespace inplace
}  // namespace diskann
