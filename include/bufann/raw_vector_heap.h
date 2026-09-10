// File-backed storage for the IVF-PQ raw-vector heap: fixed-size slots
// grouped into fixed-size pages, addressed by RawVectorRID (see ivf_pq.h).
//
// Bypasses the BufANN buffer pool entirely -- reads and writes go straight
// to disk via pread/pwrite. Wiring this into the buffer pool's page cache
// is a follow-up (see "How IVF-PQ fits with Buffer Pool Management" in the
// design doc); this class only needs to be correct as a standalone heap.
//
// Reopening an existing heap file after a restart is not implemented here:
// reconstructing the free list and next-slot cursor from persisted state is
// the durability/recovery PR's job. RawVectorHeap::open() always creates a
// fresh, empty heap.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

static constexpr uint32_t kRawVectorPageHeaderBytes = 8;

// ---------------------------------------------------------------------------
// RawVectorHeapLayout -- page/slot geometry for a given (page_size, elem_size)
// ---------------------------------------------------------------------------
// Page layout: [page header][occupancy bitmap][slot 0]...[slot N-1][pad].
// slots_per_page is derived from page_size and elem_size, not assumed to be
// any particular fixed count -- the design doc's own page-layout table (7
// slots of 512B in a 4096B page) is worked out for its dim=128/fp32 example,
// not a universal constant, since elem_size scales with the configured dim
// and element type.
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
        return page_offset(page_id) + kRawVectorPageHeaderBytes;
    }
};

// Throws ANNException if elem_size doesn't fit in page_size at all.
RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size);

// ---------------------------------------------------------------------------
// RawVectorHeap
// ---------------------------------------------------------------------------
// No synchronization between write_vector/read_vector/free_slot: a freed
// slot goes straight back onto the free list for immediate reuse, with no
// grace period. Safe only while nothing else can be concurrently reading a
// flat_slot that's being freed/reused -- once concurrent search (exact
// re-rank reads) and concurrent insert/delete coexist, this needs the same
// grace-period reclaim the design doc calls for around posting-list rebuilds.
class RawVectorHeap {
public:
    RawVectorHeap() = default;
    ~RawVectorHeap();

    RawVectorHeap(const RawVectorHeap&)            = delete;
    RawVectorHeap& operator=(const RawVectorHeap&) = delete;

    void open(const std::string& path, RawVectorHeapLayout layout);
    void close();

    // Allocates a slot, preferring a reused entry from free_list; grows the
    // heap by one slot (extending the file with a fresh page when needed)
    // otherwise. Returns the flat slot index (see RawVectorRID).
    uint32_t allocate_slot(RawVectorFreeList& free_list);

    // Writes layout().elem_size bytes from `data` into `flat_slot` and marks
    // its occupancy bit set. `flat_slot` must have come from allocate_slot().
    void write_vector(uint32_t flat_slot, const void* data);

    // Reads layout().elem_size bytes from `flat_slot` into `out`.
    void read_vector(uint32_t flat_slot, void* out) const;

    // Clears the occupancy bit for `flat_slot`. Caller is responsible for
    // pushing `flat_slot` onto a RawVectorFreeList.
    void free_slot(uint32_t flat_slot);

    const RawVectorHeapLayout& layout() const { return _layout; }

private:
    void set_occupancy_bit(uint32_t page_id, uint32_t slot_idx, bool occupied);

    int _fd = -1;
    RawVectorHeapLayout _layout;
    std::mutex _grow_mtx;
    uint32_t _next_flat_slot = 0;
    uint32_t _allocated_pages = 0;
};

}  // namespace inplace
}  // namespace diskann
