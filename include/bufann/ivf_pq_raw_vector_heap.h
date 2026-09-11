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

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

constexpr uint32_t RAW_VECTOR_PAGE_HEADER_BYTES = 8;

// Occupancy-bitmap updates are serialized on a stripe of mutexes indexed by
// page_id. Striping rather than one lock per page keeps the heap's footprint
// independent of how far it has grown; pages that collide on a stripe just
// serialize, which is correct but rarely contended at this width.
constexpr uint32_t RAW_VECTOR_BITMAP_LOCK_STRIPES = 64;

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
        return page_offset(page_id) + RAW_VECTOR_PAGE_HEADER_BYTES;
    }
};

// Throws ANNException if elem_size doesn't fit in page_size at all.
RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size);

// ---------------------------------------------------------------------------
// RawVectorHeap
// ---------------------------------------------------------------------------
// allocate_slot/write_vector/free_slot are safe to call concurrently: slot
// handout is serialized on _grow_mtx, and occupancy-bitmap updates on a
// per-page stripe of _bitmap_mtx. The bitmap needs its own locking because
// the update is a read-modify-write of one byte, and at the design doc's own
// 4096B/512B geometry every slot in a page shares a single bitmap byte -- so
// two threads writing into the same page would otherwise lose each other's
// bits, which is precisely what a parallel build loop does.
//
// What is NOT yet safe: a freed slot goes straight back onto the free list for
// immediate reuse, with no grace period, so read_vector must not race a
// free_slot of the same flat_slot. Once concurrent search (exact re-rank
// reads) and concurrent insert/delete coexist, this needs the same
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

    // Repositions the allocation cursor over a heap whose contents already
    // exist. open() starts both of these at zero; the durability/recovery path
    // is what will call this, after reading the persisted state back. Not safe
    // to call while anything else is allocating.
    void restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages);

    // Reads back `flat_slot`'s occupancy bit. The bitmap is what a future
    // reopen/recovery path has to rebuild the free list from, so it is worth
    // being able to read it and not only write it.
    bool is_slot_occupied(uint32_t flat_slot) const;

    // Clears the occupancy bit for `flat_slot` and returns it to `free_list`
    // for reuse. Both halves happen under the free list's own mutex, so this
    // mirrors allocate_slot rather than leaving the caller to push by hand.
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
