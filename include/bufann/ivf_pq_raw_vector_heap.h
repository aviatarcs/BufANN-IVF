// IVF-PQ raw-vector heap: fixed-size slots in fixed-size pages, addressed by
// flat slot index (see RawVectorRID). Plain pread/pwrite, no buffer pool.
// open() creates a fresh heap; open_existing() resumes one from the geometry
// and allocation cursor the index file recorded. Nothing else is persisted:
// the free list is rebuilt on load from the page directories (occupancy
// bitmap and slot owners, see ivf_pq_recover_deletes), and the heap file is
// only meaningful together with the index file that describes it.

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
constexpr uint32_t RAW_VECTOR_MAX_READERS         = 1024;  // ReadGuards held at once

// allocate_slot/write_vector/free_slot may be called concurrently: allocation
// is serialized on _grow_mtx, bitmap read-modify-writes on a per-page stripe.
// A freed slot is not handed out again while a ReadGuard that could still
// address it is held; see ReadGuard for what readers and freers must do.
class RawVectorHeap {
public:
    explicit RawVectorHeap(uint32_t max_readers = RAW_VECTOR_MAX_READERS);
    ~RawVectorHeap();
    RawVectorHeap(const RawVectorHeap&)            = delete;
    RawVectorHeap& operator=(const RawVectorHeap&) = delete;

    void open(const std::string& path, RawVectorHeapLayout layout);

    // Resumes a heap that open()/bulk load wrote earlier. The file must be
    // exactly `allocated_pages` pages and its first and last page headers must
    // carry their own ids and this layout's page_size/elem_size; every other
    // page is verified when it is read.
    void open_existing(const std::string& path, RawVectorHeapLayout layout,
                       uint32_t next_flat_slot, uint32_t allocated_pages);
    void close();  // afterwards next_flat_slot() == allocated_pages() == 0

    // Grace period for slot reuse. A reader holds a ReadGuard from before it
    // loads the RID of any vector until after its last read_vector; a slot
    // freed while such a guard is held is not reused until that guard is
    // released, so the bytes the reader gets are the vector whose RID it
    // loaded (or, if the vector was deleted in between, still that vector's
    // last bytes). Readers that enter after the free do not delay the reuse.
    //
    // This holds provided a deleter clears the RID's active bit with
    // store_rid before it calls free_slot and readers use load_rid (both
    // seq_cst): then a reader registered before the free, or one that loads
    // the RID after it, sees the RID inactive. Every ReadGuard occupies one
    // of max_readers registry entries; when all are taken, entering waits
    // for one to be released.
    class ReadGuard {
    public:
        explicit ReadGuard(const RawVectorHeap& heap);
        ~ReadGuard();
        ReadGuard(const ReadGuard&)            = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;

    private:
        const RawVectorHeap& _heap;
        uint32_t _reader;
    };

    // Pops from free_list.free_slots, refilling it from free_list.deferred
    // first when it is empty (reclaim_freed_slots), else grows the heap by a
    // page as needed.
    uint32_t allocate_slot(RawVectorFreeList& free_list);
    // Records `owner` (the vector's id) and the bytes, then sets the occupancy bit.
    void write_vector(uint32_t flat_slot, uint32_t owner, const void* data);
    // Returns the slot's owner; throws if the page header is not the slot's
    // page and geometry. Callers that know which vector they expect compare.
    uint32_t read_vector(uint32_t flat_slot, void* out) const;
    bool is_slot_occupied(uint32_t flat_slot) const;
    // The page's occupancy bitmap (layout().bitmap_bytes bytes, bit i for
    // slot i of the page, LSB first) and owner ids (slots_per_page of them)
    // in one read; verifies the page header.
    void read_page_directory(uint32_t page_id, uint8_t* bitmap, uint32_t* owners) const;
    // Clears the bit and defers the slot until the readers now registered have left.
    void free_slot(uint32_t flat_slot, RawVectorFreeList& free_list);
    // Moves onto free_list.free_slots every deferred slot that no registered
    // reader can still address; returns how many.
    size_t reclaim_freed_slots(RawVectorFreeList& free_list);

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
    uint32_t enter_reader() const;  // registers the current epoch; returns the registry entry
    void leave_reader(uint32_t reader) const;
    uint64_t oldest_reader_epoch() const;  // UINT64_MAX when no reader is registered
    size_t reclaim_locked(RawVectorFreeList& free_list);  // caller holds free_list.mtx

    // One cache line per entry, so readers entering and leaving do not
    // contend on each other's lines. 0 marks a free entry; epochs start at 1.
    struct alignas(64) ReaderEntry {
        std::atomic<uint64_t> epoch{0};
    };

    int _fd = -1;
    RawVectorHeapLayout _layout;
    std::mutex _grow_mtx;
    mutable std::array<std::mutex, RAW_VECTOR_BITMAP_LOCK_STRIPES> _bitmap_mtx;
    std::atomic<uint32_t> _next_flat_slot{0};
    std::atomic<uint32_t> _allocated_pages{0};
    std::atomic<uint64_t> _epoch{1};  // advanced by every free_slot
    mutable std::vector<ReaderEntry> _readers;
};

// Build-time loader for an empty heap: vector k of the load lands in flat
// slot k. Pages are assembled in memory and written `pages_per_flush` at a
// time. finish() flushes the tail and sets the heap's allocation cursor.
// Not thread-safe.
class RawVectorHeapBulkWriter {
public:
    RawVectorHeapBulkWriter(RawVectorHeap& heap, uint32_t pages_per_flush);

    uint32_t append(uint32_t owner, const void* data);  // returns the flat slot
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
