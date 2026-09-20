#include "bufann/ivf_pq_raw_vector_heap.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <thread>

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size) {
    IVF_PQ_REQUIRE(elem_size > 0, "raw-vector element size must be greater than zero");
    auto bytes_used = [&](uint32_t slots) {
        return uint64_t(RAW_VECTOR_PAGE_HEADER_BYTES) + (slots + 7) / 8 + uint64_t(slots) * elem_size;
    };

    // Start from the count that ignores the bitmap and step down until it fits.
    uint32_t slots = page_size > RAW_VECTOR_PAGE_HEADER_BYTES
                         ? (page_size - RAW_VECTOR_PAGE_HEADER_BYTES) / elem_size
                         : 0;
    while (slots > 0 && bytes_used(slots) > page_size) {
        --slots;
    }
    IVF_PQ_REQUIRE(slots > 0, "raw-vector element size does not fit in page_size");

    RawVectorHeapLayout layout;
    layout.page_size = page_size;
    layout.elem_size = elem_size;
    layout.slots_per_page = slots;
    layout.bitmap_bytes = (slots + 7) / 8;
    layout.slots_offset = RAW_VECTOR_PAGE_HEADER_BYTES + layout.bitmap_bytes;
    return layout;
}

namespace {

void stamp_page_header(void* page, const RawVectorHeapLayout& layout, uint32_t page_id) {
    RawVectorPageHeader h;
    h.magic = RAW_VECTOR_PAGE_MAGIC;
    h.page_id = page_id;
    h.page_size = layout.page_size;
    h.elem_size = layout.elem_size;
    std::memcpy(page, &h, sizeof(h));
}

void require_page_header(const void* page, const RawVectorHeapLayout& layout, uint32_t page_id) {
    RawVectorPageHeader h;
    std::memcpy(&h, page, sizeof(h));
    const std::string where = "raw-vector heap page " + std::to_string(page_id);
    IVF_PQ_REQUIRE(h.magic == RAW_VECTOR_PAGE_MAGIC, where + " does not carry a page header");
    IVF_PQ_REQUIRE(h.page_id == page_id, where + " carries the header of page " + std::to_string(h.page_id));
    IVF_PQ_REQUIRE(h.page_size == layout.page_size && h.elem_size == layout.elem_size,
                   where + " was written with page_size " + std::to_string(h.page_size) + " and elem_size " +
                       std::to_string(h.elem_size) + ", not the " + std::to_string(layout.page_size) + "/" +
                       std::to_string(layout.elem_size) + " of this layout");
}

}  // namespace

RawVectorHeap::RawVectorHeap(uint32_t max_readers) : _readers(max_readers) {
    IVF_PQ_REQUIRE(max_readers > 0, "raw-vector heap needs at least one reader registry entry");
}

RawVectorHeap::~RawVectorHeap() { close(); }

// Everything the grace period relies on is seq_cst: the epoch load and the
// CAS here, the deleter's RID store and the epoch increment in free_slot,
// and the registry scan in oldest_reader_epoch. In that single total order,
// a reclaim that found this entry empty precedes the CAS, so every RID this
// reader loads afterwards is one the deleter had already cleared; and a
// reader whose recorded epoch is newer than a free loaded the epoch after
// that free's increment, hence after its RID store. The epoch may advance
// between the load and the CAS; recording the older value only makes the
// reader block reclaims it did not need to.
//
// A full registry means max_readers searches are mid-query; waiting for one
// to finish is a stall, where throwing would abort a query (and terminate
// the process from inside ivf_pq_search_batch's OpenMP loop).
uint32_t RawVectorHeap::enter_reader() const {
    thread_local uint32_t hint = 0;
    const uint32_t n = uint32_t(_readers.size());
    for (;;) {
        const uint64_t epoch = _epoch.load();
        for (uint32_t tries = 0; tries < n; ++tries) {
            const uint32_t i = (hint + tries) % n;
            uint64_t expected = 0;
            if (_readers[i].epoch.compare_exchange_strong(expected, epoch)) {
                hint = i;
                return i;
            }
        }
        std::this_thread::yield();
    }
}

void RawVectorHeap::leave_reader(uint32_t reader) const { _readers[reader].epoch.store(0); }

uint64_t RawVectorHeap::oldest_reader_epoch() const {
    uint64_t oldest = UINT64_MAX;
    for (const ReaderEntry& r : _readers) {
        const uint64_t e = r.epoch.load();
        if (e != 0 && e < oldest) oldest = e;
    }
    return oldest;
}

RawVectorHeap::ReadGuard::ReadGuard(const RawVectorHeap& heap) : _heap(heap), _reader(heap.enter_reader()) {}

RawVectorHeap::ReadGuard::~ReadGuard() { _heap.leave_reader(_reader); }

void RawVectorHeap::set_layout(RawVectorHeapLayout layout) {
    IVF_PQ_REQUIRE(layout.slots_per_page > 0 && layout.elem_size > 0 &&
                       layout.slots_offset == RAW_VECTOR_PAGE_HEADER_BYTES + layout.bitmap_bytes &&
                       layout.bitmap_bytes * 8 >= layout.slots_per_page &&
                       uint64_t(layout.slots_offset) + uint64_t(layout.slots_per_page) * layout.elem_size <=
                           layout.page_size,
                   "raw-vector heap layout is inconsistent; use compute_raw_vector_heap_layout");
    _layout = layout;
}

void RawVectorHeap::open(const std::string& path, RawVectorHeapLayout layout) {
    close();
    set_layout(layout);

    struct stat st;
    IVF_PQ_REQUIRE(::stat(path.c_str(), &st) != 0 || st.st_size == 0,
                   "Refusing to overwrite existing non-empty raw-vector heap file "
                   "(use open_existing to resume it): " + path);

    _fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    IVF_PQ_REQUIRE(_fd >= 0, "Failed to open raw-vector heap file: " + path);
    _allocated_pages = 0;
    _next_flat_slot = 0;
}

void RawVectorHeap::open_existing(const std::string& path, RawVectorHeapLayout layout,
                                  uint32_t next_flat_slot, uint32_t allocated_pages) {
    close();
    set_layout(layout);
    IVF_PQ_REQUIRE(uint64_t(next_flat_slot) <= uint64_t(allocated_pages) * _layout.slots_per_page,
                   "slot cursor lies past the allocated pages");

    _fd = ::open(path.c_str(), O_RDWR);
    const int open_errno = errno;
    IVF_PQ_REQUIRE(_fd >= 0,
                   (open_errno == ENOENT ? "file not found: " : "Failed to open raw-vector heap file: ") + path);

    // Everything from here on is checked through _fd, not the path: the size
    // of whatever the path names now is no evidence about the file this heap
    // will read. The headers carry the geometry, so a file written with
    // another page_size or elem_size is refused even when its byte count and
    // page ids happen to line up with this layout.
    try {
        struct stat st;
        IVF_PQ_REQUIRE(::fstat(_fd, &st) == 0, "Failed to stat raw-vector heap file: " + path);
        const uint64_t expected_bytes = uint64_t(allocated_pages) * _layout.page_size;
        IVF_PQ_REQUIRE(uint64_t(st.st_size) == expected_bytes,
                       "raw-vector heap " + path + " is " + std::to_string(st.st_size) + " bytes, not the " +
                           std::to_string(allocated_pages) + " pages of " + std::to_string(_layout.page_size) +
                           " bytes recorded");
        if (allocated_pages > 0) {
            require_page_header_on_disk(0);
            require_page_header_on_disk(allocated_pages - 1);
        }
    } catch (...) {
        close();
        throw;
    }
    _allocated_pages = allocated_pages;
    _next_flat_slot = next_flat_slot;
}

// Also drops the cursor: a closed heap has nothing allocated, so a rejected
// reopen cannot leave the previous file's cursor for allocate_slot to hand
// out or for an index header to record.
void RawVectorHeap::close() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
    _allocated_pages.store(0);
    _next_flat_slot.store(0);
}

// pread/pwrite may transfer fewer bytes than asked (a signal, or a network
// filesystem); loop until done and treat 0 bytes (EOF) or an error as failure.
void RawVectorHeap::read_at(uint64_t offset, void* buf, size_t bytes, const char* what) const {
    char* p = static_cast<char*>(buf);
    while (bytes > 0) {
        ssize_t n = ::pread(_fd, p, bytes, static_cast<off_t>(offset));
        IVF_PQ_REQUIRE(n > 0, std::string("Failed to ") + what);
        p += n, offset += n, bytes -= n;
    }
}

void RawVectorHeap::write_at(uint64_t offset, const void* buf, size_t bytes, const char* what) {
    const char* p = static_cast<const char*>(buf);
    while (bytes > 0) {
        ssize_t n = ::pwrite(_fd, p, bytes, static_cast<off_t>(offset));
        IVF_PQ_REQUIRE(n > 0, std::string("Failed to ") + what);
        p += n, offset += n, bytes -= n;
    }
}

void RawVectorHeap::require_allocated(uint32_t flat_slot) const {
    IVF_PQ_REQUIRE(_layout.page_of(flat_slot) < allocated_pages(),
                   "raw-vector slot " + std::to_string(flat_slot) + " is past the allocated pages");
}

void RawVectorHeap::require_page_header_on_disk(uint32_t page_id) const {
    char header[RAW_VECTOR_PAGE_HEADER_BYTES];
    read_at(_layout.page_offset(page_id), header, sizeof(header), "read raw-vector page header");
    require_page_header(header, _layout, page_id);
}

uint32_t RawVectorHeap::allocate_slot(RawVectorFreeList& free_list) {
    {
        std::lock_guard<std::mutex> lg(free_list.mtx);
        if (free_list.free_slots.empty()) reclaim_locked(free_list);
        if (!free_list.free_slots.empty()) {
            uint32_t flat = free_list.free_slots.back();
            free_list.free_slots.pop_back();
            return flat;
        }
    }

    std::lock_guard<std::mutex> lg(_grow_mtx);
    uint32_t flat = _next_flat_slot.load();
    IVF_PQ_REQUIRE(flat <= RAW_VECTOR_RID_SLOT_MASK,
                   "raw-vector heap is full: flat slot index exceeds the 31 bits "
                   "addressable by RawVectorRID");
    // Extend the file before publishing the slot, so a reader that sees the
    // new cursor also sees the page.
    while (allocated_pages() <= _layout.page_of(flat)) {
        std::vector<char> page(_layout.page_size, 0);
        stamp_page_header(page.data(), _layout, allocated_pages());
        write_at(_layout.page_offset(allocated_pages()), page.data(), _layout.page_size,
                 "extend raw-vector heap file");
        _allocated_pages.fetch_add(1);
    }
    _next_flat_slot.store(flat + 1);
    return flat;
}

void RawVectorHeap::write_vector(uint32_t flat_slot, const void* data) {
    require_allocated(flat_slot);
    write_at(_layout.slot_offset(flat_slot), data, _layout.elem_size, "write raw vector");
    set_occupancy_bit(flat_slot, true);
}

// One pread from the page start through the slot, so checking the header
// costs no second system call; it copies at most a page instead of a slot.
void RawVectorHeap::read_vector(uint32_t flat_slot, void* out) const {
    require_allocated(flat_slot);
    const uint32_t page_id = _layout.page_of(flat_slot);
    const uint32_t slot_in_page = _layout.slot_offset_in_page(_layout.index_in_page(flat_slot));
    thread_local std::vector<char> prefix;
    prefix.resize(size_t(slot_in_page) + _layout.elem_size);
    read_at(_layout.page_offset(page_id), prefix.data(), prefix.size(), "read raw vector");
    require_page_header(prefix.data(), _layout, page_id);
    std::memcpy(out, prefix.data() + slot_in_page, _layout.elem_size);
}

bool RawVectorHeap::is_slot_occupied(uint32_t flat_slot) const {
    require_allocated(flat_slot);
    std::lock_guard<std::mutex> lg(bitmap_mutex(flat_slot));
    uint8_t byte = 0;
    read_at(_layout.bitmap_byte_offset(flat_slot), &byte, 1, "read occupancy bitmap");
    return (byte & _layout.bitmap_mask(_layout.index_in_page(flat_slot))) != 0;
}

void RawVectorHeap::free_slot(uint32_t flat_slot, RawVectorFreeList& free_list) {
    require_allocated(flat_slot);
    set_occupancy_bit(flat_slot, false);
    std::lock_guard<std::mutex> lg(free_list.mtx);
    // The epoch is taken under the lock so `deferred` stays ascending.
    free_list.deferred.push_back({_epoch.fetch_add(1), flat_slot});
}

size_t RawVectorHeap::reclaim_freed_slots(RawVectorFreeList& free_list) {
    std::lock_guard<std::mutex> lg(free_list.mtx);
    return reclaim_locked(free_list);
}

// A slot freed at epoch f may still be addressed by a reader whose recorded
// epoch is <= f, and by no other; `deferred` is ascending, so the reclaimable
// entries are a prefix.
size_t RawVectorHeap::reclaim_locked(RawVectorFreeList& free_list) {
    if (free_list.deferred.empty()) return 0;
    const uint64_t oldest = oldest_reader_epoch();
    auto& deferred = free_list.deferred;
    auto end = std::find_if(deferred.begin(), deferred.end(),
                            [&](const RawVectorFreeList::Deferred& d) { return d.epoch >= oldest; });
    for (auto it = deferred.begin(); it != end; ++it) free_list.free_slots.push_back(it->flat_slot);
    const size_t reclaimed = size_t(end - deferred.begin());
    deferred.erase(deferred.begin(), end);
    return reclaimed;
}

void RawVectorHeap::set_occupancy_bit(uint32_t flat_slot, bool occupied) {
    uint64_t offset = _layout.bitmap_byte_offset(flat_slot);
    uint8_t mask = _layout.bitmap_mask(_layout.index_in_page(flat_slot));

    std::lock_guard<std::mutex> lg(bitmap_mutex(flat_slot));
    uint8_t byte = 0;
    read_at(offset, &byte, 1, "read occupancy bitmap");
    byte = occupied ? (byte | mask) : (byte & ~mask);
    write_at(offset, &byte, 1, "write occupancy bitmap");
}

void RawVectorHeap::write_pages(uint32_t first_page_id, const void* pages, uint32_t num_pages) {
    for (uint32_t p = 0; p < num_pages; ++p) {
        require_page_header(static_cast<const char*>(pages) + size_t(p) * _layout.page_size, _layout,
                            first_page_id + p);
    }
    write_at(_layout.page_offset(first_page_id), pages, size_t(num_pages) * _layout.page_size,
             "bulk-write raw-vector heap pages");
}

void RawVectorHeap::restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages) {
    IVF_PQ_REQUIRE(uint64_t(next_flat_slot) <= uint64_t(allocated_pages) * _layout.slots_per_page,
                   "slot cursor lies past the allocated pages");
    std::lock_guard<std::mutex> lg(_grow_mtx);
    _next_flat_slot.store(next_flat_slot);
    _allocated_pages.store(allocated_pages);
}

RawVectorHeapBulkWriter::RawVectorHeapBulkWriter(RawVectorHeap& heap, uint32_t pages_per_flush)
    : _heap(heap), _layout(heap.layout()) {
    IVF_PQ_REQUIRE(pages_per_flush > 0, "pages_per_flush must be greater than zero");
    IVF_PQ_REQUIRE(heap.next_flat_slot() == 0 && heap.allocated_pages() == 0,
                   "bulk load requires an empty raw-vector heap");
    _buf.assign(size_t(pages_per_flush) * _layout.page_size, 0);
    _buf_capacity_slots = pages_per_flush * _layout.slots_per_page;
}

uint32_t RawVectorHeapBulkWriter::append(const void* data) {
    IVF_PQ_REQUIRE(!_finished, "append after finish on raw-vector bulk writer");
    IVF_PQ_REQUIRE(_next_flat_slot <= RAW_VECTOR_RID_SLOT_MASK,
                   "raw-vector heap is full: flat slot index exceeds the 31 bits "
                   "addressable by RawVectorRID");

    uint32_t flat = _next_flat_slot++;
    uint32_t in_buf = slots_in_buffer() - 1;
    uint32_t index = in_buf % _layout.slots_per_page;
    uint32_t page_in_buf = in_buf / _layout.slots_per_page;
    char* page = _buf.data() + size_t(page_in_buf) * _layout.page_size;
    if (index == 0) {
        stamp_page_header(page, _layout, _first_page_in_buf + page_in_buf);
    }
    std::memcpy(page + _layout.slot_offset_in_page(index), data, _layout.elem_size);
    page[_layout.bitmap_byte_in_page(index)] |= _layout.bitmap_mask(index);

    if (slots_in_buffer() == _buf_capacity_slots) {
        flush();
    }
    return flat;
}

void RawVectorHeapBulkWriter::flush() {
    uint32_t slots = slots_in_buffer();
    if (slots == 0) {
        return;
    }
    uint32_t pages = (slots + _layout.slots_per_page - 1) / _layout.slots_per_page;
    _heap.write_pages(_first_page_in_buf, _buf.data(), pages);
    _first_page_in_buf += pages;
    std::fill(_buf.begin(), _buf.end(), 0);
}

void RawVectorHeapBulkWriter::finish() {
    if (_finished) {
        return;
    }
    flush();
    _heap.restore_slot_cursor(_next_flat_slot, _first_page_in_buf);
    _finished = true;
}

}  // namespace inplace
}  // namespace diskann
