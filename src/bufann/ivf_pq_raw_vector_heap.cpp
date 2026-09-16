#include "bufann/ivf_pq_raw_vector_heap.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

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

RawVectorHeap::~RawVectorHeap() { close(); }

void RawVectorHeap::open(const std::string& path, RawVectorHeapLayout layout) {
    close();
    IVF_PQ_REQUIRE(layout.slots_per_page > 0 && layout.elem_size > 0 &&
                       layout.slots_offset == RAW_VECTOR_PAGE_HEADER_BYTES + layout.bitmap_bytes &&
                       layout.bitmap_bytes * 8 >= layout.slots_per_page &&
                       uint64_t(layout.slots_offset) + uint64_t(layout.slots_per_page) * layout.elem_size <=
                           layout.page_size,
                   "raw-vector heap layout is inconsistent; use compute_raw_vector_heap_layout");
    _layout = layout;

    struct stat st;
    IVF_PQ_REQUIRE(::stat(path.c_str(), &st) != 0 || st.st_size == 0,
                   "Refusing to overwrite existing non-empty raw-vector heap file "
                   "(reopen/recovery not yet supported): " + path);

    _fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    IVF_PQ_REQUIRE(_fd >= 0, "Failed to open raw-vector heap file: " + path);
    _allocated_pages = 0;
    _next_flat_slot = 0;
}

void RawVectorHeap::close() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
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

uint32_t RawVectorHeap::allocate_slot(RawVectorFreeList& free_list) {
    {
        std::lock_guard<std::mutex> lg(free_list.mtx);
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
        std::vector<char> zero_page(_layout.page_size, 0);
        write_at(_layout.page_offset(allocated_pages()), zero_page.data(), _layout.page_size,
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

void RawVectorHeap::read_vector(uint32_t flat_slot, void* out) const {
    require_allocated(flat_slot);
    read_at(_layout.slot_offset(flat_slot), out, _layout.elem_size, "read raw vector");
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
    free_list.free_slots.push_back(flat_slot);
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
    char* page = _buf.data() + size_t(in_buf / _layout.slots_per_page) * _layout.page_size;
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
