#include "bufann/ivf_pq_raw_vector_heap.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "ann_exception.h"

namespace diskann {
namespace inplace {

RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size) {
    RawVectorHeapLayout layout;
    layout.page_size = page_size;
    layout.elem_size = elem_size;

    // bitmap_bytes and slots_per_page depend on each other; iterate to a fixpoint.
    uint32_t bitmap_bytes = 0;
    uint32_t slots_per_page = 0;
    for (int iter = 0; iter < 8; ++iter) {
        uint32_t used = RAW_VECTOR_PAGE_HEADER_BYTES + bitmap_bytes;
        if (used >= page_size || elem_size == 0) {
            throw ANNException(
                "raw-vector element size does not fit in page_size",
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        uint32_t next_slots_per_page = (page_size - used) / elem_size;
        uint32_t next_bitmap_bytes = (next_slots_per_page + 7) / 8;
        if (next_slots_per_page == slots_per_page && next_bitmap_bytes == bitmap_bytes) {
            slots_per_page = next_slots_per_page;
            bitmap_bytes = next_bitmap_bytes;
            break;
        }
        slots_per_page = next_slots_per_page;
        bitmap_bytes = next_bitmap_bytes;
    }
    if (slots_per_page == 0) {
        throw ANNException(
            "raw-vector element size does not fit in page_size",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    layout.bitmap_bytes = bitmap_bytes;
    layout.slots_per_page = slots_per_page;
    layout.slots_offset = RAW_VECTOR_PAGE_HEADER_BYTES + bitmap_bytes;
    return layout;
}

RawVectorHeap::~RawVectorHeap() { close(); }

void RawVectorHeap::open(const std::string& path, RawVectorHeapLayout layout) {
    close();
    _layout = layout;

    // Reopen is not supported yet; refuse rather than silently truncate.
    struct stat st;
    if (::stat(path.c_str(), &st) == 0 && st.st_size > 0) {
        throw ANNException(
            "Refusing to overwrite existing non-empty raw-vector heap file "
            "(reopen/recovery not yet supported): " + path,
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // No O_DIRECT, so caller buffers need not be aligned.
    _fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (_fd < 0) {
        throw ANNException("Failed to open raw-vector heap file: " + path,
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    _allocated_pages = 0;
    _next_flat_slot = 0;
}

void RawVectorHeap::close() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
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
    // Must fit RawVectorRID's 31-bit slot field; past that, packing would alias.
    if (_next_flat_slot > RAW_VECTOR_RID_SLOT_MASK) {
        throw ANNException("raw-vector heap is full: flat slot index exceeds "
                           "the 31 bits addressable by RawVectorRID",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    uint32_t flat = _next_flat_slot++;
    uint32_t page_id = flat / _layout.slots_per_page;
    if (page_id >= _allocated_pages) {
        std::vector<char> zero_page(_layout.page_size, 0);
        for (uint32_t p = _allocated_pages; p <= page_id; ++p) {
            ssize_t n = ::pwrite(_fd, zero_page.data(), _layout.page_size,
                                 static_cast<off_t>(_layout.page_offset(p)));
            if (n < 0 || static_cast<uint32_t>(n) != _layout.page_size) {
                throw ANNException("Failed to extend raw-vector heap file",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        _allocated_pages = page_id + 1;
    }
    return flat;
}

void RawVectorHeap::write_vector(uint32_t flat_slot, const void* data) {
    uint32_t page_id = flat_slot / _layout.slots_per_page;
    uint32_t slot_idx = flat_slot % _layout.slots_per_page;
    uint64_t off = _layout.slot_offset(page_id, slot_idx);
    ssize_t n = ::pwrite(_fd, data, _layout.elem_size, static_cast<off_t>(off));
    if (n < 0 || static_cast<uint32_t>(n) != _layout.elem_size) {
        throw ANNException("Failed to write raw vector", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    set_occupancy_bit(page_id, slot_idx, true);
}

void RawVectorHeap::read_vector(uint32_t flat_slot, void* out) const {
    uint32_t page_id = flat_slot / _layout.slots_per_page;
    uint32_t slot_idx = flat_slot % _layout.slots_per_page;
    uint64_t off = _layout.slot_offset(page_id, slot_idx);
    ssize_t n = ::pread(_fd, out, _layout.elem_size, static_cast<off_t>(off));
    if (n < 0 || static_cast<uint32_t>(n) != _layout.elem_size) {
        throw ANNException("Failed to read raw vector", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
}

void RawVectorHeap::write_pages(uint32_t first_page_id, const void* pages, uint32_t num_pages) {
    size_t bytes = static_cast<size_t>(num_pages) * _layout.page_size;
    ssize_t n = ::pwrite(_fd, pages, bytes,
                         static_cast<off_t>(_layout.page_offset(first_page_id)));
    if (n < 0 || static_cast<size_t>(n) != bytes) {
        throw ANNException("Failed to bulk-write raw-vector heap pages", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
}

void RawVectorHeap::restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages) {
    std::lock_guard<std::mutex> lg(_grow_mtx);
    _next_flat_slot  = next_flat_slot;
    _allocated_pages = allocated_pages;
}

bool RawVectorHeap::is_slot_occupied(uint32_t flat_slot) const {
    uint32_t page_id  = flat_slot / _layout.slots_per_page;
    uint32_t slot_idx = flat_slot % _layout.slots_per_page;
    uint64_t off = _layout.bitmap_offset(page_id) + slot_idx / 8;

    std::lock_guard<std::mutex> lg(_bitmap_mtx[page_id % RAW_VECTOR_BITMAP_LOCK_STRIPES]);

    uint8_t byte = 0;
    ssize_t n = ::pread(_fd, &byte, 1, static_cast<off_t>(off));
    if (n < 0) {
        throw ANNException("Failed to read occupancy bitmap", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    return (byte & static_cast<uint8_t>(1u << (slot_idx % 8))) != 0;
}

void RawVectorHeap::free_slot(uint32_t flat_slot, RawVectorFreeList& free_list) {
    uint32_t page_id = flat_slot / _layout.slots_per_page;
    uint32_t slot_idx = flat_slot % _layout.slots_per_page;
    set_occupancy_bit(page_id, slot_idx, false);

    std::lock_guard<std::mutex> lg(free_list.mtx);
    free_list.free_slots.push_back(flat_slot);
}

void RawVectorHeap::set_occupancy_bit(uint32_t page_id, uint32_t slot_idx, bool occupied) {
    uint32_t byte_idx = slot_idx / 8;
    uint32_t bit_idx  = slot_idx % 8;
    uint64_t off = _layout.bitmap_offset(page_id) + byte_idx;

    // Slots in a page share bitmap bytes; the read-modify-write must be atomic.
    std::lock_guard<std::mutex> lg(_bitmap_mtx[page_id % RAW_VECTOR_BITMAP_LOCK_STRIPES]);

    uint8_t byte = 0;
    ssize_t n = ::pread(_fd, &byte, 1, static_cast<off_t>(off));
    if (n < 0) {
        throw ANNException("Failed to read occupancy bitmap", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (occupied) {
        byte |= static_cast<uint8_t>(1u << bit_idx);
    } else {
        byte &= static_cast<uint8_t>(~(1u << bit_idx));
    }
    n = ::pwrite(_fd, &byte, 1, static_cast<off_t>(off));
    if (n != 1) {
        throw ANNException("Failed to write occupancy bitmap", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
}

RawVectorHeapBulkWriter::RawVectorHeapBulkWriter(RawVectorHeap& heap, uint32_t pages_per_flush)
    : _heap(heap), _layout(heap.layout()), _pages_per_flush(pages_per_flush) {
    if (pages_per_flush == 0) {
        throw ANNException("pages_per_flush must be greater than zero", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (heap.next_flat_slot() != 0 || heap.allocated_pages() != 0) {
        throw ANNException("bulk load requires an empty raw-vector heap", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    _buf.assign(static_cast<size_t>(pages_per_flush) * _layout.page_size, 0);
}

uint32_t RawVectorHeapBulkWriter::append(const void* data) {
    if (_finished) {
        throw ANNException("append after finish on raw-vector bulk writer", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_next_flat_slot > RAW_VECTOR_RID_SLOT_MASK) {
        throw ANNException("raw-vector heap is full: flat slot index exceeds "
                           "the 31 bits addressable by RawVectorRID",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    uint32_t flat = _next_flat_slot++;
    uint32_t slot_idx = flat % _layout.slots_per_page;
    uint32_t page_in_buf = (flat / _layout.slots_per_page) - _first_page_in_buf;

    char* page = _buf.data() + static_cast<size_t>(page_in_buf) * _layout.page_size;
    std::memcpy(page + _layout.slots_offset + static_cast<size_t>(slot_idx) * _layout.elem_size,
                data, _layout.elem_size);
    page[RAW_VECTOR_PAGE_HEADER_BYTES + slot_idx / 8] |= static_cast<char>(1u << (slot_idx % 8));
    _buf_pages = page_in_buf + 1;

    // Flush once the last page of the buffer is full.
    if (_buf_pages == _pages_per_flush && slot_idx + 1 == _layout.slots_per_page) {
        flush();
    }
    return flat;
}

void RawVectorHeapBulkWriter::flush() {
    if (_buf_pages == 0) {
        return;
    }
    _heap.write_pages(_first_page_in_buf, _buf.data(), _buf_pages);
    _first_page_in_buf += _buf_pages;
    _buf_pages = 0;
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
