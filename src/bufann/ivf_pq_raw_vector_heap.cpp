#include "bufann/ivf_pq_raw_vector_heap.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <vector>

#include "ann_exception.h"

namespace diskann {
namespace inplace {

RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size) {
    RawVectorHeapLayout layout;
    layout.page_size = page_size;
    layout.elem_size = elem_size;

    // bitmap_bytes depends on slots_per_page and vice versa; a couple of
    // fixpoint passes converge since shrinking the bitmap by a byte can only
    // ever free up room for a bounded number of extra slots.
    uint32_t bitmap_bytes = 0;
    uint32_t slots_per_page = 0;
    for (int iter = 0; iter < 8; ++iter) {
        uint32_t used = kRawVectorPageHeaderBytes + bitmap_bytes;
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
    layout.slots_offset = kRawVectorPageHeaderBytes + bitmap_bytes;
    return layout;
}

RawVectorHeap::~RawVectorHeap() { close(); }

void RawVectorHeap::open(const std::string& path, RawVectorHeapLayout layout) {
    close();
    _layout = layout;

    // Reopening an existing non-empty heap isn't supported yet (see the
    // file-header comment), so refuse rather than silently truncating
    // whatever a caller expected to still be there.
    struct stat st;
    if (::stat(path.c_str(), &st) == 0 && st.st_size > 0) {
        throw ANNException(
            "Refusing to overwrite existing non-empty raw-vector heap file "
            "(reopen/recovery not yet supported): " + path,
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // No O_DIRECT: this heap bypasses the buffer pool entirely for now, so
    // there's no requirement that caller buffers be page-aligned.
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
    // A flat slot has to survive being packed into RawVectorRID's low 31 bits;
    // past that, make_raw_vector_rid() would silently mask and alias an
    // already-live slot.
    if (_next_flat_slot > kRawVectorRidSlotMask) {
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

void RawVectorHeap::restore_slot_cursor(uint32_t next_flat_slot, uint32_t allocated_pages) {
    std::lock_guard<std::mutex> lg(_grow_mtx);
    _next_flat_slot  = next_flat_slot;
    _allocated_pages = allocated_pages;
}

bool RawVectorHeap::is_slot_occupied(uint32_t flat_slot) const {
    uint32_t page_id  = flat_slot / _layout.slots_per_page;
    uint32_t slot_idx = flat_slot % _layout.slots_per_page;
    uint64_t off = _layout.bitmap_offset(page_id) + slot_idx / 8;

    std::lock_guard<std::mutex> lg(_bitmap_mtx[page_id % kRawVectorBitmapLockStripes]);

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

    // Slots within a page share bitmap bytes, so this read-modify-write has to
    // be atomic with respect to other slots of the same page.
    std::lock_guard<std::mutex> lg(_bitmap_mtx[page_id % kRawVectorBitmapLockStripes]);

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

}  // namespace inplace
}  // namespace diskann
