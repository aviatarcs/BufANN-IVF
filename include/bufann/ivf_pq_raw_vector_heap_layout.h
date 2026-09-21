// Page/slot geometry of the IVF-PQ raw-vector heap. Split out so the index
// file structures can describe a heap without depending on the heap itself.

#pragma once

#include <cstdint>

namespace diskann {
namespace inplace {

constexpr uint32_t RAW_VECTOR_PAGE_HEADER_BYTES = 16;
constexpr uint32_t RAW_VECTOR_PAGE_MAGIC        = 0x48465649;  // "IVFH" little-endian

// Every page starts with its own id and the geometry it was written with, so
// a page read from the wrong offset (a spliced, truncated-and-regrown, or
// foreign file) or through the wrong layout (an index file that describes a
// different heap) is detected on read. The heap file carries no header of its
// own, so this is the only place the file states its geometry.
struct RawVectorPageHeader {
    uint32_t magic     = 0;
    uint32_t page_id   = 0;
    uint32_t page_size = 0;
    uint32_t elem_size = 0;
};
static_assert(sizeof(RawVectorPageHeader) == RAW_VECTOR_PAGE_HEADER_BYTES, "page header is part of the file format");

// Page layout: [RawVectorPageHeader][occupancy bitmap, 1 bit per slot, LSB
// first][owner ids, u32 per slot][slot 0..N-1][pad]. page_size and elem_size
// determine the rest, and are what every page header records. A slot's owner
// is the id of the vector written to it, so a slot can be checked against
// the RID that led to it: on every read, and when a load reconciles the
// index file's RID table with the heap (ivf_pq_recover_deletes), where an
// occupied slot whose owner is another vector means the listed one was
// deleted and the slot reused.
constexpr uint32_t RAW_VECTOR_OWNER_BYTES = 4;

struct RawVectorHeapLayout {
    uint32_t page_size      = 0;
    uint32_t elem_size      = 0;
    uint32_t slots_per_page = 0;
    uint32_t bitmap_bytes   = 0;
    uint32_t owners_offset  = 0;  // byte offset of the owner-id array within a page
    uint32_t slots_offset   = 0;  // byte offset of slot 0 within a page

    uint32_t page_of(uint32_t flat_slot) const { return flat_slot / slots_per_page; }
    uint32_t index_in_page(uint32_t flat_slot) const { return flat_slot % slots_per_page; }

    uint64_t page_offset(uint32_t page_id) const { return uint64_t(page_id) * page_size; }
    uint32_t slot_offset_in_page(uint32_t index) const { return slots_offset + index * elem_size; }
    uint32_t owner_offset_in_page(uint32_t index) const { return owners_offset + index * RAW_VECTOR_OWNER_BYTES; }
    uint32_t bitmap_byte_in_page(uint32_t index) const { return RAW_VECTOR_PAGE_HEADER_BYTES + index / 8; }
    static uint8_t bitmap_mask(uint32_t index) { return uint8_t(1u << (index % 8)); }

    uint64_t slot_offset(uint32_t flat_slot) const {
        return page_offset(page_of(flat_slot)) + slot_offset_in_page(index_in_page(flat_slot));
    }
    uint64_t bitmap_byte_offset(uint32_t flat_slot) const {
        return page_offset(page_of(flat_slot)) + bitmap_byte_in_page(index_in_page(flat_slot));
    }
    uint64_t owner_offset(uint32_t flat_slot) const {
        return page_offset(page_of(flat_slot)) + owner_offset_in_page(index_in_page(flat_slot));
    }
};

// Largest slots_per_page whose header, bitmap and slots fit in page_size.
// Throws if not even one slot fits.
RawVectorHeapLayout compute_raw_vector_heap_layout(uint32_t page_size, uint32_t elem_size);

}  // namespace inplace
}  // namespace diskann
