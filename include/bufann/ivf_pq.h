// Data structures for the IVF-PQ index type. IVF-PQ is separate from the graph
// index (no adjacency, own raw-vector page format); BufANNConfig::index_type
// selects between them. Not yet wired into build/query/mutation paths.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "tsl/robin_set.h"

#include "bufann/ivf_pq_raw_vector_heap_layout.h"

namespace diskann {
namespace inplace {

// IVFMetadata: coarse quantizer (centroids)
struct IVFMetadata {
    uint32_t nlist       = 0;
    uint32_t dim         = 0;
    uint32_t aligned_dim = 0;
    std::vector<float> centroids;      // shape: [nlist, aligned_dim]
    std::vector<float> centroid_l2sq;  // shape: [nlist]; set by set_ivf_centroid_norms
};

// ClusterAssignments: centroid index per vector, indexed by internal ID
struct ClusterAssignments {
    std::vector<uint32_t> cluster_id;  // shape: [N]
};

// PostingLists: partition c is ids[offsets[c] : offsets[c+1]].
struct PostingLists {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> ids;
};

// PQMetadata: product-quantization pivots and per-vector codes
struct PQMetadata {
    uint32_t chunks    = 0;
    uint32_t chunk_dim = 0;
    uint32_t k         = 0;  // number of PQ centers per chunk (256 for uint8 codes)
    std::vector<float>   pivots;  // shape: [chunks, k, chunk_dim]
    std::vector<uint8_t> codes;   // shape: [N, chunks]
};

// IVFPQSearchConfig: search-time handle over the live index structures
// Atomic pointers let a background rebuild be published with one store while
// searches continue. Non-owning: a swapped-out structure may still be read by
// in-flight searches, so it must not be freed until grace-period reclaim
// exists. Publish with release, load with acquire.
struct IVFPQSearchConfig {
    bool     is_ivf_pq_index = false;
    uint32_t nprobe          = 0;
    std::atomic<const IVFMetadata*>        ivf_meta{nullptr};
    std::atomic<const ClusterAssignments*> assignments{nullptr};
    std::atomic<const PostingLists*>       posting_lists{nullptr};
    std::atomic<const PQMetadata*>         pq_meta{nullptr};
};

// RawVectorRID: bit 31 is the active flag, the low 31 bits a flat heap slot.
// A RID that searches may read concurrently with a delete or insert is
// accessed through load_rid/store_rid, whose seq_cst ordering the heap's
// free-slot grace period relies on (RawVectorHeap::ReadGuard).
struct RawVectorRID {
    uint32_t packed = 0;
};
static_assert(sizeof(RawVectorRID) == 4, "RawVectorRID must pack into 32 bits");

inline RawVectorRID load_rid(const RawVectorRID& rid) {
    return RawVectorRID{std::atomic_ref<const uint32_t>(rid.packed).load()};
}
inline void store_rid(RawVectorRID& rid, RawVectorRID value) {
    std::atomic_ref<uint32_t>(rid.packed).store(value.packed);
}

constexpr uint32_t RAW_VECTOR_RID_ACTIVE_BIT = 0x80000000u;
constexpr uint32_t RAW_VECTOR_RID_SLOT_MASK  = 0x7FFFFFFFu;

inline uint32_t rid_flat_slot(RawVectorRID rid) { return rid.packed & RAW_VECTOR_RID_SLOT_MASK; }
inline bool     rid_is_active(RawVectorRID rid) { return (rid.packed & RAW_VECTOR_RID_ACTIVE_BIT) != 0; }

inline RawVectorRID make_raw_vector_rid(uint32_t flat_slot, bool active) {
    RawVectorRID rid;
    rid.packed = (flat_slot & RAW_VECTOR_RID_SLOT_MASK) | (active ? RAW_VECTOR_RID_ACTIVE_BIT : 0u);
    return rid;
}

// Vector ID -> raw-vector slot. Needed because inserts are not ID-ordered.
struct RawVectorRIDTable {
    std::vector<RawVectorRID> rid;
};

// Slots released by deletes, reused by inserts -- but not before every
// search that may still address a slot has finished (RawVectorHeap::ReadGuard).
// free_slot appends to `deferred` with the epoch of the free; allocate_slot
// moves the entries no in-flight reader can still see onto `free_slots` and
// pops from there. Both vectors are guarded by mtx.
struct RawVectorFreeList {
    struct Deferred {
        uint64_t epoch;      // RawVectorHeap epoch at the free
        uint32_t flat_slot;  // see RawVectorRID
    };
    std::mutex mtx;
    std::vector<Deferred> deferred;    // ascending by epoch
    std::vector<uint32_t> free_slots;  // reusable now
};

// PostingListDelta: pending mutations not yet folded into PostingLists.
// A deleted base vector stays in its posting list until the rebuild, so it
// is tombstoned and searches skip it; a deleted inserted vector is simply
// removed from pending_inserts, so tombstones never holds an inserted id.
struct PostingListDelta {
    std::unordered_map<uint32_t, std::vector<uint32_t>> pending_inserts;  // cluster_id -> vector_ids
    tsl::robin_set<uint32_t> tombstones;                                  // deleted base vector_ids
};

// DynamicPQCodes: PQ codes for vectors inserted since the last rebuild
// PQMetadata::codes is a flat array and cannot grow in place, so codes for
// recent inserts live here until the next rebuild. Queries consult both.
struct DynamicPQCodes {
    std::unordered_map<uint32_t, std::vector<uint8_t>> codes;  // vector_id -> [chunks]
};

// IVFPQDelta: what mutations change that the index file does not hold, and
// the lock that orders them against searches. An insert publishes and a
// delete retracts under mtx held exclusively; a search holds it shared while
// it reads the delta and the RID table (which inserts grow) and releases it
// before its heap reads. Ids at or past the base count are inserts, with
// codes in `codes` and posting-list membership in `lists`, until the rebuild
// folds them into the index (write_ivf_pq_index refuses an index with
// unfolded inserts). Deletes are likewise in memory and the heap's
// occupancy bitmap only: the index file still lists the vector, so a reload
// of the prefix brings it back until the rebuild rewrites the file.
struct IVFPQDelta {
    mutable std::shared_mutex mtx;
    PostingListDelta lists;
    DynamicPQCodes codes;
    RawVectorFreeList free_list;
};

// IVFPQIndex: everything the combined index file persists, in memory. The
// raw vectors themselves stay in the heap file; heap_layout, heap_pages and
// heap_next_slot describe it and are what RawVectorHeap::open_existing needs.
// assignments and rid_table also cover inserted vectors; lists and pq.codes
// cover only the base vectors the file was written from.
struct IVFPQIndex {
    IVFMetadata meta;
    ClusterAssignments assignments;
    PostingLists lists;
    PQMetadata pq;
    RawVectorRIDTable rid_table;
    RawVectorHeapLayout heap_layout;
    uint32_t heap_pages     = 0;
    uint32_t heap_next_slot = 0;  // RawVectorHeap::next_flat_slot() when the file was written
};

// Vectors the posting lists and PQ codes cover; ids from here on are inserts.
inline uint32_t ivf_pq_num_base(const IVFPQIndex& index) {
    return index.lists.offsets.empty() ? 0 : index.lists.offsets.back();
}

// IVFPQIndexFileHeader: on-disk header for the combined IVF-PQ index file.
// Fixed layout, written and read as raw bytes; bump version on any change.
// Version 2 added raw_vector_next_slot and the RawVectorPageHeader; version 3
// grew that page header from 8 to 16 bytes to record page_size and elem_size,
// which moves every slot in the heap file.
constexpr uint32_t IVF_PQ_INDEX_MAGIC   = 0x51465649;  // "IVFQ" little-endian
constexpr uint32_t IVF_PQ_INDEX_VERSION = 3;

struct IVFPQIndexFileHeader {
    uint32_t magic   = 0;
    uint32_t version = 0;

    uint32_t nlist       = 0;
    uint32_t dim         = 0;
    uint32_t aligned_dim = 0;

    uint32_t pq_chunks    = 0;
    uint32_t pq_chunk_dim = 0;
    uint32_t pq_k         = 0;

    uint64_t num_vectors = 0;

    uint64_t centroids_offset = 0;
    uint64_t centroids_bytes  = 0;

    uint64_t cluster_assignments_offset = 0;
    uint64_t cluster_assignments_bytes  = 0;

    uint64_t posting_offsets_offset = 0;
    uint64_t posting_offsets_bytes  = 0;
    uint64_t posting_ids_offset     = 0;
    uint64_t posting_ids_bytes      = 0;

    uint64_t pq_pivots_offset = 0;
    uint64_t pq_pivots_bytes  = 0;
    uint64_t pq_codes_offset  = 0;
    uint64_t pq_codes_bytes   = 0;

    uint64_t rid_table_offset = 0;
    uint64_t rid_table_bytes  = 0;

    uint32_t raw_vector_elem_size = 0;
    uint32_t raw_vector_page_size = 0;
    uint64_t raw_vectors_bytes    = 0;
    uint64_t raw_vector_next_slot = 0;
};
static_assert(sizeof(IVFPQIndexFileHeader) == 176, "header layout is part of the file format");

}  // namespace inplace
}  // namespace diskann
