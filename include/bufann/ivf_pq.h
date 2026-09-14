// Data structures for the IVF-PQ index type. IVF-PQ is separate from the graph
// index (no adjacency, own raw-vector page format); BufANNConfig::index_type
// selects between them. Not yet wired into build/query/mutation paths.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tsl/robin_set.h"

namespace diskann {
namespace inplace {

// IVFMetadata: coarse quantizer (centroids)
struct IVFMetadata {
    uint32_t nlist       = 0;
    uint32_t dim         = 0;
    uint32_t aligned_dim = 0;
    std::vector<float> centroids;  // shape: [nlist, aligned_dim]
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
struct RawVectorRID {
    uint32_t packed = 0;
};
static_assert(sizeof(RawVectorRID) == 4, "RawVectorRID must pack into 32 bits");

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

// Slots released by deletes, reused by inserts.
struct RawVectorFreeList {
    std::mutex mtx;
    std::vector<uint32_t> free_slots;  // flat slot indices, see RawVectorRID
};

// PostingListDelta: pending mutations not yet folded into PostingLists
struct PostingListDelta {
    std::unordered_map<uint32_t, std::vector<uint32_t>> pending_inserts;  // cluster_id -> vector_ids
    tsl::robin_set<uint32_t> tombstones;                                  // vector_ids
};

// DynamicPQCodes: PQ codes for vectors inserted since the last rebuild
// PQMetadata::codes is a flat array and cannot grow in place, so codes for
// recent inserts live here until the next rebuild. Queries consult both.
struct DynamicPQCodes {
    std::unordered_map<uint32_t, std::vector<uint8_t>> codes;  // vector_id -> [chunks]
};

// IVFPQIndexFileHeader: on-disk header for the combined IVF-PQ index file
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
};

}  // namespace inplace
}  // namespace diskann
