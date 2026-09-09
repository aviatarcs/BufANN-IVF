// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// On-disk and in-memory data structures for the IVF-PQ index type.
//
// IVF-PQ is a separate index type from the BufANN graph index: it carries no
// graph adjacency, uses its own page format for raw vectors, and is built and
// deployed independently. Which index type is active is chosen at
// build/startup time via BufANNConfig::index_type (see bufann_api.h).
//
// This header only declares the structures used across the build, query, and
// mutation paths; it is not yet wired into any of them.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tsl/robin_set.h"

namespace diskann {
namespace inplace {

// ---------------------------------------------------------------------------
// IVFMetadata -- coarse quantizer (centroids)
// ---------------------------------------------------------------------------
struct IVFMetadata {
    uint32_t nlist       = 0;
    uint32_t dim         = 0;
    uint32_t aligned_dim = 0;
    std::vector<float> centroids;  // shape: [nlist, aligned_dim]
};

// ---------------------------------------------------------------------------
// ClusterAssignments -- centroid index per vector, indexed by internal ID
// ---------------------------------------------------------------------------
struct ClusterAssignments {
    std::vector<uint32_t> cluster_id;  // shape: [N]; cluster_id[i] = centroid index for vector i
};

// ---------------------------------------------------------------------------
// PostingLists -- CSR-style grouping of vector IDs by cluster
// ---------------------------------------------------------------------------
// offsets[i] = start of partition i in the ids array.
// ids is the flattened array of all point IDs, such that partition i is
// ids[offsets[i]:offsets[i+1]].
struct PostingLists {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> ids;
};

// ---------------------------------------------------------------------------
// PQMetadata -- product-quantization pivots and per-vector codes
// ---------------------------------------------------------------------------
struct PQMetadata {
    uint32_t chunks    = 0;
    uint32_t chunk_dim = 0;
    uint32_t k         = 0;  // number of PQ centers per chunk (256 for uint8 codes)
    std::vector<float>   pivots;  // shape: [chunks, k, chunk_dim]
    std::vector<uint8_t> codes;   // shape: [N, chunks]
};

// ---------------------------------------------------------------------------
// IVFPQSearchConfig -- search-time handle over the live index structures
// ---------------------------------------------------------------------------
// Structures are held behind atomic pointers so background rebuilds (of
// PostingLists / ClusterAssignments) can be swapped in without blocking
// concurrent searches; a rebuild is built off to the side and published with
// a single atomic store once complete.
struct IVFPQSearchConfig {
    bool     is_ivf_pq_index = false;
    uint32_t nprobe          = 0;
    std::atomic<const IVFMetadata*>        ivf_meta{nullptr};
    std::atomic<const ClusterAssignments*> assignments{nullptr};
    std::atomic<const PostingLists*>       posting_lists{nullptr};
    std::atomic<const PQMetadata*>         pq_meta{nullptr};
};

// ---------------------------------------------------------------------------
// RawVectorRID -- packed pointer from a vector ID to its raw-vector heap slot
// ---------------------------------------------------------------------------
// A RID is packed into 32 bits: bit 31 is the active flag, the low 31 bits
// are a flat slot index that decomposes into (page_id, slot_idx) given the
// heap's slots-per-page (see RawVectorHeapLayout in raw_vector_heap.h) --
// slots per page depends on the configured element size, so it is passed in
// rather than assumed fixed.
struct RawVectorRID {
    uint32_t packed = 0;
};
static_assert(sizeof(RawVectorRID) == 4, "RawVectorRID must pack into 32 bits");

constexpr uint32_t kRawVectorRidActiveBit = 0x80000000u;
constexpr uint32_t kRawVectorRidSlotMask  = 0x7FFFFFFFu;

inline uint32_t rid_flat_slot(RawVectorRID rid) { return rid.packed & kRawVectorRidSlotMask; }
inline bool     rid_is_active(RawVectorRID rid) { return (rid.packed & kRawVectorRidActiveBit) != 0; }
inline uint32_t rid_page_id(RawVectorRID rid, uint32_t slots_per_page) {
    return rid_flat_slot(rid) / slots_per_page;
}
inline uint32_t rid_slot_idx(RawVectorRID rid, uint32_t slots_per_page) {
    return rid_flat_slot(rid) % slots_per_page;
}

inline RawVectorRID make_raw_vector_rid(uint32_t flat_slot, bool active) {
    RawVectorRID rid;
    rid.packed = (flat_slot & kRawVectorRidSlotMask) | (active ? kRawVectorRidActiveBit : 0u);
    return rid;
}

// Maps vector ID -> location of its raw vector bytes on disk. Inserts are not
// necessarily ID-ordered, so this table is required to locate a vector's page
// and slot from its ID alone.
struct RawVectorRIDTable {
    std::vector<RawVectorRID> rid;
};

// Free-list of (page, slot) locations in the raw-vector heap, populated by
// deletes and drained by inserts. Mirrors the free-list used by the graph
// index's slot allocator.
struct RawVectorFreeList {
    std::mutex mtx;
    std::vector<uint32_t> free_slots;  // flat slot indices, see RawVectorRID
};

// ---------------------------------------------------------------------------
// PostingListDelta -- pending mutations not yet folded into PostingLists
// ---------------------------------------------------------------------------
struct PostingListDelta {
    std::unordered_map<uint32_t, std::vector<uint32_t>> pending_inserts;  // cluster_id -> vector_ids
    tsl::robin_set<uint32_t> tombstones;                                  // vector_ids
};

// ---------------------------------------------------------------------------
// IVFPQIndexFileHeader -- on-disk header for the combined IVF-PQ index file
// ---------------------------------------------------------------------------
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
