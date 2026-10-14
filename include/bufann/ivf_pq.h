// Data structures for the IVF-PQ index type. IVF-PQ is separate from the graph
// index (no adjacency, own raw-vector page format); BufANNConfig::index_type
// selects between them. Not yet wired into build/query/mutation paths.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <pthread.h>

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

// PostingLists: partition c is ids[offsets[c] : offsets[c+1]]. `codes` holds
// the PQ code of ids[i] at [i * chunks, (i + 1) * chunks), so a probe scans
// its partition's codes sequentially; PQMetadata::codes, indexed by id, would
// cost a cache miss per candidate. The index file stores them in this order;
// the build derives them from the by-id codes (set_ivf_posting_codes, or the
// writer). A loaded index holds them only as `blocked` and leaves `codes`
// empty, so the codes are in memory once.
// The same codes transposed for the SIMD scan (ivf_pq_fastscan.h): list p
// owns blocks [block_start[p], block_start[p+1]); its i-th vector is lane
// i % IVF_FASTSCAN_BLOCK of its (i / IVF_FASTSCAN_BLOCK)-th block, and a
// block holds `chunks` rows of IVF_FASTSCAN_BLOCK bytes, row c being
// sub-code c of the block's vectors. Lanes past the list's end are zero.
// Built at load from the file's posting-order codes, not persisted.
constexpr uint32_t IVF_FASTSCAN_BLOCK = 64;
struct IVFPQBlockedCodes {
    uint32_t chunks = 0;
    std::vector<uint32_t> block_start;  // [nlist + 1]
    std::vector<uint8_t> blocks;        // [blocks x chunks x IVF_FASTSCAN_BLOCK]
};

struct PostingLists {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> ids;
    std::vector<uint8_t>  codes;  // shape: [ids.size(), chunks]
    IVFPQBlockedCodes blocked;    // empty until build_ivf_blocked_codes
};

// PQMetadata: product-quantization pivots and per-vector codes. Chunk c
// covers dimensions [chunk_offsets[c], chunk_offsets[c+1]); when chunks does
// not divide dim, the first dim - floor(dim / chunks) * chunks chunks hold
// one dimension more, as the upstream PQ trainer splits them
// (pq_chunk_offsets). Chunk c's k pivots are contiguous, k * chunk_len(c)
// floats from pivots[k * chunk_offsets[c]] on (pq_pivot); with uniform
// chunks that is the [chunks, k, chunk_dim] layout.
struct PQMetadata {
    uint32_t chunks    = 0;
    uint32_t chunk_dim = 0;  // dim / chunks when uniform, else 0
    uint32_t k         = 0;  // number of PQ centers per chunk (256 for uint8 codes)
    std::vector<uint32_t> chunk_offsets;  // [chunks + 1]
    std::vector<float>   pivots;  // [k x dim], chunk by chunk
    // shape: [N, chunks], by vector id. Only the build holds them (from
    // load_ivf_pq); a loaded index keeps the codes once, in PostingLists::codes,
    // and leaves this empty.
    std::vector<uint8_t> codes;
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
// RIDs that searches read concurrently with mutations go through
// load_rid/store_rid; their seq_cst ordering is what RawVectorHeap::ReadGuard
// relies on.
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

// Slots released by deletes. free_slot appends to `deferred` with the heap
// epoch of the free; allocate_slot moves the entries no in-flight reader can
// still address onto `free_slots` (see RawVectorHeap::ReadGuard) and pops
// from there. Both vectors are guarded by mtx.
struct RawVectorFreeList {
    struct Deferred {
        uint64_t epoch;      // RawVectorHeap epoch at the free
        uint32_t flat_slot;  // see RawVectorRID
    };
    std::mutex mtx;
    std::vector<Deferred> deferred;    // ascending by epoch
    std::vector<uint32_t> free_slots;  // reusable now
};

// PostingListDelta: pending mutations not yet folded into PostingLists. A
// deleted base vector stays in its posting list until the rebuild, so it is
// tombstoned; a deleted insert just leaves pending_inserts.
struct PostingListDelta {
    std::unordered_map<uint32_t, std::vector<uint32_t>> pending_inserts;  // cluster_id -> vector_ids
    tsl::robin_set<uint32_t> tombstones;                                  // deleted base vector_ids
};

// DynamicPQCodes: PQ codes for vectors inserted since the last rebuild
inline std::vector<uint32_t> pq_chunk_offsets(uint32_t dim, uint32_t chunks) {
    std::vector<uint32_t> off(size_t(chunks) + 1, 0);
    const uint32_t low = chunks == 0 ? 0 : dim / chunks, high_count = chunks == 0 ? 0 : dim - low * chunks;
    for (uint32_t c = 0; c < chunks; ++c) off[c + 1] = off[c] + low + (c < high_count ? 1 : 0);
    return off;
}
inline uint32_t pq_chunk_len(const PQMetadata& pq, uint32_t c) { return pq.chunk_offsets[c + 1] - pq.chunk_offsets[c]; }
inline const float* pq_pivot(const PQMetadata& pq, uint32_t c, uint32_t j) {
    return pq.pivots.data() + size_t(pq.k) * pq.chunk_offsets[c] + size_t(j) * pq_chunk_len(pq, c);
}
// The PQ's shape is consistent with `dim`: offsets as pq_chunk_offsets gives
// them, chunk_dim matching, k pivots of every chunk.
inline bool pq_shape_ok(const PQMetadata& pq, uint32_t dim) {
    return pq.chunks > 0 && pq.chunks <= dim && pq.k > 0 && pq.chunk_offsets.size() == size_t(pq.chunks) + 1 &&
           pq.chunk_offsets.back() == dim && pq.chunk_dim == (dim % pq.chunks == 0 ? dim / pq.chunks : 0) &&
           pq.pivots.size() == size_t(pq.k) * dim;
}

// PQMetadata::codes is a flat array and cannot grow in place, so codes for
// recent inserts live here until the next rebuild. Queries consult both.
struct DynamicPQCodes {
    std::unordered_map<uint32_t, std::vector<uint8_t>> codes;  // vector_id -> [chunks]
};

// The delta's lock. std::shared_mutex is glibc's reader-preferring rwlock,
// under which a writer waits for a moment with no reader at all; a few
// threads searching back to back never leave one, and mutations stall for
// good. This one holds new readers back while a writer waits. For the same
// reason a thread must not take it shared twice without releasing.
class WriterPreferringSharedMutex {
public:
    WriterPreferringSharedMutex() {
        pthread_rwlockattr_t attr;
        pthread_rwlockattr_init(&attr);
        pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
        pthread_rwlock_init(&_lock, &attr);
        pthread_rwlockattr_destroy(&attr);
    }
    ~WriterPreferringSharedMutex() { pthread_rwlock_destroy(&_lock); }
    WriterPreferringSharedMutex(const WriterPreferringSharedMutex&)            = delete;
    WriterPreferringSharedMutex& operator=(const WriterPreferringSharedMutex&) = delete;

    void lock() { pthread_rwlock_wrlock(&_lock); }
    void unlock() { pthread_rwlock_unlock(&_lock); }
    void lock_shared() { pthread_rwlock_rdlock(&_lock); }
    void unlock_shared() { pthread_rwlock_unlock(&_lock); }

private:
    pthread_rwlock_t _lock;
};

// IVFPQDelta: what mutations change that the index file does not hold.
// Mutations take mtx exclusively; a search holds it shared while it reads
// the delta and the RID table (which inserts grow), never across heap I/O.
// Ids at or past the base count are inserts, with codes in `codes` and list
// membership in `lists`, until the rebuild folds them in (write_ivf_pq_index
// refuses an index with unfolded inserts). Deletes reach the file only as
// the heap's occupancy bits; a load recovers them (ivf_pq_recover_deletes).
struct IVFPQDelta {
    mutable WriterPreferringSharedMutex mtx;
    PostingListDelta lists;
    DynamicPQCodes codes;
    RawVectorFreeList free_list;
};

// IVFPQIndex: everything the combined index file persists, in memory. The
// raw vectors themselves stay in the heap file; heap_layout, heap_pages and
// heap_next_slot describe it and are what RawVectorHeap::open_existing needs.
// assignments and rid_table also cover inserted vectors; lists (and their
// codes) cover only the base vectors the file was written from.
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
// which moves every slot in the heap file; version 4 added the per-page
// owner-id array between the bitmap and the slots, which moves them again;
// version 5 stores the PQ codes in posting-list order, not by vector id.
constexpr uint32_t IVF_PQ_INDEX_MAGIC   = 0x51465649;  // "IVFQ" little-endian
constexpr uint32_t IVF_PQ_INDEX_VERSION = 5;

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
