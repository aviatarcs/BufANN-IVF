// bufann_api.h
//
// Clean interface for building and querying a BufANN disk-resident graph index.
// Suitable for use by serverful database engines and external benchmarks.
//
// Template parameter T: float | uint8_t | int8_t

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "distance.h"
#include "pq_table.h"
#include "utils.h"
#include "bufann/inplace_backend.h"
#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_backend.h"

namespace diskann {
namespace inplace {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Selects which index implementation BufANNConfig configures. Chosen at
// build/startup time; graph and IVF-PQ indexes are independent on-disk
// formats and are never mixed within a single index_prefix.
enum class IndexType : uint8_t {
    Graph = 0,
    IvfPq = 1,
};

struct BufANNConfig {
    // --- required ---
    uint32_t dim = 0;           // vector dimensionality

    // --- index type ---
    IndexType index_type = IndexType::Graph;

    // --- IVF-PQ parameters (only used when index_type == IvfPq) ---
    uint32_t ivf_nlist     = 0;  // number of coarse-quantizer partitions
    uint32_t ivf_nprobe    = 0;  // partitions probed per search (search_L overrides it per query)
    uint32_t ivf_pq_chunks = 0;  // PQ subvector count for IVF-PQ codes/pivots; dim must be a multiple
    uint32_t ivf_rerank_m  = 0;  // PQ candidates re-ranked exactly per query; 0 = max(100, 10 * topK)

    // --- graph parameters ---
    uint32_t R           = 64;  // max graph degree (used for build and insert pruning)
    uint32_t L           = 100; // beam width for graph construction / default search beam
    uint32_t C           = 750; // candidate list cap for RNG pruning
    float    alpha       = 1.2f;
    bool     saturate_graph = false;

    // --- threading ---
    uint32_t build_threads = 8;

    // --- buffer pool ---
    uint32_t page_size                  = 4096;
    uint32_t buffer_pool_frames         = 16384;
    uint32_t flush_budget_pages_per_cycle = 32;
    uint32_t flush_wakeup_ms            = 100;

    // --- search ---
    uint32_t beamwidth = 4;

    // --- product quantization ---
    uint32_t pq_chunks = 0;  // 0 = no PQ

    // --- metadata/PQ storage ---
    // Maximum internal-ID slot count. 0 means derive from the current dataset
    // size at build/load time; set this above the loaded size when inserts need
    // headroom.
    uint32_t max_dataset_size = 0;

    // --- delete repair ---
    // Number of replacement edges to add per affected neighbor when repairing
    // a deleted node. Ported from PathFinder online deletion (default 3).
    uint32_t c_replace = 3;
    // Beam width for the search used to discover approximate in-neighbors of
    // each deleted node during repair. 0 falls back to config.L.
    uint32_t delete_repair_L = 0;

    // --- build mode ---
    // memory_budget_gb == 0: in-memory build (full dataset must fit in RAM)
    // memory_budget_gb  > 0: out-of-core disk-stream build via merged Vamana
    float memory_budget_gb = 0.0f;

    // --- metric ---
    diskann::Metric metric = diskann::Metric::L2;
};

// ---------------------------------------------------------------------------
// Index handle
// ---------------------------------------------------------------------------

template<typename T>
struct BufANNIndex {
    InPlaceGraphStore    store;
    std::vector<unsigned> medoids;
    // Float centroids retained for on-disk format compatibility. Search now
    // starts from all live medoids rather than selecting one via its centroid.
    std::vector<float>    medoid_coords;
    BufANNConfig         config;
    diskann::Distance<T>* dist_cmp        = nullptr;
    diskann::Distance<float>* dist_cmp_float = nullptr;
    FixedChunkPQTable<T>  pq_table;
    uint32_t              aligned_dim = 0;
    std::string           pq_prefix;  // path prefix used for PQ files

    // Set when config.index_type == IndexType::IvfPq; the graph members above
    // are then unused. Search scratch is per thread.
    std::unique_ptr<IVFPQBackend> ivf;

    BufANNIndex()  = default;
    ~BufANNIndex() {
        delete dist_cmp; dist_cmp = nullptr;
        delete dist_cmp_float; dist_cmp_float = nullptr;
    }

    BufANNIndex(const BufANNIndex&)            = delete;
    BufANNIndex& operator=(const BufANNIndex&) = delete;
};

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

// Build a new index from a raw data file.
//
// data_bin      - path to the flat binary file (.bin) containing base vectors.
//                 Format: [int32 npts][int32 dim][T * npts * dim].
// index_prefix  - path prefix for the heap file and metadata files written to
//                 disk (e.g. "/data/myindex/bufann").
// config        - build and runtime parameters (see BufANNConfig above).
// build_temp_dir- scratch directory for intermediate files when using the
//                 out-of-core disk-stream build mode. Ignored for in-memory build.
//
// Returns an owning pointer to the loaded index. Caller must free with
// bufann_free().
template<typename T>
BufANNIndex<T>* bufann_build(
    const std::string& data_bin,
    const std::string& index_prefix,
    const BufANNConfig& config,
    const std::string& build_temp_dir = "");

// Load an existing index that was previously built and saved to disk.
//
// index_prefix  - same prefix used when the index was built.
// config        - must supply at least dim, pq_chunks, page_size,
//                 buffer_pool_frames, and beamwidth. Build parameters
//                 (R, L, C, alpha) are re-used for maintenance operations.
//
// Returns an owning pointer. Caller must free with bufann_free().
template<typename T>
BufANNIndex<T>* bufann_load(
    const std::string& index_prefix,
    const BufANNConfig& config);

uint32_t bufann_snapshot_active_cap(const std::string& index_prefix);

// Insert a single node into the graph index.
//
// Synchronous: on return the node is live and reachable from search. Reverse
// edges are repaired in the foreground before return.
//
// tag      - external ID for the new point (must not already be active).
// coords   - raw vector with config.dim elements. Zero-padded internally
//            to the aligned dimension.
// search_L - beam width to use when finding the new node's neighbors.
//            Pass 0 to use config.L. Unused by an IVF-PQ index, which
//            assigns the point to its nearest partition; the point is
//            searchable on return but lives only in memory and the heap
//            file until the index file is rewritten, so bufann_load of the
//            prefix is refused until then.
template<typename T>
void bufann_insert(
    BufANNIndex<T>& idx,
    TagType tag,
    const T* coords,
    uint32_t search_L = 0);

// Logically delete a node by external tag. Foreground delete records the tag in
// an exact deleted-tag set and returns immediately; query result emission filters
// that set so the tag is invisible before graph repair runs.
//
// On an IVF-PQ index the delete is immediate: the vector is invisible to
// searches on return, its heap slot is reusable once in-flight searches
// finish, and the tag may be inserted again. A tag that is not active
// throws std::invalid_argument. A delete is durable once its clear of the
// slot's occupancy bit reaches the heap file: bufann_load recovers deletes
// from the bitmap, since the index file is not rewritten until the rebuild.
template<typename T>
void bufann_delete(
    BufANNIndex<T>& idx,
    TagType tag);

// Micro-batched logical delete: records every tag in `tags` in the deleted-tag
// set. Tag-to-internal-ID resolution and graph repair happen in maintenance.
// On an IVF-PQ index it is bufann_delete per tag, stopping at the first
// tag that throws.
template<typename T>
void bufann_delete_batch(
    BufANNIndex<T>& idx,
    const TagType* tags,
    size_t count);

// Maintenance pass for logical deletes since the previous maintenance run.
// Swaps out the current deleted-tag set, scans metadata arrays to resolve tags
// to active internal IDs, transitions those IDs to DeletePending, runs repair
// in maintenance microbatches, removes stale edges, then reclaims slots/internal
// IDs.
template<typename T>
void bufann_cleanup_deleted_edges(BufANNIndex<T>& idx,
                                  uint32_t num_threads = 1,
                                  uint32_t delete_micro_batch = 1);

// Flush all dirty pages currently held by the BufANN buffer pool.
template<typename T>
void bufann_flush_dirty(BufANNIndex<T>& idx);

// Flush up to max_pages dirty pages. max_pages=0 flushes all dirty pages.
template<typename T>
uint32_t bufann_flush_dirty_budget(BufANNIndex<T>& idx, uint32_t max_pages);

// Search for the topK approximate nearest neighbors of query_vec.
//
// query_vec - raw query vector with config.dim elements. Zero-padded
//             internally to the aligned dimension.
// topK      - number of results to return.
// search_L  - beam width. Pass 0 to use config.L. For an IVF-PQ index this
//             is the number of partitions probed (0 = config.ivf_nprobe), so
//             a sweep over L sweeps nprobe.
// out_tags  - caller-provided buffer with capacity >= topK. When nullptr,
//             results are discarded and only the count is returned.
//
// Returns the number of result tags written to out_tags.
template<typename T>
uint32_t bufann_query_into(
    BufANNIndex<T>& idx,
    const T* query_vec,
    uint32_t topK,
    TagType* out_tags,
    uint32_t search_L = 0);

// Convenience wrapper that returns a vector of at most topK tags sorted
// by ascending distance.
template<typename T>
std::vector<TagType> bufann_query(
    BufANNIndex<T>& idx,
    const T* query_vec,
    uint32_t topK,
    uint32_t search_L = 0);

// Repair the graph after deletions.
//
// Performs up to `budget` pages of sweep-repair: for each live node whose
// neighbor list references a deleted node, the dead neighbor is replaced by
// candidates drawn from its neighborhood.
//
// Reverse-edge consistency is handled in insert foreground path.
//
// budget - number of pages to sweep per call. Pass 0 to sweep all pages.
template<typename T>
void bufann_cleanup(BufANNIndex<T>& idx);

// Release all resources held by the index. Sets the pointer to nullptr.
template<typename T>
void bufann_free(BufANNIndex<T>*& idx);

}  // namespace inplace
}  // namespace diskann
