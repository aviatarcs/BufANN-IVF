// IVF-PQ query path: centroid scan -> top-nprobe partitions -> PQ table
// lookup over their posting lists -> exact re-rank of the top rerank_m from
// the raw-vector heap. A batch computes every query's centroid distances
// with one GEMM and then handles the queries in parallel; a single query is
// a batch of one. ivf_pq_search_graph instead finds the partitions through a
// proximity graph over the centroids (ivf_pq_centroid_graph.h).

#pragma once

#include <cstdint>
#include <vector>

#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_centroid_graph.h"
#include "bufann/ivf_pq_fastscan.h"
#include "bufann/ivf_pq_raw_vector_heap.h"

namespace diskann {
namespace inplace {

// Neighbours ascending by dist; fewer than k when the probed partitions hold
// fewer active vectors. dists are exact squared L2 after a re-rank, PQ
// approximations without one.
struct IVFPQSearchResult {
    std::vector<uint32_t> ids;
    std::vector<float> dists;
};

// Per-query probe depth for ivf_pq_search_graph; off (tau == 0) by default.
// When on, nprobe is a cap: of the partitions the centroid search returns,
// nearest first, the first min_probes are probed and then each one whose
// squared centroid distance is at most tau times the ref-th nearest's.
// Queries whose neighbours sit in one tight region stop early; a fixed
// nprobe pays for the widest query everywhere. On SIFT100M (nlist 131072)
// it probes 1.7-2.3x fewer codes at the probe recall of a fixed nprobe.
struct IVFProbeCutoff {
    float tau = 0.0f;         // 0: probe all nprobe; otherwise >= 1
    uint32_t ref = 8;         // the reference is the ref-th nearest partition
    uint32_t min_probes = 16;
};

// How many of `n` partitions, ascending squared distances `dist`, the cutoff
// probes: n when it is off; otherwise at least min(n, min_probes).
uint32_t ivf_probe_count(const float* dist, uint32_t n, const IVFProbeCutoff& cutoff);

// How the shortlist's raw vectors are fetched for the exact re-rank. By
// default (batch 0) all of them in one buffer-pool batch, whose misses are
// read concurrently. Otherwise `batch` at a time, nearest by PQ distance
// first, each batch's misses read concurrently; with patience > 0 the
// re-rank stops once `patience` batches in a row put nothing into the exact
// top k, so a query whose PQ order is already right reads a prefix of its
// shortlist. Batches without a stop only add round trips.
struct IVFRerankSchedule {
    uint32_t batch = 0;
    uint32_t patience = 0;
};

// Where a thread's queries spent their time, summed over the queries searched
// with one scratch: choosing the partitions (GEMV or graph), building the PQ
// table, scanning the codes, picking the shortlist and its RIDs, fetching its
// raw vectors (buffer-pool pins, I/O included) and computing exact distances.
// The phases are back to back, so their sum is the search's own time.
struct IVFPQSearchTimes {
    uint64_t centroid_ns = 0, table_ns = 0, scan_ns = 0, select_ns = 0, fetch_ns = 0, rerank_ns = 0;
    uint64_t queries = 0, reranked = 0, fetch_batches = 0;

    IVFPQSearchTimes& operator+=(const IVFPQSearchTimes& o) {
        centroid_ns += o.centroid_ns, table_ns += o.table_ns, scan_ns += o.scan_ns, select_ns += o.select_ns;
        fetch_ns += o.fetch_ns, rerank_ns += o.rerank_ns;
        queries += o.queries, reranked += o.reranked, fetch_batches += o.fetch_batches;
        return *this;
    }
};

// Buffers one thread reuses across queries; sized to the index on every
// search (no-ops once they fit), so a scratch may be used on any index but
// not shared between threads.
struct IVFPQSearchScratch {
    IVFPQSearchTimes times;             // accumulated, never reset by a search
    std::vector<float> query_padded;    // [aligned_dim], zero beyond dim
    std::vector<float> centroid_dist;   // [nlist]
    std::vector<uint32_t> probe_order;  // [nlist] partition ids, nearest first
    std::vector<float> probe_dist;      // the graph search's squared distances, parallel to probe_order
    std::vector<float> pq_table;        // [chunks x k]
    std::vector<uint32_t> candidates;   // ids scanned from the probed partitions
    std::vector<float> pq_dist;         // parallel to candidates
    std::vector<uint32_t> order;        // selection workspace over candidates
    std::vector<uint32_t> shortlist;    // ids kept for the re-rank
    std::vector<uint32_t> shortlist_slot;  // their heap slots, parallel to shortlist
    std::vector<float> exact_dist;      // parallel to shortlist
    std::vector<float> top_exact;       // max-heap of the k smallest exact distances so far
    std::vector<char> raw_vectors;      // [shortlist x heap elem_size]
    std::vector<uint32_t> owners;       // parallel to shortlist
    std::vector<float> vector;          // [dim]
    IVFCentroidGraphScratch graph;      // ivf_pq_search_graph only
    // The SIMD scan (ivf_pq_fastscan.h), used when rerank_m > 0, the codes
    // are 8-bit, the index has blocked codes, the CPU has AVX-512 VBMI and
    // fastscan_enabled(). It keeps the rerank_m nearest by quantized PQ
    // distance, which can differ from the float scan's near the boundary.
    IVFPQQuantizedTable qtable;
    std::vector<float> sample;          // float PQ distances that set the table's range
    std::vector<uint16_t> block_sums;   // one list's sums, [IVF_FASTSCAN_BLOCK x its blocks]
    std::vector<std::pair<uint16_t, uint32_t>> top;  // max-heap: quantized distance, candidate id
};

// T is the raw vector element type the heap holds (float, uint8_t, int8_t);
// the query is always float, like the centroids and pivots. Centroid
// distances come from one GEMM against the padded centroids as
// ||q||^2 + ||c||^2 - 2 q.c, so partitions equidistant to within a few ulps
// of those norms may be ordered either way. nprobe is clamped to nlist.
// rerank_m == 0 skips the re-rank and ranks by PQ distance; otherwise it
// must be at least k. Vectors whose RID is inactive are dropped. `delta`,
// when given, contributes the vectors inserted since the index file was
// written (ivf_pq_insert), scanned in the probed partitions like the base
// vectors, and withholds the deleted ones (ivf_pq_delete); searches may run
// concurrently with inserts into and deletes from it. `rerank` sets how the
// re-rank fetches its vectors (IVFRerankSchedule).
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch, const IVFPQDelta* delta = nullptr,
                                const IVFRerankSchedule& rerank = {});

// As ivf_pq_search, but the probed partitions are the nprobe nearest
// centroids that a search of `graph` with a beam of centroid_L >= nprobe
// finds (nprobe clamped to nlist), in place of the exact nearest nprobe from
// a GEMV over every centroid. It visits a few hundred centroids instead of
// all nlist, and may miss a partition the GEMV would probe. `graph` must be
// built from index.meta; only its nlist is checked. With `cutoff` on, the
// nprobe partitions found are cut per query (IVFProbeCutoff); `rerank` sets
// how the re-rank fetches its vectors (IVFRerankSchedule).
template<typename T>
IVFPQSearchResult ivf_pq_search_graph(const IVFPQIndex& index, const IVFCentroidGraph& graph,
                                      const RawVectorHeap& heap, const float* query, uint32_t k, uint32_t nprobe,
                                      uint32_t centroid_L, uint32_t rerank_m, IVFPQSearchScratch& scratch,
                                      const IVFPQDelta* delta = nullptr, const IVFProbeCutoff& cutoff = {},
                                      const IVFRerankSchedule& rerank = {});

// Allocates a scratch per call; for one-off queries.
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m);

// `queries` is row-major [nq x dim]. Centroid distances come from one GEMM per
// `gemm_rows` queries; the queries of each such block are then searched in
// parallel with OpenMP, one scratch per thread. Results are in query order and
// identical to nq single-query searches up to the rounding of the batched GEMM.
const uint32_t IVF_PQ_SEARCH_GEMM_ROWS = 1024;

template<typename T>
std::vector<IVFPQSearchResult> ivf_pq_search_batch(const IVFPQIndex& index, const RawVectorHeap& heap,
                                                   const float* queries, size_t nq, uint32_t k,
                                                   uint32_t nprobe, uint32_t rerank_m,
                                                   uint32_t gemm_rows = IVF_PQ_SEARCH_GEMM_ROWS,
                                                   const IVFPQDelta* delta = nullptr);

}  // namespace inplace
}  // namespace diskann
