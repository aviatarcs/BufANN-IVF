// IVF-PQ query path: centroid scan -> top-nprobe partitions -> PQ table
// lookup over their posting lists -> exact re-rank of the top rerank_m from
// the raw-vector heap. A batch computes every query's centroid distances
// with one GEMM and then handles the queries in parallel; a single query is
// a batch of one.

#pragma once

#include <cstdint>
#include <vector>

#include "bufann/ivf_pq.h"
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

// Buffers one thread reuses across queries; sized to the index on every
// search (no-ops once they fit), so a scratch may be used on any index but
// not shared between threads.
struct IVFPQSearchScratch {
    std::vector<float> query_padded;    // [aligned_dim], zero beyond dim
    std::vector<float> centroid_dist;   // [nlist]
    std::vector<uint32_t> probe_order;  // [nlist] partition ids, nearest first
    std::vector<float> pq_table;        // [chunks x k]
    std::vector<uint32_t> candidates;   // ids scanned from the probed partitions
    std::vector<float> pq_dist;         // parallel to candidates
    std::vector<uint32_t> order;        // selection workspace over candidates
    std::vector<uint32_t> shortlist;    // ids kept for the re-rank
    std::vector<uint32_t> shortlist_slot;  // their heap slots, parallel to shortlist
    std::vector<float> exact_dist;      // parallel to shortlist
    std::vector<char> raw_vector;       // [heap elem_size]
    std::vector<float> vector;          // [dim]
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
// concurrently with inserts into and deletes from it.
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch, const IVFPQDelta* delta = nullptr);

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
