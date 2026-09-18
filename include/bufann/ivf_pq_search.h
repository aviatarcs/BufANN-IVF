// IVF-PQ query path, one query at a time: centroid scan -> top-nprobe
// partitions -> PQ table lookup over their posting lists -> exact re-rank of
// the top rerank_m from the raw-vector heap. Recall is the spec, not speed;
// the batched path builds on this.

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

// Buffers one thread reuses across queries. Holds the centroid norms of the
// index it was last prepared for; ivf_pq_search re-prepares on a different
// index, so a scratch must not be shared between threads.
class IVFPQSearchScratch {
public:
    void prepare(const IVFPQIndex& index);
    bool prepared_for(const IVFPQIndex& index) const { return _index == &index; }

    const std::vector<float>& centroid_l2sq() const { return _centroid_l2sq; }

    std::vector<float> query_padded;    // [aligned_dim], zero beyond dim
    std::vector<float> centroid_dist;   // [nlist]
    std::vector<uint32_t> probe_order;  // [nlist] partition ids, nearest first
    std::vector<float> pq_table;        // [chunks x k]
    std::vector<uint32_t> candidates;   // ids scanned from the probed partitions
    std::vector<float> pq_dist;         // parallel to candidates
    std::vector<uint32_t> order;        // selection workspace over candidates
    std::vector<uint32_t> shortlist;    // ids kept for the re-rank
    std::vector<float> exact_dist;      // parallel to shortlist
    std::vector<char> raw_vector;       // [heap elem_size]
    std::vector<float> vector;          // [dim]

private:
    const IVFPQIndex* _index = nullptr;
    std::vector<float> _centroid_l2sq;  // [nlist]
};

// T is the raw vector element type the heap holds (float, uint8_t, int8_t);
// the query is always float, like the centroids and pivots. Centroid
// distances come from one GEMM against the padded centroids as
// ||q||^2 + ||c||^2 - 2 q.c, so partitions equidistant to within a few ulps
// of those norms may be ordered either way. nprobe is clamped to nlist.
// rerank_m == 0 skips the re-rank and ranks by PQ distance; otherwise it
// must be at least k. Vectors whose RID is inactive are dropped.
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch);

// Allocates a scratch per call; for one-off queries.
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m);

}  // namespace inplace
}  // namespace diskann
