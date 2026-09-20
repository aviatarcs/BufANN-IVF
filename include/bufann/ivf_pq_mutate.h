// IVF-PQ mutations. An insert lands in the raw-vector heap and in an
// IVFPQDelta that every search consults; the index file is not touched
// (write_ivf_pq_index refuses an index with unfolded inserts) until the
// posting-list rebuild folds the delta in.

#pragma once

#include <cstdint>
#include <vector>

#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_raw_vector_heap.h"

namespace diskann {
namespace inplace {

// Partition of x: the centroid with the smallest exact squared L2 distance,
// the lowest id on a tie. x has meta.dim elements.
uint32_t ivf_nearest_centroid(const IVFMetadata& meta, const float* x);

// code[c] = argmin_j ||x[chunk c] - pivots[c][j]||^2, exact float distances,
// the lowest j on a tie. x has chunks * chunk_dim elements.
void ivf_pq_encode(const PQMetadata& pq, const float* x, uint8_t* code);

// An insert in two steps, so a caller can publish its own bookkeeping (the
// backend's tag maps) in the same critical section as the vector.
//
// prepare: partition and code of the vector (ix.meta.dim elements of T; the
// heap's elem_size must be dim * sizeof(T)), and its raw bytes written to a
// slot from the free list or the end of the heap. Nothing addresses the
// slot yet, so this takes no lock and may run concurrently with searches.
struct IVFPQPreparedInsert {
    uint32_t cluster = 0;
    uint32_t slot    = 0;
    std::vector<uint8_t> code;  // [chunks]
};
template<typename T>
IVFPQPreparedInsert ivf_pq_prepare_insert(const IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta,
                                          const T* vec);

// publish: gives the vector the next unused id (ids are never reused) and
// makes it visible to searches: assignment, active RID, delta membership
// and code, all at once. The caller holds delta.mtx exclusively.
uint32_t ivf_pq_publish_insert(IVFPQIndex& ix, IVFPQDelta& delta, IVFPQPreparedInsert&& prepared);

// Both steps; searches see the vector once the call returns.
template<typename T>
uint32_t ivf_pq_insert(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, const T* vec);

}  // namespace inplace
}  // namespace diskann
