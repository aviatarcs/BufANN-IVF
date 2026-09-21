// IVF-PQ mutations. An insert lands in the raw-vector heap and in an
// IVFPQDelta that every search consults; a delete clears the vector's RID,
// records it in the delta and gives its heap slot back. The index file is
// not touched until the posting-list rebuild folds the delta in.

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
// heap's elem_size must be dim * sizeof(T)), the next unused id, and the raw
// bytes written under that id to a slot from the free list or the end of
// the heap. The id is reserved under delta.mtx as an inactive RID that no
// list names yet; the heap write is unlocked. Ids are never reused, so a
// prepare that fails after reserving leaves a permanently inactive id.
struct IVFPQPreparedInsert {
    uint32_t id      = 0;
    uint32_t cluster = 0;
    uint32_t slot    = 0;
    std::vector<uint8_t> code;  // [chunks]
};
template<typename T>
IVFPQPreparedInsert ivf_pq_prepare_insert(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, const T* vec);

// publish: makes the vector visible to searches: active RID, delta
// membership and code, all at once. The caller holds delta.mtx exclusively.
uint32_t ivf_pq_publish_insert(IVFPQIndex& ix, IVFPQDelta& delta, IVFPQPreparedInsert&& prepared);

// Both steps; searches see the vector once the call returns.
template<typename T>
uint32_t ivf_pq_insert(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, const T* vec);

// A delete in two steps, mirroring the insert. `id` must be active.
//
// retract: makes the vector invisible to searches -- RID cleared with
// store_rid (before the slot is freed, as the heap's grace period requires),
// a base id tombstoned, an inserted id removed from the delta. Returns the
// heap slot for the caller to free once it has released delta.mtx, which it
// holds exclusively here.
uint32_t ivf_pq_retract_delete(IVFPQIndex& ix, IVFPQDelta& delta, uint32_t id);

// Both steps; the slot is on the free list's deferred entries on return.
void ivf_pq_delete(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, uint32_t id);

// Reconciles a freshly loaded index with its heap, whose occupancy bits and
// slot owners record deletes the index file does not: every listed vector
// whose slot is not occupied under its id is retracted, and every
// unoccupied slot below the cursor goes on the free list. A slot occupied
// by a vector the file does not list there is an insert since the file was
// written, and the load is refused. Returns how many vectors it retracted.
// The delta must be empty and nothing else may touch the index meanwhile.
size_t ivf_pq_recover_deletes(IVFPQIndex& ix, const RawVectorHeap& heap, IVFPQDelta& delta);

}  // namespace inplace
}  // namespace diskann
