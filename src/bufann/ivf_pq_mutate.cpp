#include "bufann/ivf_pq_mutate.h"

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

namespace {

float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

}  // namespace

uint32_t ivf_nearest_centroid(const IVFMetadata& meta, const float* x) {
    IVF_PQ_REQUIRE(meta.nlist > 0 && meta.centroids.size() == size_t(meta.nlist) * meta.aligned_dim,
                   "IVFMetadata is inconsistent");
    uint32_t best = 0;
    float best_dist = sq_dist(x, meta.centroids.data(), meta.dim);
    for (uint32_t c = 1; c < meta.nlist; ++c) {
        const float d = sq_dist(x, meta.centroids.data() + size_t(c) * meta.aligned_dim, meta.dim);
        if (d < best_dist) best = c, best_dist = d;
    }
    return best;
}

void ivf_pq_encode(const PQMetadata& pq, const float* x, uint8_t* code) {
    IVF_PQ_REQUIRE(pq.chunks > 0 && pq.k > 0 && pq.k <= 256 &&
                       pq.pivots.size() == size_t(pq.chunks) * pq.k * pq.chunk_dim,
                   "PQMetadata is inconsistent");
    for (uint32_t c = 0; c < pq.chunks; ++c) {
        const float* sub = x + size_t(c) * pq.chunk_dim;
        const float* pivots = pq.pivots.data() + size_t(c) * pq.k * pq.chunk_dim;
        uint32_t best = 0;
        float best_dist = sq_dist(sub, pivots, pq.chunk_dim);
        for (uint32_t j = 1; j < pq.k; ++j) {
            const float d = sq_dist(sub, pivots + size_t(j) * pq.chunk_dim, pq.chunk_dim);
            if (d < best_dist) best = j, best_dist = d;
        }
        code[c] = uint8_t(best);
    }
}

template<typename T>
IVFPQPreparedInsert ivf_pq_prepare_insert(const IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta,
                                          const T* vec) {
    IVF_PQ_REQUIRE(vec != nullptr, "vector is null");
    const uint32_t dim = ix.meta.dim;
    IVF_PQ_REQUIRE(heap.layout().elem_size == dim * sizeof(T),
                   "raw-vector heap elem_size does not match dim * sizeof(T)");
    IVF_PQ_REQUIRE(ix.pq.chunks * ix.pq.chunk_dim == dim, "PQMetadata does not cover dim");

    std::vector<float> x(dim);
    for (uint32_t d = 0; d < dim; ++d) x[d] = float(vec[d]);
    IVFPQPreparedInsert p;
    p.cluster = ivf_nearest_centroid(ix.meta, x.data());
    p.code.resize(ix.pq.chunks);
    ivf_pq_encode(ix.pq, x.data(), p.code.data());

    p.slot = heap.allocate_slot(delta.free_list);
    try {
        heap.write_vector(p.slot, vec);
    } catch (...) {
        heap.free_slot(p.slot, delta.free_list);  // never published, so no reader can hold it
        throw;
    }
    return p;
}

uint32_t ivf_pq_publish_insert(IVFPQIndex& ix, IVFPQDelta& delta, IVFPQPreparedInsert&& p) {
    IVF_PQ_REQUIRE(p.code.size() == ix.pq.chunks && p.cluster < ix.meta.nlist, "prepared insert does not fit the index");
    IVF_PQ_REQUIRE(ix.rid_table.rid.size() == ix.assignments.cluster_id.size(),
                   "RID table and cluster assignments disagree on the vector count");
    IVF_PQ_REQUIRE(ix.rid_table.rid.size() < 0xFFFFFFFFu, "no unused vector id is left");
    const uint32_t id = uint32_t(ix.rid_table.rid.size());
    ix.assignments.cluster_id.push_back(p.cluster);
    ix.rid_table.rid.push_back(make_raw_vector_rid(p.slot, true));
    delta.codes.codes.emplace(id, std::move(p.code));
    delta.lists.pending_inserts[p.cluster].push_back(id);
    return id;
}

template<typename T>
uint32_t ivf_pq_insert(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, const T* vec) {
    IVFPQPreparedInsert prepared = ivf_pq_prepare_insert<T>(ix, heap, delta, vec);
    std::unique_lock<std::shared_mutex> lock(delta.mtx);
    return ivf_pq_publish_insert(ix, delta, std::move(prepared));
}

#define IVF_PQ_INSTANTIATE_INSERT(T)                                                                          \
    template IVFPQPreparedInsert ivf_pq_prepare_insert<T>(const IVFPQIndex&, RawVectorHeap&, IVFPQDelta&,    \
                                                          const T*);                                          \
    template uint32_t ivf_pq_insert<T>(IVFPQIndex&, RawVectorHeap&, IVFPQDelta&, const T*);
IVF_PQ_INSTANTIATE_INSERT(float)
IVF_PQ_INSTANTIATE_INSERT(uint8_t)
IVF_PQ_INSTANTIATE_INSERT(int8_t)
#undef IVF_PQ_INSTANTIATE_INSERT

}  // namespace inplace
}  // namespace diskann
