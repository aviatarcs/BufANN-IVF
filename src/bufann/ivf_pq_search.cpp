#include "bufann/ivf_pq_search.h"

#include <mkl.h>

#include <algorithm>
#include <numeric>

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

namespace {

float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

float l2sq(const float* a, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += a[d] * a[d];
    return s;
}

// Leaves the indices of the `m` smallest values in order[0, m), ascending.
void select_smallest(const std::vector<float>& values, uint32_t m, std::vector<uint32_t>& order) {
    order.resize(values.size());
    std::iota(order.begin(), order.end(), 0u);
    m = std::min<uint32_t>(m, uint32_t(order.size()));
    std::partial_sort(order.begin(), order.begin() + m, order.end(),
                      [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    order.resize(m);
}

// O(1) shape checks, run on every query; the structural invariants (sorted
// offsets, ids in range, RIDs inside the heap) are the loader's job.
void require_consistent(const IVFPQIndex& ix) {
    const size_t n = ix.assignments.cluster_id.size();
    IVF_PQ_REQUIRE(ix.meta.nlist > 0 && ix.meta.dim > 0 && ix.meta.aligned_dim >= ix.meta.dim &&
                       ix.meta.centroids.size() == size_t(ix.meta.nlist) * ix.meta.aligned_dim &&
                       ix.meta.centroid_l2sq.size() == ix.meta.nlist,
                   "IVFMetadata is inconsistent (centroid_l2sq must be set; see set_ivf_centroid_norms)");
    IVF_PQ_REQUIRE(ix.pq.chunks > 0 && ix.pq.k > 0 && ix.pq.chunks * ix.pq.chunk_dim == ix.meta.dim &&
                       ix.pq.pivots.size() == size_t(ix.pq.chunks) * ix.pq.k * ix.pq.chunk_dim &&
                       ix.pq.codes.size() == n * ix.pq.chunks,
                   "PQMetadata is inconsistent with the index");
    IVF_PQ_REQUIRE(ix.lists.offsets.size() == size_t(ix.meta.nlist) + 1 && ix.lists.offsets.front() == 0 &&
                       ix.lists.offsets.back() == n && ix.lists.ids.size() == n,
                   "PostingLists are inconsistent with the index");
    IVF_PQ_REQUIRE(ix.rid_table.rid.size() == n, "RID table does not cover every vector");
}

void size_scratch(IVFPQSearchScratch& s, const IVFPQIndex& index) {
    const IVFMetadata& meta = index.meta;
    s.query_padded.assign(meta.aligned_dim, 0.0f);
    s.centroid_dist.resize(meta.nlist);
    s.probe_order.resize(meta.nlist);
    s.pq_table.resize(size_t(index.pq.chunks) * index.pq.k);
    s.vector.resize(meta.dim);
}

}  // namespace

template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& ix, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch) {
    IVF_PQ_REQUIRE(query != nullptr, "query is null");
    IVF_PQ_REQUIRE(k > 0, "k must be greater than zero");
    IVF_PQ_REQUIRE(nprobe > 0, "nprobe must be greater than zero");
    IVF_PQ_REQUIRE(rerank_m == 0 || rerank_m >= k, "rerank_m must be 0 (no re-rank) or at least k");
    require_consistent(ix);
    size_scratch(scratch, ix);
    const IVFMetadata& meta = ix.meta;
    const PQMetadata& pq = ix.pq;
    const uint32_t dim = meta.dim;
    IVF_PQ_REQUIRE(heap.layout().elem_size == dim * sizeof(T),
                   "raw-vector heap elem_size does not match dim * sizeof(T)");

    // dist[c] = ||q||^2 + ||c||^2 - 2 q.c over the padded dimension; the
    // padding is zero on both sides, so it adds nothing.
    std::copy_n(query, dim, scratch.query_padded.begin());
    const float q_l2sq = l2sq(query, dim);
    for (uint32_t c = 0; c < meta.nlist; ++c) {
        scratch.centroid_dist[c] = q_l2sq + meta.centroid_l2sq[c];
    }
    const MKL_INT nlist = meta.nlist, aligned_dim = meta.aligned_dim;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nlist, aligned_dim, -2.0f, scratch.query_padded.data(),
                aligned_dim, meta.centroids.data(), aligned_dim, 1.0f, scratch.centroid_dist.data(), nlist);
    select_smallest(scratch.centroid_dist, std::min(nprobe, meta.nlist), scratch.probe_order);

    for (uint32_t c = 0; c < pq.chunks; ++c) {
        for (uint32_t j = 0; j < pq.k; ++j) {
            scratch.pq_table[size_t(c) * pq.k + j] =
                sq_dist(query + c * pq.chunk_dim, pq.pivots.data() + (size_t(c) * pq.k + j) * pq.chunk_dim,
                        pq.chunk_dim);
        }
    }

    scratch.candidates.clear();
    scratch.pq_dist.clear();
    for (uint32_t part : scratch.probe_order) {
        for (uint32_t i = ix.lists.offsets[part]; i < ix.lists.offsets[part + 1]; ++i) {
            const uint32_t id = ix.lists.ids[i];
            const uint8_t* code = pq.codes.data() + size_t(id) * pq.chunks;
            float d = 0.0f;
            for (uint32_t c = 0; c < pq.chunks; ++c) d += scratch.pq_table[size_t(c) * pq.k + code[c]];
            scratch.candidates.push_back(id);
            scratch.pq_dist.push_back(d);
        }
    }

    // Inactive RIDs (deleted vectors) are dropped here rather than in the
    // scan, which would cost a random RID-table read per candidate.
    select_smallest(scratch.pq_dist, rerank_m == 0 ? k : rerank_m, scratch.order);
    scratch.shortlist.clear();
    scratch.exact_dist.clear();
    for (uint32_t i : scratch.order) {
        if (rid_is_active(ix.rid_table.rid[scratch.candidates[i]])) {
            scratch.shortlist.push_back(scratch.candidates[i]);
            scratch.exact_dist.push_back(scratch.pq_dist[i]);
        }
    }

    IVFPQSearchResult result;
    if (rerank_m == 0) {
        result.ids = scratch.shortlist;
        result.dists = scratch.exact_dist;
        return result;
    }

    scratch.raw_vector.resize(heap.layout().elem_size);
    const T* raw = reinterpret_cast<const T*>(scratch.raw_vector.data());
    for (size_t i = 0; i < scratch.shortlist.size(); ++i) {
        heap.read_vector(rid_flat_slot(ix.rid_table.rid[scratch.shortlist[i]]), scratch.raw_vector.data());
        for (uint32_t d = 0; d < dim; ++d) scratch.vector[d] = float(raw[d]);
        scratch.exact_dist[i] = sq_dist(query, scratch.vector.data(), dim);
    }
    select_smallest(scratch.exact_dist, k, scratch.order);
    for (uint32_t i : scratch.order) {
        result.ids.push_back(scratch.shortlist[i]);
        result.dists.push_back(scratch.exact_dist[i]);
    }
    return result;
}

template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m) {
    IVFPQSearchScratch scratch;
    return ivf_pq_search<T>(index, heap, query, k, nprobe, rerank_m, scratch);
}

#define IVF_PQ_INSTANTIATE_SEARCH(T)                                                                        \
    template IVFPQSearchResult ivf_pq_search<T>(const IVFPQIndex&, const RawVectorHeap&, const float*,     \
                                                uint32_t, uint32_t, uint32_t, IVFPQSearchScratch&);        \
    template IVFPQSearchResult ivf_pq_search<T>(const IVFPQIndex&, const RawVectorHeap&, const float*,     \
                                                uint32_t, uint32_t, uint32_t);
IVF_PQ_INSTANTIATE_SEARCH(float)
IVF_PQ_INSTANTIATE_SEARCH(uint8_t)
IVF_PQ_INSTANTIATE_SEARCH(int8_t)
#undef IVF_PQ_INSTANTIATE_SEARCH

}  // namespace inplace
}  // namespace diskann
