#include "bufann/ivf_pq_search.h"

#include <mkl.h>
#include <omp.h>

#include <algorithm>
#include <numeric>
#include <shared_mutex>

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
// offsets, ids in range, RIDs inside the heap) are the loader's job. The RID
// table, which inserts grow, is checked under the delta lock in the search.
void require_consistent(const IVFPQIndex& ix) {
    const size_t n_base = ivf_pq_num_base(ix);
    IVF_PQ_REQUIRE(ix.meta.nlist > 0 && ix.meta.dim > 0 && ix.meta.aligned_dim >= ix.meta.dim &&
                       ix.meta.centroids.size() == size_t(ix.meta.nlist) * ix.meta.aligned_dim &&
                       ix.meta.centroid_l2sq.size() == ix.meta.nlist,
                   "IVFMetadata is inconsistent (centroid_l2sq must be set; see set_ivf_centroid_norms)");
    IVF_PQ_REQUIRE(ix.pq.chunks > 0 && ix.pq.k > 0 && ix.pq.chunks * ix.pq.chunk_dim == ix.meta.dim &&
                       ix.pq.pivots.size() == size_t(ix.pq.chunks) * ix.pq.k * ix.pq.chunk_dim &&
                       ix.pq.codes.size() == n_base * ix.pq.chunks,
                   "PQMetadata is inconsistent with the index");
    IVF_PQ_REQUIRE(ix.lists.offsets.size() == size_t(ix.meta.nlist) + 1 && ix.lists.offsets.front() == 0 &&
                       ix.lists.ids.size() == n_base,
                   "PostingLists are inconsistent with the index");
}

void size_scratch(IVFPQSearchScratch& s, const IVFPQIndex& index) {
    const IVFMetadata& meta = index.meta;
    s.query_padded.assign(meta.aligned_dim, 0.0f);
    s.centroid_dist.resize(meta.nlist);
    s.probe_order.resize(meta.nlist);
    s.pq_table.resize(size_t(index.pq.chunks) * index.pq.k);
    s.vector.resize(meta.dim);
}

void require_search_args(const IVFPQIndex& ix, uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                         uint32_t heap_elem_size, size_t elem_bytes) {
    IVF_PQ_REQUIRE(k > 0, "k must be greater than zero");
    IVF_PQ_REQUIRE(nprobe > 0, "nprobe must be greater than zero");
    IVF_PQ_REQUIRE(rerank_m == 0 || rerank_m >= k, "rerank_m must be 0 (no re-rank) or at least k");
    require_consistent(ix);
    IVF_PQ_REQUIRE(heap_elem_size == ix.meta.dim * elem_bytes,
                   "raw-vector heap elem_size does not match dim * sizeof(T)");
}

// dist[r][c] = ||q_r||^2 + ||c||^2 - 2 q_r.c for `rows` padded queries, via
// one GEMM against the padded centroids; the padding is zero on both sides.
void centroid_distances(const IVFMetadata& meta, const float* queries_padded, uint32_t rows, float* dist) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float q_l2sq = l2sq(queries_padded + size_t(r) * meta.aligned_dim, meta.dim);
        for (uint32_t c = 0; c < meta.nlist; ++c) dist[size_t(r) * meta.nlist + c] = q_l2sq + meta.centroid_l2sq[c];
    }
    const MKL_INT m = rows, n = meta.nlist, kdim = meta.aligned_dim;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, kdim, -2.0f, queries_padded, kdim,
                meta.centroids.data(), kdim, 1.0f, dist, n);
}

// Everything after the centroid distances: probe order, PQ table, scan of
// the probed posting lists, selection, optional exact re-rank.
template<typename T>
IVFPQSearchResult search_from_centroid_dist(const IVFPQIndex& ix, const RawVectorHeap& heap,
                                            const IVFPQDelta* delta, const float* query,
                                            const float* centroid_dist, uint32_t k, uint32_t nprobe,
                                            uint32_t rerank_m, IVFPQSearchScratch& scratch) {
    const IVFMetadata& meta = ix.meta;
    const PQMetadata& pq = ix.pq;
    const uint32_t dim = meta.dim;
    size_scratch(scratch, ix);

    // Held from the first RID load through the last read_vector, so a slot a
    // concurrent delete frees is not refilled under this query's re-rank.
    RawVectorHeap::ReadGuard guard(heap);

    std::copy_n(centroid_dist, meta.nlist, scratch.centroid_dist.begin());
    select_smallest(scratch.centroid_dist, std::min(nprobe, meta.nlist), scratch.probe_order);

    for (uint32_t c = 0; c < pq.chunks; ++c) {
        for (uint32_t j = 0; j < pq.k; ++j) {
            scratch.pq_table[size_t(c) * pq.k + j] =
                sq_dist(query + c * pq.chunk_dim, pq.pivots.data() + (size_t(c) * pq.k + j) * pq.chunk_dim,
                        pq.chunk_dim);
        }
    }

    auto add_candidate = [&](uint32_t id, const uint8_t* code) {
        float d = 0.0f;
        for (uint32_t c = 0; c < pq.chunks; ++c) d += scratch.pq_table[size_t(c) * pq.k + code[c]];
        scratch.candidates.push_back(id);
        scratch.pq_dist.push_back(d);
    };
    scratch.candidates.clear();
    scratch.pq_dist.clear();
    for (uint32_t part : scratch.probe_order) {
        for (uint32_t i = ix.lists.offsets[part]; i < ix.lists.offsets[part + 1]; ++i) {
            const uint32_t id = ix.lists.ids[i];
            add_candidate(id, pq.codes.data() + size_t(id) * pq.chunks);
        }
    }

    // Inserts grow the delta and the RID table, so both are read under the
    // delta's shared lock: here for the delta, again below for the
    // shortlist's RIDs, with the selection between them unlocked so a
    // waiting writer is held up by at most a scan.
    std::shared_lock<WriterPreferringSharedMutex> delta_lock;
    if (delta != nullptr) delta_lock = std::shared_lock<WriterPreferringSharedMutex>(delta->mtx);
    IVF_PQ_REQUIRE(ix.rid_table.rid.size() >= ivf_pq_num_base(ix), "RID table does not cover every base vector");
    if (delta != nullptr) {
        // Deleted base vectors are still in the posting lists; dropped before
        // selection so they cannot crowd live vectors out of the shortlist.
        const tsl::robin_set<uint32_t>& tombstones = delta->lists.tombstones;
        if (!tombstones.empty()) {
            size_t live = 0;
            for (size_t i = 0; i < scratch.candidates.size(); ++i) {
                if (tombstones.count(scratch.candidates[i]) != 0) continue;
                scratch.candidates[live] = scratch.candidates[i];
                scratch.pq_dist[live] = scratch.pq_dist[i];
                ++live;
            }
            scratch.candidates.resize(live);
            scratch.pq_dist.resize(live);
        }
        for (uint32_t part : scratch.probe_order) {
            auto pending = delta->lists.pending_inserts.find(part);
            if (pending == delta->lists.pending_inserts.end()) continue;
            for (uint32_t id : pending->second) {
                auto code = delta->codes.codes.find(id);
                IVF_PQ_REQUIRE(code != delta->codes.codes.end() && code->second.size() == pq.chunks,
                               "inserted vector " + std::to_string(id) + " has no PQ code in the delta");
                add_candidate(id, code->second.data());
            }
        }
    }

    if (delta_lock.owns_lock()) delta_lock.unlock();

    // The active check covers a delete that landed while the lock was
    // released, and an index searched without its delta.
    select_smallest(scratch.pq_dist, rerank_m == 0 ? k : rerank_m, scratch.order);
    if (delta != nullptr) delta_lock.lock();
    scratch.shortlist.clear();
    scratch.shortlist_slot.clear();
    scratch.exact_dist.clear();
    for (uint32_t i : scratch.order) {
        const RawVectorRID rid = load_rid(ix.rid_table.rid[scratch.candidates[i]]);
        if (rid_is_active(rid)) {
            scratch.shortlist.push_back(scratch.candidates[i]);
            scratch.shortlist_slot.push_back(rid_flat_slot(rid));
            scratch.exact_dist.push_back(scratch.pq_dist[i]);
        }
    }
    if (delta_lock.owns_lock()) delta_lock.unlock();

    IVFPQSearchResult result;
    if (rerank_m == 0) {
        result.ids = scratch.shortlist;
        result.dists = scratch.exact_dist;
        return result;
    }

    scratch.raw_vector.resize(heap.layout().elem_size);
    const T* raw = reinterpret_cast<const T*>(scratch.raw_vector.data());
    for (size_t i = 0; i < scratch.shortlist.size(); ++i) {
        const uint32_t owner = heap.read_vector(scratch.shortlist_slot[i], scratch.raw_vector.data());
        IVF_PQ_REQUIRE(owner == scratch.shortlist[i], "raw-vector slot " + std::to_string(scratch.shortlist_slot[i]) +
                                                          " holds vector " + std::to_string(owner) + ", not " +
                                                          std::to_string(scratch.shortlist[i]));
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

}  // namespace

template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& ix, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch, const IVFPQDelta* delta) {
    IVF_PQ_REQUIRE(query != nullptr, "query is null");
    require_search_args(ix, k, nprobe, rerank_m, heap.layout().elem_size, sizeof(T));
    size_scratch(scratch, ix);
    std::copy_n(query, ix.meta.dim, scratch.query_padded.begin());
    // A one-row GEMM is far cheaper than MKL's thread fan-out for it: on a
    // busy 48-core host the fan-out alone cost ~3 ms per query.
    const int mkl_threads = mkl_set_num_threads_local(1);
    centroid_distances(ix.meta, scratch.query_padded.data(), 1, scratch.centroid_dist.data());
    mkl_set_num_threads_local(mkl_threads);
    return search_from_centroid_dist<T>(ix, heap, delta, query, scratch.centroid_dist.data(), k, nprobe,
                                        rerank_m, scratch);
}

template<typename T>
std::vector<IVFPQSearchResult> ivf_pq_search_batch(const IVFPQIndex& ix, const RawVectorHeap& heap,
                                                   const float* queries, size_t nq, uint32_t k,
                                                   uint32_t nprobe, uint32_t rerank_m, uint32_t gemm_rows,
                                                   const IVFPQDelta* delta) {
    IVF_PQ_REQUIRE(queries != nullptr || nq == 0, "queries is null");
    IVF_PQ_REQUIRE(gemm_rows > 0, "gemm_rows must be greater than zero");
    require_search_args(ix, k, nprobe, rerank_m, heap.layout().elem_size, sizeof(T));
    const IVFMetadata& meta = ix.meta;

    std::vector<IVFPQSearchResult> results(nq);
    std::vector<IVFPQSearchScratch> scratches(static_cast<size_t>(omp_get_max_threads()));
    std::vector<float> padded(size_t(gemm_rows) * meta.aligned_dim);
    std::vector<float> dist(size_t(gemm_rows) * meta.nlist);

    for (size_t start = 0; start < nq; start += gemm_rows) {
        const uint32_t rows = uint32_t(std::min<size_t>(gemm_rows, nq - start));
        std::fill(padded.begin(), padded.end(), 0.0f);
        for (uint32_t r = 0; r < rows; ++r) {
            std::copy_n(queries + (start + r) * meta.dim, meta.dim, padded.begin() + size_t(r) * meta.aligned_dim);
        }
        centroid_distances(meta, padded.data(), rows, dist.data());

#pragma omp parallel for schedule(dynamic, 8)
        for (int64_t r = 0; r < int64_t(rows); ++r) {
            results[start + r] = search_from_centroid_dist<T>(
                ix, heap, delta, queries + (start + r) * meta.dim, dist.data() + size_t(r) * meta.nlist, k,
                nprobe, rerank_m, scratches[size_t(omp_get_thread_num())]);
        }
    }
    return results;
}

template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m) {
    IVFPQSearchScratch scratch;
    return ivf_pq_search<T>(index, heap, query, k, nprobe, rerank_m, scratch);
}

#define IVF_PQ_INSTANTIATE_SEARCH(T)                                                                        \
    template IVFPQSearchResult ivf_pq_search<T>(const IVFPQIndex&, const RawVectorHeap&, const float*,     \
                                                uint32_t, uint32_t, uint32_t, IVFPQSearchScratch&,         \
                                                const IVFPQDelta*);                                        \
    template IVFPQSearchResult ivf_pq_search<T>(const IVFPQIndex&, const RawVectorHeap&, const float*,     \
                                                uint32_t, uint32_t, uint32_t);                             \
    template std::vector<IVFPQSearchResult> ivf_pq_search_batch<T>(const IVFPQIndex&, const RawVectorHeap&, \
                                                                   const float*, size_t, uint32_t, uint32_t, \
                                                                   uint32_t, uint32_t, const IVFPQDelta*);
IVF_PQ_INSTANTIATE_SEARCH(float)
IVF_PQ_INSTANTIATE_SEARCH(uint8_t)
IVF_PQ_INSTANTIATE_SEARCH(int8_t)
#undef IVF_PQ_INSTANTIATE_SEARCH

}  // namespace inplace
}  // namespace diskann
