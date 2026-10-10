#include "bufann/ivf_pq_centroid_graph.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <random>

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

namespace {

using Candidate = IVFCentroidGraphScratch::Candidate;

#ifdef __AVX512F__
__mmask16 tail_mask(uint32_t dim, uint32_t d) { return dim - d >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (dim - d)) - 1); }

float sq_dist(const float* a, const float* b, uint32_t dim) {
    __m512 acc = _mm512_setzero_ps();
    for (uint32_t d = 0; d < dim; d += 16) {
        const __mmask16 m = tail_mask(dim, d);
        const __m512 x = _mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + d), _mm512_maskz_loadu_ps(m, b + d));
        acc = _mm512_fmadd_ps(x, x, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

// FAISS's fvec_L2sqr_batch_4: four rows against one query with four
// independent accumulators, so one row's loads overlap the others' adds.
// Each distance is summed exactly as sq_dist sums it.
void sq_dist4(const float* q, const float* const* rows, uint32_t dim, float* out) {
    __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (uint32_t d = 0; d < dim; d += 16) {
        const __mmask16 m = tail_mask(dim, d);
        const __m512 x = _mm512_maskz_loadu_ps(m, q + d);
        for (int r = 0; r < 4; ++r) {
            const __m512 t = _mm512_sub_ps(x, _mm512_maskz_loadu_ps(m, rows[r] + d));
            acc[r] = _mm512_fmadd_ps(t, t, acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = _mm512_reduce_add_ps(acc[r]);
}
#else
float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

void sq_dist4(const float* q, const float* const* rows, uint32_t dim, float* out) {
    for (int r = 0; r < 4; ++r) out[r] = sq_dist(q, rows[r], dim);
}
#endif

const float* centroid(const IVFMetadata& meta, uint32_t c) {
    return meta.centroids.data() + size_t(c) * meta.aligned_dim;
}

void start_search(IVFCentroidGraphScratch& s, uint32_t nlist) {
    if (s.visited_epoch.size() != nlist || ++s.epoch == 0) {
        s.visited_epoch.assign(nlist, 0);
        s.epoch = 1;
    }
    s.frontier.clear();
    s.best.clear();
}

// Ties on distance go to the lower centroid id, so results do not depend on
// heap order (a duplicated centroid always loses to its first copy).
bool nearer(const Candidate& a, const Candidate& b) { return a.dist < b.dist || (a.dist == b.dist && a.id < b.id); }
bool farther(const Candidate& a, const Candidate& b) { return nearer(b, a); }

// Best-first search from `entry` keeping the L nearest centroids visited:
// expand the nearest unexpanded one, stop once it is farther than the L-th
// nearest found. The same expansions as a sorted beam of L, but an insert
// costs O(log L) rather than shifting up to L entries. `neighbors_of(c,
// out)` copies c's out-edges into out. Every expanded centroid is appended
// to `expanded` when it is non-null (the build prunes over them). Leaves
// s.best sorted ascending by distance.
template<typename NeighborsOf>
void beam_search(const IVFMetadata& meta, uint32_t entry, const float* query, uint32_t L,
                 IVFCentroidGraphScratch& s, NeighborsOf&& neighbors_of, std::vector<Candidate>* expanded) {
    start_search(s, meta.nlist);
    s.visited_epoch[entry] = s.epoch;
    const Candidate first{sq_dist(query, centroid(meta, entry), meta.dim), entry, false};
    s.frontier.push_back(first);
    s.best.push_back(first);
    while (!s.frontier.empty()) {
        std::pop_heap(s.frontier.begin(), s.frontier.end(), farther);
        const Candidate cur = s.frontier.back();
        s.frontier.pop_back();
        if (s.best.size() == L && cur.dist > s.best.front().dist) break;
        if (expanded != nullptr) expanded->push_back({cur.dist, cur.id, true});
        neighbors_of(cur.id, s.nbrs);
        // Keep the unvisited neighbours, marking them, and ask for their
        // rows before computing any distance: the centroids are far larger
        // than the caches.
        const size_t row_bytes = size_t(meta.dim) * sizeof(float);
        size_t fresh = 0;
        for (uint32_t nb : s.nbrs) {
            if (s.visited_epoch[nb] == s.epoch) continue;
            s.visited_epoch[nb] = s.epoch;
            s.nbrs[fresh++] = nb;
            const char* row = reinterpret_cast<const char*>(centroid(meta, nb));
            for (size_t off = 0; off < row_bytes; off += 64) __builtin_prefetch(row + off);
        }
        s.nbrs.resize(fresh);
        s.nbr_dist.resize(fresh);
        size_t i = 0;
        for (; i + 4 <= fresh; i += 4) {
            const float* rows[4] = {centroid(meta, s.nbrs[i]), centroid(meta, s.nbrs[i + 1]),
                                    centroid(meta, s.nbrs[i + 2]), centroid(meta, s.nbrs[i + 3])};
            sq_dist4(query, rows, meta.dim, s.nbr_dist.data() + i);
        }
        for (; i < fresh; ++i) s.nbr_dist[i] = sq_dist(query, centroid(meta, s.nbrs[i]), meta.dim);
        for (i = 0; i < fresh; ++i) {
            const uint32_t nb = s.nbrs[i];
            const float d = s.nbr_dist[i];
            if (s.best.size() == L && d >= s.best.front().dist) continue;
            s.best.push_back({d, nb, false});
            std::push_heap(s.best.begin(), s.best.end(), nearer);
            if (s.best.size() > L) {
                std::pop_heap(s.best.begin(), s.best.end(), nearer);
                s.best.pop_back();
            }
            s.frontier.push_back({d, nb, false});
            std::push_heap(s.frontier.begin(), s.frontier.end(), farther);
        }
    }
    std::sort_heap(s.best.begin(), s.best.end(), nearer);
}

// Robust prune: keeps up to R of `pool` nearest p, skipping any candidate
// c' that a kept c covers (alpha * d(c, c') <= d(p, c')).
std::vector<uint32_t> robust_prune(const IVFMetadata& meta, uint32_t p, std::vector<Candidate>& pool, float alpha,
                                   uint32_t R) {
    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
    });
    pool.erase(std::unique(pool.begin(), pool.end(),
                           [](const Candidate& a, const Candidate& b) { return a.id == b.id; }),
               pool.end());
    std::vector<uint32_t> kept;
    std::vector<bool> covered(pool.size(), false);
    for (size_t i = 0; i < pool.size() && kept.size() < R; ++i) {
        if (covered[i] || pool[i].id == p) continue;
        kept.push_back(pool[i].id);
        const float* c = centroid(meta, pool[i].id);
        for (size_t j = i + 1; j < pool.size(); ++j) {
            if (!covered[j] && alpha * sq_dist(c, centroid(meta, pool[j].id), meta.dim) <= pool[j].dist) {
                covered[j] = true;
            }
        }
    }
    return kept;
}

}  // namespace

IVFCentroidGraph build_ivf_centroid_graph(const IVFMetadata& meta, const IVFCentroidGraphParams& params) {
    IVF_PQ_REQUIRE(meta.nlist > 0 && meta.dim > 0 && meta.aligned_dim >= meta.dim &&
                       meta.centroids.size() == size_t(meta.nlist) * meta.aligned_dim,
                   "IVFMetadata is inconsistent");
    IVF_PQ_REQUIRE(params.degree > 0 && params.build_L >= params.degree && params.alpha >= 1.0f,
                   "centroid graph needs degree > 0, build_L >= degree and alpha >= 1");
    const uint32_t nlist = meta.nlist, R = params.degree;

    IVFCentroidGraph g;
    g.nlist = nlist;
    g.degree = R;
    {
        std::vector<double> mean(meta.dim, 0.0);
        for (uint32_t c = 0; c < nlist; ++c) {
            for (uint32_t d = 0; d < meta.dim; ++d) mean[d] += centroid(meta, c)[d];
        }
        std::vector<float> meanf(meta.dim);
        for (uint32_t d = 0; d < meta.dim; ++d) meanf[d] = float(mean[d] / nlist);
        float best = sq_dist(meanf.data(), centroid(meta, 0), meta.dim);
        for (uint32_t c = 1; c < nlist; ++c) {
            const float d = sq_dist(meanf.data(), centroid(meta, c), meta.dim);
            if (d < best) best = d, g.entry = c;
        }
    }

    // Built as adjacency vectors under per-node locks, flattened at the end.
    std::vector<std::vector<uint32_t>> adj(nlist);
    std::vector<std::mutex> locks(nlist);
    auto neighbors_of = [&](uint32_t c, std::vector<uint32_t>& out) {
        std::lock_guard<std::mutex> lk(locks[c]);
        out = adj[c];
    };
    std::vector<uint32_t> order(nlist);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), std::mt19937(0));

    for (const float alpha : {1.0f, params.alpha}) {
#pragma omp parallel
        {
            IVFCentroidGraphScratch s;
            std::vector<Candidate> pool;
#pragma omp for schedule(dynamic, 64)
            for (int64_t i = 0; i < int64_t(nlist); ++i) {
                const uint32_t p = order[size_t(i)];
                const float* pc = centroid(meta, p);
                pool.clear();
                beam_search(meta, g.entry, pc, params.build_L, s, neighbors_of, &pool);
                std::vector<uint32_t> current;
                neighbors_of(p, current);
                for (uint32_t nb : current) pool.push_back({sq_dist(pc, centroid(meta, nb), meta.dim), nb, true});
                const std::vector<uint32_t> kept = robust_prune(meta, p, pool, alpha, R);
                {
                    std::lock_guard<std::mutex> lk(locks[p]);
                    adj[p] = kept;
                }
                // Back edges, re-pruning a list that would exceed R.
                for (uint32_t nb : kept) {
                    std::lock_guard<std::mutex> lk(locks[nb]);
                    std::vector<uint32_t>& list = adj[nb];
                    if (std::find(list.begin(), list.end(), p) != list.end()) continue;
                    if (list.size() < R) {
                        list.push_back(p);
                        continue;
                    }
                    std::vector<Candidate> back;
                    const float* nc = centroid(meta, nb);
                    for (uint32_t x : list) back.push_back({sq_dist(nc, centroid(meta, x), meta.dim), x, true});
                    back.push_back({sq_dist(nc, pc, meta.dim), p, true});
                    list = robust_prune(meta, nb, back, alpha, R);
                }
            }
        }
    }

    g.neighbors.assign(size_t(nlist) * R, IVF_CENTROID_GRAPH_NONE);
    for (uint32_t c = 0; c < nlist; ++c) std::copy(adj[c].begin(), adj[c].end(), g.neighbors.begin() + size_t(c) * R);
    return g;
}

void search_ivf_centroid_graph(const IVFMetadata& meta, const IVFCentroidGraph& graph, const float* query,
                               uint32_t L, uint32_t n, IVFCentroidGraphScratch& scratch,
                               std::vector<uint32_t>& out, uint32_t start) {
    IVF_PQ_REQUIRE(graph.nlist == meta.nlist && graph.degree > 0 && graph.entry < graph.nlist &&
                       graph.neighbors.size() == size_t(graph.nlist) * graph.degree,
                   "centroid graph does not match the index's nlist");
    IVF_PQ_REQUIRE(n > 0 && L >= n, "centroid graph search needs 0 < n <= L");
    IVF_PQ_REQUIRE(start == IVF_CENTROID_GRAPH_NONE || start < graph.nlist, "search start is not a centroid");
    const uint32_t* all = graph.neighbors.data();
    auto neighbors_of = [&](uint32_t c, std::vector<uint32_t>& nbrs) {
        const uint32_t* list = all + size_t(c) * graph.degree;
        nbrs.assign(list, std::find(list, list + graph.degree, IVF_CENTROID_GRAPH_NONE));
    };
    beam_search(meta, start == IVF_CENTROID_GRAPH_NONE ? graph.entry : start, query, L, scratch, neighbors_of,
                nullptr);
    out.clear();
    for (size_t i = 0; i < scratch.best.size() && out.size() < n; ++i) out.push_back(scratch.best[i].id);
}

}  // namespace inplace
}  // namespace diskann
