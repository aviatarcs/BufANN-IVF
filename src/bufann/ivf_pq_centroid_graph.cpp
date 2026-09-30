#include "bufann/ivf_pq_centroid_graph.h"

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

float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

const float* centroid(const IVFMetadata& meta, uint32_t c) {
    return meta.centroids.data() + size_t(c) * meta.aligned_dim;
}

void start_search(IVFCentroidGraphScratch& s, uint32_t nlist) {
    if (s.visited_epoch.size() != nlist || ++s.epoch == 0) {
        s.visited_epoch.assign(nlist, 0);
        s.epoch = 1;
    }
    s.beam.clear();
}

// Best-first search from `entry` with a beam of L; `neighbors_of(c, out)`
// copies c's out-edges into out. Every expanded centroid is appended to
// `expanded` when it is non-null (the build prunes over them).
template<typename NeighborsOf>
void beam_search(const IVFMetadata& meta, uint32_t entry, const float* query, uint32_t L,
                 IVFCentroidGraphScratch& s, NeighborsOf&& neighbors_of, std::vector<Candidate>* expanded) {
    start_search(s, meta.nlist);
    s.visited_epoch[entry] = s.epoch;
    s.beam.push_back({sq_dist(query, centroid(meta, entry), meta.dim), entry, false});
    std::vector<uint32_t> nbrs;
    size_t next = 0;  // the first unexpanded beam position
    while (next < s.beam.size()) {
        Candidate& cur = s.beam[next];
        cur.expanded = true;
        if (expanded != nullptr) expanded->push_back(cur);
        neighbors_of(cur.id, nbrs);
        size_t lowest_insert = s.beam.size();
        for (uint32_t nb : nbrs) {
            if (s.visited_epoch[nb] == s.epoch) continue;
            s.visited_epoch[nb] = s.epoch;
            const float d = sq_dist(query, centroid(meta, nb), meta.dim);
            if (s.beam.size() == L && d >= s.beam.back().dist) continue;
            auto at = std::upper_bound(s.beam.begin(), s.beam.end(), d,
                                       [](float v, const Candidate& c) { return v < c.dist; });
            lowest_insert = std::min(lowest_insert, size_t(at - s.beam.begin()));
            s.beam.insert(at, {d, nb, false});
            if (s.beam.size() > L) s.beam.pop_back();
        }
        // Resume at the nearest unexpanded candidate, which an insert may
        // have placed before the current one.
        next = std::min(lowest_insert, next + 1);
        while (next < s.beam.size() && s.beam[next].expanded) ++next;
    }
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
                               std::vector<uint32_t>& out) {
    IVF_PQ_REQUIRE(graph.nlist == meta.nlist && graph.degree > 0 && graph.entry < graph.nlist &&
                       graph.neighbors.size() == size_t(graph.nlist) * graph.degree,
                   "centroid graph does not match the index's nlist");
    IVF_PQ_REQUIRE(n > 0 && L >= n, "centroid graph search needs 0 < n <= L");
    const uint32_t* all = graph.neighbors.data();
    auto neighbors_of = [&](uint32_t c, std::vector<uint32_t>& nbrs) {
        const uint32_t* list = all + size_t(c) * graph.degree;
        nbrs.assign(list, std::find(list, list + graph.degree, IVF_CENTROID_GRAPH_NONE));
    };
    beam_search(meta, graph.entry, query, L, scratch, neighbors_of, nullptr);
    out.clear();
    for (size_t i = 0; i < scratch.beam.size() && out.size() < n; ++i) out.push_back(scratch.beam[i].id);
}

}  // namespace inplace
}  // namespace diskann
