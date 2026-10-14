// Tests for the proximity graph over the IVF centroids: its structure
// (degree bound, no self or duplicate edges, every centroid reachable from
// the entry, the entry nearest the mean), a beam as wide as nlist returning
// the brute-force nearest centroids, recall against brute force at the
// beams the search uses, scratch reuse across graphs and across the visit
// epoch's wrap, and the guards. Centroids are synthetic blobs in a
// dimension that is not a multiple of 8, so the aligned_dim stride is used.

#include "bufann/ivf_pq_centroid_graph.h"
#include "ivf_pq_test_util.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t NLIST = 4096;
const uint32_t DIM = 20;
const uint32_t ALIGNED_DIM = 24;
const uint32_t BLOBS = 64;
const uint32_t NQ = 300;

IVFMetadata make_meta(uint32_t nlist, uint32_t seed) {
    IVFMetadata meta;
    meta.nlist = nlist;
    meta.dim = DIM;
    meta.aligned_dim = ALIGNED_DIM;
    const std::vector<float> points = draw_blobs<float>(nlist, seed, DIM, BLOBS);
    meta.centroids.assign(size_t(nlist) * ALIGNED_DIM, 0.0f);
    for (uint32_t c = 0; c < nlist; ++c) {
        std::copy_n(points.begin() + size_t(c) * DIM, DIM, meta.centroids.begin() + size_t(c) * ALIGNED_DIM);
    }
    return meta;
}

// Brute-force distances from `query` to every centroid.
std::vector<float> centroid_dists(const IVFMetadata& meta, const float* query) {
    std::vector<float> d(meta.nlist);
    for (uint32_t c = 0; c < meta.nlist; ++c) d[c] = sq_dist(query, meta.centroids.data() + size_t(c) * meta.aligned_dim, DIM);
    return d;
}

bool test_structure(const IVFMetadata& meta, const IVFCentroidGraph& g) {
    TestCase t("graph structure: bounded degree, no self or duplicate edges, all reachable, entry nearest the mean");
    t.check(g.nlist == meta.nlist && g.degree == IVFCentroidGraphParams{}.degree &&
                g.neighbors.size() == size_t(g.nlist) * g.degree,
            "graph shape does not match nlist x degree");
    size_t edges = 0;
    for (uint32_t c = 0; c < g.nlist; ++c) {
        const uint32_t* list = g.neighbors.data() + size_t(c) * g.degree;
        const uint32_t* end = std::find(list, list + g.degree, IVF_CENTROID_GRAPH_NONE);
        std::vector<uint32_t> nbrs(list, end);
        edges += nbrs.size();
        bool ok = std::all_of(end, list + g.degree, [](uint32_t v) { return v == IVF_CENTROID_GRAPH_NONE; }) &&
                  std::all_of(nbrs.begin(), nbrs.end(), [&](uint32_t v) { return v < g.nlist && v != c; });
        std::sort(nbrs.begin(), nbrs.end());
        ok = ok && std::adjacent_find(nbrs.begin(), nbrs.end()) == nbrs.end();
        if (!t.check(ok, "centroid " + std::to_string(c) + " has a bad neighbour list")) return t.done();
    }
    std::vector<uint8_t> seen(g.nlist, 0);
    std::vector<uint32_t> frontier{g.entry};
    seen[g.entry] = 1;
    size_t reached = 1;
    while (!frontier.empty()) {
        const uint32_t c = frontier.back();
        frontier.pop_back();
        for (uint32_t i = 0; i < g.degree; ++i) {
            const uint32_t v = g.neighbors[size_t(c) * g.degree + i];
            if (v == IVF_CENTROID_GRAPH_NONE) break;
            if (!seen[v]) seen[v] = 1, ++reached, frontier.push_back(v);
        }
    }
    t.check(reached == g.nlist, std::to_string(g.nlist - reached) + " centroids unreachable from the entry");

    std::vector<double> mean(DIM, 0.0);
    for (uint32_t c = 0; c < meta.nlist; ++c) {
        for (uint32_t d = 0; d < DIM; ++d) mean[d] += meta.centroids[size_t(c) * ALIGNED_DIM + d];
    }
    std::vector<float> meanf(DIM);
    for (uint32_t d = 0; d < DIM; ++d) meanf[d] = float(mean[d] / meta.nlist);
    const std::vector<float> to_mean = centroid_dists(meta, meanf.data());
    t.check(close(to_mean[g.entry], *std::min_element(to_mean.begin(), to_mean.end())),
            "entry is not the centroid nearest the mean");
    std::cout << "  average out-degree " << double(edges) / g.nlist << std::endl;
    return t.done();
}

bool test_full_beam_is_exact(const IVFMetadata& meta, const IVFCentroidGraph& g, const std::vector<float>& queries) {
    TestCase t("a beam of nlist returns the brute-force nearest centroids");
    IVFCentroidGraphScratch scratch;
    std::vector<uint32_t> got;
    for (uint32_t q = 0; q < NQ; ++q) {
        const float* query = queries.data() + size_t(q) * DIM;
        const std::vector<float> d = centroid_dists(meta, query);
        for (uint32_t n : {1u, 64u, NLIST}) {
            search_ivf_centroid_graph(meta, g, query, NLIST, n, scratch, got);
            const std::vector<uint32_t> want = top_k(d, n);
            bool ok = got.size() == n;
            for (uint32_t i = 0; ok && i < n; ++i) ok = close(d[got[i]], d[want[i]]);
            if (!t.check(ok, "query " + std::to_string(q) + " n " + std::to_string(n) + " is not the exact top-n")) {
                return t.done();
            }
        }
    }
    return t.done();
}

// The search against a beam search written here the plain way: a beam kept
// sorted, extended by inserts, expanded at its nearest unexpanded entry
// until every entry is expanded. Same graph, entry and L, so the results
// must be the same centroids at the same distances, position by position.
bool test_matches_sorted_beam(const IVFMetadata& meta, const IVFCentroidGraph& g, const std::vector<float>& queries) {
    TestCase t("search returns what a plain sorted-beam search returns");
    struct C {
        float d;
        uint32_t id;
        bool expanded;
    };
    auto reference = [&](const float* q, uint32_t L, uint32_t n) {
        auto dist = [&](uint32_t c) { return sq_dist(q, meta.centroids.data() + size_t(c) * meta.aligned_dim, DIM); };
        std::vector<C> beam{{dist(g.entry), g.entry, false}};
        std::vector<uint8_t> seen(meta.nlist, 0);
        seen[g.entry] = 1;
        for (;;) {
            auto next = std::find_if(beam.begin(), beam.end(), [](const C& c) { return !c.expanded; });
            if (next == beam.end()) break;
            next->expanded = true;
            const uint32_t id = next->id;
            for (uint32_t i = 0; i < g.degree; ++i) {
                const uint32_t nb = g.neighbors[size_t(id) * g.degree + i];
                if (nb == IVF_CENTROID_GRAPH_NONE) break;
                if (seen[nb]) continue;
                seen[nb] = 1;
                C c{dist(nb), nb, false};
                beam.insert(std::upper_bound(beam.begin(), beam.end(), c, [](const C& a, const C& b) { return a.d < b.d; }), c);
                if (beam.size() > L) beam.pop_back();
            }
        }
        std::vector<float> d;
        for (size_t i = 0; i < beam.size() && d.size() < n; ++i) d.push_back(beam[i].d);
        return d;
    };
    IVFCentroidGraphScratch scratch;
    std::vector<uint32_t> got;
    size_t compared = 0;
    for (uint32_t L : {4u, 16u, 64u, 300u}) {
        for (uint32_t q = 0; q < NQ; ++q) {
            const float* query = queries.data() + size_t(q) * DIM;
            const uint32_t n = std::max(1u, L / 2);
            const std::vector<float> want = reference(query, L, n);
            search_ivf_centroid_graph(meta, g, query, L, n, scratch, got);
            const std::vector<float> d = centroid_dists(meta, query);
            bool ok = got.size() == want.size();
            for (size_t i = 0; ok && i < got.size(); ++i) ok = close(d[got[i]], want[i]);
            ++compared;
            if (!t.check(ok, "query " + std::to_string(q) + " L " + std::to_string(L) + " differs from the sorted beam")) {
                return t.done();
            }
        }
    }
    std::cout << "  compared " << compared << " searches" << std::endl;
    return t.done();
}

// Recall of the n returned centroids against the brute-force top n, over
// the beams the search is used with (a few times nprobe).
bool test_recall(const IVFMetadata& meta, const IVFCentroidGraph& g, const std::vector<float>& queries) {
    TestCase t("recall against brute force at beams of 2 and 4 x n");
    IVFCentroidGraphScratch scratch;
    std::vector<uint32_t> got;
    for (uint32_t n : {8u, 64u}) {
        for (uint32_t factor : {2u, 4u}) {
            size_t hits = 0;
            for (uint32_t q = 0; q < NQ; ++q) {
                const float* query = queries.data() + size_t(q) * DIM;
                const std::vector<float> d = centroid_dists(meta, query);
                search_ivf_centroid_graph(meta, g, query, factor * n, n, scratch, got);
                // A returned centroid counts if it is no farther than the
                // true n-th nearest (so ties at the boundary count).
                const float nth = d[top_k(d, n).back()];
                for (uint32_t c : got) hits += d[c] <= nth || close(d[c], nth);
            }
            const double recall = double(hits) / (double(NQ) * n);
            std::cout << "  n " << n << " beam " << factor * n << ": recall " << recall << std::endl;
            t.check(recall >= (factor == 4 ? 0.99 : 0.95),
                    "recall " + std::to_string(recall) + " at n " + std::to_string(n) + " beam " +
                        std::to_string(factor * n));
        }
    }
    return t.done();
}

bool test_scratch_reuse(const IVFMetadata& meta, const IVFCentroidGraph& g, const std::vector<float>& queries) {
    TestCase t("one scratch across graphs of different nlist and across the visit epoch's wrap");
    const IVFMetadata small_meta = make_meta(64, 5);
    const IVFCentroidGraph small = build_ivf_centroid_graph(small_meta);
    const float* query = queries.data();
    std::vector<uint32_t> fresh_big, fresh_small, got;
    {
        IVFCentroidGraphScratch s1, s2;
        search_ivf_centroid_graph(meta, g, query, 256, 64, s1, fresh_big);
        search_ivf_centroid_graph(small_meta, small, query, 16, 8, s2, fresh_small);
    }
    IVFCentroidGraphScratch scratch;
    search_ivf_centroid_graph(meta, g, query, 256, 64, scratch, got);
    t.check(got == fresh_big, "first search differs from a fresh scratch");
    search_ivf_centroid_graph(small_meta, small, query, 16, 8, scratch, got);
    t.check(got == fresh_small, "search of a smaller graph after a larger one differs from a fresh scratch");
    search_ivf_centroid_graph(meta, g, query, 256, 64, scratch, got);
    t.check(got == fresh_big, "search of the larger graph again differs from a fresh scratch");

    // Epoch 0 must never be current: every centroid's mark starts at 0, and
    // after a wrap the marks of epoch 1 are stale. A query in another blob
    // visits centroids the previous searches did not, which a stale or
    // zero mark would skip.
    const float* other = nullptr;
    for (uint32_t q = 1; q < NQ && other == nullptr; ++q) {
        if (sq_dist(query, queries.data() + size_t(q) * DIM, DIM) > 10000.0f) other = queries.data() + size_t(q) * DIM;
    }
    if (!t.check(other != nullptr, "no query far from the first")) return t.done();
    std::vector<uint32_t> fresh_other;
    {
        IVFCentroidGraphScratch s;
        search_ivf_centroid_graph(meta, g, other, 256, 64, s, fresh_other);
    }
    scratch.epoch = std::numeric_limits<uint32_t>::max();
    search_ivf_centroid_graph(meta, g, other, 256, 64, scratch, got);
    t.check(got == fresh_other, "search at the epoch wrap differs from a fresh scratch");
    search_ivf_centroid_graph(meta, g, query, 256, 64, scratch, got);
    search_ivf_centroid_graph(meta, g, other, 256, 64, scratch, got);
    t.check(got == fresh_other, "searches after the epoch wrap differ from a fresh scratch");
    return t.done();
}

bool test_single_centroid() {
    TestCase t("a single centroid: no edges, and every search returns it");
    const IVFMetadata one = make_meta(1, 6);
    const IVFCentroidGraph g = build_ivf_centroid_graph(one);
    IVFCentroidGraphScratch scratch;
    std::vector<uint32_t> got;
    search_ivf_centroid_graph(one, g, one.centroids.data(), 4, 1, scratch, got);
    t.check(g.entry == 0 && g.neighbors[0] == IVF_CENTROID_GRAPH_NONE && got == std::vector<uint32_t>{0},
            "one-centroid graph or search is wrong");
    return t.done();
}

bool test_guards(const IVFMetadata& meta, const IVFCentroidGraph& g, const std::vector<float>& queries) {
    TestCase t("build and search guards");
    auto build_with = [&](uint32_t degree, uint32_t build_L, float alpha) {
        IVFCentroidGraphParams p;
        p.degree = degree, p.build_L = build_L, p.alpha = alpha;
        build_ivf_centroid_graph(meta, p);
    };
    t.expect_throw("degree 0", [&] { build_with(0, 100, 1.2f); });
    t.expect_throw("build_L < degree", [&] { build_with(32, 31, 1.2f); });
    t.expect_throw("alpha < 1", [&] { build_with(32, 100, 0.9f); });
    IVFMetadata short_meta = meta;
    short_meta.centroids.pop_back();
    t.expect_throw("centroids short by one", [&] { build_ivf_centroid_graph(short_meta); });

    IVFCentroidGraphScratch scratch;
    std::vector<uint32_t> out;
    const float* query = queries.data();
    t.expect_throw("n = 0", [&] { search_ivf_centroid_graph(meta, g, query, 8, 0, scratch, out); });
    t.expect_throw("L < n", [&] { search_ivf_centroid_graph(meta, g, query, 7, 8, scratch, out); });
    auto corrupt = [&](const std::string& what, auto mutate) {
        IVFCentroidGraph copy = g;
        mutate(copy);
        t.expect_throw(what, [&] { search_ivf_centroid_graph(meta, copy, query, 16, 8, scratch, out); });
    };
    corrupt("graph for another nlist", [](IVFCentroidGraph& c) { c.nlist -= 1; });
    corrupt("neighbour table short by one", [](IVFCentroidGraph& c) { c.neighbors.pop_back(); });
    corrupt("entry out of range", [](IVFCentroidGraph& c) { c.entry = c.nlist; });
    corrupt("degree 0", [](IVFCentroidGraph& c) { c.degree = 0; });
    return t.done();
}

}  // namespace

int main() {
    bool all_pass = true;
    try {
        const IVFMetadata meta = make_meta(NLIST, 1);
        const std::vector<float> queries = draw_blobs<float>(NQ, 2, DIM, BLOBS);
        const IVFCentroidGraph g = build_ivf_centroid_graph(meta);
        all_pass &= test_structure(meta, g);
        all_pass &= test_full_beam_is_exact(meta, g, queries);
        all_pass &= test_matches_sorted_beam(meta, g, queries);
        all_pass &= test_recall(meta, g, queries);
        all_pass &= test_scratch_reuse(meta, g, queries);
        all_pass &= test_single_centroid();
        all_pass &= test_guards(meta, g, queries);
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }
    return all_pass ? 0 : 1;
}
