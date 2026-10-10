// Tests for ivf_pq_search against oracles it does not share code with: a
// reference search written out longhand (direct centroid distances, no
// GEMM), brute force over the base file for exact distances and for the
// nprobe = nlist top-k, and the argument/consistency guards. Runs on float
// and uint8 bases so the raw-vector type conversion is covered.

#include "bufann/inplace_backend.h"
#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_search.h"
#include "ivf_pq_test_util.h"
#include "utils.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t N = 20000;
const uint32_t NQ = 200;
const uint32_t DIM = 20;  // not a multiple of 8: aligned_dim padding is exercised
const uint32_t BLOBS = 16;
const uint32_t NLIST = 32;
const uint32_t CHUNKS = 4;
const uint32_t K = 10;
const uint32_t RERANK_M = 50;
const uint32_t TRAIN_SEED = 7;
const float TIE_ULPS = 8.0f;

// The reference search, written without the GEMM expansion. rerank_m == 0
// ranks by PQ distance, as in ivf_pq_search.
template<typename T>
IVFPQSearchResult reference_search(const IVFPQIndex& ix, const RawVectorHeap& heap, const float* q, uint32_t k,
                                   uint32_t nprobe, uint32_t rerank_m) {
    std::vector<float> centroid_dist(ix.meta.nlist);
    for (uint32_t c = 0; c < ix.meta.nlist; ++c) {
        centroid_dist[c] = sq_dist(q, ix.meta.centroids.data() + size_t(c) * ix.meta.aligned_dim, DIM);
    }
    std::vector<float> table(size_t(ix.pq.chunks) * ix.pq.k);
    for (uint32_t c = 0; c < ix.pq.chunks; ++c) {
        for (uint32_t j = 0; j < ix.pq.k; ++j) {
            table[size_t(c) * ix.pq.k + j] = sq_dist(
                q + c * ix.pq.chunk_dim, ix.pq.pivots.data() + (size_t(c) * ix.pq.k + j) * ix.pq.chunk_dim,
                ix.pq.chunk_dim);
        }
    }
    std::vector<uint32_t> candidates;
    std::vector<float> pq_dist;
    for (uint32_t part : top_k(centroid_dist, nprobe)) {
        for (uint32_t i = ix.lists.offsets[part]; i < ix.lists.offsets[part + 1]; ++i) {
            uint32_t id = ix.lists.ids[i];
            const uint8_t* code = ix.pq.codes.data() + size_t(id) * ix.pq.chunks;
            float d = 0.0f;
            for (uint32_t c = 0; c < ix.pq.chunks; ++c) d += table[size_t(c) * ix.pq.k + code[c]];
            candidates.push_back(id);
            pq_dist.push_back(d);
        }
    }
    IVFPQSearchResult result;
    if (rerank_m == 0) {
        for (uint32_t i : top_k(pq_dist, k)) {
            result.ids.push_back(candidates[i]);
            result.dists.push_back(pq_dist[i]);
        }
        return result;
    }
    std::vector<uint32_t> shortlist;
    std::vector<float> exact;
    std::vector<T> raw(DIM);
    for (uint32_t i : top_k(pq_dist, rerank_m)) {
        heap.read_vector(rid_flat_slot(ix.rid_table.rid[candidates[i]]), raw.data());
        shortlist.push_back(candidates[i]);
        exact.push_back(sq_dist(q, to_float(raw.data(), DIM).data(), DIM));
    }
    for (uint32_t i : top_k(exact, k)) {
        result.ids.push_back(shortlist[i]);
        result.dists.push_back(exact[i]);
    }
    return result;
}

// True when the nprobe-th and (nprobe+1)-th nearest centroids are within a
// few ulps of the norms the GEMM expansion adds and subtracts, so the two
// searches may legitimately probe different partitions.
bool probe_boundary_is_tied(const IVFPQIndex& ix, const float* q, uint32_t nprobe) {
    if (nprobe >= ix.meta.nlist) return false;
    std::vector<float> centroid_dist(ix.meta.nlist), norms(ix.meta.nlist);
    for (uint32_t c = 0; c < ix.meta.nlist; ++c) {
        const float* centroid = ix.meta.centroids.data() + size_t(c) * ix.meta.aligned_dim;
        centroid_dist[c] = sq_dist(q, centroid, DIM);
        norms[c] = l2sq(centroid, DIM);
    }
    std::vector<uint32_t> order = top_k(centroid_dist, nprobe + 1);
    uint32_t last_in = order[nprobe - 1], first_out = order[nprobe];
    float tol = TIE_ULPS * std::numeric_limits<float>::epsilon() *
                (l2sq(q, DIM) + std::max(norms[last_in], norms[first_out]));
    return centroid_dist[first_out] - centroid_dist[last_in] <= tol;
}

template<typename T>
bool test_matches_reference(const std::string& tag, const Built& b, const std::vector<float>& queries) {
    TestCase t(tag + ": ivf_pq_search matches the reference search within ties");
    IVFPQSearchScratch scratch;
    size_t compared = 0, skipped = 0;
    for (uint32_t nprobe : {1u, 3u, 8u, NLIST, 2 * NLIST}) {
        for (uint32_t rerank_m : {0u, RERANK_M}) {
            for (uint32_t q = 0; q < NQ; ++q) {
                const float* query = queries.data() + size_t(q) * DIM;
                if (probe_boundary_is_tied(b.index, query, nprobe)) {
                    ++skipped;
                    continue;
                }
                ++compared;
                IVFPQSearchResult want = reference_search<T>(b.index, b.heap, query, K, nprobe, rerank_m);
                IVFPQSearchResult got = ivf_pq_search<T>(b.index, b.heap, query, K, nprobe, rerank_m, scratch);
                if (!t.check(same_within_ties(got, want), "query " + std::to_string(q) + " nprobe " +
                                                              std::to_string(nprobe) + " rerank_m " +
                                                              std::to_string(rerank_m) + " differs")) {
                    return t.done();
                }
            }
        }
    }
    std::cout << "  compared " << compared << " searches, skipped " << skipped << " with a tied probe boundary"
              << std::endl;
    t.check(skipped < compared / 20, "too many tied probe boundaries for the comparison to mean anything");
    return t.done();
}

// The index searched through a buffer pool of 128 frames over its heap file
// (fewer than the f32 heap's pages, so re-rank reads evict and re-read pages,
// two per batch) must return what the reference returns through the page
// cache, single queries and a batch over 4 threads alike.
template<typename T>
bool test_buffer_pool_matches_reference(const std::string& tag, const Built& b, const std::string& prefix,
                                        const std::vector<float>& queries) {
    TestCase t(tag + ": searching through a 128-frame buffer pool matches the reference through the page cache");
    InPlaceIOStats stats;
    RawVectorHeap pooled;
    const uint32_t frames = 128;
    pooled.open_existing(ivf_raw_vectors_path(prefix), b.index.heap_layout, b.index.heap_next_slot,
                         b.index.heap_pages, RawVectorHeapCache{frames, &stats});
    IVFPQSearchScratch scratch;
    const int threads = omp_get_max_threads();
    omp_set_num_threads(4);  // 4 x 2 pinned pages cannot fill a 32-frame shard
    for (uint32_t nprobe : {1u, 8u, NLIST}) {
        std::vector<IVFPQSearchResult> batch =
            ivf_pq_search_batch<T>(b.index, pooled, queries.data(), NQ, K, nprobe, RERANK_M);
        for (uint32_t q = 0; q < NQ; ++q) {
            const float* query = queries.data() + size_t(q) * DIM;
            IVFPQSearchResult want = reference_search<T>(b.index, b.heap, query, K, nprobe, RERANK_M);
            IVFPQSearchResult got = ivf_pq_search<T>(b.index, pooled, query, K, nprobe, RERANK_M, scratch);
            t.check(same_within_ties(got, want),
                    "query " + std::to_string(q) + " nprobe " + std::to_string(nprobe) + " differs");
            if (!probe_boundary_is_tied(b.index, query, nprobe)) {
                t.check(same_within_ties(batch[q], want),
                        "batched query " + std::to_string(q) + " nprobe " + std::to_string(nprobe) + " differs");
            }
        }
    }
    omp_set_num_threads(threads);
    t.check(b.index.heap_pages <= frames || stats.evictions.load() > 0,
            "the heap outgrew the pool but nothing was evicted");
    pooled.close();
    return t.done();
}

// The batch path must agree with single-query searches on every query, in
// query order, for a batch that spans several GEMM blocks, with and without
// the re-rank; an empty batch is empty.
template<typename T>
bool test_batch_matches_single(const std::string& tag, const Built& b, const std::vector<float>& queries) {
    TestCase t(tag + ": ivf_pq_search_batch matches single-query searches across GEMM blocks");
    IVFPQSearchScratch scratch;
    const uint32_t gemm_rows = 37;  // does not divide NQ, so the last block is partial
    for (uint32_t nprobe : {1u, 8u}) {
        for (uint32_t rerank_m : {0u, RERANK_M}) {
            std::vector<IVFPQSearchResult> batch =
                ivf_pq_search_batch<T>(b.index, b.heap, queries.data(), NQ, K, nprobe, rerank_m, gemm_rows);
            if (!t.check(batch.size() == NQ, "batch returned the wrong number of results")) return t.done();
            for (uint32_t q = 0; q < NQ; ++q) {
                // A batched GEMM rounds differently from a one-row one, so a
                // probe boundary tied to within ulps can go either way.
                if (probe_boundary_is_tied(b.index, queries.data() + size_t(q) * DIM, nprobe)) continue;
                IVFPQSearchResult single =
                    ivf_pq_search<T>(b.index, b.heap, queries.data() + size_t(q) * DIM, K, nprobe, rerank_m, scratch);
                if (!t.check(same_within_ties(batch[q], single), "query " + std::to_string(q) + " nprobe " +
                                                                     std::to_string(nprobe) + " rerank_m " +
                                                                     std::to_string(rerank_m) + " differs")) {
                    return t.done();
                }
            }
        }
    }
    t.check(ivf_pq_search_batch<T>(b.index, b.heap, queries.data(), 0, K, 8, RERANK_M).empty(),
            "empty batch is not empty");
    t.expect_throw("gemm_rows == 0",
                   [&] { ivf_pq_search_batch<T>(b.index, b.heap, queries.data(), NQ, K, 8, RERANK_M, 0); });
    t.expect_throw("null queries with nq > 0", [&] { ivf_pq_search_batch<T>(b.index, b.heap, nullptr, NQ, K, 8, 0); });
    return t.done();
}

template<typename T>
bool test_exact_distances_and_full_probe(const std::string& tag, const Built& b, const std::vector<T>& base,
                                         const std::vector<float>& queries) {
    TestCase t(tag + ": re-ranked distances are brute force; nprobe = nlist with a full re-rank is exact top-k");
    const std::vector<float> basef = to_float(base.data(), base.size());
    IVFPQSearchScratch scratch;
    std::vector<float> all(N);
    for (uint32_t q = 0; q < NQ; ++q) {
        const float* query = queries.data() + size_t(q) * DIM;
        IVFPQSearchResult got = ivf_pq_search<T>(b.index, b.heap, query, K, 4, RERANK_M, scratch);
        bool ok = got.ids.size() == K && got.dists.size() == K && std::is_sorted(got.dists.begin(), got.dists.end());
        for (size_t i = 0; ok && i < K; ++i) {
            ok = got.ids[i] < N && close(got.dists[i], sq_dist(query, basef.data() + size_t(got.ids[i]) * DIM, DIM)) &&
                 std::count(got.ids.begin(), got.ids.end(), got.ids[i]) == 1;
        }
        if (!t.check(ok, "query " + std::to_string(q) + ": result is not K distinct ids with brute-force distances")) {
            return t.done();
        }

        for (uint32_t i = 0; i < N; ++i) all[i] = sq_dist(query, basef.data() + size_t(i) * DIM, DIM);
        IVFPQSearchResult want;
        for (uint32_t i : top_k(all, K)) {
            want.ids.push_back(i);
            want.dists.push_back(all[i]);
        }
        IVFPQSearchResult full = ivf_pq_search<T>(b.index, b.heap, query, K, NLIST, N, scratch);
        if (!t.check(same_within_ties(full, want),
                     "query " + std::to_string(q) + ": full probe and re-rank is not the exact top-k")) {
            return t.done();
        }
    }
    return t.done();
}

template<typename T>
bool test_result_size_follows_candidates(const std::string& tag, const Built& b, const std::vector<float>& queries) {
    TestCase t(tag + ": k beyond the probed candidates returns every candidate, and no more");
    const IVFPQIndex& ix = b.index;
    const float* query = queries.data();
    IVFPQSearchResult one = ivf_pq_search<T>(ix, b.heap, query, N + 5, 1, 0);
    std::vector<float> centroid_dist(NLIST);
    for (uint32_t c = 0; c < NLIST; ++c) {
        centroid_dist[c] = sq_dist(query, ix.meta.centroids.data() + size_t(c) * ix.meta.aligned_dim, DIM);
    }
    uint32_t nearest = top_k(centroid_dist, 1)[0];
    uint32_t partition_size = ix.lists.offsets[nearest + 1] - ix.lists.offsets[nearest];
    t.check(one.ids.size() == partition_size,
            "nprobe 1 returned " + std::to_string(one.ids.size()) + " ids from a partition of " +
                std::to_string(partition_size));
    IVFPQSearchResult all = ivf_pq_search<T>(ix, b.heap, query, N + 5, NLIST, N + 5);
    std::vector<uint32_t> ids = all.ids;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    t.check(all.ids.size() == N && ids.size() == N, "full probe with k > N did not return every vector once");
    return t.done();
}

template<typename T>
bool test_inactive_rid_is_dropped(const std::string& tag, const Built& b, const std::vector<float>& queries) {
    TestCase t(tag + ": a vector whose RID is inactive is dropped from both PQ-only and re-ranked results");
    const float* query = queries.data();
    IVFPQSearchResult before = ivf_pq_search<T>(b.index, b.heap, query, K, 4, RERANK_M);
    IVFPQIndex copy = b.index;
    uint32_t victim = before.ids[0];
    copy.rid_table.rid[victim] = make_raw_vector_rid(rid_flat_slot(copy.rid_table.rid[victim]), false);
    for (uint32_t rerank_m : {0u, RERANK_M}) {
        IVFPQSearchResult after = ivf_pq_search<T>(copy, b.heap, query, K, 4, rerank_m);
        t.check(std::find(after.ids.begin(), after.ids.end(), victim) == after.ids.end(),
                "inactive id " + std::to_string(victim) + " returned with rerank_m " + std::to_string(rerank_m));
    }
    // The shortlist loses one member, so the next K-1 move up and a new id fills the last place.
    IVFPQSearchResult after = ivf_pq_search<T>(copy, b.heap, query, K, 4, RERANK_M);
    t.check(after.ids.size() == K && std::equal(before.ids.begin() + 1, before.ids.end(), after.ids.begin()),
            "dropping the nearest id did not shift the re-ranked result by one");
    return t.done();
}

bool test_scratch_follows_index(const Built& a, const Built& b, const std::vector<float>& queries) {
    TestCase t("one scratch used across two indexes gives each index's own results");
    const float* query = queries.data();
    IVFPQSearchScratch scratch;
    IVFPQSearchResult on_a = ivf_pq_search<float>(a.index, a.heap, query, K, 4, RERANK_M, scratch);
    IVFPQSearchResult on_b = ivf_pq_search<uint8_t>(b.index, b.heap, query, K, 4, RERANK_M, scratch);
    IVFPQSearchResult fresh_b = ivf_pq_search<uint8_t>(b.index, b.heap, query, K, 4, RERANK_M);
    IVFPQSearchResult on_a_again = ivf_pq_search<float>(a.index, a.heap, query, K, 4, RERANK_M, scratch);
    IVFPQSearchResult fresh_a = ivf_pq_search<float>(a.index, a.heap, query, K, 4, RERANK_M);
    t.check(on_b.ids == fresh_b.ids && on_b.dists == fresh_b.dists, "reused scratch gave a different result on b");
    t.check(on_a.ids == fresh_a.ids && on_a_again.ids == fresh_a.ids && on_a_again.dists == fresh_a.dists,
            "reused scratch gave a different result back on a");

    // Same index object, centroids replaced in place (what a rebuild that
    // reuses storage does): nothing cached per scratch may survive that.
    IVFPQIndex& ix = const_cast<IVFPQIndex&>(a.index);
    const IVFMetadata original = ix.meta;
    for (float& v : ix.meta.centroids) v = -v;
    set_ivf_centroid_norms(ix.meta);
    IVFPQSearchResult mutated = ivf_pq_search<float>(ix, a.heap, query, K, 4, RERANK_M, scratch);
    IVFPQSearchResult fresh_mutated = ivf_pq_search<float>(ix, a.heap, query, K, 4, RERANK_M);
    ix.meta = original;
    t.check(mutated.ids == fresh_mutated.ids && mutated.dists == fresh_mutated.dists,
            "reused scratch gave a different result after the index's centroids changed in place");
    t.check(mutated.ids != fresh_a.ids, "negated centroids should probe different partitions");
    return t.done();
}

// A beam as wide as nlist makes the graph's probe set the exact nearest
// nprobe, so ivf_pq_search_graph must then match the reference search; a
// narrow beam must still return well-formed results from nprobe partitions.
template<typename T>
bool test_graph_matches_reference(const std::string& tag, const Built& b, const std::vector<float>& queries) {
    TestCase t(tag + ": ivf_pq_search_graph with a beam of nlist matches the reference search within ties");
    const IVFCentroidGraph graph = build_ivf_centroid_graph(b.index.meta);
    IVFPQSearchScratch scratch;
    size_t compared = 0, skipped = 0;
    for (uint32_t nprobe : {1u, 3u, 8u, NLIST, 2 * NLIST}) {
        for (uint32_t rerank_m : {0u, RERANK_M}) {
            for (uint32_t q = 0; q < NQ; ++q) {
                const float* query = queries.data() + size_t(q) * DIM;
                if (probe_boundary_is_tied(b.index, query, nprobe)) {
                    ++skipped;
                    continue;
                }
                ++compared;
                IVFPQSearchResult want = reference_search<T>(b.index, b.heap, query, K, nprobe, rerank_m);
                IVFPQSearchResult got =
                    ivf_pq_search_graph<T>(b.index, graph, b.heap, query, K, nprobe, NLIST, rerank_m, scratch);
                if (!t.check(same_within_ties(got, want), "query " + std::to_string(q) + " nprobe " +
                                                              std::to_string(nprobe) + " rerank_m " +
                                                              std::to_string(rerank_m) + " differs")) {
                    return t.done();
                }
            }
        }
    }
    t.check(skipped < compared / 20, "too many tied probe boundaries for the comparison to mean anything");

    // nprobe = 1 with the narrowest beam: every result comes from the one
    // partition the graph returned, which is a real partition.
    std::vector<uint32_t> part_of(N);
    for (uint32_t p = 0; p < NLIST; ++p) {
        for (uint32_t i = b.index.lists.offsets[p]; i < b.index.lists.offsets[p + 1]; ++i) part_of[b.index.lists.ids[i]] = p;
    }
    for (uint32_t q = 0; q < NQ; ++q) {
        IVFPQSearchResult got =
            ivf_pq_search_graph<T>(b.index, graph, b.heap, queries.data() + size_t(q) * DIM, K, 1, 1, RERANK_M, scratch);
        bool ok = !got.ids.empty() && std::is_sorted(got.dists.begin(), got.dists.end());
        for (uint32_t id : got.ids) ok = ok && part_of[id] == part_of[got.ids[0]];
        if (!t.check(ok, "query " + std::to_string(q) + ": nprobe 1 result is not from a single partition")) break;
    }

    t.expect_throw("centroid_L < nprobe",
                   [&] { ivf_pq_search_graph<T>(b.index, graph, b.heap, queries.data(), K, 8, 7, RERANK_M, scratch); });
    t.expect_throw("null query",
                   [&] { ivf_pq_search_graph<T>(b.index, graph, b.heap, nullptr, K, 8, 16, RERANK_M, scratch); });
    t.expect_throw("k = 0",
                   [&] { ivf_pq_search_graph<T>(b.index, graph, b.heap, queries.data(), 0, 8, 16, RERANK_M, scratch); });
    IVFCentroidGraph other = graph;
    other.nlist = NLIST / 2;
    other.neighbors.resize(size_t(other.nlist) * other.degree);
    other.entry = 0;
    t.expect_throw("graph built for another nlist",
                   [&] { ivf_pq_search_graph<T>(b.index, other, b.heap, queries.data(), K, 8, 16, RERANK_M, scratch); });
    return t.done();
}

// The SIMD scan keeps the rerank_m nearest by quantized PQ distance, so near
// the boundary it may shortlist differently from the float scan. Checked
// against brute force and the float reference: every returned distance is
// exact; with every candidate re-ranked the result is the exact top-k; at a
// normal rerank_m the top-k agrees with the float reference's on at least
// 99% of ids, and the batch path returns what single queries do.
template<typename T>
bool test_fastscan_end_to_end(const std::string& tag, const Built& b, const std::vector<T>& base,
                              const std::vector<float>& queries) {
    TestCase t(tag + ": the SIMD scan returns exact distances, the exact top-k at a full re-rank, "
               "and the float scan's top-k to within 1%");
    if (!fastscan_simd_available()) {
        std::cout << "  (no AVX-512 VBMI on this CPU: the search uses the float scan; nothing to compare)" << std::endl;
        return t.done();
    }
    set_fastscan_enabled(true);
    const std::vector<float> basef = to_float(base.data(), base.size());
    IVFPQSearchScratch scratch, float_scratch;
    size_t agree = 0, total = 0;
    std::vector<float> all(N);
    for (uint32_t nprobe : {1u, 3u, 8u, NLIST}) {
        std::vector<IVFPQSearchResult> batch = ivf_pq_search_batch<T>(b.index, b.heap, queries.data(), NQ, K, nprobe, RERANK_M);
        for (uint32_t q = 0; q < NQ; ++q) {
            const float* query = queries.data() + size_t(q) * DIM;
            IVFPQSearchResult got = ivf_pq_search<T>(b.index, b.heap, query, K, nprobe, RERANK_M, scratch);
            bool exact = std::is_sorted(got.dists.begin(), got.dists.end());
            for (size_t i = 0; exact && i < got.ids.size(); ++i) {
                exact = close(got.dists[i], sq_dist(query, basef.data() + size_t(got.ids[i]) * DIM, DIM));
            }
            if (!t.check(exact, "query " + std::to_string(q) + ": a returned distance is not the exact one")) return t.done();
            if (!probe_boundary_is_tied(b.index, query, nprobe)) {
                t.check(same_within_ties(batch[q], got), "batched query " + std::to_string(q) + " differs from single");
            }
            set_fastscan_enabled(false);
            IVFPQSearchResult want = ivf_pq_search<T>(b.index, b.heap, query, K, nprobe, RERANK_M, float_scratch);
            set_fastscan_enabled(true);
            for (uint32_t id : got.ids) agree += std::find(want.ids.begin(), want.ids.end(), id) != want.ids.end();
            total += want.ids.size();
        }
    }
    std::cout << "  top-k agreement with the float scan: " << double(agree) / double(total) << std::endl;
    t.check(agree >= 0.99 * double(total), "SIMD and float top-k agree on fewer than 99% of ids");
    for (uint32_t q = 0; q < NQ; ++q) {
        const float* query = queries.data() + size_t(q) * DIM;
        for (uint32_t i = 0; i < N; ++i) all[i] = sq_dist(query, basef.data() + size_t(i) * DIM, DIM);
        IVFPQSearchResult want;
        for (uint32_t i : top_k(all, K + 1)) want.ids.push_back(i), want.dists.push_back(all[i]);
        IVFPQSearchResult full = ivf_pq_search<T>(b.index, b.heap, query, K, NLIST, N, scratch);
        if (!t.check(same_within_ties(full, want), "query " + std::to_string(q) + ": full re-rank is not exact")) break;
    }
    set_fastscan_enabled(false);
    return t.done();
}

bool test_guards(const Built& b, const std::vector<float>& queries) {
    TestCase t("argument and index-consistency guards");
    const IVFPQIndex& ix = b.index;
    const float* query = queries.data();
    t.expect_throw("k = 0", [&] { ivf_pq_search<float>(ix, b.heap, query, 0, 4, RERANK_M); });
    t.expect_throw("nprobe = 0", [&] { ivf_pq_search<float>(ix, b.heap, query, K, 0, RERANK_M); });
    t.expect_throw("0 < rerank_m < k", [&] { ivf_pq_search<float>(ix, b.heap, query, K, 4, K - 1); });
    t.expect_throw("null query", [&] { ivf_pq_search<float>(ix, b.heap, nullptr, K, 4, RERANK_M); });
    t.expect_throw("heap elem_size for the wrong T", [&] { ivf_pq_search<uint8_t>(ix, b.heap, query, K, 4, RERANK_M); });

    auto corrupt = [&](const std::string& what, auto mutate) {
        IVFPQIndex copy = ix;
        mutate(copy);
        t.expect_throw(what, [&] { ivf_pq_search<float>(copy, b.heap, query, K, 4, RERANK_M); });
    };
    corrupt("RID table short by one", [](IVFPQIndex& c) { c.rid_table.rid.pop_back(); });
    corrupt("posting-list codes short by one row",
            [](IVFPQIndex& c) { c.lists.codes.resize(c.lists.codes.size() - CHUNKS); });
    corrupt("posting-list codes never set", [](IVFPQIndex& c) { c.lists.codes.clear(); });
    corrupt("PQ chunks do not cover dim", [](IVFPQIndex& c) { c.pq.chunk_dim -= 1; });
    corrupt("posting offsets end early", [](IVFPQIndex& c) { c.lists.offsets.back() -= 1; });
    corrupt("posting offsets missing a partition", [](IVFPQIndex& c) { c.lists.offsets.pop_back(); });
    corrupt("centroids short by one", [](IVFPQIndex& c) { c.meta.centroids.pop_back(); });
    return t.done();
}

}  // namespace

int main() {
    const std::string prefix_f = temp_path("ivf_pq_search_f32");
    const std::string prefix_u = temp_path("ivf_pq_search_u8");
    bool all_pass = true;
    try {
        std::vector<float> base_f = draw_blobs<float>(N, 1, DIM, BLOBS);
        std::vector<uint8_t> base_u = draw_blobs<uint8_t>(N, 1, DIM, BLOBS);
        std::vector<float> queries = draw_blobs<float>(NQ, 2, DIM, BLOBS);
        // These tests compare against a float reference exactly; the SIMD
        // scan, whose shortlist may differ near the boundary, has its own.
        set_fastscan_enabled(false);
        std::unique_ptr<Built> f = build_index<float>(prefix_f, base_f, N, DIM, NLIST, CHUNKS, TRAIN_SEED);
        std::unique_ptr<Built> u = build_index<uint8_t>(prefix_u, base_u, N, DIM, NLIST, CHUNKS, TRAIN_SEED);

        all_pass &= test_matches_reference<float>("f32", *f, queries);
        all_pass &= test_matches_reference<uint8_t>("u8", *u, queries);
        all_pass &= test_batch_matches_single<float>("f32", *f, queries);
        all_pass &= test_batch_matches_single<uint8_t>("u8", *u, queries);
        all_pass &= test_exact_distances_and_full_probe<float>("f32", *f, base_f, queries);
        all_pass &= test_exact_distances_and_full_probe<uint8_t>("u8", *u, base_u, queries);
        all_pass &= test_result_size_follows_candidates<float>("f32", *f, queries);
        all_pass &= test_inactive_rid_is_dropped<float>("f32", *f, queries);
        all_pass &= test_inactive_rid_is_dropped<uint8_t>("u8", *u, queries);
        all_pass &= test_graph_matches_reference<float>("f32", *f, queries);
        all_pass &= test_graph_matches_reference<uint8_t>("u8", *u, queries);
        all_pass &= test_buffer_pool_matches_reference<float>("f32", *f, prefix_f, queries);
        all_pass &= test_buffer_pool_matches_reference<uint8_t>("u8", *u, prefix_u, queries);
        all_pass &= test_fastscan_end_to_end<float>("f32", *f, base_f, queries);
        all_pass &= test_fastscan_end_to_end<uint8_t>("u8", *u, base_u, queries);
        all_pass &= test_scratch_follows_index(*f, *u, queries);
        all_pass &= test_guards(*f, queries);

        f->heap.close();
        u->heap.close();
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }
    remove_index_files(prefix_f);
    remove_index_files(prefix_u);
    return all_pass ? 0 : 1;
}
