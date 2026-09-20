// Tests for ivf_pq_insert against oracles it does not share code with: the
// build's own cluster assignments and PQ codes (upstream's GEMM-based
// encoders), brute force over base + inserted vectors, a PQ-only reference
// computed here from both code tables, and the delta's contents read back
// directly. Also inserts racing searches, and the guards.

#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_mutate.h"
#include "bufann/ivf_pq_search.h"
#include "ivf_pq_test_util.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t N = 8000;
const uint32_t NI = 400;  // inserted after the build
const uint32_t NQ = 100;
const uint32_t DIM = 20;
const uint32_t BLOBS = 16;
const uint32_t NLIST = 32;
const uint32_t CHUNKS = 4;
const uint32_t K = 10;
const uint32_t TRAIN_SEED = 7;
const float TIE_ULPS = 8.0f;
const float DIST_REL_TOL = 1e-5f;

bool close(float a, float b) { return std::fabs(a - b) <= DIST_REL_TOL * std::max(std::fabs(a), std::fabs(b)); }

template<typename T>
T clamp_to(float v) {
    float lo = float(std::numeric_limits<T>::lowest()), hi = float(std::numeric_limits<T>::max());
    return T(std::round(std::min(hi, std::max(lo, v))));
}
template<>
float clamp_to<float>(float v) { return v; }

// The search test's fixture: 16 Gaussian blobs in [40, 215].
template<typename T>
std::vector<T> draw(uint32_t count, uint32_t seed) {
    std::mt19937 gen(seed);
    std::mt19937 center_gen(99);
    std::uniform_real_distribution<float> spread(40.0f, 215.0f);
    std::vector<float> centers(size_t(BLOBS) * DIM);
    for (float& v : centers) v = spread(center_gen);
    std::normal_distribution<float> noise(0.0f, 6.0f);
    std::vector<T> out(size_t(count) * DIM);
    for (uint32_t i = 0; i < count; ++i) {
        const float* c = centers.data() + size_t(gen() % BLOBS) * DIM;
        for (uint32_t d = 0; d < DIM; ++d) out[size_t(i) * DIM + d] = clamp_to<T>(c[d] + noise(gen));
    }
    return out;
}

template<typename T>
std::vector<float> to_float(const T* v, size_t n) {
    return std::vector<float>(v, v + n);
}

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

std::vector<uint32_t> top_k(const std::vector<float>& values, uint32_t k) {
    std::vector<uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), 0u);
    k = std::min<uint32_t>(k, uint32_t(order.size()));
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    order.resize(k);
    return order;
}

// Same distances position by position, and same id wherever the distance is
// not shared with a neighbouring position.
bool same_within_ties(const IVFPQSearchResult& got, const IVFPQSearchResult& want) {
    if (got.ids.size() != want.ids.size()) return false;
    for (size_t i = 0; i < got.ids.size(); ++i) {
        if (!close(got.dists[i], want.dists[i])) return false;
        bool tied = (i > 0 && close(want.dists[i], want.dists[i - 1])) ||
                    (i + 1 < want.ids.size() && close(want.dists[i], want.dists[i + 1]));
        if (!tied && got.ids[i] != want.ids[i]) return false;
    }
    return true;
}

struct Built {
    IVFPQIndex index;
    RawVectorHeap heap;
    IVFPQDelta delta;
};

template<typename T>
std::unique_ptr<Built> build(const std::string& prefix, const std::vector<T>& base) {
    const std::string base_bin = prefix + "_base.bin";
    diskann::save_bin<T>(base_bin, const_cast<T*>(base.data()), N, DIM);
    {
        IVFPQIndex ix;
        ix.meta = train_ivf_centroids<T>(base_bin, NLIST, 0.0, NUM_K_MEANS_ITERS, TRAIN_SEED);
        ix.heap_layout = compute_raw_vector_heap_layout(4096, DIM * sizeof(T));
        RawVectorHeap heap;
        heap.open(ivf_raw_vectors_path(prefix), ix.heap_layout);
        assign_ivf_clusters<T>(base_bin, ix.meta, heap, ix.assignments, ix.rid_table);
        ix.lists = build_ivf_posting_lists(ix.assignments, NLIST);
        train_ivf_pq_pivots<T>(base_bin, prefix, CHUNKS, 1.0, NUM_K_MEANS_ITERS, TRAIN_SEED);
        encode_ivf_pq_codes<T>(base_bin, prefix, CHUNKS);
        ix.pq = load_ivf_pq(prefix);
        ix.heap_pages = heap.allocated_pages();
        ix.heap_next_slot = heap.next_flat_slot();
        write_ivf_pq_index(prefix, ix);
    }
    auto built = std::make_unique<Built>();
    built->index = load_ivf_pq_index(prefix);
    built->heap.open_existing(ivf_raw_vectors_path(prefix), built->index.heap_layout, built->index.heap_next_slot,
                              built->index.heap_pages);
    ::unlink(base_bin.c_str());
    return built;
}

void remove_index_files(const std::string& prefix) {
    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
}

// Whether two squared distances are within the float rounding of the
// ||x||^2 + ||c||^2 - 2x.c expansion the build's encoders use.
bool within_expansion_rounding(float d_a, float d_b, float x_l2sq, float c_l2sq) {
    return std::fabs(d_a - d_b) <= TIE_ULPS * std::numeric_limits<float>::epsilon() * (x_l2sq + c_l2sq);
}

// The build assigned and encoded every base vector with upstream's
// GEMM-based routines; the exact argmins must agree except on near-ties.
template<typename T>
bool test_partition_and_code_match_the_build(const std::string& tag, const Built& b, const std::vector<T>& base) {
    TestCase t(tag + ": ivf_nearest_centroid and ivf_pq_encode agree with the build's assignments and codes");
    const IVFMetadata& meta = b.index.meta;
    const PQMetadata& pq = b.index.pq;
    size_t cluster_ties = 0, code_ties = 0;
    std::vector<uint8_t> code(CHUNKS);
    for (uint32_t i = 0; i < N; ++i) {
        const std::vector<float> x = to_float(base.data() + size_t(i) * DIM, DIM);
        const uint32_t got = ivf_nearest_centroid(meta, x.data());
        const uint32_t want = b.index.assignments.cluster_id[i];
        if (got != want) {
            const float* cg = meta.centroids.data() + size_t(got) * meta.aligned_dim;
            const float* cw = meta.centroids.data() + size_t(want) * meta.aligned_dim;
            const float dg = sq_dist(x.data(), cg, DIM), dw = sq_dist(x.data(), cw, DIM);
            if (!t.check(dg <= dw && within_expansion_rounding(dg, dw, l2sq(x.data(), DIM),
                                                               std::max(l2sq(cg, DIM), l2sq(cw, DIM))),
                         "vector " + std::to_string(i) + ": partition " + std::to_string(got) + " (dist " +
                             std::to_string(dg) + ") vs the build's " + std::to_string(want) + " (dist " +
                             std::to_string(dw) + ")")) {
                return t.done();
            }
            ++cluster_ties;
        }

        ivf_pq_encode(pq, x.data(), code.data());
        for (uint32_t c = 0; c < CHUNKS; ++c) {
            const uint8_t want_code = pq.codes[size_t(i) * CHUNKS + c];
            if (code[c] == want_code) continue;
            const float* sub = x.data() + c * pq.chunk_dim;
            const float* pg = pq.pivots.data() + (size_t(c) * pq.k + code[c]) * pq.chunk_dim;
            const float* pw = pq.pivots.data() + (size_t(c) * pq.k + want_code) * pq.chunk_dim;
            const float dg = sq_dist(sub, pg, pq.chunk_dim), dw = sq_dist(sub, pw, pq.chunk_dim);
            if (!t.check(dg <= dw && within_expansion_rounding(dg, dw, l2sq(sub, pq.chunk_dim),
                                                               std::max(l2sq(pg, pq.chunk_dim), l2sq(pw, pq.chunk_dim))),
                         "vector " + std::to_string(i) + " chunk " + std::to_string(c) + ": code " +
                             std::to_string(code[c]) + " (dist " + std::to_string(dg) + ") vs the build's " +
                             std::to_string(want_code) + " (dist " + std::to_string(dw) + ")")) {
                return t.done();
            }
            ++code_ties;
        }
    }
    std::cout << "  near-ties: " << cluster_ties << " partitions, " << code_ties << " codes of " << N * CHUNKS
              << std::endl;
    t.check(cluster_ties < N / 100 && code_ties < N * CHUNKS / 100, "too many disagreements to be rounding");
    return t.done();
}

// Inserts NI vectors into a copy of the index and checks what ivf_pq_insert
// wrote, that each is found by a search for itself at the nprobe that covers
// its partition, and that searches over base + inserts match brute force
// (re-ranked) and a two-table PQ reference (PQ-only).
template<typename T>
bool test_inserted_vectors_are_found(const std::string& tag, Built& b, const std::vector<T>& base,
                                     const std::vector<T>& extra, const std::vector<float>& queries) {
    TestCase t(tag + ": inserted vectors are indexed, found at the covering nprobe, and ranked with the base");
    IVFPQIndex ix = b.index;
    IVFPQDelta delta;
    const IVFMetadata& meta = ix.meta;
    const PQMetadata& pq = ix.pq;
    const uint32_t slots_before = b.heap.next_flat_slot();

    for (uint32_t i = 0; i < NI; ++i) {
        const T* v = extra.data() + size_t(i) * DIM;
        const uint32_t id = ivf_pq_insert<T>(ix, b.heap, delta, v);
        if (!t.check(id == N + i, "insert " + std::to_string(i) + " got id " + std::to_string(id))) return t.done();
    }
    t.check(ix.rid_table.rid.size() == N + NI && ix.assignments.cluster_id.size() == N + NI &&
                delta.codes.codes.size() == NI && ix.lists.ids.size() == N && pq.codes.size() == size_t(N) * CHUNKS,
            "tables did not grow by NI (and the base tables must not grow)");
    t.check(b.heap.next_flat_slot() == slots_before + NI, "inserts on an empty free list did not take fresh slots");

    // What the insert wrote: partition, code, raw bytes, delta membership.
    std::vector<uint8_t> code(CHUNKS);
    std::vector<T> raw(DIM);
    size_t listed = 0;
    for (uint32_t i = 0; i < NI; ++i) {
        const uint32_t id = N + i;
        const T* v = extra.data() + size_t(i) * DIM;
        const std::vector<float> x = to_float(v, DIM);
        const uint32_t cluster = ix.assignments.cluster_id[id];
        ivf_pq_encode(pq, x.data(), code.data());
        const RawVectorRID rid = ix.rid_table.rid[id];
        b.heap.read_vector(rid_flat_slot(rid), raw.data());
        auto pending = delta.lists.pending_inserts.find(cluster);
        listed += pending != delta.lists.pending_inserts.end() &&
                  std::count(pending->second.begin(), pending->second.end(), id) == 1;
        if (!t.check(cluster == ivf_nearest_centroid(meta, x.data()) && delta.codes.codes.at(id) == code &&
                         rid_is_active(rid) && std::memcmp(raw.data(), v, DIM * sizeof(T)) == 0 &&
                         b.heap.is_slot_occupied(rid_flat_slot(rid)),
                     "inserted vector " + std::to_string(id) + " was not recorded as prepared")) {
            return t.done();
        }
    }
    t.check(listed == NI, "not every inserted id is listed once under its partition in the delta");
    size_t delta_ids = 0;
    for (const auto& kv : delta.lists.pending_inserts) delta_ids += kv.second.size();
    t.check(delta_ids == NI, "the delta lists hold " + std::to_string(delta_ids) + " ids, not NI");

    // Each inserted vector is the exact nearest neighbour of itself; nprobe 1
    // covers its partition unless the two nearest centroids are tied within
    // the GEMM expansion's rounding, then nprobe 2 does.
    IVFPQSearchScratch scratch;
    std::vector<float> centroid_dist(NLIST), norms(NLIST);
    for (uint32_t c = 0; c < NLIST; ++c) norms[c] = l2sq(meta.centroids.data() + size_t(c) * meta.aligned_dim, DIM);
    size_t at_two = 0, without_delta = 0;
    for (uint32_t i = 0; i < NI; ++i) {
        const uint32_t id = N + i;
        const std::vector<float> x = to_float(extra.data() + size_t(i) * DIM, DIM);
        for (uint32_t c = 0; c < NLIST; ++c) {
            centroid_dist[c] = sq_dist(x.data(), meta.centroids.data() + size_t(c) * meta.aligned_dim, DIM);
        }
        std::vector<uint32_t> near = top_k(centroid_dist, 2);
        const bool tied = within_expansion_rounding(centroid_dist[near[0]], centroid_dist[near[1]],
                                                    l2sq(x.data(), DIM), std::max(norms[near[0]], norms[near[1]]));
        at_two += tied;
        IVFPQSearchResult r = ivf_pq_search<T>(ix, b.heap, x.data(), 1, tied ? 2 : 1, 10, scratch, &delta);
        if (!t.check(r.ids.size() == 1 && r.ids[0] == id && r.dists[0] == 0.0f,
                     "inserted vector " + std::to_string(id) + " not found as its own nearest neighbour at nprobe " +
                         std::to_string(tied ? 2 : 1))) {
            return t.done();
        }
        IVFPQSearchResult no_delta = ivf_pq_search<T>(ix, b.heap, x.data(), 1, NLIST, 10, scratch);
        without_delta += no_delta.ids.size() == 1 && no_delta.ids[0] < N;
    }
    t.check(without_delta == NI, "a search without the delta returned an inserted id");
    std::cout << "  " << at_two << " of " << NI << " inserted vectors sit on a tied partition boundary" << std::endl;

    // Base + inserted, brute force (full probe, full re-rank) and PQ-only
    // (full probe, reference over both code tables).
    std::vector<float> all = to_float(base.data(), size_t(N) * DIM);
    const std::vector<float> extraf = to_float(extra.data(), size_t(NI) * DIM);
    all.insert(all.end(), extraf.begin(), extraf.end());
    std::vector<float> exact(N + NI), approx(N + NI), table(size_t(CHUNKS) * pq.k);
    size_t inserted_in_topk = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        const float* query = queries.data() + size_t(q) * DIM;
        for (uint32_t i = 0; i < N + NI; ++i) exact[i] = sq_dist(query, all.data() + size_t(i) * DIM, DIM);
        IVFPQSearchResult want;
        for (uint32_t i : top_k(exact, K)) want.ids.push_back(i), want.dists.push_back(exact[i]);
        IVFPQSearchResult got = ivf_pq_search<T>(ix, b.heap, query, K, NLIST, N + NI, scratch, &delta);
        if (!t.check(same_within_ties(got, want),
                     "query " + std::to_string(q) + ": full probe over base + inserts is not the exact top-k")) {
            return t.done();
        }
        for (uint32_t id : got.ids) inserted_in_topk += id >= N;

        for (uint32_t c = 0; c < CHUNKS; ++c) {
            for (uint32_t j = 0; j < pq.k; ++j) {
                table[size_t(c) * pq.k + j] = sq_dist(query + c * pq.chunk_dim,
                                                      pq.pivots.data() + (size_t(c) * pq.k + j) * pq.chunk_dim,
                                                      pq.chunk_dim);
            }
        }
        for (uint32_t i = 0; i < N + NI; ++i) {
            const uint8_t* c = i < N ? pq.codes.data() + size_t(i) * CHUNKS : delta.codes.codes.at(i).data();
            float d = 0.0f;
            for (uint32_t k = 0; k < CHUNKS; ++k) d += table[size_t(k) * pq.k + c[k]];
            approx[i] = d;
        }
        IVFPQSearchResult want_pq;
        for (uint32_t i : top_k(approx, K)) want_pq.ids.push_back(i), want_pq.dists.push_back(approx[i]);
        IVFPQSearchResult got_pq = ivf_pq_search<T>(ix, b.heap, query, K, NLIST, 0, scratch, &delta);
        if (!t.check(same_within_ties(got_pq, want_pq),
                     "query " + std::to_string(q) + ": PQ-only ranking over base + inserts is not the reference")) {
            return t.done();
        }
    }
    t.check(inserted_in_topk > 0, "no inserted vector ever made a top-k; the merge was not exercised");
    std::cout << "  inserted vectors in the exact top-" << K << ": " << inserted_in_topk << " of " << NQ * K << std::endl;

    // The batched path consults the delta too.
    std::vector<IVFPQSearchResult> batch =
        ivf_pq_search_batch<T>(ix, b.heap, queries.data(), NQ, K, NLIST, N + NI, IVF_PQ_SEARCH_GEMM_ROWS, &delta);
    size_t batch_mismatches = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        IVFPQSearchResult single = ivf_pq_search<T>(ix, b.heap, queries.data() + size_t(q) * DIM, K, NLIST, N + NI,
                                                    scratch, &delta);
        batch_mismatches += !same_within_ties(batch[q], single);
    }
    t.check(batch_mismatches == 0, std::to_string(batch_mismatches) + " batched results differ from single-query");

    // The index file cannot describe the inserts, so writing it is refused.
    const std::string refused = temp_path("ivf_pq_mutate_refused");
    t.expect_throw("write_ivf_pq_index with unfolded inserts", [&] { write_ivf_pq_index(refused, ix); });
    ::unlink(ivf_pq_index_path(refused).c_str());
    return t.done();
}

// Inserts from two threads while four search, each inserter checking its
// vector is findable as soon as its insert returns. The reallocation hazard
// this guards (rid_table growing under a search) shows as a crash or a
// bad id, not a wrong distance.
bool test_inserts_race_searches(Built& b, const std::vector<float>& queries) {
    TestCase t("f32: inserts concurrent with searches are visible on return and never break a search");
    IVFPQIndex ix = b.index;
    IVFPQDelta delta;
    const uint32_t per_thread = 500;
    std::vector<float> extra = draw<float>(2 * per_thread, 11);
    std::atomic<bool> stop{false};
    std::atomic<size_t> failures{0}, searches{0};

    auto inserter = [&](uint32_t which) {
        IVFPQSearchScratch scratch;
        for (uint32_t i = 0; i < per_thread; ++i) {
            const float* v = extra.data() + size_t(which * per_thread + i) * DIM;
            try {
                const uint32_t id = ivf_pq_insert<float>(ix, b.heap, delta, v);
                IVFPQSearchResult r = ivf_pq_search<float>(ix, b.heap, v, 1, NLIST, 10, scratch, &delta);
                if (r.ids.size() != 1 || r.ids[0] != id || r.dists[0] != 0.0f) failures.fetch_add(1);
            } catch (...) {
                failures.fetch_add(1);
            }
        }
    };
    auto searcher = [&](uint32_t seed) {
        IVFPQSearchScratch scratch;
        std::mt19937 gen(seed);
        while (!stop.load()) {
            const float* q = queries.data() + size_t(gen() % NQ) * DIM;
            try {
                IVFPQSearchResult r = ivf_pq_search<float>(ix, b.heap, q, K, 4, 50, scratch, &delta);
                bool ok = r.ids.size() <= K && std::is_sorted(r.dists.begin(), r.dists.end());
                for (size_t i = 0; ok && i < r.ids.size(); ++i) {
                    ok = r.ids[i] < N + 2 * per_thread && std::count(r.ids.begin(), r.ids.end(), r.ids[i]) == 1;
                }
                if (!ok) failures.fetch_add(1);
                searches.fetch_add(1);
            } catch (...) {
                failures.fetch_add(1);
            }
        }
    };
    std::vector<std::thread> threads;
    for (uint32_t s = 0; s < 4; ++s) threads.emplace_back(searcher, 100 + s);
    std::thread a(inserter, 0), c(inserter, 1);
    a.join();
    c.join();
    stop.store(true);
    for (auto& th : threads) th.join();

    t.check(failures.load() == 0, std::to_string(failures.load()) + " inserts or searches failed");
    t.check(ix.rid_table.rid.size() == N + 2 * per_thread, "not every concurrent insert got an id");
    t.check(searches.load() > 0, "no search ran during the inserts");
    std::cout << "  " << searches.load() << " searches ran alongside " << 2 * per_thread << " inserts" << std::endl;
    return t.done();
}

bool test_guards(Built& b, const std::vector<float>& extra) {
    TestCase t("insert guards: null vector, wrong element type, prepared insert that does not fit");
    IVFPQIndex ix = b.index;
    IVFPQDelta delta;
    const uint32_t slots_before = b.heap.next_flat_slot();
    t.expect_throw("null vector", [&] { ivf_pq_insert<float>(ix, b.heap, delta, nullptr); });
    std::vector<uint8_t> as_u8(DIM, 1);
    t.expect_throw("uint8 vector into a float heap", [&] { ivf_pq_insert<uint8_t>(ix, b.heap, delta, as_u8.data()); });
    IVFPQPreparedInsert bad;
    bad.cluster = 0;
    bad.slot = 0;
    bad.code.assign(CHUNKS + 1, 0);
    t.expect_throw("code of the wrong length", [&] { ivf_pq_publish_insert(ix, delta, std::move(bad)); });
    IVFPQPreparedInsert far;
    far.cluster = NLIST;
    far.code.assign(CHUNKS, 0);
    t.expect_throw("cluster >= nlist", [&] { ivf_pq_publish_insert(ix, delta, std::move(far)); });
    t.check(ix.rid_table.rid.size() == N && delta.codes.codes.empty() && delta.lists.pending_inserts.empty() &&
                b.heap.next_flat_slot() == slots_before,
            "a rejected insert left something behind");

    // A prepared insert is invisible until published, then visible.
    IVFPQSearchScratch scratch;
    IVFPQPreparedInsert p = ivf_pq_prepare_insert<float>(ix, b.heap, delta, extra.data());
    IVFPQSearchResult before = ivf_pq_search<float>(ix, b.heap, extra.data(), 1, NLIST, 10, scratch, &delta);
    t.check(before.ids.size() == 1 && before.ids[0] < N && b.heap.next_flat_slot() == slots_before + 1,
            "a prepared but unpublished insert was found, or took no slot");
    uint32_t id;
    {
        std::unique_lock<std::shared_mutex> lock(delta.mtx);
        id = ivf_pq_publish_insert(ix, delta, std::move(p));
    }
    IVFPQSearchResult after = ivf_pq_search<float>(ix, b.heap, extra.data(), 1, NLIST, 10, scratch, &delta);
    t.check(id == N && after.ids.size() == 1 && after.ids[0] == N, "the published insert was not found");
    return t.done();
}

}  // namespace

int main() {
    const std::string prefix_f = temp_path("ivf_pq_mutate_f32");
    const std::string prefix_u = temp_path("ivf_pq_mutate_u8");
    bool all_pass = true;
    try {
        std::vector<float> base_f = draw<float>(N, 1), extra_f = draw<float>(NI, 3);
        std::vector<uint8_t> base_u = draw<uint8_t>(N, 1), extra_u = draw<uint8_t>(NI, 3);
        std::vector<float> queries = draw<float>(NQ, 2);
        std::unique_ptr<Built> f = build<float>(prefix_f, base_f);
        std::unique_ptr<Built> u = build<uint8_t>(prefix_u, base_u);

        all_pass &= test_partition_and_code_match_the_build<float>("f32", *f, base_f);
        all_pass &= test_partition_and_code_match_the_build<uint8_t>("u8", *u, base_u);
        all_pass &= test_inserted_vectors_are_found<float>("f32", *f, base_f, extra_f, queries);
        all_pass &= test_inserted_vectors_are_found<uint8_t>("u8", *u, base_u, extra_u, queries);
        all_pass &= test_inserts_race_searches(*f, queries);
        all_pass &= test_guards(*f, extra_f);

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
