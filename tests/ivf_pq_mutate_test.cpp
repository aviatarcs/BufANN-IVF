// Tests for ivf_pq_insert and ivf_pq_delete against oracles they do not
// share code with: the build's own cluster assignments and PQ codes
// (upstream's GEMM-based encoders), brute force over the live base +
// inserted vectors, a PQ-only reference computed here from both code
// tables, and the delta's and heap's contents read back directly. Also
// inserts and deletes racing searches, and the guards.

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
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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
// not shared with a neighbouring position. `want` may carry one entry more
// than `got`: the runner-up, which only serves the tie test of the last
// position (uint8 data has integer distances, so ties there are common).
bool same_within_ties(const IVFPQSearchResult& got, const IVFPQSearchResult& want) {
    if (want.ids.size() != got.ids.size() && want.ids.size() != got.ids.size() + 1) return false;
    for (size_t i = 0; i < got.ids.size(); ++i) {
        if (!close(got.dists[i], want.dists[i])) return false;
        bool tied = (i > 0 && close(want.dists[i], want.dists[i - 1])) ||
                    (i + 1 < want.ids.size() && close(want.dists[i], want.dists[i + 1]));
        if (!tied && got.ids[i] != want.ids[i]) return false;
    }
    return true;
}

// Exact top-k over the rows of `all` (row i is vector id i) not marked dead.
IVFPQSearchResult brute_force(const std::vector<float>& all, const std::vector<uint8_t>& dead, const float* query,
                              uint32_t k) {
    const size_t n = all.size() / DIM;
    std::vector<float> exact(n, std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < n; ++i) {
        if (!dead[i]) exact[i] = sq_dist(query, all.data() + size_t(i) * DIM, DIM);
    }
    IVFPQSearchResult want;
    for (uint32_t i : top_k(exact, k)) {
        if (dead[i]) break;
        want.ids.push_back(i), want.dists.push_back(exact[i]);
    }
    return want;
}

// PQ-only top-k over the live vectors, from the base code table and the
// delta's codes.
IVFPQSearchResult pq_reference(const IVFPQIndex& ix, const IVFPQDelta& delta, const std::vector<uint8_t>& dead,
                               const float* query, uint32_t k) {
    const PQMetadata& pq = ix.pq;
    std::vector<float> table(size_t(CHUNKS) * pq.k);
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        for (uint32_t j = 0; j < pq.k; ++j) {
            table[size_t(c) * pq.k + j] = sq_dist(query + c * pq.chunk_dim,
                                                  pq.pivots.data() + (size_t(c) * pq.k + j) * pq.chunk_dim, pq.chunk_dim);
        }
    }
    const size_t n = ix.rid_table.rid.size();
    std::vector<float> approx(n, std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < n; ++i) {
        if (dead[i]) continue;
        const uint8_t* c = i < N ? pq.codes.data() + size_t(i) * CHUNKS : delta.codes.codes.at(i).data();
        float d = 0.0f;
        for (uint32_t k = 0; k < CHUNKS; ++k) d += table[size_t(k) * pq.k + c[k]];
        approx[i] = d;
    }
    IVFPQSearchResult want;
    for (uint32_t i : top_k(approx, k)) {
        if (dead[i]) break;
        want.ids.push_back(i), want.dists.push_back(approx[i]);
    }
    return want;
}

struct Built {
    IVFPQIndex index;
    RawVectorHeap heap;
    IVFPQDelta delta;
};

// A private copy of the built heap for a test that frees and refills slots:
// the tests share `Built` and each starts from a copy of its index, whose
// base ids would otherwise address slots an earlier test refilled.
struct HeapCopy {
    std::string path;
    RawVectorHeap heap;
    HeapCopy(const std::string& prefix, const RawVectorHeap& shared, const std::string& name) : path(prefix + "_" + name + "_heap.bin") {
        std::filesystem::copy_file(ivf_raw_vectors_path(prefix), path, std::filesystem::copy_options::overwrite_existing);
        heap.open_existing(path, shared.layout(), shared.next_flat_slot(), shared.allocated_pages());
    }
    ~HeapCopy() {
        heap.close();
        ::unlink(path.c_str());
    }
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
    const std::vector<uint8_t> none_dead(N + NI, 0);
    size_t inserted_in_topk = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        const float* query = queries.data() + size_t(q) * DIM;
        IVFPQSearchResult want = brute_force(all, none_dead, query, K + 1);
        IVFPQSearchResult got = ivf_pq_search<T>(ix, b.heap, query, K, NLIST, N + NI, scratch, &delta);
        if (!t.check(same_within_ties(got, want),
                     "query " + std::to_string(q) + ": full probe over base + inserts is not the exact top-k")) {
            return t.done();
        }
        for (uint32_t id : got.ids) inserted_in_topk += id >= N;

        IVFPQSearchResult want_pq = pq_reference(ix, delta, none_dead, query, K + 1);
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

// Deletes every 7th base vector and every 3rd insert, then checks what
// ivf_pq_delete left behind (RID, heap bit, delta), that no search returns a
// deleted id, that full-probe searches are brute force / the PQ reference
// over the live vectors, that a query whose whole PQ shortlist was deleted
// still gets k live answers, and that freed slots are reused by inserts only
// once no reader can still address them.
template<typename T>
bool test_deleted_vectors_are_gone(const std::string& tag, const std::string& prefix, Built& b,
                                   const std::vector<T>& base, const std::vector<T>& extra,
                                   const std::vector<float>& queries) {
    TestCase t(tag + ": deleted vectors leave the RID table, heap and delta, and never come back from a search");
    IVFPQIndex ix = b.index;
    IVFPQDelta delta;
    HeapCopy own(prefix, b.heap, "delete");
    RawVectorHeap& heap = own.heap;
    for (uint32_t i = 0; i < NI; ++i) ivf_pq_insert<T>(ix, heap, delta, extra.data() + size_t(i) * DIM);
    std::vector<float> all = to_float(base.data(), size_t(N) * DIM);
    const std::vector<float> extraf = to_float(extra.data(), size_t(NI) * DIM);
    all.insert(all.end(), extraf.begin(), extraf.end());

    std::vector<uint8_t> dead(N + NI, 0);
    std::vector<uint32_t> deleted, deleted_slot;
    for (uint32_t id = 0; id < N + NI; ++id) {
        if ((id < N && id % 7 == 0) || (id >= N && (id - N) % 3 == 0)) {
            deleted.push_back(id);
            deleted_slot.push_back(rid_flat_slot(ix.rid_table.rid[id]));
            dead[id] = 1;
        }
    }
    const uint32_t slots_before = heap.next_flat_slot();
    // Registered before the first free, so every slot freed below stays out
    // of reach of inserts until it is released (checked at the end).
    std::optional<RawVectorHeap::ReadGuard> reader;
    reader.emplace(heap);
    for (uint32_t id : deleted) ivf_pq_delete(ix, heap, delta, id);
    size_t base_deleted = 0;
    for (uint32_t id : deleted) base_deleted += id < N;

    // What the delete wrote.
    size_t retracted = 0, still_listed = 0;
    for (size_t i = 0; i < deleted.size(); ++i) {
        const uint32_t id = deleted[i];
        const RawVectorRID rid = ix.rid_table.rid[id];
        retracted += !rid_is_active(rid) && rid_flat_slot(rid) == deleted_slot[i] && !heap.is_slot_occupied(deleted_slot[i]);
        if (id < N) {
            still_listed += delta.lists.tombstones.count(id) == 0;
        } else {
            auto pending = delta.lists.pending_inserts.find(ix.assignments.cluster_id[id]);
            still_listed += delta.lists.tombstones.count(id) != 0 || delta.codes.codes.count(id) != 0 ||
                            (pending != delta.lists.pending_inserts.end() &&
                             std::count(pending->second.begin(), pending->second.end(), id) != 0);
        }
    }
    t.check(retracted == deleted.size(), "a deleted vector is still active, moved slot, or still occupies its slot");
    t.check(still_listed == 0, "a deleted vector is still listed in the delta, or an insert was tombstoned");
    t.check(delta.lists.tombstones.size() == base_deleted && delta.codes.codes.size() == NI - (deleted.size() - base_deleted),
            "tombstones or delta codes have the wrong size");
    size_t live_ok = 0;
    for (uint32_t id = 0; id < N + NI; ++id) {
        if (dead[id]) continue;
        const RawVectorRID rid = ix.rid_table.rid[id];
        live_ok += rid_is_active(rid) && heap.is_slot_occupied(rid_flat_slot(rid));
    }
    t.check(live_ok == N + NI - deleted.size(), "a live vector was disturbed by the deletes");
    t.check(ix.rid_table.rid.size() == N + NI && ix.assignments.cluster_id.size() == N + NI &&
                heap.next_flat_slot() == slots_before,
            "a delete changed the vector count or the heap cursor");

    // A search for a deleted vector returns the nearest live one instead
    // (every 4th of them: the full re-rank makes each query cost N heap reads).
    IVFPQSearchScratch scratch;
    size_t came_back = 0, not_nearest_live = 0;
    for (size_t i = 0; i < deleted.size(); i += 4) {
        const uint32_t id = deleted[i];
        const float* x = all.data() + size_t(id) * DIM;
        IVFPQSearchResult r = ivf_pq_search<T>(ix, heap, x, 1, NLIST, N + NI, scratch, &delta);
        came_back += r.ids.size() != 1 || dead[r.ids[0]];
        not_nearest_live += !same_within_ties(r, brute_force(all, dead, x, 2));
    }
    t.check(came_back == 0, std::to_string(came_back) + " searches for a deleted vector returned it (or nothing)");
    t.check(not_nearest_live == 0, std::to_string(not_nearest_live) + " searches did not return the nearest live vector");

    // Full probe over the live vectors, re-ranked (brute force) and PQ-only
    // (reference), and the batched path.
    for (uint32_t q = 0; q < NQ; ++q) {
        const float* query = queries.data() + size_t(q) * DIM;
        IVFPQSearchResult got = ivf_pq_search<T>(ix, heap, query, K, NLIST, N + NI, scratch, &delta);
        if (!t.check(same_within_ties(got, brute_force(all, dead, query, K + 1)),
                     "query " + std::to_string(q) + ": full probe over the live vectors is not the exact top-k")) {
            return t.done();
        }
        IVFPQSearchResult got_pq = ivf_pq_search<T>(ix, heap, query, K, NLIST, 0, scratch, &delta);
        if (!t.check(same_within_ties(got_pq, pq_reference(ix, delta, dead, query, K + 1)),
                     "query " + std::to_string(q) + ": PQ-only ranking over the live vectors is not the reference")) {
            return t.done();
        }
    }
    std::vector<IVFPQSearchResult> batch =
        ivf_pq_search_batch<T>(ix, heap, queries.data(), NQ, K, NLIST, N + NI, IVF_PQ_SEARCH_GEMM_ROWS, &delta);
    size_t batch_mismatches = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        batch_mismatches += !same_within_ties(batch[q], brute_force(all, dead, queries.data() + size_t(q) * DIM, K + 1));
    }
    t.check(batch_mismatches == 0, std::to_string(batch_mismatches) + " batched results are not brute force over the live vectors");

    // Delete a query's entire PQ shortlist (its 100 nearest by PQ distance):
    // the search must still fill k from the live vectors behind them, which
    // it cannot if deleted candidates are only dropped after selection.
    const uint32_t M = 100;
    const float* crowded = queries.data();
    IVFPQSearchResult nearest = ivf_pq_search<T>(ix, heap, crowded, M, NLIST, 0, scratch, &delta);
    t.check(nearest.ids.size() == M, "the crowding query did not find M candidates");
    for (uint32_t id : nearest.ids) ivf_pq_delete(ix, heap, delta, id), dead[id] = 1;
    IVFPQSearchResult behind_pq = ivf_pq_search<T>(ix, heap, crowded, K, NLIST, 0, scratch, &delta);
    IVFPQSearchResult behind = ivf_pq_search<T>(ix, heap, crowded, K, NLIST, K, scratch, &delta);
    t.check(same_within_ties(behind_pq, pq_reference(ix, delta, dead, crowded, K + 1)),
            "after deleting the PQ shortlist, the PQ-only search is not the reference over the live vectors");
    size_t behind_live = 0;
    for (uint32_t id : behind.ids) behind_live += !dead[id];
    t.check(behind.ids.size() == K && behind_live == K,
            "after deleting the PQ shortlist, a re-ranked search with rerank_m = k returned " +
                std::to_string(behind_live) + " live vectors, not " + std::to_string(K));

    // Freed slots are reused by inserts, but not while a reader registered
    // before the free is still around.
    std::vector<uint32_t> all_freed = deleted_slot;
    for (uint32_t id : nearest.ids) all_freed.push_back(rid_flat_slot(ix.rid_table.rid[id]));
    const uint32_t freed = uint32_t(all_freed.size());
    std::vector<T> fresh = draw<T>(freed + 1, 17);
    const uint32_t held_id = ivf_pq_insert<T>(ix, heap, delta, fresh.data() + size_t(freed) * DIM);
    t.check(held_id == N + NI && heap.next_flat_slot() == slots_before + 1 &&
                std::count(all_freed.begin(), all_freed.end(), rid_flat_slot(ix.rid_table.rid[held_id])) == 0,
            "an insert reused a freed slot while a reader from before the free was registered");
    reader.reset();
    std::vector<uint32_t> reused;
    for (uint32_t i = 0; i < freed; ++i) {
        const uint32_t id = ivf_pq_insert<T>(ix, heap, delta, fresh.data() + size_t(i) * DIM);
        reused.push_back(rid_flat_slot(ix.rid_table.rid[id]));
    }
    std::sort(reused.begin(), reused.end());
    std::sort(all_freed.begin(), all_freed.end());
    t.check(heap.next_flat_slot() == slots_before + 1 && reused == all_freed,
            "the inserts after the reader left did not take exactly the freed slots");
    std::vector<T> raw(DIM);
    size_t refilled = 0;
    for (uint32_t i = 0; i < freed; ++i) {
        const uint32_t id = held_id + 1 + i;
        heap.read_vector(rid_flat_slot(ix.rid_table.rid[id]), raw.data());
        const std::vector<float> x = to_float(fresh.data() + size_t(i) * DIM, DIM);
        IVFPQSearchResult r = ivf_pq_search<T>(ix, heap, x.data(), 1, NLIST, 10, scratch, &delta);
        refilled += std::memcmp(raw.data(), fresh.data() + size_t(i) * DIM, DIM * sizeof(T)) == 0 &&
                    r.ids.size() == 1 && r.ids[0] == id && r.dists[0] == 0.0f;
    }
    t.check(refilled == freed, "a vector inserted into a freed slot has the wrong bytes or is not found");
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

// One thread deletes base vectors and inserts fresh ones (which refill the
// freed slots once reclaimable) while four search with a re-rank. Every
// returned distance must be the exact distance to the returned id's vector:
// a slot refilled under a search's re-rank would show as the replacement's
// distance under the old id. A vector whose delete returned before a search
// began must not be returned by it.
bool test_churn_races_searches(const std::string& prefix, Built& b, const std::vector<float>& base,
                               const std::vector<float>& queries) {
    TestCase t("f32: deletes and slot-refilling inserts concurrent with searches never surface a deleted id or stale bytes");
    IVFPQIndex ix = b.index;
    IVFPQDelta delta;
    HeapCopy own(prefix, b.heap, "churn");
    RawVectorHeap& heap = own.heap;
    const uint32_t churn = 1500;
    std::vector<float> all = base;
    all.resize(size_t(N + churn) * DIM);
    const std::vector<float> fresh = draw<float>(churn, 13);
    // deleted_seq[id] is the delete's position in the churn thread's order,
    // 0 for a live vector; seq counts the deletes that have returned.
    std::vector<std::atomic<uint32_t>> deleted_seq(N + churn);
    std::atomic<uint32_t> seq{0};
    std::atomic<bool> stop{false};
    std::atomic<size_t> failures{0}, searches{0}, refills{0};
    std::mt19937 pick(5);
    std::vector<uint32_t> victims(N);
    std::iota(victims.begin(), victims.end(), 0u);
    std::shuffle(victims.begin(), victims.end(), pick);
    victims.resize(churn);

    auto churner = [&] {
        std::vector<uint8_t> freed(heap.allocated_pages() * size_t(heap.layout().slots_per_page) + churn, 0);
        for (uint32_t i = 0; i < churn; ++i) {
            try {
                const uint32_t victim = victims[i];
                freed[rid_flat_slot(ix.rid_table.rid[victim])] = 1;
                ivf_pq_delete(ix, heap, delta, victim);
                deleted_seq[victim].store(seq.fetch_add(1) + 1);
                const uint32_t id = N + i;
                std::copy_n(fresh.data() + size_t(i) * DIM, DIM, all.data() + size_t(id) * DIM);
                if (ivf_pq_insert<float>(ix, heap, delta, all.data() + size_t(id) * DIM) != id) failures.fetch_add(1);
                // Not necessarily the slot just freed: a search from before
                // the free may still hold that one back.
                refills.fetch_add(freed[rid_flat_slot(load_rid(ix.rid_table.rid[id]))]);
            } catch (...) {
                failures.fetch_add(1);
            }
        }
    };
    auto searcher = [&](uint32_t s) {
        IVFPQSearchScratch scratch;
        std::mt19937 gen(s);
        while (!stop.load()) {
            const float* q = queries.data() + size_t(gen() % NQ) * DIM;
            const uint32_t before = seq.load();
            try {
                IVFPQSearchResult r = ivf_pq_search<float>(ix, heap, q, K, 4, 50, scratch, &delta);
                bool ok = r.ids.size() <= K && std::is_sorted(r.dists.begin(), r.dists.end());
                for (size_t i = 0; ok && i < r.ids.size(); ++i) {
                    const uint32_t id = r.ids[i];
                    if (!(ok = id < N + churn && std::count(r.ids.begin(), r.ids.end(), id) == 1)) break;
                    const uint32_t deleted_at = deleted_seq[id].load();
                    ok = (deleted_at == 0 || deleted_at > before) &&
                         close(r.dists[i], sq_dist(q, all.data() + size_t(id) * DIM, DIM));
                }
                if (!ok) failures.fetch_add(1);
                searches.fetch_add(1);
            } catch (...) {
                failures.fetch_add(1);
            }
        }
    };
    std::vector<std::thread> threads;
    for (uint32_t s = 0; s < 4; ++s) threads.emplace_back(searcher, 200 + s);
    std::thread c(churner);
    c.join();
    stop.store(true);
    for (auto& th : threads) th.join();

    t.check(failures.load() == 0, std::to_string(failures.load()) + " mutations or searches failed");
    t.check(ix.rid_table.rid.size() == N + churn, "not every insert got an id");
    t.check(searches.load() > 0, "no search ran during the churn");
    t.check(refills.load() > 0, "no insert refilled a freed slot; the grace period was not exercised");
    std::cout << "  " << searches.load() << " searches ran alongside " << churn << " deletes and inserts, " << refills.load()
              << " of which refilled a freed slot" << std::endl;
    return t.done();
}

bool test_guards(const std::string& prefix, Built& b, const std::vector<float>& extra) {
    TestCase t("mutation guards: null vector, wrong element type, prepared insert that does not fit, bad deletes");
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

    HeapCopy own(prefix, b.heap, "guards");
    t.expect_throw("delete of an id past the table", [&] { ivf_pq_delete(ix, own.heap, delta, N + 1); });
    ivf_pq_delete(ix, own.heap, delta, 3);
    t.expect_throw("delete of a deleted id", [&] { ivf_pq_delete(ix, own.heap, delta, 3); });
    t.check(delta.lists.tombstones.size() == 1 && delta.free_list.deferred.size() == 1 &&
                !rid_is_active(ix.rid_table.rid[3]) && rid_is_active(ix.rid_table.rid[N]),
            "a rejected delete changed the delta, or the accepted one did not");
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
        all_pass &= test_deleted_vectors_are_gone<float>("f32", prefix_f, *f, base_f, extra_f, queries);
        all_pass &= test_deleted_vectors_are_gone<uint8_t>("u8", prefix_u, *u, base_u, extra_u, queries);
        all_pass &= test_inserts_race_searches(*f, queries);
        all_pass &= test_churn_races_searches(prefix_f, *f, base_f, queries);
        all_pass &= test_guards(prefix_f, *f, extra_f);

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
