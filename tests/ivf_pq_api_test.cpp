// IVF-PQ through the public BufANN API: bufann_build with
// IndexType::IvfPq, bufann_query / bufann_query_into, bufann_load,
// bufann_insert / bufann_delete with caller tags, the search_L-as-nprobe
// convention, config validation, and the graph-only operations refusing an
// IVF-PQ index.

#include "bufann/bufann_api.h"
#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_search.h"
#include "ivf_pq_test_util.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t N = 20000, NQ = 100, DIM = 32, NLIST = 32, CHUNKS = 8, K = 10;

template<typename T>
T clamp_to(float v) {
    float lo = float(std::numeric_limits<T>::lowest()), hi = float(std::numeric_limits<T>::max());
    return T(std::round(std::min(hi, std::max(lo, v))));
}
template<>
float clamp_to<float>(float v) { return v; }

// 8 intrinsic dimensions embedded in DIM (see the recall test for why).
template<typename T>
std::vector<T> synthetic(size_t count, uint32_t seed, float scale) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::mt19937 fixed(3);
    const uint32_t intrinsic = 8, clusters = 64;
    std::vector<float> centers(size_t(clusters) * intrinsic), embed(size_t(intrinsic) * DIM);
    for (float& v : centers) v = 20.0f * unit(fixed);
    for (float& v : embed) v = unit(fixed) / std::sqrt(float(intrinsic));
    std::vector<T> out(count * DIM);
    std::vector<float> z(intrinsic);
    for (size_t i = 0; i < count; ++i) {
        const float* c = centers.data() + size_t(gen() % clusters) * intrinsic;
        for (uint32_t k = 0; k < intrinsic; ++k) z[k] = c[k] + 4.0f * unit(gen);
        for (uint32_t d = 0; d < DIM; ++d) {
            float v = 0.5f * unit(gen);
            for (uint32_t k = 0; k < intrinsic; ++k) v += z[k] * embed[size_t(k) * DIM + d];
            out[i * DIM + d] = clamp_to<T>(scale * v + (std::is_same<T, uint8_t>::value ? 128.0f : 0.0f));
        }
    }
    return out;
}

template<typename T>
BufANNConfig ivf_config() {
    BufANNConfig c;
    c.dim = DIM;
    c.index_type = IndexType::IvfPq;
    c.ivf_nlist = NLIST;
    c.ivf_nprobe = 4;
    c.ivf_pq_chunks = CHUNKS;
    return c;
}

template<typename T>
float sq_dist(const T* a, const T* q) {
    float s = 0.0f;
    for (uint32_t j = 0; j < DIM; ++j) s += (float(a[j]) - float(q[j])) * (float(a[j]) - float(q[j]));
    return s;
}

// Rows of `base` (any count) nearest to q, as row indices, skipping the
// rows marked in `dead`.
template<typename T>
std::vector<TagType> brute_force(const std::vector<T>& base, const T* q, uint32_t k,
                                 const std::vector<uint8_t>& dead = {}) {
    const uint32_t n = uint32_t(base.size() / DIM);
    std::vector<float> d(n);
    for (uint32_t i = 0; i < n; ++i) {
        d[i] = i < dead.size() && dead[i] ? std::numeric_limits<float>::infinity() : sq_dist(base.data() + size_t(i) * DIM, q);
    }
    std::vector<TagType> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](TagType a, TagType b) { return d[a] < d[b]; });
    order.resize(k);
    return order;
}

template<typename T>
bool test_build_query_load(const std::string& tag, const std::string& prefix) {
    TestCase t(tag + ": bufann_build/query/load with IndexType::IvfPq");
    const std::string base_bin = prefix + "_base.bin";
    std::vector<T> base = synthetic<T>(N, 1, 1.0f), queries = synthetic<T>(NQ, 2, 1.0f);
    diskann::save_bin<T>(base_bin, base.data(), N, DIM);
    BufANNConfig cfg = ivf_config<T>();

    BufANNIndex<T>* idx = bufann_build<T>(base_bin, prefix, cfg);
    t.check(idx != nullptr && idx->ivf != nullptr, "build did not produce an IVF-PQ backend");

    // The API result equals a direct ivf_pq_search with the same parameters
    // (nprobe from config, rerank_m = max(100, 10k)), and is a strong top-k.
    size_t api_hits = 0, mismatches = 0;
    IVFPQSearchScratch scratch;
    std::vector<float> qf(DIM);
    for (uint32_t q = 0; q < NQ; ++q) {
        const T* query = queries.data() + size_t(q) * DIM;
        std::vector<TagType> got = bufann_query<T>(*idx, query, K);
        for (uint32_t d = 0; d < DIM; ++d) qf[d] = float(query[d]);
        IVFPQSearchResult direct = ivf_pq_search<T>(idx->ivf->index, idx->ivf->heap, qf.data(), K, cfg.ivf_nprobe,
                                                    std::max<uint32_t>(100, 10 * K), scratch);
        mismatches += !(got.size() == direct.ids.size() && std::equal(got.begin(), got.end(), direct.ids.begin()));
        std::vector<TagType> truth = brute_force<T>(base, query, K);
        for (TagType id : got) api_hits += std::find(truth.begin(), truth.end(), id) != truth.end();
    }
    t.check(mismatches == 0, std::to_string(mismatches) + " queries differ from a direct ivf_pq_search");
    const float recall = float(api_hits) / float(NQ * K);
    t.check(recall >= 0.85f, "recall@10 through the API is only " + std::to_string(recall));

    // search_L is nprobe: probing everything with a full re-rank must beat
    // the default, and bufann_query_into with a null buffer just counts.
    size_t full_hits = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        const T* query = queries.data() + size_t(q) * DIM;
        std::vector<TagType> got = bufann_query<T>(*idx, query, K, NLIST);
        std::vector<TagType> truth = brute_force<T>(base, query, K);
        for (TagType id : got) full_hits += std::find(truth.begin(), truth.end(), id) != truth.end();
    }
    t.check(full_hits >= api_hits, "nprobe = nlist via search_L did not reach the default's recall");
    t.check(bufann_query_into<T>(*idx, queries.data(), K, nullptr) == K, "query_into with nullptr did not return K");

    // Graph-only operations refuse the index; flushes are no-ops.
    t.expect_throw_any("bufann_cleanup_deleted_edges", [&] { bufann_cleanup_deleted_edges<T>(*idx); });
    bufann_flush_dirty<T>(*idx);
    t.check(bufann_flush_dirty_budget<T>(*idx, 0) == 0, "flush budget on IVF-PQ should flush nothing");

    // Reload from disk and get identical answers.
    bufann_free<T>(idx);
    t.check(idx == nullptr, "bufann_free did not null the pointer");
    BufANNIndex<T>* loaded = bufann_load<T>(prefix, cfg);
    size_t reload_mismatches = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        const T* query = queries.data() + size_t(q) * DIM;
        std::vector<TagType> a = bufann_query<T>(*loaded, query, K);
        for (uint32_t d = 0; d < DIM; ++d) qf[d] = float(query[d]);
        IVFPQSearchResult direct = ivf_pq_search<T>(loaded->ivf->index, loaded->ivf->heap, qf.data(), K,
                                                    cfg.ivf_nprobe, std::max<uint32_t>(100, 10 * K), scratch);
        reload_mismatches += !(a.size() == direct.ids.size() && std::equal(a.begin(), a.end(), direct.ids.begin()));
    }
    t.check(reload_mismatches == 0, "reloaded index answers differently");

    // Inserts under caller tags (distinct from the base rows): each is its
    // own nearest neighbour by tag, and a full probe over base + inserts is
    // the brute-force top-k, compared by distance so ties do not matter.
    const uint32_t NI = 200;
    const TagType FIRST_TAG = 1000000;
    std::vector<T> extra = synthetic<T>(NI, 5, 1.0f);
    for (uint32_t i = 0; i < NI; ++i) bufann_insert<T>(*loaded, FIRST_TAG + i, extra.data() + size_t(i) * DIM);
    size_t self_hits = 0;
    for (uint32_t i = 0; i < NI; ++i) {
        std::vector<TagType> got = bufann_query<T>(*loaded, extra.data() + size_t(i) * DIM, 1, NLIST);
        self_hits += got.size() == 1 && got[0] == FIRST_TAG + i;
    }
    t.check(self_hits == NI, std::to_string(NI - self_hits) + " inserted vectors are not their own nearest neighbour");
    std::vector<T> combined = base;
    combined.insert(combined.end(), extra.begin(), extra.end());
    auto row_of = [&](TagType tag) -> const T* {
        return tag >= FIRST_TAG ? extra.data() + size_t(tag - FIRST_TAG) * DIM : base.data() + size_t(tag) * DIM;
    };
    loaded->config.ivf_rerank_m = N + NI;  // exact: every candidate is re-ranked
    size_t insert_mismatches = 0, inserted_returned = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        const T* query = queries.data() + size_t(q) * DIM;
        std::vector<TagType> got = bufann_query<T>(*loaded, query, K, NLIST);
        std::vector<TagType> truth = brute_force<T>(combined, query, K);
        bool ok = got.size() == K;
        for (uint32_t i = 0; ok && i < K; ++i) {
            const float want = sq_dist(combined.data() + size_t(truth[i]) * DIM, query);
            const float have = sq_dist(row_of(got[i]), query);
            ok = std::fabs(have - want) <= 1e-5f * std::max(have, want) &&
                 std::count(got.begin(), got.end(), got[i]) == 1;
            inserted_returned += got[i] >= FIRST_TAG;
        }
        insert_mismatches += !ok;
    }
    t.check(insert_mismatches == 0, std::to_string(insert_mismatches) + " queries are not brute force over base + inserts");
    t.check(inserted_returned > 0, "no inserted tag was ever returned");
    // Concurrent inserts publish out of id order; every tag must still
    // resolve to its own vector.
    const uint32_t NC = 800, CT = 8;
    const TagType CONC_TAG = FIRST_TAG + NI;
    std::vector<T> conc = synthetic<T>(NC, 9, 1.0f);
    {
        std::vector<std::thread> threads;
        for (uint32_t th = 0; th < CT; ++th) {
            threads.emplace_back([&, th] {
                for (uint32_t i = th; i < NC; i += CT) bufann_insert<T>(*loaded, CONC_TAG + i, conc.data() + size_t(i) * DIM);
            });
        }
        for (auto& th : threads) th.join();
    }
    size_t conc_self = 0;
    loaded->config.ivf_rerank_m = 0;  // the default; a full re-rank per self-lookup is slow and not needed
    for (uint32_t i = 0; i < NC; ++i) {
        std::vector<TagType> got = bufann_query<T>(*loaded, conc.data() + size_t(i) * DIM, 1, NLIST);
        conc_self += got.size() == 1 && got[0] == CONC_TAG + i;
    }
    loaded->config.ivf_rerank_m = N + NI;
    t.check(conc_self == NC, std::to_string(NC - conc_self) + " concurrently inserted vectors do not resolve to their tag");
    for (uint32_t i = 0; i < NC; ++i) bufann_delete<T>(*loaded, CONC_TAG + i);

    t.expect_throw_any("re-inserting an inserted tag", [&] { bufann_insert<T>(*loaded, FIRST_TAG, extra.data()); });
    t.expect_throw_any("inserting an active base tag", [&] { bufann_insert<T>(*loaded, 0, extra.data()); });
    t.expect_throw_any("inserting INVALID_TAG", [&] { bufann_insert<T>(*loaded, INVALID_TAG, extra.data()); });
    t.check(bufann_query<T>(*loaded, extra.data(), 1, NLIST) == std::vector<TagType>{FIRST_TAG},
            "a rejected insert disturbed the index");

    // Deletes by tag, single and batched: a deleted tag is never returned,
    // searches are brute force over the live vectors, and the tag is free to
    // insert again.
    std::vector<TagType> gone_base, gone_inserted;
    for (TagType tag = 0; tag < N; tag += 5) gone_base.push_back(tag);
    for (uint32_t i = 0; i < NI; i += 4) gone_inserted.push_back(FIRST_TAG + i);
    for (TagType tag : gone_base) bufann_delete<T>(*loaded, tag);
    bufann_delete_batch<T>(*loaded, gone_inserted.data(), gone_inserted.size());
    std::vector<uint8_t> dead(N + NI, 0);
    for (TagType tag : gone_base) dead[tag] = 1;
    for (TagType tag : gone_inserted) dead[N + (tag - FIRST_TAG)] = 1;
    auto row_index = [&](TagType tag) { return tag >= FIRST_TAG ? N + (tag - FIRST_TAG) : tag; };
    size_t returned_deleted = 0, delete_mismatches = 0;
    loaded->config.ivf_rerank_m = 0;  // the default: a full re-rank per query is slow, and not needed here
    for (size_t i = 0; i < gone_base.size(); i += 10) {
        std::vector<TagType> got = bufann_query<T>(*loaded, row_of(gone_base[i]), 1, NLIST);
        returned_deleted += got.size() != 1 || got[0] == gone_base[i];
    }
    for (TagType tag : gone_inserted) {
        std::vector<TagType> got = bufann_query<T>(*loaded, row_of(tag), 1, NLIST);
        returned_deleted += got.size() != 1 || got[0] == tag;
    }
    t.check(returned_deleted == 0, std::to_string(returned_deleted) + " deleted tags were returned (or nothing was)");
    loaded->config.ivf_rerank_m = N + NI;
    for (uint32_t q = 0; q < NQ; ++q) {
        const T* query = queries.data() + size_t(q) * DIM;
        std::vector<TagType> got = bufann_query<T>(*loaded, query, K, NLIST);
        std::vector<TagType> truth = brute_force<T>(combined, query, K, dead);
        bool ok = got.size() == K;
        for (uint32_t i = 0; ok && i < K; ++i) {
            const float want = sq_dist(combined.data() + size_t(truth[i]) * DIM, query);
            const float have = sq_dist(row_of(got[i]), query);
            ok = !dead[row_index(got[i])] && std::fabs(have - want) <= 1e-5f * std::max(have, want) &&
                 std::count(got.begin(), got.end(), got[i]) == 1;
        }
        delete_mismatches += !ok;
    }
    t.check(delete_mismatches == 0, std::to_string(delete_mismatches) + " queries are not brute force over the live vectors");
    {
        const TagType batch[] = {11, 21, gone_base[0], 31};  // gone_base[0] is already deleted
        t.expect_throw_any("delete_batch with a deleted tag in it", [&] { bufann_delete_batch<T>(*loaded, batch, 4); });
        t.expect_throw_any("tag before the bad one was deleted", [&] { bufann_delete<T>(*loaded, 11); });
        t.expect_throw_any("tag before the bad one was deleted", [&] { bufann_delete<T>(*loaded, 21); });
        bufann_delete<T>(*loaded, 31);  // the one after it was not
        dead[11] = dead[21] = dead[31] = 1;
    }
    t.expect_throw_any("deleting a deleted base tag", [&] { bufann_delete<T>(*loaded, gone_base[0]); });
    t.expect_throw_any("deleting a deleted inserted tag", [&] { bufann_delete<T>(*loaded, gone_inserted[0]); });
    t.expect_throw_any("deleting a tag never inserted", [&] { bufann_delete<T>(*loaded, FIRST_TAG + NI); });
    t.expect_throw_any("deleting INVALID_TAG", [&] { bufann_delete<T>(*loaded, INVALID_TAG); });
    t.check(bufann_query<T>(*loaded, row_of(gone_base[1]), 1, NLIST)[0] != gone_base[1],
            "a rejected delete disturbed the index");

    // A deleted tag, base or inserted, resolves to the vector re-inserted
    // under it (two live inserts, deleted first so the query cannot tie).
    const T* moved_a = extra.data() + size_t(NI - 1) * DIM;
    const T* moved_b = extra.data() + size_t(NI - 2) * DIM;
    bufann_delete<T>(*loaded, FIRST_TAG + NI - 1);
    bufann_delete<T>(*loaded, FIRST_TAG + NI - 2);
    bufann_insert<T>(*loaded, gone_base[0], moved_a);
    bufann_insert<T>(*loaded, gone_inserted[0], moved_b);
    t.check(bufann_query<T>(*loaded, moved_a, 1, NLIST) == std::vector<TagType>{gone_base[0]} &&
                bufann_query<T>(*loaded, moved_b, 1, NLIST) == std::vector<TagType>{gone_inserted[0]},
            "a re-inserted tag does not resolve to its new vector");
    t.expect_throw_any("re-inserting a re-inserted tag", [&] { bufann_insert<T>(*loaded, gone_base[0], moved_a); });

    // The heap has outgrown what the index file records, so until the
    // rebuild rewrites the file a reload is refused rather than wrong.
    bufann_free<T>(loaded);
    t.expect_throw_any("reload after inserts", [&] { bufann_load<T>(prefix, cfg); });

    ::unlink(base_bin.c_str());
    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
    return t.done();
}

// Build, delete, free, load: the deletes hold and their slots are reused.
template<typename T>
bool test_deletes_survive_reload(const std::string& tag, const std::string& prefix) {
    TestCase t(tag + ": bufann_load recovers deletes from the heap's occupancy bitmap");
    const std::string base_bin = prefix + "_base.bin";
    std::vector<T> base = synthetic<T>(N, 1, 1.0f), queries = synthetic<T>(NQ, 2, 1.0f);
    diskann::save_bin<T>(base_bin, base.data(), N, DIM);
    BufANNConfig cfg = ivf_config<T>();
    cfg.ivf_rerank_m = N;

    BufANNIndex<T>* idx = bufann_build<T>(base_bin, prefix, cfg);
    std::vector<uint8_t> dead(N, 0);
    for (TagType tag = 3; tag < N; tag += 9) bufann_delete<T>(*idx, tag), dead[tag] = 1;
    const uint32_t slots = idx->ivf->heap.next_flat_slot();
    bufann_free<T>(idx);

    BufANNIndex<T>* loaded = bufann_load<T>(prefix, cfg);
    size_t mismatches = 0;
    for (uint32_t q = 0; q < NQ; ++q) {
        const T* query = queries.data() + size_t(q) * DIM;
        std::vector<TagType> got = bufann_query<T>(*loaded, query, K, NLIST);
        std::vector<TagType> truth = brute_force<T>(base, query, K, dead);
        bool ok = got.size() == K;
        for (uint32_t i = 0; ok && i < K; ++i) {
            const float want = sq_dist(base.data() + size_t(truth[i]) * DIM, query);
            const float have = sq_dist(base.data() + size_t(got[i]) * DIM, query);
            ok = !dead[got[i]] && std::fabs(have - want) <= 1e-5f * std::max(have, want);
        }
        mismatches += !ok;
    }
    t.check(mismatches == 0, std::to_string(mismatches) + " queries after the reload are not brute force over the live vectors");
    t.expect_throw_any("deleting a tag deleted before the reload", [&] { bufann_delete<T>(*loaded, 3); });
    bufann_insert<T>(*loaded, 3, queries.data());
    t.check(loaded->ivf->heap.next_flat_slot() == slots && bufann_query<T>(*loaded, queries.data(), 1, NLIST) == std::vector<TagType>{3},
            "re-inserting a deleted tag after the reload grew the heap or is not found");
    // That insert did not grow the heap, so only the slot's owner tells a
    // load that the file is behind; the load must be refused, not lose it.
    bufann_free<T>(loaded);
    t.expect_throw_any("reload after an insert into a freed slot", [&] { bufann_load<T>(prefix, cfg); });

    ::unlink(base_bin.c_str());
    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
    return t.done();
}

// Four threads insert, delete and re-insert vectors under a shared pool of
// tags while four query: no result may be INVALID_TAG or a tag outside the
// pool, and afterwards every tag's state must match what its last mutation
// left: live tags are their own nearest neighbour, dead ones refuse a delete.
template<typename T>
bool test_tag_churn_under_queries(const std::string& tag_name, const std::string& prefix) {
    TestCase t(tag_name + ": concurrent inserts, deletes and queries by tag never surface a bad tag");
    const std::string base_bin = prefix + "_base.bin";
    std::vector<T> base = synthetic<T>(N, 1, 1.0f), queries = synthetic<T>(NQ, 2, 1.0f);
    diskann::save_bin<T>(base_bin, base.data(), N, DIM);
    BufANNIndex<T>* idx = bufann_build<T>(base_bin, prefix, ivf_config<T>());

    // Pool: base tags [0, POOL) and new tags [FIRST, FIRST + POOL); each
    // mutator owns a disjoint slice of both, so the oracle needs no lock.
    const uint32_t POOL = 2000, MUTATORS = 4, ROUNDS = 6;
    const TagType FIRST = 5000000;
    std::vector<T> fresh = synthetic<T>(POOL, 8, 1.0f);
    std::vector<uint8_t> live_base(POOL, 1), live_new(POOL, 0);
    std::atomic<bool> stop{false};
    std::atomic<size_t> failures{0}, searches{0};
    auto mutator = [&](uint32_t m) {
        std::mt19937 gen(m);
        try {
            for (uint32_t round = 0; round < ROUNDS; ++round) {
                for (uint32_t i = m; i < POOL; i += MUTATORS) {
                    // Toggle a base tag or a new tag; a new tag gets its own vector.
                    if (gen() % 2 == 0) {
                        if (live_base[i]) bufann_delete<T>(*idx, i);
                        else bufann_insert<T>(*idx, i, base.data() + size_t(i) * DIM);
                        live_base[i] = !live_base[i];
                    } else {
                        if (live_new[i]) bufann_delete<T>(*idx, FIRST + i);
                        else bufann_insert<T>(*idx, FIRST + i, fresh.data() + size_t(i) * DIM);
                        live_new[i] = !live_new[i];
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cout << "  mutator: " << e.what() << std::endl;
            failures.fetch_add(1);
        }
    };
    auto searcher = [&](uint32_t s) {
        std::mt19937 gen(100 + s);
        while (!stop.load()) {
            try {
                std::vector<TagType> got = bufann_query<T>(*idx, queries.data() + size_t(gen() % NQ) * DIM, K, 4);
                bool ok = got.size() <= K;
                for (TagType g : got) ok = ok && g != INVALID_TAG && (g < N || (g >= FIRST && g < FIRST + POOL));
                if (!ok) failures.fetch_add(1);
                searches.fetch_add(1);
            } catch (...) {
                failures.fetch_add(1);
            }
        }
    };
    std::vector<std::thread> threads;
    for (uint32_t s = 0; s < 4; ++s) threads.emplace_back(searcher, s);
    std::vector<std::thread> mutators;
    for (uint32_t m = 0; m < MUTATORS; ++m) mutators.emplace_back(mutator, m);
    for (auto& th : mutators) th.join();
    stop.store(true);
    for (auto& th : threads) th.join();
    t.check(failures.load() == 0, std::to_string(failures.load()) + " mutations or queries failed");
    t.check(searches.load() > 0, "no query ran during the churn");

    size_t wrong = 0;
    for (uint32_t i = 0; i < POOL; ++i) {
        for (int which = 0; which < 2; ++which) {
            const TagType tag = which == 0 ? i : FIRST + i;
            const T* vec = which == 0 ? base.data() + size_t(i) * DIM : fresh.data() + size_t(i) * DIM;
            const bool live = which == 0 ? live_base[i] : live_new[i];
            if (live) {
                std::vector<TagType> got = bufann_query<T>(*idx, vec, 1, NLIST);
                wrong += !(got.size() == 1 && got[0] == tag);
            } else {
                try {
                    bufann_delete<T>(*idx, tag);
                    ++wrong;
                } catch (const std::invalid_argument&) {
                }
            }
        }
    }
    t.check(wrong == 0, std::to_string(wrong) + " tags do not match the state their last mutation left");
    std::cout << "  " << searches.load() << " queries ran alongside " << ROUNDS * POOL << " mutations" << std::endl;

    bufann_free<T>(idx);
    ::unlink(base_bin.c_str());
    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
    return t.done();
}

bool test_config_validation(const std::string& prefix) {
    TestCase t("IVF-PQ config validation and type mismatch on load");
    const std::string base_bin = prefix + "_base.bin";
    std::vector<float> base = synthetic<float>(N, 1, 1.0f);
    diskann::save_bin<float>(base_bin, base.data(), N, DIM);

    auto bad = [&](const char* what, auto&& mutate) {
        BufANNConfig c = ivf_config<float>();
        mutate(c);
        t.expect_throw_any(what, [&] { bufann_build<float>(base_bin, prefix, c); });
    };
    bad("ivf_nlist == 0", [](BufANNConfig& c) { c.ivf_nlist = 0; });
    bad("ivf_pq_chunks == 0", [](BufANNConfig& c) { c.ivf_pq_chunks = 0; });
    bad("dim % ivf_pq_chunks != 0", [](BufANNConfig& c) { c.ivf_pq_chunks = 5; });
    bad("config.dim != data dim", [](BufANNConfig& c) { c.dim = DIM + 1; });

    // nprobe is only needed at query time; 0 there is an error, search_L cures it.
    BufANNConfig c = ivf_config<float>();
    c.ivf_nprobe = 0;
    BufANNIndex<float>* idx = bufann_build<float>(base_bin, prefix, c);
    t.expect_throw_any("query with ivf_nprobe == 0 and search_L == 0", [&] { bufann_query<float>(*idx, base.data(), K); });
    t.check(bufann_query<float>(*idx, base.data(), K, 2).size() == K, "search_L did not stand in for nprobe");
    bufann_free<float>(idx);

    // A float index loaded as uint8 is refused (elem_size 4 * dim vs 1 * dim).
    t.expect_throw_any("load with the wrong T", [&] { bufann_load<uint8_t>(prefix, ivf_config<uint8_t>()); });
    t.expect_throw_any("load from a prefix with no index", [&] { bufann_load<float>(prefix + "_nope", ivf_config<float>()); });

    ::unlink(base_bin.c_str());
    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
    return t.done();
}

}  // namespace

int main() {
    const std::string prefix = temp_path("ivf_pq_api");
    bool all_pass = true;
    try {
        all_pass &= test_build_query_load<float>("float", prefix + "_f32");
        all_pass &= test_build_query_load<uint8_t>("uint8", prefix + "_u8");
        all_pass &= test_deletes_survive_reload<float>("float", prefix + "_reload");
        all_pass &= test_tag_churn_under_queries<float>("float", prefix + "_churn");
        all_pass &= test_config_validation(prefix + "_cfg");
    } catch (const std::exception& e) {
        std::cout << "  FAIL: " << e.what() << std::endl;
        all_pass = false;
    }
    return all_pass ? 0 : 1;
}
