// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bufann/inplace_graph_ops.h"
#include "bufann/inplace_backend.h"
#include "logger.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>

#include "tsl/robin_map.h"

namespace diskann {
namespace inplace {

namespace {

struct MaintenanceScratchCache {
    std::unique_ptr<InPlaceSearchScratch> scratch;
    uint32_t aligned_dim = 0;
    uint32_t n_chunks = 0;
    uint32_t elem_size = 0;
    uint32_t list_size = 0;
    uint32_t max_degree = 0;
    uint32_t beamwidth = 0;
};

InPlaceSearchScratch& maintenance_scratch_for(uint32_t aligned_dim,
                                              uint32_t n_chunks,
                                              uint32_t elem_size,
                                              uint32_t list_size,
                                              uint32_t max_degree,
                                              uint32_t beamwidth) {
    thread_local MaintenanceScratchCache cache;
    const bool needs_init =
        !cache.scratch || cache.aligned_dim != aligned_dim ||
        cache.n_chunks != n_chunks || cache.elem_size != elem_size ||
        cache.list_size != list_size || cache.max_degree != max_degree ||
        cache.beamwidth != beamwidth;
    if (needs_init) {
        cache.scratch.reset(new InPlaceSearchScratch());
        cache.scratch->init(aligned_dim, n_chunks, elem_size, list_size,
                            max_degree, beamwidth);
        cache.aligned_dim = aligned_dim;
        cache.n_chunks = n_chunks;
        cache.elem_size = elem_size;
        cache.list_size = list_size;
        cache.max_degree = max_degree;
        cache.beamwidth = beamwidth;
    } else {
        cache.scratch->reset();
    }
    return *cache.scratch;
}

static inline uint32_t page_bitmap_bytes(uint32_t slots_per_page) {
    return (slots_per_page + 7) / 8;
}

static inline uint32_t page_header_bytes(uint32_t slots_per_page) {
    return 8 + page_bitmap_bytes(slots_per_page);
}

struct RepairIOPhaseStats {
    std::atomic<uint64_t> page_accesses{0};
    std::atomic<uint64_t> page_misses{0};
};

struct DeleteRepairIOStats {
    std::atomic<uint64_t> n_deletes{0};
    RepairIOPhaseStats step1;      // read tombstoned slot
    RepairIOPhaseStats step2;      // beam search + candidate coord fetch
    RepairIOPhaseStats step3;      // approx_in discovery
    RepairIOPhaseStats step4;      // repair in-neighbors
    RepairIOPhaseStats step4a_z_read;
    RepairIOPhaseStats step4b_prune;
    RepairIOPhaseStats step4c_write;
    RepairIOPhaseStats step5;      // bridge propagation
    RepairIOPhaseStats step5a_w_fetch;
    RepairIOPhaseStats step5b_y_read;
    RepairIOPhaseStats step5c_y_prune;
    RepairIOPhaseStats step5d_y_write;
    std::atomic<uint64_t> step4b_prune_calls{0};
    std::atomic<uint64_t> step5c_prune_calls{0};
    std::atomic<uint64_t> step5_unique_y_total{0};
    std::atomic<uint64_t> n_repair_calls{0};
    std::atomic<uint64_t> delete_batch_size_total{0};
    std::atomic<uint64_t> step5_call_unique_y_total{0};
    std::atomic<uint64_t> step4_call_unique_pruned_y_total{0};
    std::atomic<uint64_t> step5_call_unique_pruned_y_total{0};
};
DeleteRepairIOStats g_del_repair_io_stats;

struct InsertIOStats {
    std::atomic<uint64_t> n_inserts{0};
    RepairIOPhaseStats phases[InsertIOPhaseScope::NUM_PHASES];
};
InsertIOStats g_insert_io_stats;

struct RepairIODelta {
    uint64_t page_accesses = 0;
    uint64_t page_misses = 0;
};

static inline RepairIODelta repair_io_checkpoint() {
    return RepairIODelta{thread_page_access_count(),
                         thread_physical_reads_count()};
}

static inline void add_repair_io_delta(RepairIOPhaseStats& dst,
                                       const RepairIODelta& start) {
    dst.page_accesses.fetch_add(thread_page_access_count() - start.page_accesses,
                                std::memory_order_relaxed);
    dst.page_misses.fetch_add(thread_physical_reads_count() - start.page_misses,
                              std::memory_order_relaxed);
}

class RepairIOPhaseScope {
 public:
    explicit RepairIOPhaseScope(RepairIOPhaseStats& stats)
        : _stats(stats), _start(repair_io_checkpoint()) {}
    ~RepairIOPhaseScope() { add_repair_io_delta(_stats, _start); }

 private:
    RepairIOPhaseStats& _stats;
    RepairIODelta _start;
};

static inline RepairIODelta load_and_reset(RepairIOPhaseStats& phase) {
    return RepairIODelta{
        phase.page_accesses.exchange(0, std::memory_order_relaxed),
        phase.page_misses.exchange(0, std::memory_order_relaxed)};
}

}  // namespace

void dump_and_reset_delete_phase_stats() {
    uint64_t n = g_del_repair_io_stats.n_deletes.exchange(0, std::memory_order_relaxed);
    if (n == 0) return;
    double denom = static_cast<double>(n);

    auto print_phase = [denom](const char* label, const RepairIODelta& io) {
        double miss_rate = io.page_accesses == 0
                               ? 0.0
                               : 100.0 * static_cast<double>(io.page_misses) /
                                     static_cast<double>(io.page_accesses);
        fprintf(stderr,
                "  %-20s accesses=%12llu  misses=%10llu  "
                "accesses/delete=%8.2f  misses/delete=%8.4f  miss_rate=%7.4f%%\n",
                label,
                (unsigned long long)io.page_accesses,
                (unsigned long long)io.page_misses,
                io.page_accesses / denom,
                io.page_misses / denom,
                miss_rate);
    };

    RepairIODelta s1 = load_and_reset(g_del_repair_io_stats.step1);
    RepairIODelta s2 = load_and_reset(g_del_repair_io_stats.step2);
    RepairIODelta s3 = load_and_reset(g_del_repair_io_stats.step3);
    RepairIODelta s4 = load_and_reset(g_del_repair_io_stats.step4);
    RepairIODelta s4a = load_and_reset(g_del_repair_io_stats.step4a_z_read);
    RepairIODelta s4b = load_and_reset(g_del_repair_io_stats.step4b_prune);
    RepairIODelta s4c = load_and_reset(g_del_repair_io_stats.step4c_write);
    RepairIODelta s5 = load_and_reset(g_del_repair_io_stats.step5);
    RepairIODelta s5a = load_and_reset(g_del_repair_io_stats.step5a_w_fetch);
    RepairIODelta s5b = load_and_reset(g_del_repair_io_stats.step5b_y_read);
    RepairIODelta s5c = load_and_reset(g_del_repair_io_stats.step5c_y_prune);
    RepairIODelta s5d = load_and_reset(g_del_repair_io_stats.step5d_y_write);

    uint64_t n_dist_comp = diskann::collect_n_dist_comp_total_and_reset();
    fprintf(stderr,
            "=== delete repair buffer-pool page IO (totals over %llu deletes) ===\n",
            (unsigned long long)n);
    fprintf(stderr,
            "  pq_dist_comp        total=%12llu  per_delete=%12.2f\n",
            (unsigned long long)n_dist_comp,
            (double)n_dist_comp / denom);
    print_phase("step1 read_tomb", s1);
    print_phase("step2 beam_search", s2);
    print_phase("step3 approx_in", s3);
    print_phase("step4 in_nbr", s4);
    print_phase("  step4(A) z read", s4a);
    print_phase("  step4(B) prune", s4b);
    print_phase("  step4(C) write", s4c);
    print_phase("step5 bridge", s5);
    print_phase("  step5(A) w fetch", s5a);
    print_phase("  step5(B) y read", s5b);
    print_phase("  step5(C) y prune", s5c);
    print_phase("  step5(D) y write", s5d);

    uint64_t s4b_calls = g_del_repair_io_stats.step4b_prune_calls.exchange(
        0, std::memory_order_relaxed);
    uint64_t s5c_calls = g_del_repair_io_stats.step5c_prune_calls.exchange(
        0, std::memory_order_relaxed);
    fprintf(stderr,
            "  prune_calls         step4b=%12llu  step5c=%12llu  "
            "per_delete: 4b=%8.4f  5c=%8.4f  total=%8.4f\n",
            (unsigned long long)s4b_calls,
            (unsigned long long)s5c_calls,
            s4b_calls / denom, s5c_calls / denom,
            (s4b_calls + s5c_calls) / denom);
    uint64_t s5_uniq_y = g_del_repair_io_stats.step5_unique_y_total.exchange(
        0, std::memory_order_relaxed);
    fprintf(stderr,
            "  step5 unique_y      total=%12llu  per_delete=%8.4f\n",
            (unsigned long long)s5_uniq_y,
            s5_uniq_y / denom);

    uint64_t n_calls = g_del_repair_io_stats.n_repair_calls.exchange(
        0, std::memory_order_relaxed);
    uint64_t batch_total = g_del_repair_io_stats.delete_batch_size_total.exchange(
        0, std::memory_order_relaxed);
    uint64_t call_uniq_y = g_del_repair_io_stats.step5_call_unique_y_total.exchange(
        0, std::memory_order_relaxed);
    uint64_t s4_call_pruned = g_del_repair_io_stats
        .step4_call_unique_pruned_y_total.exchange(0, std::memory_order_relaxed);
    uint64_t s5_call_pruned = g_del_repair_io_stats
        .step5_call_unique_pruned_y_total.exchange(0, std::memory_order_relaxed);
    if (n_calls > 0) {
        double call_denom = static_cast<double>(n_calls);
        double dedup_factor = call_uniq_y > 0
            ? static_cast<double>(s5_uniq_y) / static_cast<double>(call_uniq_y)
            : 0.0;
        fprintf(stderr,
                "  repair_calls        n=%llu  avg_batch_size=%8.4f  "
                "avg_unique_y_per_call=%10.2f  "
                "cross_batch_dedup=%6.3fx\n",
                (unsigned long long)n_calls,
                batch_total / call_denom,
                call_uniq_y / call_denom,
                dedup_factor);
        fprintf(stderr,
                "  call_unique_pruned  step4=%12llu  step5=%12llu  "
                "per_delete: 4=%8.4f  5=%8.4f  total=%8.4f\n",
                (unsigned long long)s4_call_pruned,
                (unsigned long long)s5_call_pruned,
                s4_call_pruned / denom,
                s5_call_pruned / denom,
                (s4_call_pruned + s5_call_pruned) / denom);
    }
}

InsertIOPhaseScope::InsertIOPhaseScope(Phase phase)
    : _phase(phase),
      _start_accesses(thread_page_access_count()),
      _start_misses(thread_physical_reads_count()) {}

InsertIOPhaseScope::~InsertIOPhaseScope() {
    auto& s = g_insert_io_stats.phases[_phase];
    s.page_accesses.fetch_add(thread_page_access_count() - _start_accesses,
                              std::memory_order_relaxed);
    s.page_misses.fetch_add(thread_physical_reads_count() - _start_misses,
                            std::memory_order_relaxed);
}

void insert_stats_increment_n() {
    g_insert_io_stats.n_inserts.fetch_add(1, std::memory_order_relaxed);
}

void dump_and_reset_insert_phase_stats() {
    uint64_t n = g_insert_io_stats.n_inserts.exchange(0, std::memory_order_relaxed);
    if (n == 0) return;
    double denom = static_cast<double>(n);

    auto print_phase = [denom](const char* label, const RepairIODelta& io) {
        double miss_rate = io.page_accesses == 0
                               ? 0.0
                               : 100.0 * static_cast<double>(io.page_misses) /
                                     static_cast<double>(io.page_accesses);
        fprintf(stderr,
                "  %-20s accesses=%12llu  misses=%10llu  "
                "accesses/insert=%8.2f  misses/insert=%8.4f  miss_rate=%7.4f%%\n",
                label,
                (unsigned long long)io.page_accesses,
                (unsigned long long)io.page_misses,
                io.page_accesses / denom,
                io.page_misses / denom,
                miss_rate);
    };

    RepairIODelta beam = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::BEAM_SEARCH]);
    RepairIODelta prune = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::PRUNE]);
    RepairIODelta write_new = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::WRITE_NEW_NODE]);
    RepairIODelta bidir = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::BIDIRECTIONAL]);

    uint64_t n_dist_comp = diskann::collect_n_dist_comp_total_and_reset();
    fprintf(stderr,
            "=== insert buffer-pool page IO (totals over %llu inserts) ===\n",
            (unsigned long long)n);
    fprintf(stderr,
            "  pq_dist_comp        total=%12llu  per_insert=%12.2f\n",
            (unsigned long long)n_dist_comp,
            (double)n_dist_comp / denom);
    print_phase("step1 beam_search", beam);
    print_phase("step2 prune", prune);
    print_phase("step3 write_new_node", write_new);
    print_phase("step4 bidirectional", bidir);
}

// ===========================================================================
// PQ helpers (local copies from pq_flash_index.cpp anonymous namespace)
// ===========================================================================
static void pq_dist_lookup(const _u8* pq_ids, const _u64 n_pts,
                            const _u64 pq_nchunks, const float* pq_dists,
                            float* dists_out) {
    _mm_prefetch((char*)dists_out, _MM_HINT_T0);
    _mm_prefetch((char*)pq_ids, _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const float* chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
            _mm_prefetch((char*)(chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
            _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
            dists_out[idx] += chunk_dists[pq_centerid];
        }
    }
}

static inline uint64_t elapsed_ns(
    const std::chrono::steady_clock::time_point& start) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

// ===========================================================================
// graph_iterate_to_fixed_point -- PQ two-tier search
// ===========================================================================
template<typename T>
static
std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point_impl(
    const T* query, unsigned L,
    const std::vector<unsigned>& init_ids,
    unsigned beamwidth,
    InPlaceGraphStore* store,
    unsigned aligned_dim,
    Distance<T>* dist_cmp,
    InPlaceSearchScratch* scratch,
    std::vector<Neighbor>& results,
    bool use_deferred_overlay,
    TraversalScope scope,
    FixedChunkPQTable<T>* pq_table) {
    (void)use_deferred_overlay;
    const bool record_search_stats = scope == TraversalScope::SEARCH;
    const uint32_t n_chunks = store->n_chunks();
    bool use_pq = false;

    // Pre-compute PQ chunk distances only when PQ is enabled.
    if (n_chunks > 0 && pq_table != nullptr) {
        pq_table->populate_chunk_distances(query, scratch->pq_dists);
        use_pq = true;
    }

    // retset: beam-search working set, PQ-scored, size L.
    auto& retset = scratch->traversal_retset;
    retset.clear();
    retset.resize(L + 1);
    unsigned cur_list_size = 0;

    // visited: nodes already enqueued into retset (checked when processing
    // neighbors, not when inserting seeds).
    auto& visited = scratch->visited;
    visited.clear();

    // results: all expanded frontier nodes with exact full-vector distances.
    results.clear();
    if (results.capacity() < 4096) {
        results.reserve(4096);
    }

    // Seed initialization: PQ distance (or exact when no PQ), no visited check.
    for (const unsigned id : init_ids) {
        float seed_dist;
        if (!use_pq) {
            // No PQ: compute exact distance by reading the seed's coords.
            NodeRID seed_rid;
            PinnedFrame seed_frame = store->pin_page_for_node(id, READ, &seed_rid);
            if (!seed_frame.valid()) continue;
            auto seed_guard = seed_frame.read_guard();
            ConstNodeRef seed_node = seed_guard.read_node(seed_rid);
            if (!seed_node.valid()) continue;
            if (!store->is_active(id)) {
                continue;
            }
            auto inserted = visited.insert(id);
            if (!inserted.second) {
                continue;
            }
            seed_dist = dist_cmp->compare(
                query, seed_node.coords<T>(), aligned_dim);
        } else {
            if (!store->is_active(id)) continue;
            auto inserted = visited.insert(id);
            if (!inserted.second) continue;
            store->gather_pq_codes(&id, 1, scratch->pq_coord_scratch);
            pq_dist_lookup(scratch->pq_coord_scratch, 1, n_chunks,
                           scratch->pq_dists, scratch->dist_scratch);
            seed_dist = scratch->dist_scratch[0];
        }
        if (cur_list_size == 0) {
            retset[0] = Neighbor(id, seed_dist, true);
            cur_list_size = 1;
        } else {
            InsertIntoPoolNoDup(
                retset.data(), cur_list_size, Neighbor(id, seed_dist, true));
            if (cur_list_size < L) ++cur_list_size;
        }
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    auto& frontier_nodes = scratch->frontier_nodes;
    frontier_nodes.clear();
    if (frontier_nodes.capacity() < std::max<unsigned>(1, beamwidth)) {
        frontier_nodes.reserve(std::max<unsigned>(1, beamwidth));
    }
    uint32_t hops = 0;
    uint32_t cmps = 0;
    unsigned k = 0;

    while (k < cur_list_size) {
        unsigned nk = cur_list_size;
        frontier_nodes.clear();
        unsigned cursor = k;
        while (cursor < cur_list_size &&
               frontier_nodes.size() < std::max<unsigned>(1, beamwidth)) {
            if (retset[cursor].flag) {
                retset[cursor].flag = false;
                frontier_nodes.push_back(retset[cursor]);
            }
            cursor++;
        }
        if (frontier_nodes.empty()) {
            ++k;
            continue;
        }

        // Batched libaio prefetch+pin of all frontier pages. PinnedFrame ownership
        // is per-iteration; we move-assign default at the end of each iteration
        // to release the pin back to the pool as soon as we're done with it.
        assert(frontier_nodes.size() <= BufferPool::kPinBatchMax);
        std::array<uint32_t,    BufferPool::kPinBatchMax> batch_ids;
        std::array<PinnedFrame, BufferPool::kPinBatchMax> batch_frames;
        std::array<NodeRID,     BufferPool::kPinBatchMax> batch_rids;
        for (size_t i = 0; i < frontier_nodes.size(); ++i) {
            batch_ids[i] = frontier_nodes[i].id;
        }
        store->pin_pages_for_nodes_batch(
            batch_ids.data(), frontier_nodes.size(), READ,
            batch_frames.data(), batch_rids.data());
        hops += static_cast<uint32_t>(frontier_nodes.size());

        for (size_t fi = 0; fi < frontier_nodes.size(); ++fi) {
            const Neighbor& frontier = frontier_nodes[fi];
            const unsigned n = frontier.id;
            // Internal tombstones are gone from traversal. Logically deleted
            // tags still expand normally and are filtered only from results.
            if (!store->is_active(n)) { batch_frames[fi] = PinnedFrame(); continue; }
            PinnedFrame& frame = batch_frames[fi];
            if (!frame.valid()) continue;
            const NodeRID& rid = batch_rids[fi];
            auto guard = frame.read_guard();
            ConstNodeRef node = guard.read_node(rid);
            if (!node.valid() || !store->is_active(n)) {
                batch_frames[fi] = PinnedFrame();
                continue;
            }

            const float exact_dist =
                !use_pq
                    ? frontier.distance
                    : dist_cmp->compare(
                          query, node.coords<T>(),
                          aligned_dim);
            if (!(scope == TraversalScope::SEARCH &&
                  store->is_tag_deleted(store->node_tag(n)))) {
                results.emplace_back(n, exact_dist, true);
            }

            scratch->frontier_candidate_ids.clear();
            if (scratch->frontier_candidate_ids.capacity() <
                scratch->frontier_candidate_ids.size() + node.degree()) {
                scratch->frontier_candidate_ids.reserve(
                    scratch->frontier_candidate_ids.size() + node.degree());
            }
            const uint32_t* nbrs = node.neighbors();
            for (uint16_t j = 0; j < node.degree(); ++j) {
                const uint32_t nbr_id = nbrs[j];
                auto inserted = visited.insert(nbr_id);
                if (inserted.second) {
                    scratch->frontier_candidate_ids.push_back(nbr_id);
                }
            }

            const uint32_t expanded_nbr_count =
                static_cast<uint32_t>(scratch->frontier_candidate_ids.size());
            if (expanded_nbr_count == 0) {
                batch_frames[fi] = PinnedFrame();
                continue;
            }
            auto start = std::chrono::steady_clock::now();

            if (use_pq) {
                uint32_t processed = 0;
                while (processed < expanded_nbr_count) {
                    const uint32_t batch = std::min<uint32_t>(
                        InPlaceSearchScratch::MAX_SCRATCH_NODES,
                        expanded_nbr_count - processed);
                    store->gather_pq_codes(
                        reinterpret_cast<const unsigned*>(
                            scratch->frontier_candidate_ids.data() + processed),
                        batch, scratch->pq_coord_scratch);
                    pq_dist_lookup(scratch->pq_coord_scratch, batch, n_chunks,
                                   scratch->pq_dists, scratch->dist_scratch);
                    for (uint32_t m = 0; m < batch; ++m) {
                        unsigned nbr_id = scratch->frontier_candidate_ids[processed + m];
                        cmps++;
                        float dist = scratch->dist_scratch[m];
                        if (dist >= retset[cur_list_size - 1].distance &&
                            cur_list_size == L)
                            continue;
                        unsigned r = InsertIntoPoolNoDup(
                            retset.data(), cur_list_size,
                            Neighbor(nbr_id, dist, true));
                        if (cur_list_size < L) ++cur_list_size;
                        if (r < nk) nk = r;
                    }
                    processed += batch;
                }
                if (record_search_stats) {
                    scratch->record_comparable_distance(
                        DistanceScope::SEARCH, elapsed_ns(start), expanded_nbr_count);
                }
            } else {
                store->template batch_compute_dists<T>(
                    scratch->frontier_candidate_ids, query, dist_cmp,
                    scratch->frontier_candidate_dists,
                    scratch->frontier_found,
                    scratch);
                for (uint32_t m = 0; m < expanded_nbr_count; ++m) {
                    unsigned nbr_id = scratch->frontier_candidate_ids[m];
                    cmps++;
                    if (!scratch->frontier_found[m]) continue;
                    float dist = scratch->frontier_candidate_dists[m];
                    if (dist >= retset[cur_list_size - 1].distance &&
                        cur_list_size == L)
                        continue;
                    unsigned r = InsertIntoPoolNoDup(
                        retset.data(), cur_list_size,
                        Neighbor(nbr_id, dist, true));
                    if (cur_list_size < L) ++cur_list_size;
                    if (r < nk) nk = r;
                }
                if (record_search_stats) {
                    scratch->record_comparable_distance(
                        DistanceScope::SEARCH, elapsed_ns(start), expanded_nbr_count);
                }
            }
            // Release this frame's pin now that we're done expanding from it.
            batch_frames[fi] = PinnedFrame();
        }

        if (nk <= k) k = nk;
        else ++k;
    }

    if (scope != TraversalScope::INSERT) {
        std::sort(results.begin(), results.end());
    }
    return std::make_pair(hops, cmps);
}

template<typename T>
std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point(
    const T* query, unsigned L,
    const std::vector<unsigned>& init_ids,
    unsigned beamwidth,
    InPlaceGraphStore* store,
    unsigned aligned_dim,
    Distance<T>* dist_cmp,
    InPlaceSearchScratch* scratch,
    std::vector<Neighbor>& results,
    bool use_deferred_overlay,
    TraversalScope scope,
    FixedChunkPQTable<T>* pq_table) {
    return graph_iterate_to_fixed_point_impl<T>(
        query, L, init_ids, beamwidth, store, aligned_dim, dist_cmp,
        scratch, results, use_deferred_overlay, scope, pq_table);
}

// ===========================================================================
// graph_occlude_list_pq
// ===========================================================================
template<typename T>
void graph_occlude_list_pq(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    DistanceScope scope,
    FixedChunkPQTable<T>* pq_table) {
    if (pool.empty()) return;
    assert(std::is_sorted(pool.begin(), pool.end()));

    std::vector<unsigned> batch_ids;
    batch_ids.reserve(maxc);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
        uint32_t start = 0;
        while (result.size() < degree && start < pool.size() && start < maxc) {
            auto& p = pool[start];
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result.push_back(p);

            // Each time we compute pq dists in a batch: one src and many dsts.
            batch_ids.clear();
            for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                if (occlude_factor[t] <= alpha) batch_ids.push_back(pool[t].id);
            }
            if (!batch_ids.empty() && pq_table != nullptr) {
                store->compute_pq_dists_src(
                    p.id, batch_ids.data(), scratch->dist_scratch,
                    (uint32_t)batch_ids.size(), scratch->pq_coord_scratch,
                    *pq_table);
                uint32_t batch_idx = 0;
                for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                    if (occlude_factor[t] > alpha) continue;
                    float djk = scratch->dist_scratch[batch_idx++];
                    occlude_factor[t] =
                        std::max(occlude_factor[t], pool[t].distance / djk);
                }
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }
}

// ===========================================================================
// graph_occlude_list (full-precision)
// ===========================================================================
template<typename T>
void graph_occlude_list(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim,
    Distance<T>* distance,
    DistanceScope scope) {
    (void)scratch;
    (void)scope;

    if (pool.empty()) return;
    assert(std::is_sorted(pool.begin(), pool.end()));

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
        uint32_t start = 0;
        while (result.size() < degree && start < pool.size() && start < maxc) {
            auto& p = pool[start];
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result.push_back(p);

            std::vector<T> p_coords_copy;
            bool p_valid = false;
            const T* p_coords = nullptr;
            NodeRID p_rid;
            PinnedFrame p_frame = store->pin_page_for_node(p.id, READ, &p_rid);
            if (p_frame.valid()) {
                auto p_guard = p_frame.read_guard();
                ConstNodeRef p_node = p_guard.read_node(p_rid);
                if (p_node.valid() && store->is_active(p.id)) {
                    const T* src = p_node.coords<T>();
                    p_coords_copy.assign(src, src + aligned_dim);
                    p_coords = p_coords_copy.data();
                    p_valid = true;
                }
            }

            for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                if (occlude_factor[t] > alpha) continue;
                if (!p_valid) continue;
                float djk = std::numeric_limits<float>::max();
                NodeRID t_rid;
                PinnedFrame t_frame = store->pin_page_for_node(pool[t].id, READ, &t_rid);
                if (t_frame.valid()) {
                    auto t_guard = t_frame.read_guard();
                    ConstNodeRef t_node = t_guard.read_node(t_rid);
                    if (t_node.valid() && store->is_active(pool[t].id)) {
                        djk = distance->compare(p_coords, t_node.coords<T>(), aligned_dim);
                    }
                }
                occlude_factor[t] =
                    std::max(occlude_factor[t], pool[t].distance / djk);
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }
}

template<typename T>
void graph_prune_neighbors_cached_exact(
    unsigned location,
    std::vector<Neighbor>& sorted_pool,
    const std::vector<T>& sorted_coords,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<unsigned>& pruned_list,
    unsigned aligned_dim,
    Distance<T>* distance) {
    if (sorted_pool.empty()) return;
    assert(std::is_sorted(sorted_pool.begin(), sorted_pool.end()));

    std::vector<uint32_t> result_indices;
    result_indices.reserve(degree);
    std::vector<float> occlude_factor(sorted_pool.size(), 0);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result_indices.size() < degree) {
        uint32_t start = 0;
        while (result_indices.size() < degree &&
               start < sorted_pool.size() && start < maxc) {
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result_indices.push_back(start);

            const T* p_coords =
                sorted_coords.data() + static_cast<size_t>(start) * aligned_dim;
            for (uint32_t t = start + 1; t < sorted_pool.size() && t < maxc; t++) {
                if (occlude_factor[t] > alpha) continue;
                const T* t_coords =
                    sorted_coords.data() + static_cast<size_t>(t) * aligned_dim;
                const float djk = distance->compare(p_coords, t_coords, aligned_dim);
                occlude_factor[t] =
                    std::max(occlude_factor[t], sorted_pool[t].distance / djk);
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }

    pruned_list.clear();
    for (uint32_t idx : result_indices) {
        if (sorted_pool[idx].id != location) {
            pruned_list.push_back(sorted_pool[idx].id);
        }
    }
}

// ===========================================================================
// graph_prune_neighbors_pq
// ===========================================================================
template<typename T>
void graph_prune_neighbors_pq(
    unsigned location,
    std::vector<Neighbor>& pool,
    unsigned R, unsigned C, float alpha,
    std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim,
    Distance<T>* distance,
    DistanceScope scope,
    FixedChunkPQTable<T>* pq_table) {
    if (pool.empty()) return;
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(R);
    std::vector<float> occlude_factor(pool.size(), 0);

    if (store->n_chunks() > 0 && scratch && scratch->pq_coord_scratch &&
        pq_table != nullptr) {
        graph_occlude_list_pq<T>(pool, alpha, R, C, result, occlude_factor,
                                 store, scratch, scope, pq_table);
    } else if (distance && aligned_dim > 0) {
        graph_occlude_list<T>(pool, alpha, R, C, result, occlude_factor,
                              store, scratch, aligned_dim, distance, scope);
    } else {
        // No PQ and no distance function: just take top-R by distance
        for (auto& p : pool) {
            if (p.id != location) {
                result.push_back(p);
                if (result.size() >= R) break;
            }
        }
    }

    pruned_list.clear();
    for (auto& iter : result) {
        if (iter.id != location)
            pruned_list.push_back(iter.id);
    }
}

template<typename T>
void graph_inter_insert_immediate(
    unsigned new_node,
    const std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim,
    Distance<T>* dist,
    FixedChunkPQTable<T>* pq_table) {
    if (store == nullptr || dist == nullptr || !store->is_active(new_node) ||
        pruned_list.empty()) {
        return;
    }

    struct RepairTarget {
        uint32_t node_id;
        uint32_t page_id;
    };
    std::vector<RepairTarget> targets;
    targets.reserve(pruned_list.size());
    tsl::robin_set<uint32_t> seen_targets;
    seen_targets.reserve(pruned_list.size() * 2 + 1);
    for (uint32_t target : pruned_list) {
        if (target == new_node || !seen_targets.insert(target).second ||
            !store->is_active(target)) {
            continue;
        }
        targets.push_back(RepairTarget{target, store->get_page_id(target)});
    }
    if (targets.empty()) return;

    std::sort(targets.begin(), targets.end(),
              [](const RepairTarget& a, const RepairTarget& b) {
                  if (a.page_id != b.page_id) return a.page_id < b.page_id;
                  return a.node_id < b.node_id;
              });

    InPlaceSearchScratch& local_scratch =
        maintenance_scratch_for(aligned_dim, store->n_chunks(), sizeof(T),
                                C, store->max_degree(), 1);
    std::vector<unsigned> current_nbrs;
    current_nbrs.reserve(store->max_degree() + 1);
    tsl::robin_set<unsigned> existing;
    existing.reserve(store->max_degree() + 1);
    std::vector<Neighbor> pool;
    pool.reserve(store->max_degree() + 1);
    std::vector<unsigned> pruned;
    pruned.reserve(store->max_degree());
    std::vector<uint8_t> pq_scratch_buf;
    std::vector<float> dist_buf;
    std::vector<T> target_vec_copy;

    size_t ti = 0;
    while (ti < targets.size()) {
        const uint32_t cur_page = targets[ti].page_id;
        const size_t page_start = ti;
        while (ti < targets.size() && targets[ti].page_id == cur_page) {
            ++ti;
        }
        if (cur_page == INVALID_PAGE) continue;

        for (size_t idx = page_start; idx < ti; ++idx) {
            const uint32_t target = targets[idx].node_id;
            if (!store->is_active(target)) continue;

            NodeRID rid;
            PinnedFrame frame = store->pin_page_for_node(target, WRITE, &rid);
            if (!frame.valid() || rid.page_id != cur_page) continue;

            current_nbrs.clear();
            target_vec_copy.clear();
            const bool use_pq_dist =
                store->n_chunks() > 0 && pq_table != nullptr &&
                store->pq_codes_for_node(target) != nullptr;
            bool needs_prune = false;

            {
                auto guard = frame.write_guard();
                MutableNodeRef node = guard.write_node(rid);
                if (!node.valid() || node.node_id() != target ||
                    !store->is_active(target)) {
                    continue;
                }

                const uint32_t* nbrs = node.neighbors();
                current_nbrs.reserve(std::max<size_t>(
                    current_nbrs.capacity(), node.degree() + 1));
                for (uint16_t j = 0; j < node.degree(); ++j) {
                    const uint32_t nbr = nbrs[j];
                    if (nbr != target && store->is_active(nbr)) {
                        current_nbrs.push_back(nbr);
                    }
                }

                existing.clear();
                existing.insert(current_nbrs.begin(), current_nbrs.end());
                if (existing.find(new_node) != existing.end()) {
                    continue;
                }
                current_nbrs.push_back(new_node);

                if (current_nbrs.size() <= static_cast<size_t>(R)) {
                    node.set_neighbors(current_nbrs.data(), current_nbrs.size());
                    continue;
                }

                needs_prune = true;
                if (!use_pq_dist) {
                    const T* target_coords = node.coords<T>();
                    target_vec_copy.assign(target_coords,
                                           target_coords + aligned_dim);
                }
            }

            if (!needs_prune) continue;

            pool.clear();
            pool.reserve(std::max<size_t>(pool.capacity(), current_nbrs.size()));
            if (use_pq_dist) {
                pq_scratch_buf.resize(current_nbrs.size() * store->n_chunks() + 64);
                dist_buf.resize(current_nbrs.size());
                store->compute_pq_dists_src(
                    target, current_nbrs.data(), dist_buf.data(),
                    static_cast<uint32_t>(current_nbrs.size()),
                    pq_scratch_buf.data(), *pq_table);
                for (size_t j = 0; j < current_nbrs.size(); ++j) {
                    pool.emplace_back(current_nbrs[j], dist_buf[j], true);
                }
            } else {
                const T* target_vec = target_vec_copy.data();
                for (uint32_t nbr : current_nbrs) {
                    float d = std::numeric_limits<float>::max();
                    NodeRID nbr_rid;
                    PinnedFrame nbr_frame =
                        store->pin_page_for_node(nbr, READ, &nbr_rid);
                    if (nbr_frame.valid()) {
                        auto nbr_guard = nbr_frame.read_guard();
                        ConstNodeRef nbr_node = nbr_guard.read_node(nbr_rid);
                        if (nbr_node.valid() &&
                            store->is_active(nbr)) {
                            d = dist->compare(target_vec, nbr_node.coords<T>(),
                                              aligned_dim);
                        }
                    }
                    pool.emplace_back(nbr, d, true);
                }
            }

            pruned.clear();
            graph_prune_neighbors_pq<T>(target, pool, R, C, alpha,
                                        pruned, store, &local_scratch,
                                        aligned_dim, dist,
                                        DistanceScope::MAINTENANCE,
                                        pq_table);

            {
                auto guard = frame.write_guard();
                MutableNodeRef node = guard.write_node(rid);
                if (!node.valid() || node.node_id() != target ||
                    !store->is_active(target)) {
                    continue;
                }
                node.set_neighbors(pruned.data(), pruned.size());
            }
        }
    }
}

template void graph_inter_insert_immediate<float>(
    unsigned, const std::vector<unsigned>&, InPlaceGraphStore*,
    unsigned, unsigned, float, unsigned, Distance<float>*,
    FixedChunkPQTable<float>*);
template void graph_inter_insert_immediate<uint8_t>(
    unsigned, const std::vector<unsigned>&, InPlaceGraphStore*,
    unsigned, unsigned, float, unsigned, Distance<uint8_t>*,
    FixedChunkPQTable<uint8_t>*);
template void graph_inter_insert_immediate<int8_t>(
    unsigned, const std::vector<unsigned>&, InPlaceGraphStore*,
    unsigned, unsigned, float, unsigned, Distance<int8_t>*,
    FixedChunkPQTable<int8_t>*);

uint32_t InPlaceGraphStore::remove_deleted_edges_full_scan(uint32_t num_threads) {
    std::vector<uint32_t> delete_pending_ids;
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        delete_pending_ids.swap(_delete_pending_since_maintenance);
    }
    if (delete_pending_ids.empty()) return 0;

    std::sort(delete_pending_ids.begin(), delete_pending_ids.end());
    delete_pending_ids.erase(
        std::unique(delete_pending_ids.begin(), delete_pending_ids.end()),
        delete_pending_ids.end());

    std::unordered_set<uint32_t> delete_pending_set;
    delete_pending_set.reserve(delete_pending_ids.size() * 2 + 1);
    for (uint32_t node_id : delete_pending_ids) {
        delete_pending_set.insert(node_id);
    }

    std::vector<PageDir> pages;
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        pages = _page_dir;
    }

    const std::unordered_set<uint32_t>& delete_pending_set_ref =
        delete_pending_set;
    auto process_page = [&](const PageDir& page, PinnedFrame& frame) -> uint32_t {
        uint32_t page_removed_edges = 0;
        auto guard = frame.write_guard();
        for (uint16_t slot = 0; slot < page.slots_per_page; ++slot) {
            MutableNodeRef node = guard.write_node(slot);
            if (!node.valid()) continue;

            const uint32_t node_id = node.node_id();
            if (node_id == INVALID_NODE || !is_active(node_id)) {
                continue;
            }

            const uint16_t degree = node.degree();
            if (degree == 0) continue;

            uint32_t* neighbors = node.neighbors();
            uint16_t write_idx = 0;
            uint32_t removed_from_node = 0;
            for (uint16_t read_idx = 0; read_idx < degree; ++read_idx) {
                const uint32_t neighbor = neighbors[read_idx];
                if (delete_pending_set_ref.find(neighbor) !=
                    delete_pending_set_ref.end()) {
                    ++removed_from_node;
                    continue;
                }
                if (write_idx != read_idx) {
                    neighbors[write_idx] = neighbor;
                }
                ++write_idx;
            }

            if (removed_from_node == 0) continue;
            node.set_degree(write_idx);
            page_removed_edges += removed_from_node;
        }
        return page_removed_edges;
    };

    num_threads = std::max<uint32_t>(1, num_threads);
    const size_t page_count = pages.size();
    if (page_count == 0) {
        reclaim_delete_pending_ids(delete_pending_ids);
        return 0;
    }

    uint32_t removed_edges = 0;
    constexpr size_t kBatch = 64;
    const size_t batch_count = (page_count + kBatch - 1) / kBatch;

#pragma omp parallel for num_threads(static_cast<int>(num_threads)) \
    schedule(dynamic, 1) reduction(+:removed_edges)
    for (size_t batch_id = 0; batch_id < batch_count; ++batch_id) {
        const size_t begin_page = batch_id * kBatch;
        const size_t end_page = std::min(begin_page + kBatch, page_count);
        uint32_t batch_pages[kBatch];
        PinnedFrame batch_frames[kBatch];
        size_t batch_size = 0;

        for (size_t page_id = begin_page; page_id < end_page; ++page_id) {
            const PageDir& page = pages[page_id];
            if (page.slots_per_page == 0 || page.num_occupied == 0) {
                continue;
            }

            batch_pages[batch_size++] = static_cast<uint32_t>(page_id);
        }

        if (batch_size > 0) {
            try {
                pin_pages_batch_transient(batch_pages, batch_size, WRITE,
                                          batch_frames,
                                          true /* account_sequential_read */);
                for (size_t i = 0; i < batch_size; ++i) {
                    PinnedFrame& frame = batch_frames[i];
                    if (!frame.valid()) continue;
                    const uint32_t page_id = batch_pages[i];
                    removed_edges += process_page(pages[page_id], frame);
                    frame.release();
                }
            } catch (...) {
                std::terminate();
            }
        }
    }

    reclaim_delete_pending_ids(delete_pending_ids);
    return removed_edges;
}

std::vector<uint32_t> InPlaceGraphStore::reclaim_delete_pending_ids(
    const std::vector<uint32_t>& node_ids) {
    struct ReclaimItem {
      uint32_t node_id;
      RID      rid;
    };
    std::vector<ReclaimItem> reclaim_items;
    reclaim_items.reserve(node_ids.size());

    // Collect valid reclaim items. DeletePending nodes still keep their slots
    // until this maintenance commit point, so readers cannot confuse a reused
    // internal ID with an old graph reference.
    {
        uint32_t page_dir_size = 0;
        {
            std::lock_guard<std::mutex> lk(_pages_mtx);
            page_dir_size = static_cast<uint32_t>(_page_dir.size());
        }
        for (uint32_t node_id : node_ids) {
            if (!node_present(node_id) ||
                _node_rids[node_id].page_id == INVALID_PAGE) continue;
            if (_node_states[node_id].load(std::memory_order_acquire) !=
                NodeState::DeletePending) continue;
            if (_node_rids[node_id].page_id >= page_dir_size) continue;
            reclaim_items.push_back(ReclaimItem{node_id, _node_rids[node_id]});
        }
    }

    std::sort(reclaim_items.begin(), reclaim_items.end(),
              [](const ReclaimItem& a, const ReclaimItem& b) {
                  if (a.rid.page_id != b.rid.page_id) {
                      return a.rid.page_id < b.rid.page_id;
                  }
                  return a.rid.slot_idx < b.rid.slot_idx;
              });

    struct PageFreedCount { uint32_t page_id; uint16_t freed; };
    std::vector<PageFreedCount> freed_counts;
    std::vector<uint32_t> reclaimed_ids;
    reclaimed_ids.reserve(reclaim_items.size());

    {
        size_t cursor = 0;
        while (cursor < reclaim_items.size()) {
            const uint32_t page_id = reclaim_items[cursor].rid.page_id;
            PinnedFrame frame = pin_page_transient(page_id, WRITE);
            if (!frame.valid()) {
                while (cursor < reclaim_items.size() &&
                       reclaim_items[cursor].rid.page_id == page_id) {
                    ++cursor;
                }
                continue;
            }
            auto guard = frame.write_guard();
            char* page_data = frame._frame->data;
            uint8_t* bitmap = reinterpret_cast<uint8_t*>(page_data + 8);
            uint16_t freed_on_page = 0;
            while (cursor < reclaim_items.size() &&
                   reclaim_items[cursor].rid.page_id == page_id) {
                const ReclaimItem& item = reclaim_items[cursor++];
                if (!node_present(item.node_id) ||
                    _node_states[item.node_id].load(std::memory_order_acquire) !=
                        NodeState::DeletePending ||
                    _node_rids[item.node_id].page_id != item.rid.page_id ||
                    _node_rids[item.node_id].slot_idx != item.rid.slot_idx) {
                    continue;
                }
                MutableNodeRef node = guard.write_node(item.rid);
                if (!node.valid() || node.node_id() != item.node_id) continue;
                bitmap[item.rid.slot_idx / 8] &=
                    static_cast<uint8_t>(~(1u << (item.rid.slot_idx % 8)));
                reset_node_slot(item.node_id);
                _node_states[item.node_id].store(NodeState::Free,
                                                 std::memory_order_release);
                reclaimed_ids.push_back(item.node_id);
                ++freed_on_page;
            }
            if (freed_on_page > 0) {
                frame.mark_dirty();
                freed_counts.push_back(PageFreedCount{page_id, freed_on_page});
            }
        }
    }

    if (!freed_counts.empty()) {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        for (const auto& fc : freed_counts) {
            if (fc.page_id < _page_dir.size()) {
                PageDir& dir = _page_dir[fc.page_id];
                dir.num_occupied = (dir.num_occupied >= fc.freed)
                                       ? dir.num_occupied - fc.freed
                                       : 0;
                _pages_with_space.insert(fc.page_id);
            }
        }
    }

    if (!reclaimed_ids.empty()) {
        {
            std::lock_guard<std::mutex> lk(_freelist_mtx);
            _freelist.insert(_freelist.end(), reclaimed_ids.begin(),
                             reclaimed_ids.end());
        }
    }

    return reclaimed_ids;
}

template<typename T>
uint32_t InPlaceGraphStore::repair_deleted_single(
    uint32_t del_id,
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim, diskann::Distance<T>* dist,
    uint32_t c_replace,
    uint32_t delete_repair_L,
    uint32_t beamwidth,
    FixedChunkPQTable<T>* pq_table) {
    std::vector<uint32_t> batch{del_id};
    return repair_deleted_ids_impl<T>(batch, R, C, alpha, aligned_dim,
                                      dist, c_replace, delete_repair_L,
                                      beamwidth, pq_table);
}

template<typename T>
uint32_t InPlaceGraphStore::repair_deleted_explicit(
    const std::vector<uint32_t>& delete_batch,
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim, diskann::Distance<T>* dist,
    uint32_t c_replace,
    uint32_t delete_repair_L,
    uint32_t beamwidth,
    FixedChunkPQTable<T>* pq_table) {
    return repair_deleted_ids_impl<T>(delete_batch, R, C, alpha, aligned_dim,
                                      dist, c_replace, delete_repair_L,
                                      beamwidth, pq_table);
}

template<typename T>
uint32_t InPlaceGraphStore::repair_deleted_ids_impl(
    const std::vector<uint32_t>& delete_batch,
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim, diskann::Distance<T>* dist,
    uint32_t c_replace,
    uint32_t delete_repair_L,
    uint32_t beamwidth,
    FixedChunkPQTable<T>* pq_table) {
    if (delete_batch.empty()) return 0;
    if (c_replace == 0) c_replace = 3;
    unsigned search_L = delete_repair_L;
    const unsigned bw = std::max<unsigned>(1u, beamwidth);

    tsl::robin_map<uint32_t, uint32_t> delete_to_batch_idx;
    delete_to_batch_idx.reserve(delete_batch.size() * 2 + 1);
    for (uint32_t i = 0; i < delete_batch.size(); ++i) {
        delete_to_batch_idx.insert_or_assign(delete_batch[i], i);
    }

    InPlaceSearchScratch& search_scratch =
        maintenance_scratch_for(aligned_dim, _n_chunks, sizeof(T),
                                search_L, _Mmax, bw);

    constexpr size_t kMaxCandidates = 50;

    auto write_neighbors = [&](MutableNodeRef& node,
                               const std::vector<unsigned>& ids) {
        const size_t n = std::min<size_t>(_Mmax, ids.size());
        uint32_t* dst = node.neighbors();
        for (size_t i = 0; i < n; ++i) dst[i] = ids[i];
        node.set_degree(static_cast<uint16_t>(n));
    };

    auto live_writable_node = [&](uint32_t node_id,
                                  PinnedFrame& frame,
                                  FrameWriteGuard& guard,
                                  MutableNodeRef& node,
                                  NodeRID& rid) -> bool {
        frame = pin_page_for_node(node_id, WRITE, &rid);
        if (!frame.valid()) return false;
        guard = frame.write_guard();
        node = guard.write_node(rid);
        return node.valid() && node.node_id() == node_id &&
               is_active(node_id);
    };

    // Sort a pool by PQ distance to target_id, then run PQ Vamana occlusion.
    auto prune_to_R_pq = [&](uint32_t target_id,
                              const std::vector<unsigned>& pool_ids,
                              std::vector<unsigned>& out_pruned) {
        out_pruned.clear();
        if (pool_ids.empty()) return;

        if (_n_chunks == 0 || search_scratch.pq_coord_scratch == nullptr ||
            search_scratch.dist_scratch == nullptr ||
            pq_table == nullptr ||
            pq_codes_for_node(target_id) == nullptr) {
            throw std::runtime_error(
                "repair_deleted_ids_impl: PQ pruning requested without PQ codes");
        }
        std::vector<unsigned> ready_ids;
        ready_ids.reserve(pool_ids.size());
        for (unsigned id : pool_ids) {
            if (pq_codes_for_node(id) != nullptr) {
                ready_ids.push_back(id);
            }
        }
        if (ready_ids.empty()) return;

        const size_t n = ready_ids.size();
        std::vector<uint8_t> pq_scratch(n * static_cast<size_t>(_n_chunks));
        std::vector<float> dists(n);
        compute_pq_dists_src(target_id, ready_ids.data(), dists.data(),
                             static_cast<uint32_t>(n), pq_scratch.data(),
                             *pq_table);

        std::vector<Neighbor> pool;
        pool.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            pool.emplace_back(ready_ids[i], dists[i], true);
        }
        graph_prune_neighbors_pq<T>(
            target_id, pool, R, C, alpha, out_pruned, this, &search_scratch,
            /*aligned_dim=*/0, /*distance=*/nullptr,
            DistanceScope::MAINTENANCE, pq_table);
    };

    std::vector<T> del_coord_buf(aligned_dim);
    std::vector<unsigned> del_out_nbrs;
    std::vector<Neighbor> expanded;
    std::vector<unsigned> candidate_ids;
    std::vector<T> candidate_coords;
    std::vector<uint32_t> approx_in;

    uint32_t repaired = 0;

    // Call-scope step-5 pending: y -> list of w's queued for splicing,
    // accumulated across every deletion in this batch. Flushed once at
    // end-of-call so each unique y is read/maybe-pruned/written once
    // regardless of how many deletions proposed it.
    std::unordered_map<uint32_t, std::vector<uint32_t>> step5_pending;

    // Cross-batch (call-scope) union of y's that actually triggered a
    // prune_to_R_pq invocation. Strict subset of call_unique_y. Difference
    // between this and step{4b,5c}_prune_calls is the cross-deletion repeat
    // count -- the headroom that future call-scope coalescing would close.
    std::unordered_set<uint32_t> call_pruned_y_step4;
    std::unordered_set<uint32_t> call_pruned_y_step5;

    for (uint32_t del_id : delete_batch) {
        g_del_repair_io_stats.n_deletes.fetch_add(1, std::memory_order_relaxed);
        // Step 1: read tombstoned slot (coords + out-neighbors still present).
        {
            RepairIOPhaseScope io_step1(g_del_repair_io_stats.step1);
            NodeRID del_rid;
            PinnedFrame del_frame =
                pin_page_for_node(del_id, READ, &del_rid);
            if (!del_frame.valid()) continue;
            auto del_guard = del_frame.read_guard();
            ConstNodeRef del_node = del_guard.read_node(del_rid);
            if (!del_node.valid() || del_node.node_id() != del_id) continue;
            std::memcpy(del_coord_buf.data(), del_node.coords<T>(),
                        static_cast<size_t>(aligned_dim) * sizeof(T));
            del_out_nbrs.clear();
            const uint32_t* nbrs = del_node.neighbors();
            const uint16_t deg = del_node.degree();
            for (uint16_t j = 0; j < deg; ++j) {
                const uint32_t cand = nbrs[j];
                if (delete_to_batch_idx.find(cand) != delete_to_batch_idx.end() ||
                    !is_active(cand)) continue;
                del_out_nbrs.push_back(cand);
            }
        }

        // Step 2: beam search toward del_coords to find local neighborhood.
      {
        RepairIOPhaseScope io_step2(g_del_repair_io_stats.step2);
        std::vector<uint32_t> entry_pool;
        get_entry_points(entry_pool);
        if (entry_pool.empty()) continue;
        std::vector<unsigned> init_ids(entry_pool.begin(), entry_pool.end());

        search_scratch.reset();
        expanded.clear();
        graph_iterate_to_fixed_point<T>(
          del_coord_buf.data(), search_L, init_ids, bw,
          this, aligned_dim, dist, &search_scratch, expanded,
            /*use_deferred_overlay=*/false,
            TraversalScope::INSERT, pq_table);
        if (expanded.empty()) continue;

        // Top-kMaxCandidates live expanded nodes form the candidate pool.
        candidate_ids.clear();
        for (const Neighbor& ne : expanded) {
            if (ne.id == del_id) continue;
            if (delete_to_batch_idx.find(ne.id) != delete_to_batch_idx.end())
                continue;
            if (!is_active(ne.id)) continue;
            candidate_ids.push_back(ne.id);
            if (candidate_ids.size() >= kMaxCandidates) break;
        }
        if (candidate_ids.empty()) continue;
      }

        // Step 3: approx in-neighbors = expanded nodes whose edge list
        // still points to del_id. While those pages are pinned, also fetch
        // coords for the already-selected candidate pool and compact away
        // candidates whose slots can no longer be read.
        {
          RepairIOPhaseScope io_step3(g_del_repair_io_stats.step3);
          approx_in.clear();
          candidate_coords.clear();
          candidate_coords.reserve(candidate_ids.size() * static_cast<size_t>(aligned_dim));
          size_t candidate_read = 0;
          size_t candidate_write = 0;
          for (const Neighbor& ne : expanded) {
            const bool is_candidate =
                candidate_read < candidate_ids.size() &&
                candidate_ids[candidate_read] == ne.id;
            if (ne.id == del_id) continue;
            if (delete_to_batch_idx.find(ne.id) != delete_to_batch_idx.end())
                continue;
            if (!is_active(ne.id)) {
                if (is_candidate) ++candidate_read;
                continue;
            }
            NodeRID zrid;
            PinnedFrame zframe = pin_page_for_node(ne.id, READ, &zrid);
            if (!zframe.valid()) {
                if (is_candidate) ++candidate_read;
                continue;
            }
            auto zguard = zframe.read_guard();
            ConstNodeRef znode = zguard.read_node(zrid);
            if (!znode.valid() || znode.node_id() != ne.id) {
                if (is_candidate) ++candidate_read;
                continue;
            }
            if (is_candidate) {
                candidate_ids[candidate_write++] = ne.id;
                const T* coords = znode.coords<T>();
                candidate_coords.insert(candidate_coords.end(), coords, coords + aligned_dim);
                ++candidate_read;
            }
            const uint32_t* edges = znode.neighbors();
            const uint16_t deg = znode.degree();
            for (uint16_t j = 0; j < deg; ++j) {
                if (edges[j] == del_id) {
                    approx_in.push_back(ne.id);
                    break;
                }
            }
          }
          candidate_ids.resize(candidate_write);
        }
        if (candidate_ids.empty()) continue;

        // Step 4: repair each in-neighbor. Two-phase: (A) lock-free plan,
        // (B) brief WRITE pin to apply.
        {
          RepairIOPhaseScope io_step4(g_del_repair_io_stats.step4);
          for (uint32_t z : approx_in) {
            // --- Phase A: read z snapshot under READ pin, then release ---
            std::vector<unsigned>    existing;
            tsl::robin_set<unsigned> existing_set;
            std::vector<T>           zcoord_buf(aligned_dim);
            {
                RepairIOPhaseScope io_step4a(g_del_repair_io_stats.step4a_z_read);
                NodeRID zrid;
                PinnedFrame zframe = pin_page_for_node(z, READ, &zrid);
                if (!zframe.valid()) continue;
                auto zguard = zframe.read_guard();
                ConstNodeRef znode = zguard.read_node(zrid);
                if (!znode.valid() || znode.node_id() != z ||
                    !is_active(z)) continue;
                existing.reserve(znode.degree());
                existing_set.reserve(static_cast<size_t>(znode.degree()) * 2);
                const uint32_t* nbrs = znode.neighbors();
                const uint16_t  deg = znode.degree();
                for (uint16_t j = 0; j < deg; ++j) {
                    const uint32_t e = nbrs[j];
                    if (delete_to_batch_idx.find(e) !=
                        delete_to_batch_idx.end()) continue;
                    if (!is_active(e)) continue;   // drop foreign tombstones
                    existing.push_back(e);
                    existing_set.insert(e);
                }
                std::memcpy(zcoord_buf.data(), znode.coords<T>(),
                            static_cast<size_t>(aligned_dim) * sizeof(T));
            }

            std::vector<unsigned> plan;
            {
                RepairIOPhaseScope io_step4b(g_del_repair_io_stats.step4b_prune);
            // Pick up to c_replace closest candidates not already in z.
            std::vector<std::pair<float, size_t>> cand_ranked;
            cand_ranked.reserve(candidate_ids.size());
            for (size_t i = 0; i < candidate_ids.size(); ++i) {
                const uint32_t cid = candidate_ids[i];
                if (cid == z) continue;
                if (existing_set.count(cid)) continue;
                const T* cv = candidate_coords.data() + i * aligned_dim;
                float d = dist->compare(zcoord_buf.data(), cv, aligned_dim);
                cand_ranked.emplace_back(d, i);
            }
            std::sort(cand_ranked.begin(), cand_ranked.end());

            std::vector<unsigned> additions;
            additions.reserve(c_replace);
            for (const auto& cb : cand_ranked) {
                if (additions.size() >= c_replace) break;
                additions.push_back(candidate_ids[cb.second]);
            }

            // Decide plan: either append (no overflow) or prune.
            const size_t new_total = existing.size() + additions.size();
            if (new_total <= R) {
                plan = existing;
                plan.insert(plan.end(), additions.begin(), additions.end());
            } else {
                std::vector<unsigned> pool_ids = existing;
                for (size_t k = 0; k < additions.size(); ++k) {
                    pool_ids.push_back(additions[k]);
                }
                g_del_repair_io_stats.step4b_prune_calls.fetch_add(
                    1, std::memory_order_relaxed);
                call_pruned_y_step4.insert(z);
                prune_to_R_pq(z, pool_ids, plan);
                if (plan.empty()) plan = existing;
            }
            }

            // --- Phase B: WRITE pin and apply ---
            {
                RepairIOPhaseScope io_step4c(g_del_repair_io_stats.step4c_write);
                NodeRID         zrid;
                PinnedFrame     zframe;
                FrameWriteGuard zguard;
                MutableNodeRef  znode;
                if (!live_writable_node(z, zframe, zguard, znode, zrid))
                  continue;
                write_neighbors(znode, plan);
                repaired++;
            }
          }
        }

        // Step 5: bridge propagation -- proposal phase only. For each
        // out-neighbor w of del_id, pick up to c_replace candidate y's to
        // absorb w and append w into step5_pending[y]. Flushing is deferred
        // to a single call-scope pass after the for(del_id) loop, so each
        // unique y across the whole batch is read/maybe-pruned/written once.
        {
          RepairIOPhaseScope io_step5(g_del_repair_io_stats.step5);

          // Track this deletion's distinct y proposals for the existing
          // per-deletion unique-y counter (semantic preserved across the
          // call-scope coalescing change).
          std::unordered_set<uint32_t> per_del_unique_y;
          per_del_unique_y.reserve(del_out_nbrs.size() * c_replace);

          for (uint32_t w : del_out_nbrs) {                                                         // <-------- for (w)
            if (!is_active(w))
              continue;

            std::vector<float> w_pq_dists(candidate_ids.size());
            {
                RepairIOPhaseScope io_step5a(g_del_repair_io_stats.step5a_w_fetch);
                if (pq_codes_for_node(w) == nullptr) continue;
                std::vector<uint8_t> w_pq_scratch(candidate_ids.size() * static_cast<size_t>(_n_chunks));
                compute_pq_dists_src(w, candidate_ids.data(), w_pq_dists.data(), static_cast<uint32_t>(candidate_ids.size()), w_pq_scratch.data(), *pq_table);
            }

            std::vector<std::pair<float, size_t>> w_ranked;
            w_ranked.reserve(candidate_ids.size());
            for (size_t i = 0; i < candidate_ids.size(); ++i) {
                const uint32_t yid = candidate_ids[i];
                if (yid == w) continue;
                if (pq_codes_for_node(yid) == nullptr) continue;
                w_ranked.emplace_back(w_pq_dists[i], i);
            }
            std::sort(w_ranked.begin(), w_ranked.end());

            uint32_t added = 0;
            for (const auto& wr : w_ranked) {
                if (added >= c_replace) break;
                const uint32_t y = candidate_ids[wr.second];
                if (y == w) continue;
                if (!is_active(y)) continue;   // covers other batch members
                step5_pending[y].push_back(w);
                per_del_unique_y.insert(y);
                ++added;
            } // for (wr :  w_ranked)
          } // for (w)

          g_del_repair_io_stats.step5_unique_y_total.fetch_add(
              per_del_unique_y.size(), std::memory_order_relaxed);
        } // step 5 proposal end
    } // for (del_id : delete_batch)

    // Call-scope step-5 flush: each y proposed by any deletion in the batch
    // is read, deduped, maybe-pruned, and written exactly once.
    {
      RepairIOPhaseScope io_step5(g_del_repair_io_stats.step5);
      for (auto& kv : step5_pending) {
        const uint32_t y = kv.first;
        std::vector<uint32_t>& proposed_ws = kv.second;

        std::vector<unsigned> existing;
        std::unordered_map<uint32_t, char> existing_set;
        {
            NodeRID            yrid;
            RepairIOPhaseScope io_step5b(g_del_repair_io_stats.step5b_y_read);
            PinnedFrame yframe = pin_page_for_node(y, READ, &yrid);
            if (!yframe.valid()) continue;
            auto         yguard = yframe.read_guard();
            ConstNodeRef ynode = yguard.read_node(yrid);
            if (!ynode.valid() || ynode.node_id() != y ||
                !is_active(y)) continue;
            existing.reserve(ynode.degree());
            existing_set.reserve(ynode.degree() * 2);
            const uint32_t* nbrs = ynode.neighbors();
            const uint16_t  deg = ynode.degree();
            for (uint16_t j = 0; j < deg; ++j) {
                const uint32_t e = nbrs[j];
                if (delete_to_batch_idx.find(e) !=
                    delete_to_batch_idx.end()) continue;
                if (!is_active(e)) continue;   // drop foreign tombstones
                existing.push_back(e);
                existing_set[e] = 1;
            }
        }

        // Dedupe proposed w's (possibly from multiple deletions) against
        // y's existing list and against each other.
        std::vector<unsigned> ws_to_add;
        ws_to_add.reserve(proposed_ws.size());
        for (uint32_t w : proposed_ws) {
            if (existing_set.find(w) != existing_set.end()) continue;
            existing_set[w] = 1;
            ws_to_add.push_back(w);
        }
        if (ws_to_add.empty()) continue;

        std::vector<unsigned> plan;
        if (existing.size() + ws_to_add.size() <= R) {
            plan = existing;
            plan.insert(plan.end(), ws_to_add.begin(), ws_to_add.end());
        } else {
            std::vector<unsigned> pool = existing;
            pool.insert(pool.end(), ws_to_add.begin(), ws_to_add.end());
            {
                RepairIOPhaseScope io_step5c(g_del_repair_io_stats.step5c_y_prune);
                g_del_repair_io_stats.step5c_prune_calls.fetch_add(
                    1, std::memory_order_relaxed);
                call_pruned_y_step5.insert(y);
                prune_to_R_pq(y, pool, plan);
            }
        }
        if (plan.empty()) continue;

        NodeRID         yrid2;
        PinnedFrame     yframe2;
        FrameWriteGuard yguard2;
        MutableNodeRef  ynode2;
        bool            y_writable = false;
        {
            RepairIOPhaseScope io_step5d(g_del_repair_io_stats.step5d_y_write);
            y_writable = live_writable_node(y, yframe2, yguard2, ynode2, yrid2);
        }
        if (!y_writable) continue;
        write_neighbors(ynode2, plan);
        ++repaired;
      } // for (y in step5_pending)
    } // call-scope step 5 flush end

    g_del_repair_io_stats.n_repair_calls.fetch_add(1, std::memory_order_relaxed);
    g_del_repair_io_stats.delete_batch_size_total.fetch_add(
        delete_batch.size(), std::memory_order_relaxed);
    g_del_repair_io_stats.step5_call_unique_y_total.fetch_add(
        step5_pending.size(), std::memory_order_relaxed);
    g_del_repair_io_stats.step4_call_unique_pruned_y_total.fetch_add(
        call_pruned_y_step4.size(), std::memory_order_relaxed);
    g_del_repair_io_stats.step5_call_unique_pruned_y_total.fetch_add(
        call_pruned_y_step5.size(), std::memory_order_relaxed);

    return repaired;
}

template uint32_t InPlaceGraphStore::repair_deleted_single<float>(
    uint32_t, unsigned, unsigned, float, unsigned, diskann::Distance<float>*,
    uint32_t, uint32_t, uint32_t, FixedChunkPQTable<float>*);
template uint32_t InPlaceGraphStore::repair_deleted_single<uint8_t>(
    uint32_t, unsigned, unsigned, float, unsigned, diskann::Distance<uint8_t>*,
    uint32_t, uint32_t, uint32_t, FixedChunkPQTable<uint8_t>*);
template uint32_t InPlaceGraphStore::repair_deleted_single<int8_t>(
    uint32_t, unsigned, unsigned, float, unsigned, diskann::Distance<int8_t>*,
    uint32_t, uint32_t, uint32_t, FixedChunkPQTable<int8_t>*);
template uint32_t InPlaceGraphStore::repair_deleted_explicit<float>(
    const std::vector<uint32_t>&, unsigned, unsigned, float, unsigned,
    diskann::Distance<float>*, uint32_t, uint32_t, uint32_t,
    FixedChunkPQTable<float>*);
template uint32_t InPlaceGraphStore::repair_deleted_explicit<uint8_t>(
    const std::vector<uint32_t>&, unsigned, unsigned, float, unsigned,
    diskann::Distance<uint8_t>*, uint32_t, uint32_t, uint32_t,
    FixedChunkPQTable<uint8_t>*);
template uint32_t InPlaceGraphStore::repair_deleted_explicit<int8_t>(
    const std::vector<uint32_t>&, unsigned, unsigned, float, unsigned,
    diskann::Distance<int8_t>*, uint32_t, uint32_t, uint32_t,
    FixedChunkPQTable<int8_t>*);

// ===========================================================================
// Explicit template instantiations for graph ops
// ===========================================================================
template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<float>(
    const float*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<float>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&, bool, TraversalScope,
    FixedChunkPQTable<float>*);

template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<uint8_t>(
    const uint8_t*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<uint8_t>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&, bool, TraversalScope,
    FixedChunkPQTable<uint8_t>*);

template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<int8_t>(
    const int8_t*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<int8_t>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&, bool, TraversalScope,
    FixedChunkPQTable<int8_t>*);

template void graph_occlude_list_pq<float>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope,
    FixedChunkPQTable<float>*);
template void graph_occlude_list_pq<uint8_t>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope,
    FixedChunkPQTable<uint8_t>*);
template void graph_occlude_list_pq<int8_t>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope,
    FixedChunkPQTable<int8_t>*);

template void graph_occlude_list<float>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, unsigned, Distance<float>*,
    DistanceScope);

template void graph_prune_neighbors_pq<float>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<float>*, DistanceScope, FixedChunkPQTable<float>*);
template void graph_prune_neighbors_pq<uint8_t>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<uint8_t>*, DistanceScope, FixedChunkPQTable<uint8_t>*);
template void graph_prune_neighbors_pq<int8_t>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<int8_t>*, DistanceScope, FixedChunkPQTable<int8_t>*);

}  // namespace inplace
}  // namespace diskann
