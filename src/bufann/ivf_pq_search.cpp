#include "bufann/ivf_pq_search.h"

#include <mkl.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <shared_mutex>

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

namespace {

using Clock = std::chrono::steady_clock;

// Adds the time since `t` to `acc` and moves `t` to now.
void lap(Clock::time_point& t, uint64_t& acc) {
    const Clock::time_point now = Clock::now();
    acc += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count());
    t = now;
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
    IVF_PQ_REQUIRE(pq_shape_ok(ix.pq, ix.meta.dim),
                   "PQMetadata is inconsistent with the index");
    IVF_PQ_REQUIRE(ix.lists.offsets.size() == size_t(ix.meta.nlist) + 1 && ix.lists.offsets.front() == 0 &&
                       ix.lists.ids.size() == n_base,
                   "PostingLists are inconsistent with the index");
    IVF_PQ_REQUIRE(ix.lists.codes.size() == n_base * ix.pq.chunks ||
                       ivf_blocked_codes_cover(ix.lists.blocked, ix.lists, ix.pq.chunks),
                   "neither posting-list nor blocked codes cover the lists (see set_ivf_posting_codes)");
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

// Candidate generation by the SIMD scan: the rerank_m live codes with the
// smallest quantized PQ distances in the probed lists (and the delta's
// inserts there) into s.candidates / s.pq_dist, taking `delta_lock` (shared)
// when there is a delta. The table's range is set from the rerank_m-th
// smallest float distance among the first 2 x rerank_m live codes in probe
// order, an upper bound on the final rerank_m-th; codes are kept on a
// running threshold, so nothing beyond the top rerank_m is stored.
void fast_candidates(const IVFPQIndex& ix, const IVFPQDelta* delta, uint32_t R, IVFPQSearchScratch& s,
                     std::shared_lock<WriterPreferringSharedMutex>& delta_lock) {
    const IVFPQBlockedCodes& bc = ix.lists.blocked;
    const uint32_t chunks = ix.pq.chunks;
    const size_t block_bytes = size_t(chunks) * IVF_FASTSCAN_BLOCK;
    const float* table = s.pq_table.data();
    if (delta != nullptr) delta_lock = std::shared_lock<WriterPreferringSharedMutex>(delta->mtx);
    IVF_PQ_REQUIRE(ix.rid_table.rid.size() >= ivf_pq_num_base(ix), "RID table does not cover every base vector");
    const tsl::robin_set<uint32_t>* dead =
        delta != nullptr && !delta->lists.tombstones.empty() ? &delta->lists.tombstones : nullptr;
    auto list_blocks = [&](uint32_t part) { return bc.blocks.data() + size_t(bc.block_start[part]) * block_bytes; };

    s.sample.clear();
    const size_t want = 2 * size_t(R);
    for (uint32_t part : s.probe_order) {
        const uint32_t begin = ix.lists.offsets[part], n = ix.lists.offsets[part + 1] - begin;
        const uint8_t* first = list_blocks(part);
        for (uint32_t i = 0; i < n && s.sample.size() < want; ++i) {
            if (dead != nullptr && dead->count(ix.lists.ids[begin + i]) != 0) continue;
            s.sample.push_back(adc_block_lane(first + size_t(i / IVF_FASTSCAN_BLOCK) * block_bytes, chunks,
                                              i % IVF_FASTSCAN_BLOCK, table));
        }
        if (s.sample.size() >= want) break;
    }
    // Fewer live codes than R in the sample means fewer than R in the probed
    // lists' first 2R: keep everything rather than bound by a distance.
    const bool bounded = s.sample.size() >= R;
    float dmax = 0.0f;
    if (bounded) {
        std::nth_element(s.sample.begin(), s.sample.begin() + (R - 1), s.sample.end());
        dmax = s.sample[R - 1];
    } else if (!s.sample.empty()) {
        dmax = *std::max_element(s.sample.begin(), s.sample.end());
    }
    quantize_pq_table(table, chunks, dmax, s.qtable);
    // A code no farther than dmax sums to at most 255, plus half a bin of
    // rounding per sub-quantizer.
    uint32_t threshold = bounded ? 256 + chunks / 2 : 0x10000;
    s.top.clear();
    auto push = [&](uint16_t q, uint64_t id) {
        s.top.emplace_back(q, id);
        std::push_heap(s.top.begin(), s.top.end());
        if (s.top.size() > R) {
            std::pop_heap(s.top.begin(), s.top.end());
            s.top.pop_back();
        }
        if (s.top.size() == R) threshold = s.top.front().first;
    };
    // The heap orders by (distance, id) as it always has; the posting
    // position rides below the id (IVF_NO_POSTING_POS for inserts).
    auto entry = [](uint32_t id, uint32_t pos) { return (uint64_t(id) << 32) | pos; };
    for (size_t i = 0; i < s.probe_order.size(); ++i) {
        // The next list is a random jump in memory; ask for its start now.
        if (i + 1 < s.probe_order.size()) {
            const uint8_t* next = list_blocks(s.probe_order[i + 1]);
            for (size_t off = 0; off < 2 * block_bytes; off += 64) __builtin_prefetch(next + off);
        }
        const uint32_t part = s.probe_order[i];
        const uint32_t begin = ix.lists.offsets[part], n = ix.lists.offsets[part + 1] - begin;
        const uint32_t nblocks = (n + IVF_FASTSCAN_BLOCK - 1) / IVF_FASTSCAN_BLOCK;
        if (s.block_sums.size() < size_t(nblocks) * IVF_FASTSCAN_BLOCK) s.block_sums.resize(size_t(nblocks) * IVF_FASTSCAN_BLOCK);
        scan_blocks(list_blocks(part), nblocks, s.qtable, s.block_sums.data());
        for (uint32_t b = 0; b < nblocks; ++b) {
            const uint16_t* sums = s.block_sums.data() + size_t(b) * IVF_FASTSCAN_BLOCK;
            const uint32_t* ids = ix.lists.ids.data() + begin + size_t(b) * IVF_FASTSCAN_BLOCK;
            // Most lanes are over the threshold; visit only those under it.
            uint64_t under = lanes_below(sums, std::min(IVF_FASTSCAN_BLOCK, n - b * IVF_FASTSCAN_BLOCK), threshold);
            for (; under != 0; under &= under - 1) {
                const uint32_t lane = uint32_t(__builtin_ctzll(under));
                // An earlier push in this block may have lowered the threshold.
                if (sums[lane] >= threshold) continue;
                if (dead != nullptr && dead->count(ids[lane]) != 0) continue;
                push(sums[lane], entry(ids[lane], begin + b * IVF_FASTSCAN_BLOCK + lane));
            }
        }
    }
    if (delta != nullptr) {
        const float inv = 1.0f / s.qtable.delta;
        for (uint32_t part : s.probe_order) {
            auto pending = delta->lists.pending_inserts.find(part);
            if (pending == delta->lists.pending_inserts.end()) continue;
            for (uint32_t id : pending->second) {
                auto code = delta->codes.codes.find(id);
                IVF_PQ_REQUIRE(code != delta->codes.codes.end() && code->second.size() == chunks,
                               "inserted vector " + std::to_string(id) + " has no PQ code in the delta");
                float d = 0.0f;
                for (uint32_t c = 0; c < chunks; ++c) d += table[size_t(c) * 256 + code->second[c]];
                const float q = std::nearbyint((d - s.qtable.dmin) * inv);
                const uint16_t qq = uint16_t(std::clamp(q, 0.0f, 65535.0f));
                if (qq < threshold) push(qq, entry(id, IVF_NO_POSTING_POS));
            }
        }
    }
    s.candidates.clear();
    s.candidate_pos.clear();
    s.pq_dist.clear();
    for (const auto& [q, v] : s.top) {
        s.candidates.push_back(uint32_t(v >> 32));
        s.candidate_pos.push_back(uint32_t(v));
        s.pq_dist.push_back(s.qtable.dmin + float(q) * s.qtable.delta);
    }
}

// How many distinct pages the first n of `slots` lie on.
uint32_t distinct_pages(const RawVectorHeapLayout& layout, const uint32_t* slots, size_t n,
                        std::vector<uint32_t>& pages) {
    pages.resize(n);
    for (size_t i = 0; i < n; ++i) pages[i] = layout.page_of(slots[i]);
    std::sort(pages.begin(), pages.end());
    return uint32_t(std::unique(pages.begin(), pages.end()) - pages.begin());
}

void require_ex_args(const IVFPQIndex& ix, const IVFExRerank& ex, uint32_t k) {
    if (ex.store == nullptr) return;
    IVF_PQ_REQUIRE(ix.excodes.bits == IVF_EXCODE_BITS, "ex-code re-rank on an index without an ex-code store");
    IVF_PQ_REQUIRE(ex.store->layout().elem_size == ix.excodes.layout.elem_size &&
                       ex.store->layout().page_size == ix.excodes.layout.page_size &&
                       ex.store->next_flat_slot() == ivf_pq_num_base(ix),
                   "ex-code store does not match the index's (open it with index.excodes' geometry)");
    IVF_PQ_REQUIRE(ex.exact == 0 || ex.exact >= k, "ex-code exact re-rank count must be 0 or at least k");
}

// The ex-code stage of a re-rank (IVFExRerank): estimates the shortlist's
// vectors that have records, read in posting order so a page is pinned once
// per query, then leaves in the shortlist (nearest estimate first) only the
// `exact` nearest of them and the inserts, which have no record; with exact
// 0 the others go to approx_ids/approx_dists with their estimates.
void excode_stage(const IVFPQIndex& ix, const RawVectorHeap& store, uint32_t exact, const float* query,
                  IVFPQSearchScratch& s, IVFPQQueryIO& io, Clock::time_point& t) {
    const uint32_t dim = ix.meta.dim;
    const size_t rec = store.layout().elem_size;
    IVFPQSearchTimes& times = s.times;
    s.ex_index.clear();
    for (uint32_t i = 0; i < s.shortlist.size(); ++i) {
        if (s.shortlist_pos[i] != IVF_NO_POSTING_POS) s.ex_index.push_back(i);
    }
    std::sort(s.ex_index.begin(), s.ex_index.end(),
              [&](uint32_t a, uint32_t b) { return s.shortlist_pos[a] < s.shortlist_pos[b]; });
    const size_t m = s.ex_index.size();
    s.ex_slots.resize(m);
    for (size_t j = 0; j < m; ++j) s.ex_slots[j] = s.shortlist_pos[s.ex_index[j]];
    s.ex_records.resize(m * rec);
    s.ex_owners.resize(m);
    store.read_vectors(s.ex_slots.data(), m, s.ex_records.data(), s.ex_owners.data());
    io.ex_records = uint32_t(m);
    io.ex_pages = distinct_pages(store.layout(), s.ex_slots.data(), m, s.pages);
    times.ex_records += io.ex_records;
    times.ex_pages += io.ex_pages;
    lap(t, times.ex_fetch_ns);

    ivf_excode_prepare_query(ix.excodes, dim, query, s.ex_query);
    s.ex_est.resize(m);
    // Positions ascend, so the partition only moves forward.
    uint32_t list = 0;
    float qc = 0.0f;
    bool have_qc = false;
    for (size_t j = 0; j < m; ++j) {
        const uint32_t i = s.ex_index[j];
        IVF_PQ_REQUIRE(s.ex_owners[j] == s.shortlist[i], "ex-code slot " + std::to_string(s.ex_slots[j]) +
                                                             " holds vector " + std::to_string(s.ex_owners[j]) +
                                                             ", not " + std::to_string(s.shortlist[i]));
        const uint32_t pos = s.ex_slots[j];
        while (ix.lists.offsets[list + 1] <= pos) ++list, have_qc = false;
        if (!have_qc) {
            qc = sq_dist(query, ix.meta.centroids.data() + size_t(list) * ix.meta.aligned_dim, dim);
            have_qc = true;
        }
        s.ex_est[j] = ivf_excode_estimate(reinterpret_cast<const uint8_t*>(s.ex_records.data() + j * rec), dim,
                                          s.ex_query, qc);
    }
    // Shortlist entries by estimate: the exact set first, inserts after it.
    select_smallest(s.ex_est, uint32_t(m), s.order);
    const size_t keep = exact == 0 ? 0 : std::min<size_t>(exact, m);
    std::vector<uint32_t> ids, slots;
    std::vector<float> dists;
    for (size_t r = 0; r < m; ++r) {
        const uint32_t j = s.order[r], i = s.ex_index[j];
        if (r < keep) {
            ids.push_back(s.shortlist[i]), slots.push_back(s.shortlist_slot[i]), dists.push_back(s.ex_est[j]);
        } else if (exact == 0) {
            s.approx_ids.push_back(s.shortlist[i]), s.approx_dists.push_back(s.ex_est[j]);
        }
    }
    for (uint32_t i = 0; i < s.shortlist.size(); ++i) {
        if (s.shortlist_pos[i] != IVF_NO_POSTING_POS) continue;
        ids.push_back(s.shortlist[i]), slots.push_back(s.shortlist_slot[i]), dists.push_back(s.exact_dist[i]);
    }
    s.shortlist.swap(ids);
    s.shortlist_slot.swap(slots);
    s.exact_dist.swap(dists);
    lap(t, times.ex_estimate_ns);
}

// Everything after choosing the partitions in scratch.probe_order: PQ table,
// scan of the probed posting lists, selection, optional exact re-rank. `t`
// is when the partitions were chosen; the phases are timed from there.
template<typename T>
IVFPQSearchResult search_probes(const IVFPQIndex& ix, const RawVectorHeap& heap, const IVFPQDelta* delta,
                                const float* query, uint32_t k, uint32_t rerank_m, const IVFRerankSchedule& schedule,
                                const IVFExRerank& ex, IVFPQSearchScratch& scratch, Clock::time_point t) {
    const IVFMetadata& meta = ix.meta;
    const PQMetadata& pq = ix.pq;
    const uint32_t dim = meta.dim;
    IVFPQSearchTimes& times = scratch.times;
    ++times.queries;

    // Held from the first RID load through the last read_vector, so a slot a
    // concurrent delete frees is not refilled under this query's re-rank.
    RawVectorHeap::ReadGuard guard(heap);

    for (uint32_t c = 0; c < pq.chunks; ++c) {
        const uint32_t len = pq_chunk_len(pq, c);
        for (uint32_t j = 0; j < pq.k; ++j) {
            scratch.pq_table[size_t(c) * pq.k + j] = sq_dist(query + pq.chunk_offsets[c], pq_pivot(pq, c, j), len);
        }
    }
    lap(t, times.table_ns);

    const bool fast = rerank_m > 0 && pq.k == 256 && fastscan_enabled() && fastscan_simd_available() &&
                      ix.lists.blocked.chunks == pq.chunks && !ix.lists.blocked.blocks.empty();
    std::shared_lock<WriterPreferringSharedMutex> delta_lock;
    if (fast) {
        fast_candidates(ix, delta, rerank_m, scratch, delta_lock);
    } else {
        auto add_candidate = [&](uint32_t id, const uint8_t* code, uint32_t pos) {
            float d = 0.0f;
            for (uint32_t c = 0; c < pq.chunks; ++c) d += scratch.pq_table[size_t(c) * pq.k + code[c]];
            scratch.candidates.push_back(id);
            scratch.candidate_pos.push_back(pos);
            scratch.pq_dist.push_back(d);
        };
        scratch.candidates.clear();
        scratch.candidate_pos.clear();
        scratch.pq_dist.clear();
        if (!ix.lists.codes.empty()) {
            for (uint32_t part : scratch.probe_order) {
                for (uint32_t i = ix.lists.offsets[part]; i < ix.lists.offsets[part + 1]; ++i) {
                    add_candidate(ix.lists.ids[i], ix.lists.codes.data() + size_t(i) * pq.chunks, i);
                }
            }
        } else {
            // A loaded index keeps only the blocked layout.
            const size_t block_bytes = size_t(pq.chunks) * IVF_FASTSCAN_BLOCK;
            for (uint32_t part : scratch.probe_order) {
                const uint32_t begin = ix.lists.offsets[part], n = ix.lists.offsets[part + 1] - begin;
                const uint8_t* first = ivf_list_blocks(ix.lists.blocked, part);
                for (uint32_t i = 0; i < n; ++i) {
                    scratch.candidates.push_back(ix.lists.ids[begin + i]);
                    scratch.candidate_pos.push_back(begin + i);
                    scratch.pq_dist.push_back(adc_block_lane(first + size_t(i / IVF_FASTSCAN_BLOCK) * block_bytes,
                                                             pq.chunks, i % IVF_FASTSCAN_BLOCK,
                                                             scratch.pq_table.data()));
                }
            }
        }

        // Inserts grow the delta and the RID table, so both are read under the
        // delta's shared lock: here for the delta, again below for the
        // shortlist's RIDs, with the selection between them unlocked so a
        // waiting writer is held up by at most a scan.
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
                    scratch.candidate_pos[live] = scratch.candidate_pos[i];
                    scratch.pq_dist[live] = scratch.pq_dist[i];
                    ++live;
                }
                scratch.candidates.resize(live);
                scratch.candidate_pos.resize(live);
                scratch.pq_dist.resize(live);
            }
            for (uint32_t part : scratch.probe_order) {
                auto pending = delta->lists.pending_inserts.find(part);
                if (pending == delta->lists.pending_inserts.end()) continue;
                for (uint32_t id : pending->second) {
                    auto code = delta->codes.codes.find(id);
                    IVF_PQ_REQUIRE(code != delta->codes.codes.end() && code->second.size() == pq.chunks,
                                   "inserted vector " + std::to_string(id) + " has no PQ code in the delta");
                    add_candidate(id, code->second.data(), IVF_NO_POSTING_POS);
                }
            }
        }
    }
    if (delta_lock.owns_lock()) delta_lock.unlock();
    lap(t, times.scan_ns);

    // The active check covers a delete that landed while the lock was
    // released, and an index searched without its delta.
    select_smallest(scratch.pq_dist, rerank_m == 0 ? k : rerank_m, scratch.order);
    if (delta != nullptr) delta_lock.lock();
    scratch.shortlist.clear();
    scratch.shortlist_slot.clear();
    scratch.shortlist_pos.clear();
    scratch.exact_dist.clear();
    for (uint32_t i : scratch.order) {
        const RawVectorRID rid = load_rid(ix.rid_table.rid[scratch.candidates[i]]);
        if (rid_is_active(rid)) {
            scratch.shortlist.push_back(scratch.candidates[i]);
            scratch.shortlist_slot.push_back(rid_flat_slot(rid));
            scratch.shortlist_pos.push_back(scratch.candidate_pos[i]);
            scratch.exact_dist.push_back(scratch.pq_dist[i]);
        }
    }
    if (delta_lock.owns_lock()) delta_lock.unlock();
    lap(t, times.select_ns);

    IVFPQSearchResult result;
    if (rerank_m == 0) {
        result.ids = scratch.shortlist;
        result.dists = scratch.exact_dist;
        return result;
    }
    scratch.approx_ids.clear();
    scratch.approx_dists.clear();
    if (ex.store != nullptr) excode_stage(ix, *ex.store, ex.exact, query, scratch, result.io, t);

    // Each fetch is one buffer-pool batch, whose misses are read
    // concurrently, as BufANN's beam search and LindormVector's batch get do.
    // The shortlist is nearest by PQ distance first, so a stopped re-rank
    // has read the prefix most likely to hold the top k.
    const size_t elem_size = heap.layout().elem_size;
    const size_t n = scratch.shortlist.size();
    const size_t step = schedule.batch == 0 ? std::max<size_t>(n, 1) : schedule.batch;
    scratch.raw_vectors.resize(n * elem_size);
    scratch.owners.resize(n);
    // Only to tell whether a batch improved the top k.
    std::vector<float>& top = scratch.top_exact;
    top.clear();
    size_t done = 0;
    uint32_t idle = 0;
    while (done < n) {
        const size_t m = std::min(step, n - done);
        heap.read_vectors(scratch.shortlist_slot.data() + done, m, scratch.raw_vectors.data() + done * elem_size,
                          scratch.owners.data() + done);
        ++times.fetch_batches;
        lap(t, times.fetch_ns);
        bool improved = false;
        for (size_t i = done; i < done + m; ++i) {
            const uint32_t owner = scratch.owners[i];
            IVF_PQ_REQUIRE(owner == scratch.shortlist[i], "raw-vector slot " + std::to_string(scratch.shortlist_slot[i]) +
                                                              " holds vector " + std::to_string(owner) + ", not " +
                                                              std::to_string(scratch.shortlist[i]));
            const T* raw = reinterpret_cast<const T*>(scratch.raw_vectors.data() + i * elem_size);
            for (uint32_t d = 0; d < dim; ++d) scratch.vector[d] = float(raw[d]);
            const float d = sq_dist(query, scratch.vector.data(), dim);
            scratch.exact_dist[i] = d;
            if (top.size() < k || d < top.front()) {
                improved = true;
                top.push_back(d);
                std::push_heap(top.begin(), top.end());
                if (top.size() > k) {
                    std::pop_heap(top.begin(), top.end());
                    top.pop_back();
                }
            }
        }
        done += m;
        lap(t, times.rerank_ns);
        idle = improved ? 0 : idle + 1;
        if (schedule.patience > 0 && idle == schedule.patience) break;
    }
    times.reranked += done;
    result.io.raw_vectors = uint32_t(done);
    result.io.raw_pages = distinct_pages(heap.layout(), scratch.shortlist_slot.data(), done, scratch.pages);
    times.raw_pages += result.io.raw_pages;
    scratch.shortlist.resize(done);
    scratch.exact_dist.resize(done);
    scratch.shortlist.insert(scratch.shortlist.end(), scratch.approx_ids.begin(), scratch.approx_ids.end());
    scratch.exact_dist.insert(scratch.exact_dist.end(), scratch.approx_dists.begin(), scratch.approx_dists.end());
    select_smallest(scratch.exact_dist, k, scratch.order);
    for (uint32_t i : scratch.order) {
        result.ids.push_back(scratch.shortlist[i]);
        result.dists.push_back(scratch.exact_dist[i]);
    }
    lap(t, times.rerank_ns);
    return result;
}

// The nprobe partitions nearest by `centroid_dist` ([nlist]), then the rest;
// `t` is when the query's centroid search began.
template<typename T>
IVFPQSearchResult search_from_centroid_dist(const IVFPQIndex& ix, const RawVectorHeap& heap,
                                            const IVFPQDelta* delta, const float* query,
                                            const float* centroid_dist, uint32_t k, uint32_t nprobe,
                                            uint32_t rerank_m, const IVFRerankSchedule& schedule,
                                            const IVFExRerank& ex, IVFPQSearchScratch& scratch,
                                            Clock::time_point t) {
    size_scratch(scratch, ix);
    std::copy_n(centroid_dist, ix.meta.nlist, scratch.centroid_dist.begin());
    select_smallest(scratch.centroid_dist, std::min(nprobe, ix.meta.nlist), scratch.probe_order);
    lap(t, scratch.times.centroid_ns);
    return search_probes<T>(ix, heap, delta, query, k, rerank_m, schedule, ex, scratch, t);
}

}  // namespace

template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& ix, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch, const IVFPQDelta* delta,
                                const IVFRerankSchedule& rerank, const IVFExRerank& ex) {
    IVF_PQ_REQUIRE(query != nullptr, "query is null");
    require_search_args(ix, k, nprobe, rerank_m, heap.layout().elem_size, sizeof(T));
    require_ex_args(ix, ex, k);
    const Clock::time_point t = Clock::now();
    size_scratch(scratch, ix);
    std::copy_n(query, ix.meta.dim, scratch.query_padded.begin());
    // A one-row GEMM is far cheaper than MKL's thread fan-out for it: on a
    // busy 48-core host the fan-out alone cost ~3 ms per query.
    const int mkl_threads = mkl_set_num_threads_local(1);
    centroid_distances(ix.meta, scratch.query_padded.data(), 1, scratch.centroid_dist.data());
    mkl_set_num_threads_local(mkl_threads);
    return search_from_centroid_dist<T>(ix, heap, delta, query, scratch.centroid_dist.data(), k, nprobe,
                                        rerank_m, rerank, ex, scratch, t);
}

uint32_t ivf_probe_count(const float* dist, uint32_t n, const IVFProbeCutoff& cutoff) {
    if (cutoff.tau == 0.0f || n == 0) return n;
    const float bound = cutoff.tau * dist[std::min(cutoff.ref, n) - 1];
    uint32_t m = std::min(cutoff.min_probes, n);
    while (m < n && dist[m] <= bound) ++m;
    return m;
}

template<typename T>
IVFPQSearchResult ivf_pq_search_graph(const IVFPQIndex& ix, const IVFCentroidGraph& graph, const RawVectorHeap& heap,
                                      const float* query, uint32_t k, uint32_t nprobe, uint32_t centroid_L,
                                      uint32_t rerank_m, IVFPQSearchScratch& scratch, const IVFPQDelta* delta,
                                      const IVFProbeCutoff& cutoff, const IVFRerankSchedule& rerank,
                                      const IVFExRerank& ex) {
    IVF_PQ_REQUIRE(query != nullptr, "query is null");
    require_search_args(ix, k, nprobe, rerank_m, heap.layout().elem_size, sizeof(T));
    require_ex_args(ix, ex, k);
    Clock::time_point t = Clock::now();
    // tau < 1 would cut below the reference itself. NaN is tested on the
    // bits: -Ofast lets the compiler assume float comparisons see no NaN.
    uint32_t tau_bits;
    std::memcpy(&tau_bits, &cutoff.tau, sizeof(tau_bits));
    const bool tau_finite = (tau_bits & 0x7F800000u) != 0x7F800000u;
    IVF_PQ_REQUIRE(tau_finite && (cutoff.tau == 0.0f || cutoff.tau >= 1.0f),
                   "probe cutoff tau must be 0 (off) or a finite value of at least 1");
    IVF_PQ_REQUIRE(cutoff.ref > 0, "probe cutoff reference rank must be at least 1");
    size_scratch(scratch, ix);
    // Throws unless centroid_L >= nprobe (clamped to nlist) and the graph
    // has the index's nlist.
    search_ivf_centroid_graph(ix.meta, graph, query, centroid_L, std::min(nprobe, ix.meta.nlist), scratch.graph,
                              scratch.probe_order, IVF_CENTROID_GRAPH_NONE, &scratch.probe_dist);
    scratch.probe_order.resize(
        ivf_probe_count(scratch.probe_dist.data(), uint32_t(scratch.probe_dist.size()), cutoff));
    lap(t, scratch.times.centroid_ns);
    return search_probes<T>(ix, heap, delta, query, k, rerank_m, rerank, ex, scratch, t);
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
                nprobe, rerank_m, IVFRerankSchedule{}, IVFExRerank{}, scratches[size_t(omp_get_thread_num())],
                Clock::now());
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
                                                const IVFPQDelta*, const IVFRerankSchedule&,               \
                                                const IVFExRerank&);                                       \
    template IVFPQSearchResult ivf_pq_search<T>(const IVFPQIndex&, const RawVectorHeap&, const float*,     \
                                                uint32_t, uint32_t, uint32_t);                             \
    template IVFPQSearchResult ivf_pq_search_graph<T>(const IVFPQIndex&, const IVFCentroidGraph&,           \
                                                      const RawVectorHeap&, const float*, uint32_t, uint32_t, \
                                                      uint32_t, uint32_t, IVFPQSearchScratch&,             \
                                                      const IVFPQDelta*, const IVFProbeCutoff&,            \
                                                      const IVFRerankSchedule&, const IVFExRerank&);       \
    template std::vector<IVFPQSearchResult> ivf_pq_search_batch<T>(const IVFPQIndex&, const RawVectorHeap&, \
                                                                   const float*, size_t, uint32_t, uint32_t, \
                                                                   uint32_t, uint32_t, const IVFPQDelta*);
IVF_PQ_INSTANTIATE_SEARCH(float)
IVF_PQ_INSTANTIATE_SEARCH(uint8_t)
IVF_PQ_INSTANTIATE_SEARCH(int8_t)
#undef IVF_PQ_INSTANTIATE_SEARCH

}  // namespace inplace
}  // namespace diskann
