// IVF-PQ query path: centroid scan -> top-nprobe partitions -> PQ table
// lookup over their posting lists -> exact re-rank of the top rerank_m from
// the raw-vector heap. A batch computes every query's centroid distances
// with one GEMM and then handles the queries in parallel; a single query is
// a batch of one. ivf_pq_search_graph instead finds the partitions through a
// proximity graph over the centroids (ivf_pq_centroid_graph.h).

#pragma once

#include <cstdint>
#include <vector>

#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_centroid_graph.h"
#include "bufann/ivf_pq_excode.h"
#include "bufann/ivf_pq_fastscan.h"
#include "bufann/ivf_pq_raw_vector_heap.h"

namespace diskann {
namespace inplace {

// What one query read from the two on-SSD stores: records and the distinct
// pages holding them, i.e. the pages a reader with no cache would fetch
// (whether the buffer pool had them is in its own hit/miss counters).
constexpr uint32_t IVF_NO_POSTING_POS = 0xFFFFFFFFu;

struct IVFPQQueryIO {
    uint32_t ex_records = 0, ex_pages = 0;    // 9-bit ex-code store
    uint32_t raw_vectors = 0, raw_pages = 0;  // raw-vector heap
};

// Neighbours ascending by dist; fewer than k when the probed partitions hold
// fewer active vectors. dists are exact squared L2 after a re-rank, PQ
// approximations without one, and 9-bit estimates for the vectors an ex-code
// re-rank does not re-rank exactly.
struct IVFPQSearchResult {
    std::vector<uint32_t> ids;
    std::vector<float> dists;
    IVFPQQueryIO io;
};

// Re-rank through the index's 9-bit ex-code store (ivf_pq_excode.h). With
// `store` set, the rerank_m PQ candidates are first ranked by their 9-bit
// estimates, read from the store's pages (a partition's candidates share
// them), and only the `exact` nearest by estimate (0: none) are then fetched
// from the raw-vector heap and re-ranked exactly; the rest keep their
// estimates. Vectors inserted since the build have no record and are always
// re-ranked exactly. `store` must be the heap at ivf_excodes_path opened with
// index.excodes' layout; exact must be 0 or at least k.
struct IVFExRerank {
    const RawVectorHeap* store = nullptr;
    uint32_t exact = 0;
};

// Per-query probe depth for ivf_pq_search_graph; off (tau == 0) by default.
// When on, nprobe is a cap: of the partitions the centroid search returns,
// nearest first, the first min_probes are probed and then each one whose
// squared centroid distance is at most tau times the ref-th nearest's.
// Queries whose neighbours sit in one tight region stop early; a fixed
// nprobe pays for the widest query everywhere. On SIFT100M (nlist 131072)
// it probes 1.7-2.3x fewer codes at the probe recall of a fixed nprobe.
struct IVFProbeCutoff {
    float tau = 0.0f;         // 0: probe all nprobe; otherwise >= 1
    uint32_t ref = 8;         // the reference is the ref-th nearest partition
    uint32_t min_probes = 16;
};

// How many of `n` partitions, ascending squared distances `dist`, the cutoff
// probes: n when it is off; otherwise at least min(n, min_probes).
uint32_t ivf_probe_count(const float* dist, uint32_t n, const IVFProbeCutoff& cutoff);

// How the shortlist's raw vectors are fetched for the exact re-rank. By
// default (batch 0) all of them in one buffer-pool batch, whose misses are
// read concurrently. Otherwise `batch` at a time, nearest by PQ distance
// first, each batch's misses read concurrently; with patience > 0 the
// re-rank stops once `patience` batches in a row put nothing into the exact
// top k, so a query whose PQ order is already right reads a prefix of its
// shortlist. Batches without a stop only add round trips.
struct IVFRerankSchedule {
    uint32_t batch = 0;
    uint32_t patience = 0;
};

// Where a thread's queries spent their time, summed over the queries searched
// with one scratch: choosing the partitions (GEMV or graph), building the PQ
// table, scanning the codes, picking the shortlist and its RIDs, fetching its
// raw vectors (buffer-pool pins, I/O included) and computing exact distances.
// The phases are back to back, so their sum is the search's own time.
// With an ex-code re-rank, ex_fetch is reading its records and ex_estimate
// rotating the query and estimating. The I/O counts sum IVFPQQueryIO.
struct IVFPQSearchTimes {
    uint64_t centroid_ns = 0, table_ns = 0, scan_ns = 0, select_ns = 0, fetch_ns = 0, rerank_ns = 0;
    uint64_t ex_fetch_ns = 0, ex_estimate_ns = 0;
    uint64_t queries = 0, reranked = 0, fetch_batches = 0;
    uint64_t ex_records = 0, ex_pages = 0, raw_pages = 0;

    IVFPQSearchTimes& operator+=(const IVFPQSearchTimes& o) {
        centroid_ns += o.centroid_ns, table_ns += o.table_ns, scan_ns += o.scan_ns, select_ns += o.select_ns;
        fetch_ns += o.fetch_ns, rerank_ns += o.rerank_ns;
        ex_fetch_ns += o.ex_fetch_ns, ex_estimate_ns += o.ex_estimate_ns;
        queries += o.queries, reranked += o.reranked, fetch_batches += o.fetch_batches;
        ex_records += o.ex_records, ex_pages += o.ex_pages, raw_pages += o.raw_pages;
        return *this;
    }
};

// Buffers one thread reuses across queries; sized to the index on every
// search (no-ops once they fit), so a scratch may be used on any index but
// not shared between threads.
struct IVFPQSearchScratch {
    IVFPQSearchTimes times;             // accumulated, never reset by a search
    std::vector<float> query_padded;    // [aligned_dim], zero beyond dim
    std::vector<float> centroid_dist;   // [nlist]
    std::vector<uint32_t> probe_order;  // [nlist] partition ids, nearest first
    std::vector<float> probe_dist;      // the graph search's squared distances, parallel to probe_order
    std::vector<float> pq_table;        // [chunks x k]
    std::vector<uint32_t> candidates;   // ids scanned from the probed partitions
    std::vector<uint32_t> candidate_pos;  // their posting positions; IVF_NO_POSTING_POS for inserts
    std::vector<float> pq_dist;         // parallel to candidates
    std::vector<uint32_t> order;        // selection workspace over candidates
    std::vector<uint32_t> shortlist;    // ids kept for the re-rank
    std::vector<uint32_t> shortlist_slot;  // their heap slots, parallel to shortlist
    std::vector<uint32_t> shortlist_pos;   // their posting positions, parallel to shortlist
    std::vector<float> exact_dist;      // parallel to shortlist
    std::vector<float> top_exact;       // max-heap of the k smallest exact distances so far
    std::vector<char> raw_vectors;      // [shortlist x heap elem_size]
    std::vector<uint32_t> owners;       // parallel to shortlist
    std::vector<float> vector;          // [dim]
    std::vector<uint32_t> pages;        // page ids, for counting distinct ones
    // The ex-code re-rank (IVFExRerank).
    IVFExQuery ex_query;
    std::vector<uint32_t> ex_index;     // shortlist entries with a record, by posting position
    std::vector<uint32_t> ex_slots;     // their store slots
    std::vector<char> ex_records;       // [ex_index x record bytes]
    std::vector<uint32_t> ex_owners;
    std::vector<float> ex_est;          // parallel to ex_index
    std::vector<uint32_t> approx_ids;   // kept with their estimates, not re-ranked exactly
    std::vector<float> approx_dists;
    IVFCentroidGraphScratch graph;      // ivf_pq_search_graph only
    // The SIMD scan (ivf_pq_fastscan.h), used when rerank_m > 0, the codes
    // are 8-bit, the index has blocked codes, the CPU has AVX-512 VBMI and
    // fastscan_enabled(). It keeps the rerank_m nearest by quantized PQ
    // distance, which can differ from the float scan's near the boundary.
    IVFPQQuantizedTable qtable;
    std::vector<float> sample;          // float PQ distances that set the table's range
    std::vector<uint16_t> block_sums;   // one list's sums, [IVF_FASTSCAN_BLOCK x its blocks]
    // max-heap: quantized distance, then candidate id << 32 | its posting position
    std::vector<std::pair<uint16_t, uint64_t>> top;
};

// T is the raw vector element type the heap holds (float, uint8_t, int8_t);
// the query is always float, like the centroids and pivots. Centroid
// distances come from one GEMM against the padded centroids as
// ||q||^2 + ||c||^2 - 2 q.c, so partitions equidistant to within a few ulps
// of those norms may be ordered either way. nprobe is clamped to nlist.
// rerank_m == 0 skips the re-rank and ranks by PQ distance; otherwise it
// must be at least k. Vectors whose RID is inactive are dropped. `delta`,
// when given, contributes the vectors inserted since the index file was
// written (ivf_pq_insert), scanned in the probed partitions like the base
// vectors, and withholds the deleted ones (ivf_pq_delete); searches may run
// concurrently with inserts into and deletes from it. `rerank` sets how the
// re-rank fetches its vectors (IVFRerankSchedule), and `ex` whether it goes
// through the 9-bit ex-code store first (IVFExRerank).
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m,
                                IVFPQSearchScratch& scratch, const IVFPQDelta* delta = nullptr,
                                const IVFRerankSchedule& rerank = {}, const IVFExRerank& ex = {});

// As ivf_pq_search, but the probed partitions are the nprobe nearest
// centroids that a search of `graph` with a beam of centroid_L >= nprobe
// finds (nprobe clamped to nlist), in place of the exact nearest nprobe from
// a GEMV over every centroid. It visits a few hundred centroids instead of
// all nlist, and may miss a partition the GEMV would probe. `graph` must be
// built from index.meta; only its nlist is checked. With `cutoff` on, the
// nprobe partitions found are cut per query (IVFProbeCutoff); `rerank` sets
// how the re-rank fetches its vectors (IVFRerankSchedule), `ex` as for
// ivf_pq_search.
template<typename T>
IVFPQSearchResult ivf_pq_search_graph(const IVFPQIndex& index, const IVFCentroidGraph& graph,
                                      const RawVectorHeap& heap, const float* query, uint32_t k, uint32_t nprobe,
                                      uint32_t centroid_L, uint32_t rerank_m, IVFPQSearchScratch& scratch,
                                      const IVFPQDelta* delta = nullptr, const IVFProbeCutoff& cutoff = {},
                                      const IVFRerankSchedule& rerank = {}, const IVFExRerank& ex = {});

// Allocates a scratch per call; for one-off queries.
template<typename T>
IVFPQSearchResult ivf_pq_search(const IVFPQIndex& index, const RawVectorHeap& heap, const float* query,
                                uint32_t k, uint32_t nprobe, uint32_t rerank_m);

// `queries` is row-major [nq x dim]. Centroid distances come from one GEMM per
// `gemm_rows` queries; the queries of each such block are then searched in
// parallel with OpenMP, one scratch per thread. Results are in query order and
// identical to nq single-query searches up to the rounding of the batched GEMM.
const uint32_t IVF_PQ_SEARCH_GEMM_ROWS = 1024;

template<typename T>
std::vector<IVFPQSearchResult> ivf_pq_search_batch(const IVFPQIndex& index, const RawVectorHeap& heap,
                                                   const float* queries, size_t nq, uint32_t k,
                                                   uint32_t nprobe, uint32_t rerank_m,
                                                   uint32_t gemm_rows = IVF_PQ_SEARCH_GEMM_ROWS,
                                                   const IVFPQDelta* delta = nullptr);

}  // namespace inplace
}  // namespace diskann
