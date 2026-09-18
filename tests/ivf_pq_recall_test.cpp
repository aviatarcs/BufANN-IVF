// End-to-end quality check: builds an IVF-PQ index and measures recall@K of
// ivf_pq_search, with and without the exact re-rank, against ground truth.
// ivf_pq_search_test checks the search against a reference loop; this test
// checks that the whole pipeline finds true neighbours.
//
//   ivf_pq_recall_test                                     synthetic base, exact GT computed here
//   ivf_pq_recall_test base.bin query.bin gt nlist chunks  gt is a DiskANN truthset .bin or an .ivecs
//
// Prints recall@10 for several nprobe and asserts the largest reaches
// RECALL_FLOOR.

#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_search.h"
#include "ivf_pq_test_util.h"
#include "utils.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t K = 10;
const uint32_t RERANK_M = 100;
const float RECALL_FLOOR = 0.90f;
const std::vector<uint32_t> NPROBES = {1, 4, 16, 64};

float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

// Indices of the `k` smallest values, ascending by value.
std::vector<uint32_t> top_k(const std::vector<float>& values, uint32_t k) {
    std::vector<uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), 0u);
    k = std::min<uint32_t>(k, uint32_t(order.size()));
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    order.resize(k);
    return order;
}

// Ground truth as [nq][>= K] ids, from a DiskANN truthset .bin or an .ivecs.
std::vector<std::vector<uint32_t>> load_gt(const std::string& path, size_t nq) {
    std::vector<std::vector<uint32_t>> gt(nq);
    if (path.size() > 6 && path.compare(path.size() - 6, 6, ".ivecs") == 0) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw diskann::ANNException("ground truth file not found: " + path, -1);
        for (size_t i = 0; i < nq; ++i) {
            int32_t d = 0;
            in.read(reinterpret_cast<char*>(&d), 4);
            if (!in || d < int32_t(K)) {
                throw diskann::ANNException("ground truth row " + std::to_string(i) + " is missing or shorter than K", -1);
            }
            gt[i].resize(d);
            in.read(reinterpret_cast<char*>(gt[i].data()), size_t(d) * 4);
        }
        return gt;
    }
    uint32_t* ids = nullptr;
    float* dists = nullptr;
    size_t rows = 0, k = 0;
    diskann::load_truthset(path, ids, dists, rows, k);
    for (size_t i = 0; i < nq; ++i) gt[i].assign(ids + i * k, ids + i * k + k);
    delete[] ids;
    delete[] dists;
    return gt;
}

// 100k points with 8 intrinsic dimensions embedded in 64: z drawn from 64
// Gaussian clusters in R^8, x = z * A + small isotropic noise. Like real
// embeddings, neighbours are determined by low-dimensional structure that
// 4-dim PQ chunks can quantize; isotropic 64-d noise has no such structure
// and PQ cannot rank it. 1000 queries drawn the same way; exact ground truth.
void write_synthetic(const std::string& base_bin, const std::string& query_bin,
                     std::vector<std::vector<uint32_t>>& gt) {
    const uint32_t n = 100000, nq = 1000, dim = 64, intrinsic = 8, clusters = 64;
    std::mt19937 gen(3);
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> spread(-20.0f, 20.0f);
    std::vector<float> centers(size_t(clusters) * intrinsic);
    for (float& v : centers) v = spread(gen);
    std::vector<float> embed(size_t(intrinsic) * dim);
    for (float& v : embed) v = unit(gen) / std::sqrt(float(intrinsic));

    auto draw = [&](std::vector<float>& out, size_t count) {
        out.assign(count * dim, 0.0f);
        std::vector<float> z(intrinsic);
        for (size_t i = 0; i < count; ++i) {
            const float* c = centers.data() + size_t(gen() % clusters) * intrinsic;
            for (uint32_t k = 0; k < intrinsic; ++k) z[k] = c[k] + 4.0f * unit(gen);
            for (uint32_t d = 0; d < dim; ++d) {
                float v = 0.5f * unit(gen);
                for (uint32_t k = 0; k < intrinsic; ++k) v += z[k] * embed[size_t(k) * dim + d];
                out[i * dim + d] = v;
            }
        }
    };
    std::vector<float> base, queries;
    draw(base, n);
    draw(queries, nq);
    diskann::save_bin<float>(base_bin, base.data(), n, dim);
    diskann::save_bin<float>(query_bin, queries.data(), nq, dim);

    gt.assign(nq, {});
    std::vector<float> dist(n);
    for (uint32_t q = 0; q < nq; ++q) {
        for (uint32_t i = 0; i < n; ++i) dist[i] = sq_dist(queries.data() + size_t(q) * dim, base.data() + size_t(i) * dim, dim);
        gt[q] = top_k(dist, K);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string prefix = temp_path("ivf_pq_recall");
    std::string base_bin, query_bin;
    std::vector<std::vector<uint32_t>> gt;
    uint32_t nlist = 256, chunks = 16;
    bool synthetic = argc < 6;
    if (synthetic) {
        base_bin = prefix + "_base.bin";
        query_bin = prefix + "_query.bin";
        write_synthetic(base_bin, query_bin, gt);
    } else {
        base_bin = argv[1];
        query_bin = argv[2];
        nlist = uint32_t(std::stoul(argv[4]));
        chunks = uint32_t(std::stoul(argv[5]));
    }

    bool all_pass = true;
    try {
        float* qraw = nullptr;
        size_t nq = 0, dim = 0;
        diskann::load_bin<float>(query_bin, qraw, nq, dim);
        std::vector<float> queries(qraw, qraw + nq * dim);
        delete[] qraw;
        if (!synthetic) gt = load_gt(argv[3], nq);

        // Build, persist as the combined file, and search what loads back --
        // the heap is closed and reopened from what the index file records.
        auto t0 = std::chrono::steady_clock::now();
        {
            IVFPQIndex built;
            built.meta = train_ivf_centroids<float>(base_bin, nlist);
            built.heap_layout = compute_raw_vector_heap_layout(4096, uint32_t(dim) * sizeof(float));
            RawVectorHeap build_heap;
            build_heap.open(ivf_raw_vectors_path(prefix), built.heap_layout);
            assign_ivf_clusters<float>(base_bin, built.meta, build_heap, built.assignments, built.rid_table);
            built.lists = build_ivf_posting_lists(built.assignments, nlist);
            train_ivf_pq_pivots<float>(base_bin, prefix, chunks, synthetic ? 0.3 : 0.0);
            encode_ivf_pq_codes<float>(base_bin, prefix, chunks);
            built.pq = load_ivf_pq(prefix);
            built.heap_pages = build_heap.allocated_pages();
            built.heap_next_slot = build_heap.next_flat_slot();
            write_ivf_pq_index(prefix, built);
        }
        IVFPQIndex ix = load_ivf_pq_index(prefix);
        RawVectorHeap heap;
        heap.open_existing(ivf_raw_vectors_path(prefix), ix.heap_layout, ix.heap_next_slot, ix.heap_pages);
        std::cout << "build: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s (nlist " << nlist << ", " << chunks << " PQ chunks, N " << ix.assignments.cluster_id.size()
                  << ")" << std::endl;

        TestCase t("recall@" + std::to_string(K) + " over " + std::to_string(nq) + " queries");
        auto recall_of = [&](const std::vector<IVFPQSearchResult>& results) {
            size_t hits = 0;
            for (size_t q = 0; q < results.size(); ++q) {
                for (uint32_t id : results[q].ids) {
                    hits += std::find(gt[q].begin(), gt[q].begin() + K, id) != gt[q].begin() + K;
                }
            }
            return float(hits) / float(nq * K);
        };
        const int max_threads = omp_get_max_threads();
        IVFPQSearchScratch scratch;
        float best_recall = 0.0f;
        for (uint32_t nprobe : NPROBES) {
            for (bool pq_only : {true, false}) {
                const uint32_t rerank_m = pq_only ? 0 : RERANK_M;
                auto s0 = std::chrono::steady_clock::now();
                std::vector<IVFPQSearchResult> single(nq);
                for (size_t q = 0; q < nq; ++q) {
                    single[q] = ivf_pq_search<float>(ix, heap, queries.data() + q * dim, K, nprobe, rerank_m, scratch);
                }
                double single_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();

                omp_set_num_threads(max_threads);
                s0 = std::chrono::steady_clock::now();
                std::vector<IVFPQSearchResult> batch =
                    ivf_pq_search_batch<float>(ix, heap, queries.data(), nq, K, nprobe, rerank_m);
                double batch_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();

                float recall = recall_of(single), batch_recall = recall_of(batch);
                std::cout << "  nprobe " << nprobe << (pq_only ? " PQ only " : " re-rank ") << "recall " << recall
                          << "  single-query " << long(nq / single_s) << " q/s (1 thread), batched "
                          << long(nq / batch_s) << " q/s (" << max_threads << " threads)" << std::endl;
                t.check(std::fabs(batch_recall - recall) <= 0.002f,
                        "batched recall " + std::to_string(batch_recall) + " differs from single-query " +
                            std::to_string(recall));
                if (!pq_only) best_recall = std::max(best_recall, recall);
            }
        }
        t.check(best_recall >= RECALL_FLOOR,
                "recall " + std::to_string(best_recall) + " below " + std::to_string(RECALL_FLOOR));
        all_pass &= t.done();
        heap.close();
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }

    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
    if (synthetic) {
        ::unlink(base_bin.c_str());
        ::unlink(query_bin.c_str());
    }
    return all_pass ? 0 : 1;
}
