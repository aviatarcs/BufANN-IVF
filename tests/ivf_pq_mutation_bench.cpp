// Insert/delete benchmark for the IVF-PQ index through the public API, with
// no posting-list rebuild: how fast mutations are, and what an unfolded
// delta and a growing tombstone set cost a search.
//
//   ivf_pq_mutation_bench --data_file base.bin --query_file query.bin --dim N
//                         --ivf_nlist N --ivf_pq_chunks N [--base_pts N]
//                         [--queries N] [--nprobe N] [--k N] [--threads N]
//                         [--work_dir DIR] [--reuse_index]
//
// Builds on the first base_pts rows (default 90%), then inserts the rest in
// steps and deletes random base rows in steps, measuring after each step:
// search throughput (all threads, and one thread) and recall@k against an
// exact brute-force top-k over the rows live at that moment, computed here.
// Mutation throughput is measured alone and with searches running.
// Everything is in memory except the heap; nothing here is persisted.

#include "bufann/bufann_api.h"
#include "utils.h"

#include <mkl.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace diskann::inplace;

namespace {

std::string get_arg(int argc, char** argv, const char* name, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}
bool has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double rss_gb() {
    std::ifstream in("/proc/self/statm");
    long pages = 0, resident = 0;
    in >> pages >> resident;
    return double(resident) * double(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0 * 1024.0);
}

// Exact top-k over the live rows of `base` for every query: one GEMM per
// block of rows, a bounded max-heap per query, queries in parallel.
std::vector<std::vector<uint32_t>> brute_force(const float* base, size_t n, const std::vector<uint8_t>& live,
                                               const float* queries, size_t nq, uint32_t dim, uint32_t k) {
    std::vector<float> base_l2(n), q_l2(nq);
    for (size_t i = 0; i < n; ++i) base_l2[i] = cblas_sdot(int(dim), base + i * dim, 1, base + i * dim, 1);
    for (size_t q = 0; q < nq; ++q) q_l2[q] = cblas_sdot(int(dim), queries + q * dim, 1, queries + q * dim, 1);
    using Entry = std::pair<float, uint32_t>;
    std::vector<std::priority_queue<Entry>> heaps(nq);
    const size_t block = 16384;
    std::vector<float> dot(nq * block);
    for (size_t start = 0; start < n; start += block) {
        const size_t rows = std::min(block, n - start);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(nq), int(rows), int(dim), 1.0f, queries, int(dim),
                    base + start * dim, int(dim), 0.0f, dot.data(), int(rows));
#pragma omp parallel for schedule(static)
        for (int64_t q = 0; q < int64_t(nq); ++q) {
            auto& heap = heaps[q];
            const float* d = dot.data() + size_t(q) * rows;
            for (size_t r = 0; r < rows; ++r) {
                if (!live[start + r]) continue;
                const float dist = q_l2[q] + base_l2[start + r] - 2.0f * d[r];
                if (heap.size() < k) {
                    heap.emplace(dist, uint32_t(start + r));
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.emplace(dist, uint32_t(start + r));
                }
            }
        }
    }
    std::vector<std::vector<uint32_t>> gt(nq);
    for (size_t q = 0; q < nq; ++q) {
        while (!heaps[q].empty()) gt[q].push_back(heaps[q].top().second), heaps[q].pop();
    }
    return gt;
}

struct Bench {
    BufANNIndex<float>* idx = nullptr;
    const float* base = nullptr;
    size_t n_total = 0;
    const float* queries = nullptr;
    size_t nq = 0;
    uint32_t dim = 0, k = 10, nprobe = 16, threads = 1;
    std::vector<uint8_t> live;  // by row = tag

    // Throughput over the query set, repeated until ~2 s have passed.
    double qps(uint32_t num_threads) {
        std::vector<std::vector<TagType>> out(num_threads, std::vector<TagType>(k));
        size_t done = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (seconds_since(t0) < 2.0) {
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads)
            for (int64_t q = 0; q < int64_t(nq); ++q) {
                bufann_query_into<float>(*idx, queries + size_t(q) * dim, k, out[size_t(omp_get_thread_num())].data(),
                                         nprobe);
            }
            done += nq;
        }
        return double(done) / seconds_since(t0);
    }

    // Overall recall@k, and the recall of the ground-truth entries that are
    // inserted vectors on their own (NaN when there are none): inserted
    // vectors take a different scan path, so this shows whether they are
    // found as reliably as base vectors.
    std::pair<double, double> recall() {
        std::vector<std::vector<uint32_t>> gt = brute_force(base, n_total, live, queries, nq, dim, k);
        const uint32_t n_base = ivf_pq_num_base(idx->ivf->index);
        std::atomic<size_t> hits{0}, inserted_hits{0}, inserted_gt{0};
#pragma omp parallel for schedule(dynamic, 16)
        for (int64_t q = 0; q < int64_t(nq); ++q) {
            std::vector<TagType> got = bufann_query<float>(*idx, queries + size_t(q) * dim, k, nprobe);
            size_t h = 0, ih = 0, ig = 0;
            for (uint32_t g : gt[q]) {
                const bool hit = std::find(got.begin(), got.end(), g) != got.end();
                h += hit;
                if (g >= n_base) ig += 1, ih += hit;
            }
            hits.fetch_add(h), inserted_hits.fetch_add(ih), inserted_gt.fetch_add(ig);
        }
        return {double(hits.load()) / double(nq * k), double(inserted_hits.load()) / double(inserted_gt.load())};
    }

    void report(const std::string& phase) {
        const size_t inserted = idx->ivf->index.rid_table.rid.size() - ivf_pq_num_base(idx->ivf->index);
        size_t live_count = std::count(live.begin(), live.end(), uint8_t(1));
        const double qps_all = qps(threads), qps_one = qps(1);
        const auto r = recall();
        std::printf("%-34s %9zu %9zu %9zu %10.0f %9.0f %9.4f %9.4f %7.2f\n", phase.c_str(), live_count, inserted,
                    idx->ivf->delta.lists.tombstones.size(), qps_all, qps_one, r.first, r.second, rss_gb());
        std::fflush(stdout);
    }
};

// Inserts rows [from, to) from `num_threads` threads; returns inserts/s.
double insert_rows(Bench& b, size_t from, size_t to, uint32_t num_threads) {
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
    for (int64_t row = int64_t(from); row < int64_t(to); ++row) {
        bufann_insert<float>(*b.idx, TagType(row), b.base + size_t(row) * b.dim);
    }
    for (size_t row = from; row < to; ++row) b.live[row] = 1;
    return double(to - from) / seconds_since(t0);
}

double delete_rows(Bench& b, const std::vector<uint32_t>& rows, uint32_t num_threads) {
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
    for (int64_t i = 0; i < int64_t(rows.size()); ++i) bufann_delete<float>(*b.idx, rows[size_t(i)]);
    for (uint32_t row : rows) b.live[row] = 0;
    return double(rows.size()) / seconds_since(t0);
}

// Mutation throughput while `searchers` threads query continuously; also
// reports the search throughput they achieved meanwhile.
template<typename Fn>
std::pair<double, double> with_searchers(Bench& b, uint32_t searchers, Fn&& mutate) {
    std::atomic<bool> stop{false};
    std::atomic<size_t> searches{0};
    std::vector<std::thread> pool;
    for (uint32_t s = 0; s < searchers; ++s) {
        pool.emplace_back([&, s] {
            std::vector<TagType> out(b.k);
            std::mt19937 gen(s);
            while (!stop.load()) {
                bufann_query_into<float>(*b.idx, b.queries + size_t(gen() % b.nq) * b.dim, b.k, out.data(), b.nprobe);
                searches.fetch_add(1);
            }
        });
    }
    auto t0 = std::chrono::steady_clock::now();
    const double rate = mutate();
    const double elapsed = seconds_since(t0);
    stop.store(true);
    for (auto& th : pool) th.join();
    return {rate, double(searches.load()) / elapsed};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_file = get_arg(argc, argv, "--data_file", "");
    const std::string query_file = get_arg(argc, argv, "--query_file", "");
    const std::string work_dir = get_arg(argc, argv, "--work_dir", "/tmp");
    BufANNConfig cfg;
    cfg.index_type = IndexType::IvfPq;
    cfg.dim = uint32_t(std::stoul(get_arg(argc, argv, "--dim", "0")));
    cfg.ivf_nlist = uint32_t(std::stoul(get_arg(argc, argv, "--ivf_nlist", "0")));
    cfg.ivf_pq_chunks = uint32_t(std::stoul(get_arg(argc, argv, "--ivf_pq_chunks", "0")));
    cfg.ivf_nprobe = uint32_t(std::stoul(get_arg(argc, argv, "--nprobe", "16")));
    if (data_file.empty() || query_file.empty() || cfg.dim == 0 || cfg.ivf_nlist == 0 || cfg.ivf_pq_chunks == 0) {
        std::cerr << "Usage: ivf_pq_mutation_bench --data_file base.bin --query_file query.bin --dim N --ivf_nlist N "
                     "--ivf_pq_chunks N [--base_pts N] [--queries N] [--nprobe N] [--k N] [--threads N] "
                     "[--work_dir DIR] [--reuse_index]"
                  << std::endl;
        return 2;
    }

    float* base = nullptr;
    float* queries = nullptr;
    size_t n_total, nq_file, dim_base, dim_q;
    diskann::load_bin<float>(data_file, base, n_total, dim_base);
    diskann::load_bin<float>(query_file, queries, nq_file, dim_q);
    if (dim_base != cfg.dim || dim_q != cfg.dim) {
        std::cerr << "dim mismatch: base " << dim_base << ", queries " << dim_q << ", --dim " << cfg.dim << std::endl;
        return 2;
    }
    Bench b;
    b.base = base;
    b.n_total = n_total;
    b.queries = queries;
    b.nq = std::min(nq_file, size_t(std::stoul(get_arg(argc, argv, "--queries", "1000"))));
    b.dim = cfg.dim;
    b.k = uint32_t(std::stoul(get_arg(argc, argv, "--k", "10")));
    b.nprobe = cfg.ivf_nprobe;
    b.threads = uint32_t(std::stoul(get_arg(argc, argv, "--threads", std::to_string(omp_get_max_threads()))));
    const size_t n_base = std::stoul(get_arg(argc, argv, "--base_pts", std::to_string(n_total * 9 / 10)));
    b.live.assign(n_total, 0);
    std::fill(b.live.begin(), b.live.begin() + n_base, 1);

    // The run mutates the heap file (inserts grow it), after which the index
    // file no longer describes it and a load is refused; a pristine copy of
    // the heap taken right after the build is restored on --reuse_index.
    const std::string prefix = work_dir + "/ivf_pq_bench_" + std::to_string(n_base);
    const std::string base_bin = prefix + "_base.bin";
    const std::string heap_path = prefix + "_ivf_raw_vectors.bin", pristine = heap_path + ".pristine";
    if (has_flag(argc, argv, "--reuse_index") && file_exists(prefix + "_ivf_pq_index.bin") && file_exists(pristine)) {
        std::filesystem::copy_file(pristine, heap_path, std::filesystem::copy_options::overwrite_existing);
        auto t0 = std::chrono::steady_clock::now();
        b.idx = bufann_load<float>(prefix, cfg);
        std::printf("loaded %s: %zu vectors in %.1f s\n", prefix.c_str(), n_base, seconds_since(t0));
    } else {
        diskann::save_bin<float>(base_bin, base, n_base, cfg.dim);
        auto t0 = std::chrono::steady_clock::now();
        b.idx = bufann_build<float>(base_bin, prefix, cfg);
        std::printf("built %s: %zu vectors, nlist %u, %u chunks in %.1f s\n", prefix.c_str(), n_base, cfg.ivf_nlist,
                    cfg.ivf_pq_chunks, seconds_since(t0));
        std::filesystem::copy_file(heap_path, pristine, std::filesystem::copy_options::overwrite_existing);
    }
    std::printf("queries %zu, k %u, nprobe %u, threads %u, rerank_m %u\n\n", b.nq, b.k, b.nprobe, b.threads,
                std::max<uint32_t>(100, 10 * b.k));
    std::printf("%-34s %9s %9s %9s %10s %9s %9s %9s %7s\n", "phase", "live", "inserted", "tombst", "qps(all)", "qps(1t)",
                "recall", "rec(ins)", "rss_gb");
    b.report("base");

    // Inserts in steps: the delta grows and the search scans it per id.
    const size_t n_insert = n_total - n_base;
    const size_t steps[] = {n_insert / 10, n_insert / 4, n_insert / 2, n_insert};
    size_t next = n_base;
    for (size_t target : steps) {
        const size_t to = n_base + target;
        if (to <= next) continue;
        const size_t half = next + (to - next) / 2;
        const double one = insert_rows(b, next, half, 1);
        const double many = insert_rows(b, half, to, b.threads);
        std::printf("  inserts/s: %.0f (1 thread), %.0f (%u threads)\n", one, many, b.threads);
        b.report("+" + std::to_string(target) + " inserted");
        next = to;
    }

    // Deletes of random base rows in steps: dead candidates are still
    // scanned before the tombstone filter drops them.
    std::vector<uint32_t> order(n_base);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), std::mt19937(42));
    const double fractions[] = {0.05, 0.10, 0.25, 0.50};
    size_t deleted = 0;
    for (double f : fractions) {
        const size_t to = size_t(double(n_base) * f);
        std::vector<uint32_t> rows(order.begin() + deleted, order.begin() + to);
        const size_t half = rows.size() / 2;
        const double one = delete_rows(b, std::vector<uint32_t>(rows.begin(), rows.begin() + half), 1);
        const double many = delete_rows(b, std::vector<uint32_t>(rows.begin() + half, rows.end()), b.threads);
        std::printf("  deletes/s: %.0f (1 thread), %.0f (%u threads)\n", one, many, b.threads);
        b.report("base " + std::to_string(int(f * 100)) + "% deleted");
        deleted = to;
    }

    // Mutations under a search load: re-insert the deleted rows (they take
    // the freed slots) and delete them again, with 8 searchers running.
    {
        std::vector<uint32_t> rows(order.begin(), order.begin() + deleted);
        auto ins = with_searchers(b, 8, [&] {
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 64) num_threads(4)
            for (int64_t i = 0; i < int64_t(rows.size()); ++i) {
                bufann_insert<float>(*b.idx, rows[size_t(i)], b.base + size_t(rows[size_t(i)]) * b.dim);
            }
            for (uint32_t row : rows) b.live[row] = 1;
            return double(rows.size()) / seconds_since(t0);
        });
        std::printf("  with 8 searchers: %.0f inserts/s (4 threads) while searches ran at %.0f q/s\n", ins.first,
                    ins.second);
        auto del = with_searchers(b, 8, [&] { return delete_rows(b, rows, 4); });
        std::printf("  with 8 searchers: %.0f deletes/s (4 threads) while searches ran at %.0f q/s\n", del.first,
                    del.second);
        b.report("after churn under load");
    }

    bufann_free<float>(b.idx);
    delete[] base;
    delete[] queries;
    return 0;
}
