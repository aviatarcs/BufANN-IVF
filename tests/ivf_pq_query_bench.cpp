// Query benchmark for a persisted IVF-PQ index (from ivf_pq_build_index), in
// the shape of the BufANN query workload so the two can be compared: every
// thread runs whole queries, and throughput, per-query latency and recall@k
// against a DiskANN truthset are reported per nprobe, plus the batched path.
//
//   ivf_pq_query_bench --data_type float|int8|uint8 --index_prefix <prefix>
//                      --query_file <query.bin> --gt_file <truthset.bin>
//                      [--nprobes 8,16,32,64,128] [--rerank_m 100] [--k 10]
//                      [--threads N] [--warmup N]
//
// The heap is read with pread through the page cache (not O_DIRECT) and has
// no bounded buffer pool, so after the warmup its touched pages are cached.

#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_search.h"
#include "utils.h"

#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

std::string get_arg(int argc, char** argv, const char* name, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double rss_mb() {
    std::ifstream in("/proc/self/statm");
    size_t pages = 0, resident = 0;
    in >> pages >> resident;
    return double(resident) * double(sysconf(_SC_PAGESIZE)) / double(1 << 20);
}

double file_mb(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? double(st.st_size) / double(1 << 20) : 0.0;
}

double percentile_us(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size())))] * 1e6;
}

template<typename T>
int run(int argc, char** argv) {
    const std::string prefix = get_arg(argc, argv, "--index_prefix", "");
    const std::string query_file = get_arg(argc, argv, "--query_file", "");
    const std::string gt_file = get_arg(argc, argv, "--gt_file", "");
    const uint32_t k = uint32_t(std::stoul(get_arg(argc, argv, "--k", "10")));
    const uint32_t rerank_m = uint32_t(std::stoul(get_arg(argc, argv, "--rerank_m", "100")));
    const int threads = std::stoi(get_arg(argc, argv, "--threads", std::to_string(omp_get_max_threads())));
    std::vector<uint32_t> nprobes;
    {
        std::stringstream ss(get_arg(argc, argv, "--nprobes", "8,16,32,64,128"));
        for (std::string tok; std::getline(ss, tok, ',');) nprobes.push_back(uint32_t(std::stoul(tok)));
    }
    for (const std::string& f : {ivf_pq_index_path(prefix), query_file, gt_file}) {
        if (!file_exists(f)) throw diskann::ANNException("missing file: " + f, -1);
    }

    T* qraw = nullptr;
    size_t nq = 0, dim = 0;
    diskann::load_bin<T>(query_file, qraw, nq, dim);
    std::vector<float> queries(qraw, qraw + nq * dim);
    delete[] qraw;
    // The benchmark scripts' truthsets may carry a tags block after the
    // distances; load_truthset writes through `tags` when it is present.
    uint32_t* gt_ids = nullptr;
    float* gt_dists = nullptr;
    uint32_t* gt_tags = nullptr;
    size_t gt_rows = 0, gt_k = 0;
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_rows, gt_k, &gt_tags);
    if (gt_tags != nullptr && !std::equal(gt_ids, gt_ids + gt_rows * gt_k, gt_tags)) {
        throw diskann::ANNException("truthset tags differ from its ids; this index is searched by row id", -1);
    }
    if (gt_rows < nq || gt_k < k) throw diskann::ANNException("truthset smaller than queries x k", -1);
    const size_t warmup = std::min(nq, size_t(std::stoul(get_arg(argc, argv, "--warmup", "1000"))));

    auto t0 = std::chrono::steady_clock::now();
    IVFPQIndex ix = load_ivf_pq_index(prefix);
    RawVectorHeap heap;
    heap.open_existing(ivf_raw_vectors_path(prefix), ix.heap_layout, ix.heap_next_slot, ix.heap_pages);
    const size_t n = ix.assignments.cluster_id.size();
    std::printf("loaded %s: N %zu, nlist %u, %u PQ chunks in %.1f s; RSS %.0f MB; index file %.0f MB, "
                "heap %.0f MB, PQ codes %.0f MB\n",
                prefix.c_str(), n, ix.meta.nlist, ix.pq.chunks, seconds_since(t0), rss_mb(),
                file_mb(ivf_pq_index_path(prefix)), file_mb(ivf_raw_vectors_path(prefix)),
                file_mb(ivf_pq_codes_path(prefix)));

    auto recall_of = [&](const std::vector<IVFPQSearchResult>& res) {
        size_t hits = 0;
        for (size_t q = 0; q < nq; ++q) {
            const uint32_t* g = gt_ids + q * gt_k;
            for (uint32_t id : res[q].ids) hits += std::find(g, g + k, id) != g + k;
        }
        return 100.0 * double(hits) / double(nq * k);
    };

    omp_set_num_threads(threads);
    for (uint32_t nprobe : nprobes) {
        std::vector<IVFPQSearchResult> res(nq);
        std::vector<double> lat(nq);
        auto concurrent = [&](size_t count) {
#pragma omp parallel
            {
                IVFPQSearchScratch scratch;
#pragma omp for schedule(dynamic, 1)
                for (size_t q = 0; q < count; ++q) {
                    auto s = std::chrono::steady_clock::now();
                    res[q] = ivf_pq_search<T>(ix, heap, queries.data() + q * dim, k, nprobe, rerank_m, scratch);
                    lat[q] = seconds_since(s);
                }
            }
        };
        concurrent(warmup);
        t0 = std::chrono::steady_clock::now();
        concurrent(nq);
        const double wall = seconds_since(t0);
        const double recall = recall_of(res);
        double avg = 0;
        for (double l : lat) avg += l;
        avg /= double(nq);

        t0 = std::chrono::steady_clock::now();
        std::vector<IVFPQSearchResult> batch = ivf_pq_search_batch<T>(ix, heap, queries.data(), nq, k, nprobe, rerank_m);
        const double batch_wall = seconds_since(t0);

        std::printf("{\"baseline\":\"IVF-PQ\",\"N\":%zu,\"nlist\":%u,\"pq_chunks\":%u,\"nprobe\":%u,\"rerank_m\":%u,"
                    "\"query_threads\":%d,\"query_count\":%zu,\"query_qps\":%.1f,\"query_lat_avg_us\":%.1f,"
                    "\"query_lat_p50_us\":%.1f,\"query_lat_p99_us\":%.1f,\"recall\":%.3f,"
                    "\"batched_qps\":%.1f,\"batched_recall\":%.3f,\"rss_mb\":%.0f}\n",
                    n, ix.meta.nlist, ix.pq.chunks, nprobe, rerank_m, threads, nq, double(nq) / wall,
                    avg * 1e6, percentile_us(lat, 0.50), percentile_us(lat, 0.99), recall,
                    double(nq) / batch_wall, recall_of(batch), rss_mb());
        std::fflush(stdout);
    }
    delete[] gt_ids;
    delete[] gt_dists;
    delete[] gt_tags;
    heap.close();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_type = get_arg(argc, argv, "--data_type", "float");
    if (get_arg(argc, argv, "--index_prefix", "").empty() || get_arg(argc, argv, "--query_file", "").empty() ||
        get_arg(argc, argv, "--gt_file", "").empty()) {
        std::cerr << "Usage: ivf_pq_query_bench --data_type float|int8|uint8 --index_prefix <prefix> "
                     "--query_file <query.bin> --gt_file <truthset.bin> [--nprobes 8,16,32,64,128] "
                     "[--rerank_m 100] [--k 10] [--threads N] [--warmup N]"
                  << std::endl;
        return 2;
    }
    try {
        if (data_type == "float") return run<float>(argc, argv);
        if (data_type == "uint8") return run<uint8_t>(argc, argv);
        if (data_type == "int8") return run<int8_t>(argc, argv);
        std::cerr << "ERROR: --data_type must be float, int8 or uint8" << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
