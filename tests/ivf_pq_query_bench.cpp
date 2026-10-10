// Query benchmark for a persisted IVF-PQ index (from ivf_pq_build_index), in
// the shape of the BufANN query workload so the two can be compared: every
// thread runs whole queries, and throughput, per-query latency and recall@k
// against a DiskANN truthset are reported per nprobe, plus the batched path.
// With --graph_beams, each nprobe is also run through a centroid graph
// (built after load) with a beam of ceil(factor x nprobe), factors possibly
// fractional (e.g. --graph_beams 1,1.25,2), reporting in addition
// the probe-set recall: the share of probed partitions no farther than the
// exact nprobe-th nearest centroid.
//
//   ivf_pq_query_bench --data_type float|int8|uint8 --index_prefix <prefix>
//                      --query_file <query.bin> --gt_file <truthset.bin>
//                      [--nprobes 8,16,32,64,128] [--rerank_m 100] [--k 10]
//                      [--threads N] [--warmup N] [--graph_beams 2,4]
//                      [--warmup_query_file <warmup.bin>] [--skip_exact 1]
//                      [--buffer_pool_frames 262144]
//
// The heap is read through a BufANN buffer pool of --buffer_pool_frames
// page frames (default 262144, 1 GB of 4 KB pages: BufANN's benchmark
// default), O_DIRECT underneath, so the numbers are taken under the memory
// budget BufANN's are, and each line reports the pool's hits, misses and I/O
// time for its timed queries under BufANN's driver's field names. 0 frames
// reads the heap through the unbounded OS page cache instead, as before.

#include "bufann/inplace_backend.h"
#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_centroid_graph.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_search.h"
#include "utils.h"

#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
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

std::vector<uint32_t> parse_list(const std::string& s) {
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');) out.push_back(uint32_t(std::stoul(tok)));
    return out;
}

template<typename T>
int run(int argc, char** argv) {
    const std::string prefix = get_arg(argc, argv, "--index_prefix", "");
    const std::string query_file = get_arg(argc, argv, "--query_file", "");
    const std::string gt_file = get_arg(argc, argv, "--gt_file", "");
    const uint32_t k = uint32_t(std::stoul(get_arg(argc, argv, "--k", "10")));
    const uint32_t rerank_m = uint32_t(std::stoul(get_arg(argc, argv, "--rerank_m", "100")));
    const int threads = std::stoi(get_arg(argc, argv, "--threads", std::to_string(omp_get_max_threads())));
    const std::vector<uint32_t> nprobes = parse_list(get_arg(argc, argv, "--nprobes", "8,16,32,64,128"));
    // Beam factors may be fractional; the beam is ceil(factor x nprobe), at least nprobe.
    std::vector<double> graph_beams;
    {
        std::stringstream ss(get_arg(argc, argv, "--graph_beams", ""));
        for (std::string tok; std::getline(ss, tok, ',');) graph_beams.push_back(std::stod(tok));
        for (double f : graph_beams) {
            if (!(f >= 1.0)) throw diskann::ANNException("--graph_beams factors must be at least 1", -1);
        }
    }
    const uint32_t pool_frames = uint32_t(std::stoul(get_arg(argc, argv, "--buffer_pool_frames", "262144")));
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
    if (nq == 0) throw diskann::ANNException("query file holds no queries: " + query_file, -1);
    if (gt_rows < nq || gt_k < k) throw diskann::ANNException("truthset smaller than queries x k", -1);
    // Warmup queries: the first --warmup of the timed ones, or, as
    // bufann_driver's --warmup_query_file, up to --warmup rows of a separate
    // file (BufANN's scripts sample them from the base), so the pool is not
    // warmed by the very queries that are then timed.
    const std::string warmup_file = get_arg(argc, argv, "--warmup_query_file", "");
    const bool skip_exact = get_arg(argc, argv, "--skip_exact", "0") != "0";
    if (skip_exact && graph_beams.empty()) throw diskann::ANNException("--skip_exact needs --graph_beams", -1);
    std::vector<float> warmup_queries;
    size_t warmup = std::min(nq, size_t(std::stoul(get_arg(argc, argv, "--warmup", "1000"))));
    if (warmup_file.empty()) {
        warmup_queries.assign(queries.begin(), queries.begin() + warmup * dim);
    } else {
        if (!file_exists(warmup_file)) throw diskann::ANNException("missing file: " + warmup_file, -1);
        T* wraw = nullptr;
        size_t wn = 0, wdim = 0;
        diskann::load_bin<T>(warmup_file, wraw, wn, wdim);
        std::unique_ptr<T[]> owned(wraw);
        if (wdim != dim) throw diskann::ANNException("warmup query dim differs from the queries'", -1);
        warmup = std::min(wn, size_t(std::stoul(get_arg(argc, argv, "--warmup", "1000"))));
        warmup_queries.assign(wraw, wraw + warmup * dim);
    }

    auto t0 = std::chrono::steady_clock::now();
    IVFPQIndex ix = load_ivf_pq_index(prefix);
    InPlaceIOStats io;
    RawVectorHeap heap;
    heap.open_existing(ivf_raw_vectors_path(prefix), ix.heap_layout, ix.heap_next_slot, ix.heap_pages,
                       RawVectorHeapCache{pool_frames, &io});
    const size_t n = ix.assignments.cluster_id.size();
    if (dim != ix.meta.dim) {
        throw diskann::ANNException("query dim " + std::to_string(dim) + " != index dim " +
                                        std::to_string(ix.meta.dim), -1);
    }
    std::printf("loaded %s: N %zu, nlist %u, %u PQ chunks in %.1f s; RSS %.0f MB; index file %.0f MB, "
                "heap %.0f MB, PQ codes %.0f MB\n",
                prefix.c_str(), n, ix.meta.nlist, ix.pq.chunks, seconds_since(t0), rss_mb(),
                file_mb(ivf_pq_index_path(prefix)), file_mb(ivf_raw_vectors_path(prefix)),
                file_mb(ivf_pq_codes_path(prefix)));
    if (pool_frames == 0) {
        std::printf("heap: %u pages of %u B through the OS page cache\n", ix.heap_pages, ix.heap_layout.page_size);
    } else {
        std::printf("heap: %u pages of %u B through a buffer pool of %u frames (%.0f MB, %.1f%% of the heap)\n",
                    ix.heap_pages, ix.heap_layout.page_size, pool_frames,
                    double(pool_frames) * ix.heap_layout.page_size / double(1 << 20),
                    100.0 * double(pool_frames) / double(std::max(1u, ix.heap_pages)));
    }

    omp_set_num_threads(threads);
    IVFCentroidGraph graph;
    // exact_nth[q * max_nprobe + j]: distance from query q to its (j+1)-th
    // nearest centroid, by brute force, for the probe-set recall.
    std::vector<float> exact_nth;
    const uint32_t max_nprobe = std::min(ix.meta.nlist, *std::max_element(nprobes.begin(), nprobes.end()));
    if (!graph_beams.empty()) {
        t0 = std::chrono::steady_clock::now();
        graph = build_ivf_centroid_graph(ix.meta);
        std::printf("centroid graph: degree %u, built in %.2f s\n", graph.degree, seconds_since(t0));
        exact_nth.resize(nq * max_nprobe);
#pragma omp parallel
        {
            std::vector<float> d(ix.meta.nlist);
#pragma omp for schedule(dynamic, 16)
            for (size_t q = 0; q < nq; ++q) {
                const float* query = queries.data() + q * dim;
                for (uint32_t c = 0; c < ix.meta.nlist; ++c) {
                    const float* cen = ix.meta.centroids.data() + size_t(c) * ix.meta.aligned_dim;
                    float s = 0.0f;
                    for (size_t j = 0; j < dim; ++j) s += (query[j] - cen[j]) * (query[j] - cen[j]);
                    d[c] = s;
                }
                std::partial_sort(d.begin(), d.begin() + max_nprobe, d.end());
                std::copy_n(d.begin(), max_nprobe, exact_nth.begin() + q * max_nprobe);
            }
        }
    }

    // The pool's counters over one timed phase, as bufann_driver reports them:
    // I/O time is the single-page pins' plus the batch pins' wait.
    auto reset_io = [&] {
        drain_query_cache_hits(io);
        io.reset();
    };
    auto io_json = [&](size_t queries, double avg_latency_s) {
        drain_query_cache_hits(io);
        const uint64_t hits = io.cache_hits.load(), misses = io.cache_misses.load();
        const double io_us = double(io.query_io_ns.load() + io.batch_io_ns_total.load()) / 1e3 / double(queries);
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "\"buffer_pool_frames\":%u,\"cache_hits\":%llu,\"cache_misses\":%llu,"
                      "\"cache_miss_rate\":%.6f,\"misses_per_query\":%.2f,\"io_read_random_ios\":%llu,"
                      "\"query_io_avg_us_per_query\":%.1f,\"query_io_fraction_of_latency\":%.4f",
                      pool_frames, (unsigned long long)hits, (unsigned long long)misses,
                      hits + misses == 0 ? 0.0 : double(misses) / double(hits + misses),
                      double(misses) / double(queries),
                      (unsigned long long)(io.page_fault_total.load() + io.batch_misses_total.load()), io_us,
                      avg_latency_s > 0 ? io_us / (avg_latency_s * 1e6) : 0.0);
        return std::string(buf);
    };

    auto recall_of = [&](const std::vector<IVFPQSearchResult>& res) {
        size_t hits = 0;
        for (size_t q = 0; q < nq; ++q) {
            const uint32_t* g = gt_ids + q * gt_k;
            for (uint32_t id : res[q].ids) hits += std::find(g, g + k, id) != g + k;
        }
        return 100.0 * double(hits) / double(nq * k);
    };

    for (uint32_t nprobe : nprobes) {
        std::vector<IVFPQSearchResult> res(nq);
        std::vector<double> lat(nq);
        // Searches `count` rows of `qs`, recording results and latencies
        // only when they are the timed queries.
        auto concurrent = [&](const float* qs, size_t count, bool timed) {
#pragma omp parallel
            {
                IVFPQSearchScratch scratch;
#pragma omp for schedule(dynamic, 1)
                for (size_t q = 0; q < count; ++q) {
                    auto s = std::chrono::steady_clock::now();
                    IVFPQSearchResult r = ivf_pq_search<T>(ix, heap, qs + q * dim, k, nprobe, rerank_m, scratch);
                    if (timed) res[q] = std::move(r), lat[q] = seconds_since(s);
                }
            }
        };
        if (!skip_exact) {
            concurrent(warmup_queries.data(), warmup, false);
            reset_io();
            t0 = std::chrono::steady_clock::now();
            concurrent(queries.data(), nq, true);
            const double wall = seconds_since(t0);
            const double recall = recall_of(res);
            double avg = 0;
            for (double l : lat) avg += l;
            avg /= double(nq);
            const std::string io_fields = io_json(nq, avg);

            t0 = std::chrono::steady_clock::now();
            std::vector<IVFPQSearchResult> batch =
                ivf_pq_search_batch<T>(ix, heap, queries.data(), nq, k, nprobe, rerank_m);
            const double batch_wall = seconds_since(t0);

            std::printf("{\"baseline\":\"IVF-PQ\",\"centroid_search\":\"exact\",\"N\":%zu,\"nlist\":%u,\"pq_chunks\":%u,"
                        "\"nprobe\":%u,\"rerank_m\":%u,"
                        "\"query_threads\":%d,\"query_count\":%zu,\"query_qps\":%.1f,\"query_lat_avg_us\":%.1f,"
                        "\"query_lat_p50_us\":%.1f,\"query_lat_p99_us\":%.1f,\"recall\":%.3f,"
                        "\"batched_qps\":%.1f,\"batched_recall\":%.3f,\"rss_mb\":%.0f,%s}\n",
                        n, ix.meta.nlist, ix.pq.chunks, nprobe, rerank_m, threads, nq, double(nq) / wall,
                        avg * 1e6, percentile_us(lat, 0.50), percentile_us(lat, 0.99), recall,
                        double(nq) / batch_wall, recall_of(batch), rss_mb(), io_fields.c_str());
            std::fflush(stdout);
        }

        for (double factor : graph_beams) {
            const uint32_t probes = std::min(nprobe, ix.meta.nlist);
            const uint32_t beam = std::max(probes, uint32_t(std::ceil(factor * probes)));
            auto concurrent_graph = [&](const float* qs, size_t count, bool timed) {
#pragma omp parallel
                {
                    IVFPQSearchScratch scratch;
#pragma omp for schedule(dynamic, 1)
                    for (size_t q = 0; q < count; ++q) {
                        auto s = std::chrono::steady_clock::now();
                        IVFPQSearchResult r = ivf_pq_search_graph<T>(ix, graph, heap, qs + q * dim, k, nprobe, beam,
                                                                     rerank_m, scratch);
                        if (timed) res[q] = std::move(r), lat[q] = seconds_since(s);
                    }
                }
            };
            concurrent_graph(warmup_queries.data(), warmup, false);
            reset_io();
            t0 = std::chrono::steady_clock::now();
            concurrent_graph(queries.data(), nq, true);
            const double graph_wall = seconds_since(t0);
            double graph_avg = 0;
            for (double l : lat) graph_avg += l;
            graph_avg /= double(nq);
            const std::string graph_io_fields = io_json(nq, graph_avg);

            size_t probe_hits = 0;
#pragma omp parallel reduction(+ : probe_hits)
            {
                IVFCentroidGraphScratch gs;
                std::vector<uint32_t> probed;
#pragma omp for schedule(dynamic, 16)
                for (size_t q = 0; q < nq; ++q) {
                    const float* query = queries.data() + q * dim;
                    search_ivf_centroid_graph(ix.meta, graph, query, beam, probes, gs, probed);
                    const float nth = exact_nth[q * max_nprobe + probes - 1];
                    for (uint32_t c : probed) {
                        const float* cen = ix.meta.centroids.data() + size_t(c) * ix.meta.aligned_dim;
                        float s = 0.0f;
                        for (size_t j = 0; j < dim; ++j) s += (query[j] - cen[j]) * (query[j] - cen[j]);
                        probe_hits += s <= nth;
                    }
                }
            }
            std::printf("{\"baseline\":\"IVF-PQ\",\"centroid_search\":\"graph\",\"centroid_L\":%u,\"N\":%zu,"
                        "\"nlist\":%u,\"pq_chunks\":%u,\"nprobe\":%u,\"rerank_m\":%u,\"query_threads\":%d,"
                        "\"query_count\":%zu,\"query_qps\":%.1f,\"query_lat_avg_us\":%.1f,"
                        "\"query_lat_p50_us\":%.1f,\"query_lat_p99_us\":%.1f,\"recall\":%.3f,"
                        "\"probe_recall\":%.4f,\"rss_mb\":%.0f,%s}\n",
                        beam, n, ix.meta.nlist, ix.pq.chunks, nprobe, rerank_m, threads, nq,
                        double(nq) / graph_wall, graph_avg * 1e6, percentile_us(lat, 0.50),
                        percentile_us(lat, 0.99), recall_of(res), double(probe_hits) / double(nq * probes),
                        rss_mb(), graph_io_fields.c_str());
            std::fflush(stdout);
        }
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
                     "[--rerank_m 100] [--k 10] [--threads N] [--warmup N] [--graph_beams 2,4] "
                     "[--warmup_query_file <warmup.bin>] [--skip_exact 1] "
                     "[--buffer_pool_frames 262144]"
                  << std::endl;
        return 2;
    }
    try {
        if (data_type == "float") return run<float>(argc, argv);
        if (data_type == "uint8") return run<uint8_t>(argc, argv);
        if (data_type == "int8") return run<int8_t>(argc, argv);
        std::cerr << "ERROR: --data_type must be float, int8 or uint8" << std::endl;
        return 2;
    } catch (const diskann::ANNException& e) {  // not a std::exception
        std::cerr << "ERROR: " << e.message() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
