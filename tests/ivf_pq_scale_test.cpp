// Runs the whole IVF-PQ build on a synthetic clustered base at realistic
// shapes and the library's default block/flush sizes, checks the invariants
// the unit tests check but over every vector, and reports time and memory.
//
//   ivf_pq_scale_test [N=1000000] [dim=128] [nlist=1024] [chunks=32] [dir=/tmp]
//
// CTest registers a small smoke configuration; run the default by hand.

#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "ivf_pq_test_util.h"
#include "utils.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

struct Params {
    size_t n = 1000000;
    uint32_t dim = 128, nlist = 1024, chunks = 32;
    std::string dir = "/tmp";
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double peak_rss_gb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return double(ru.ru_maxrss) / (1024.0 * 1024.0);
}

// 256 Gaussian clusters with unit noise, written in blocks so the base
// never has to fit in memory.
void write_clustered_base(const std::string& path, const Params& p, uint32_t seed) {
    const uint32_t num_clusters = 256;
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> spread(-50.0f, 50.0f);
    std::vector<float> centers(size_t(num_clusters) * p.dim);
    for (float& v : centers) v = spread(gen);

    std::ofstream out(path, std::ios::binary);
    uint32_t hdr[2] = {uint32_t(p.n), p.dim};
    out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    const size_t block = 1 << 16;
    std::vector<float> buf(block * p.dim);
    for (size_t start = 0; start < p.n; start += block) {
        size_t cur = std::min(block, p.n - start);
        for (size_t i = 0; i < cur; ++i) {
            const float* c = centers.data() + size_t(gen() % num_clusters) * p.dim;
            for (uint32_t d = 0; d < p.dim; ++d) buf[i * p.dim + d] = c[d] + noise(gen);
        }
        out.write(reinterpret_cast<const char*>(buf.data()), cur * p.dim * sizeof(float));
    }
}

std::vector<float> read_rows(const std::string& path, const std::vector<size_t>& ids, uint32_t dim) {
    std::ifstream in(path, std::ios::binary);
    std::vector<float> rows(ids.size() * dim);
    for (size_t k = 0; k < ids.size(); ++k) {
        in.seekg(std::streamoff(8 + ids[k] * dim * sizeof(float)));
        in.read(reinterpret_cast<char*>(rows.data() + k * dim), dim * sizeof(float));
    }
    return rows;
}

float sq_norm(const float* a, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += a[d] * a[d];
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    Params p;
    if (argc > 1) p.n = std::stoull(argv[1]);
    if (argc > 2) p.dim = uint32_t(std::stoul(argv[2]));
    if (argc > 3) p.nlist = uint32_t(std::stoul(argv[3]));
    if (argc > 4) p.chunks = uint32_t(std::stoul(argv[4]));
    if (argc > 5) p.dir = argv[5];
    std::cout << "N=" << p.n << " dim=" << p.dim << " nlist=" << p.nlist << " chunks=" << p.chunks
              << std::endl;

    std::string prefix = p.dir + "/ivf_pq_scale_" + std::to_string(uint64_t(getpid()));
    std::string base_bin = prefix + "_base.bin";
    auto step = [&](const char* name, auto&& fn) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        std::cout << "  [" << name << "] " << seconds_since(t0) << " s, peak RSS " << peak_rss_gb()
                  << " GB" << std::endl;
    };

    bool all_pass = true;
    try {
        step("generate base", [&] { write_clustered_base(base_bin, p, 1); });

        IVFMetadata meta;
        step("1 train centroids", [&] { meta = train_ivf_centroids<float>(base_bin, p.nlist); });

        RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, p.dim * sizeof(float));
        RawVectorHeap heap;
        heap.open(ivf_raw_vectors_path(prefix), layout);
        ClusterAssignments assignments;
        RawVectorRIDTable rid_table;
        step("2 assign + bulk load", [&] {
            assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table);
        });

        PostingLists lists;
        step("3 posting lists", [&] { lists = build_ivf_posting_lists(assignments, p.nlist); });

        step("4 train PQ pivots", [&] { train_ivf_pq_pivots<float>(base_bin, prefix, p.chunks); });
        step("5 encode PQ codes", [&] { encode_ivf_pq_codes<float>(base_bin, prefix, p.chunks); });

        PQMetadata pq;
        bool round_trip = false;
        step("6 write index file", [&] {
            IVFPQIndex ix;
            ix.meta = meta;
            ix.assignments = assignments;
            ix.lists = lists;
            ix.pq = load_ivf_pq(prefix);
            ix.rid_table = rid_table;
            ix.heap_layout = layout;
            ix.heap_pages = heap.allocated_pages();
            ix.heap_next_slot = heap.next_flat_slot();
            write_ivf_pq_index(prefix, ix);
        });
        step("load index file (validating)", [&] {
            IVFPQIndex ix = load_ivf_pq_index(prefix);
            pq = std::move(ix.pq);
            round_trip = ix.meta.centroids == meta.centroids && ix.assignments.cluster_id == assignments.cluster_id &&
                         ix.rid_table.rid.size() == rid_table.rid.size() && ix.lists.ids == lists.ids &&
                         ix.lists.offsets == lists.offsets && ix.heap_pages == heap.allocated_pages() &&
                         ix.heap_next_slot == heap.next_flat_slot();
        });

        TestCase t("invariants over every vector");
        t.check(round_trip, "index file round-trip mismatch");
        uint32_t pages = uint32_t((p.n + layout.slots_per_page - 1) / layout.slots_per_page);
        t.check(heap.next_flat_slot() == p.n && heap.allocated_pages() == pages, "heap cursor");
        t.check(size_t(get_file_size(ivf_raw_vectors_path(prefix))) == size_t(pages) * layout.page_size,
                "heap file size");
        size_t bad_rid = 0;
        for (size_t i = 0; i < p.n; ++i) {
            bad_rid += !rid_is_active(rid_table.rid[i]) || rid_flat_slot(rid_table.rid[i]) != i;
        }
        t.check(bad_rid == 0, std::to_string(bad_rid) + " RIDs are not (active, i)");
        std::vector<uint8_t> seen(p.n, 0);
        size_t bad_posting = 0;
        for (uint32_t c = 0; c < p.nlist; ++c) {
            for (uint32_t k = lists.offsets[c]; k < lists.offsets[c + 1]; ++k) {
                uint32_t id = lists.ids[k];
                bad_posting += assignments.cluster_id[id] != c || (k > lists.offsets[c] && lists.ids[k - 1] >= id);
                seen[id] = 1;
            }
        }
        t.check(bad_posting == 0 && lists.offsets.back() == p.n &&
                    std::all_of(seen.begin(), seen.end(), [](uint8_t s) { return s == 1; }),
                "posting lists are not the exact inverse of the assignments");
        t.check(pq.chunks == p.chunks && pq.codes.size() == p.n * p.chunks, "PQ shape");
        all_pass &= t.done();

        TestCase s("brute-force spot checks on 1000 random vectors");
        std::mt19937 gen(7);
        std::vector<size_t> ids(1000);
        for (size_t& id : ids) id = gen() % p.n;
        std::vector<float> rows = read_rows(base_bin, ids, p.dim);
        // compute_closest_centers expands ||x-c||^2 in float, so near-ties can
        // resolve either way within a few ulps of the norms involved. The
        // synthetic base is centred near the origin, so ||x||^2 + ||c||^2
        // bounds the scale for both the raw and the mean-centred PQ path.
        const float eps = std::numeric_limits<float>::epsilon();
        auto within_rounding = [&](float chosen, float best, const float* x, const float* c, uint32_t d) {
            return chosen <= best + 16 * eps * (sq_norm(x, d) + sq_norm(c, d));
        };
        size_t bad_heap = 0, bad_cluster = 0, bad_code = 0;
        std::vector<float> got(p.dim);
        const uint32_t chunk_dim = p.dim / p.chunks;
        for (size_t k = 0; k < ids.size(); ++k) {
            const float* x = rows.data() + k * p.dim;
            heap.read_vector(rid_flat_slot(rid_table.rid[ids[k]]), got.data());
            bad_heap += std::memcmp(got.data(), x, p.dim * sizeof(float)) != 0;

            const float* chosen_c = meta.centroids.data() + size_t(assignments.cluster_id[ids[k]]) * meta.aligned_dim;
            float chosen = sq_dist(x, chosen_c, p.dim), best = chosen;
            for (uint32_t c = 0; c < p.nlist; ++c) {
                best = std::min(best, sq_dist(x, meta.centroids.data() + size_t(c) * meta.aligned_dim, p.dim));
            }
            bad_cluster += !within_rounding(chosen, best, x, chosen_c, p.dim);

            for (uint32_t c = 0; c < p.chunks; ++c) {
                const float* xc = x + c * chunk_dim;
                const float* pivots = pq.pivots.data() + size_t(c) * pq.k * chunk_dim;
                const float* chosen_p = pivots + pq.codes[ids[k] * p.chunks + c] * chunk_dim;
                float chosen_d = sq_dist(xc, chosen_p, chunk_dim), best_d = chosen_d;
                for (uint32_t j = 0; j < pq.k; ++j) best_d = std::min(best_d, sq_dist(xc, pivots + j * chunk_dim, chunk_dim));
                bad_code += !within_rounding(chosen_d, best_d, xc, chosen_p, chunk_dim);
            }
        }
        s.check(bad_heap == 0, std::to_string(bad_heap) + " heap vectors differ from the base");
        s.check(bad_cluster == 0, std::to_string(bad_cluster) + " assignments are not the nearest centroid");
        s.check(bad_code == 0, std::to_string(bad_code) + " PQ codes are not a nearest pivot");
        all_pass &= s.done();
        heap.close();
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }

    for (const std::string& f : {base_bin, ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix),
                                 ivf_pq_pivots_path(prefix), ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
    std::cout << "peak RSS " << peak_rss_gb() << " GB" << std::endl;
    return all_pass ? 0 : 1;
}
