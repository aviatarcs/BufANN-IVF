// Tests for the IVF-PQ build steps on a synthetic-blob base: centroid
// training (shape, padding, blob recovery, seeding, save/load), cluster
// assignment + raw-vector bulk load (exact nearest-centroid agreement,
// RID/heap contents, multi-block streaming, sidecar round-trips, error paths),
// and posting-list construction (exact inverse of the assignments).

#include "bufann/ivf_pq_build.h"
#include "ann_exception.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t NUM_BLOBS = 8;
const uint32_t TRAIN_SEED = 12345;  // pinned: k-means++ init must not flake
const uint32_t POINTS_PER_BLOB = 500;
const uint32_t NUM_POINTS = NUM_BLOBS * POINTS_PER_BLOB;
const uint32_t BLOB_DIM = 12;  // not a multiple of 8, so padding is exercised
const float BLOB_SPACING_F32 = 50.0f;
const float BLOB_SPACING_U8 = 30.0f;  // 8 blobs at 30 fit in [0, 255]

// Blob b is centered at b * spacing in every dimension.
std::vector<float> blob_center(uint32_t b, float spacing) {
    return std::vector<float>(BLOB_DIM, static_cast<float>(b) * spacing);
}

template<typename T>
T clamp_to(float v) {
    float lo = static_cast<float>(std::numeric_limits<T>::lowest());
    float hi = static_cast<float>(std::numeric_limits<T>::max());
    return static_cast<T>(std::round(std::min(hi, std::max(lo, v))));
}
template<>
float clamp_to<float>(float v) { return v; }

// Points of blob b are rows [b*POINTS_PER_BLOB, (b+1)*POINTS_PER_BLOB).
template<typename T>
std::vector<T> write_synthetic_base(const std::string& path, float spacing) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<T> data;
    data.reserve(static_cast<size_t>(NUM_POINTS) * BLOB_DIM);
    for (uint32_t b = 0; b < NUM_BLOBS; ++b) {
        std::vector<float> center = blob_center(b, spacing);
        for (uint32_t i = 0; i < POINTS_PER_BLOB; ++i) {
            for (uint32_t d = 0; d < BLOB_DIM; ++d) {
                data.push_back(clamp_to<T>(center[d] + noise(gen)));
            }
        }
    }
    diskann::save_bin<T>(path, data.data(), NUM_POINTS, BLOB_DIM);
    return data;
}

float dist_to_centroid(const float* point, const IVFMetadata& meta, uint32_t c) {
    float sum = 0.0f;
    for (uint32_t d = 0; d < meta.dim; ++d) {
        float diff = point[d] - meta.centroids[static_cast<size_t>(c) * meta.aligned_dim + d];
        sum += diff * diff;
    }
    return sum;
}

uint32_t nearest_centroid(const float* point, const IVFMetadata& meta) {
    uint32_t best = 0;
    float best_dist = dist_to_centroid(point, meta, 0);
    for (uint32_t c = 1; c < meta.nlist; ++c) {
        float d = dist_to_centroid(point, meta, c);
        if (d < best_dist) {
            best_dist = d;
            best = c;
        }
    }
    return best;
}

bool report(bool pass) {
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Centroid training
// ---------------------------------------------------------------------------

bool test_metadata_shape_and_padding(const IVFMetadata& meta) {
    std::cout << "[Test] metadata shape and aligned-dim zero padding..." << std::endl;
    bool pass = true;

    if (meta.nlist != NUM_BLOBS || meta.dim != BLOB_DIM || meta.aligned_dim != 16) {
        std::cout << "  FAIL: unexpected shape nlist=" << meta.nlist << " dim=" << meta.dim
                  << " aligned_dim=" << meta.aligned_dim << std::endl;
        pass = false;
    }
    if (meta.centroids.size() != static_cast<size_t>(meta.nlist) * meta.aligned_dim) {
        std::cout << "  FAIL: centroid buffer size does not match nlist*aligned_dim" << std::endl;
        pass = false;
    }
    for (uint32_t c = 0; c < meta.nlist && pass; ++c) {
        for (uint32_t d = meta.dim; d < meta.aligned_dim; ++d) {
            if (meta.centroids[static_cast<size_t>(c) * meta.aligned_dim + d] != 0.0f) {
                std::cout << "  FAIL: padding not zero at centroid " << c << " dim " << d << std::endl;
                pass = false;
            }
        }
    }
    return report(pass);
}

// Every blob must have a distinct centroid within noise distance of its center.
bool test_centroids_recover_blobs(const IVFMetadata& meta, float spacing) {
    std::cout << "[Test] trained centroids recover the synthetic blobs..." << std::endl;
    bool pass = true;

    std::vector<uint32_t> claimed;
    for (uint32_t b = 0; b < NUM_BLOBS; ++b) {
        std::vector<float> center = blob_center(b, spacing);
        uint32_t best = nearest_centroid(center.data(), meta);
        float best_dist = dist_to_centroid(center.data(), meta, best);
        // The mean of 500 unit-variance samples per dim sits ~0.05 from the
        // true center; 25 (squared, over 12 dims) is far above that but far
        // below the spacing^2 * 12 that a wrong blob would give.
        if (best_dist > 25.0f) {
            std::cout << "  FAIL: blob " << b << " nearest centroid is " << best_dist
                      << " away (squared)" << std::endl;
            pass = false;
        }
        if (std::find(claimed.begin(), claimed.end(), best) != claimed.end()) {
            std::cout << "  FAIL: centroid " << best << " claimed by two blobs" << std::endl;
            pass = false;
        }
        claimed.push_back(best);
    }
    return report(pass);
}

bool test_save_load_round_trip(const std::string& prefix, const IVFMetadata& meta) {
    std::cout << "[Test] centroid save/load round-trip..." << std::endl;
    save_ivf_centroids(prefix, meta);
    IVFMetadata loaded = load_ivf_centroids(prefix, meta.dim);

    bool pass = loaded.nlist == meta.nlist && loaded.dim == meta.dim &&
                loaded.aligned_dim == meta.aligned_dim && loaded.centroids == meta.centroids;
    if (!pass) {
        std::cout << "  FAIL: loaded centroids differ from what was saved" << std::endl;
    }

    // Loading with a dim whose padding does not match the file must be refused.
    bool threw = false;
    try {
        load_ivf_centroids(prefix, meta.dim + 8);
    } catch (const diskann::ANNException&) {
        threw = true;
    }
    if (!threw) {
        std::cout << "  FAIL: load_ivf_centroids accepted a mismatched dim" << std::endl;
        pass = false;
    }

    ::unlink(ivf_centroids_path(prefix).c_str());
    return report(pass);
}

// The seeded primitives are single-threaded, so they must be bit-exact.
bool test_seeded_primitives_are_deterministic(const std::string& base_bin) {
    std::cout << "[Test] seeded sampling and k-means++ init are bit-exact..." << std::endl;
    bool pass = true;

    auto sample = [&](uint32_t seed) {
        float* raw = nullptr;
        size_t n = 0, d = 0;
        gen_random_slice<float>(base_bin, 0.5, raw, n, d, seed);
        std::vector<float> out(raw, raw + n * d);
        delete[] raw;
        return out;
    };
    std::vector<float> s1 = sample(7), s2 = sample(7), s3 = sample(8);
    if (s1.empty() || s1 != s2) {
        std::cout << "  FAIL: gen_random_slice with the same seed drew different samples" << std::endl;
        pass = false;
    }
    if (s1 == s3) {
        std::cout << "  FAIL: gen_random_slice with different seeds drew the same sample" << std::endl;
        pass = false;
    }

    std::vector<float> p1(NUM_BLOBS * BLOB_DIM), p2(NUM_BLOBS * BLOB_DIM), p3(NUM_BLOBS * BLOB_DIM);
    size_t n = s1.size() / BLOB_DIM;
    kmeans::kmeanspp_selecting_pivots(s1.data(), n, BLOB_DIM, p1.data(), NUM_BLOBS, 7);
    kmeans::kmeanspp_selecting_pivots(s1.data(), n, BLOB_DIM, p2.data(), NUM_BLOBS, 7);
    kmeans::kmeanspp_selecting_pivots(s1.data(), n, BLOB_DIM, p3.data(), NUM_BLOBS, 8);
    if (p1 != p2) {
        std::cout << "  FAIL: kmeanspp_selecting_pivots with the same seed picked different pivots"
                  << std::endl;
        pass = false;
    }
    if (p1 == p3) {
        std::cout << "  FAIL: kmeanspp_selecting_pivots with different seeds picked the same pivots"
                  << std::endl;
        pass = false;
    }
    return report(pass);
}

// End to end, Lloyd's reduces cluster sums in OpenMP order, so two runs can
// differ by float rounding; the seed guarantees the same sample and init,
// which on this data means the same optimum to within that rounding.
bool test_same_seed_reproduces_centroids(const std::string& base_bin, const IVFMetadata& meta) {
    std::cout << "[Test] same seed reproduces the centroids to float rounding..." << std::endl;
    IVFMetadata again = train_ivf_centroids<float>(base_bin, NUM_BLOBS, 1.0,
                                                   NUM_K_MEANS_ITERS, TRAIN_SEED);
    bool pass = again.centroids.size() == meta.centroids.size();
    float max_diff = 0.0f;
    for (size_t i = 0; pass && i < meta.centroids.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(again.centroids[i] - meta.centroids[i]));
    }
    if (!pass || max_diff > 1e-3f) {
        std::cout << "  FAIL: second run differs by up to " << max_diff << std::endl;
        pass = false;
    }
    return report(pass);
}

bool test_training_error_paths(const std::string& base_bin) {
    std::cout << "[Test] training rejects nlist == 0, nlist > N, and too small a sample..."
              << std::endl;
    bool pass = true;
    auto expect_throw = [&](const char* what, auto&& fn) {
        try {
            fn();
            std::cout << "  FAIL: " << what << " did not throw" << std::endl;
            pass = false;
        } catch (const diskann::ANNException&) {
        }
    };
    expect_throw("nlist == 0", [&] { train_ivf_centroids<float>(base_bin, 0); });
    expect_throw("nlist > N", [&] { train_ivf_centroids<float>(base_bin, NUM_POINTS + 1); });
    // 1e-6 of 4000 points samples ~0 rows, far short of NUM_BLOBS centroids.
    expect_throw("sample < nlist", [&] {
        train_ivf_centroids<float>(base_bin, NUM_BLOBS, 1e-6, NUM_K_MEANS_ITERS, TRAIN_SEED);
    });
    return report(pass);
}

// ---------------------------------------------------------------------------
// Cluster assignment + raw-vector bulk load
// ---------------------------------------------------------------------------

// Ground truth is a brute-force argmin over the same padded centroids the
// build uses. `max_block_points` does not divide NUM_POINTS, so the streaming
// loop runs several full blocks and a short tail.
template<typename T>
bool test_assign_clusters_and_load_heap(const std::string& tag, const std::string& prefix,
                                        const std::string& base_bin, const std::vector<T>& base,
                                        const IVFMetadata& meta) {
    std::cout << "[Test] " << tag << ": assignment is the exact nearest centroid and the heap "
              << "holds every raw vector..." << std::endl;
    bool pass = true;
    const size_t max_block_points = 333;

    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, meta.dim * sizeof(T));
    std::string heap_path = ivf_raw_vectors_path(prefix);
    ::unlink(heap_path.c_str());
    RawVectorHeap heap;
    heap.open(heap_path, layout);

    ClusterAssignments assignments;
    RawVectorRIDTable rid_table;
    assign_ivf_clusters<T>(base_bin, meta, heap, assignments, rid_table, max_block_points);

    if (assignments.cluster_id.size() != NUM_POINTS || rid_table.rid.size() != NUM_POINTS) {
        std::cout << "  FAIL: expected " << NUM_POINTS << " assignments/RIDs, got "
                  << assignments.cluster_id.size() << "/" << rid_table.rid.size() << std::endl;
        return report(false);
    }

    size_t wrong_cluster = 0, wrong_rid = 0, wrong_bytes = 0;
    std::vector<uint32_t> per_cluster(meta.nlist, 0);
    std::vector<float> point(meta.dim);
    std::vector<T> got(meta.dim);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        for (uint32_t d = 0; d < meta.dim; ++d) {
            point[d] = static_cast<float>(base[i * meta.dim + d]);
        }
        if (assignments.cluster_id[i] != nearest_centroid(point.data(), meta)) {
            ++wrong_cluster;
        }
        if (assignments.cluster_id[i] < meta.nlist) {
            ++per_cluster[assignments.cluster_id[i]];
        }
        RawVectorRID rid = rid_table.rid[i];
        if (!rid_is_active(rid) || rid_flat_slot(rid) != i) {
            ++wrong_rid;
        }
        heap.read_vector(rid_flat_slot(rid), got.data());
        if (std::memcmp(got.data(), base.data() + i * meta.dim, meta.dim * sizeof(T)) != 0) {
            ++wrong_bytes;
        }
    }
    if (wrong_cluster || wrong_rid || wrong_bytes) {
        std::cout << "  FAIL: " << wrong_cluster << " misassigned, " << wrong_rid << " bad RIDs, "
                  << wrong_bytes << " raw-vector mismatches" << std::endl;
        pass = false;
    }
    // With one centroid per blob, every cluster holds exactly one blob.
    for (uint32_t c = 0; c < meta.nlist; ++c) {
        if (per_cluster[c] != POINTS_PER_BLOB) {
            std::cout << "  FAIL: cluster " << c << " holds " << per_cluster[c] << " points, expected "
                      << POINTS_PER_BLOB << std::endl;
            pass = false;
        }
    }

    // Posting lists built from these assignments must be exactly the blobs'
    // row ranges, since blob b occupies rows [b*500, (b+1)*500).
    PostingLists lists = build_ivf_posting_lists(assignments, meta.nlist);
    for (uint32_t c = 0; c < meta.nlist && pass; ++c) {
        uint32_t begin = lists.offsets[c], end = lists.offsets[c + 1];
        if (end - begin != POINTS_PER_BLOB || lists.ids[begin] % POINTS_PER_BLOB != 0) {
            std::cout << "  FAIL: partition " << c << " is not one whole blob" << std::endl;
            pass = false;
            break;
        }
        for (uint32_t k = begin; k < end; ++k) {
            if (lists.ids[k] != lists.ids[begin] + (k - begin)) {
                std::cout << "  FAIL: partition " << c << " is not the contiguous blob rows"
                          << std::endl;
                pass = false;
                break;
            }
        }
    }

    uint32_t expected_pages = (NUM_POINTS + layout.slots_per_page - 1) / layout.slots_per_page;
    if (heap.next_flat_slot() != NUM_POINTS || heap.allocated_pages() != expected_pages) {
        std::cout << "  FAIL: heap cursor slot/page " << heap.next_flat_slot() << "/"
                  << heap.allocated_pages() << ", expected " << NUM_POINTS << "/" << expected_pages
                  << std::endl;
        pass = false;
    }

    // Sidecar round-trips.
    save_ivf_cluster_assignments(prefix, assignments);
    save_ivf_rid_table(prefix, rid_table);
    ClusterAssignments assignments2 = load_ivf_cluster_assignments(prefix);
    RawVectorRIDTable rid_table2 = load_ivf_rid_table(prefix);
    bool rids_equal = rid_table2.rid.size() == rid_table.rid.size() &&
                      std::equal(rid_table.rid.begin(), rid_table.rid.end(), rid_table2.rid.begin(),
                                 [](RawVectorRID a, RawVectorRID b) { return a.packed == b.packed; });
    if (assignments2.cluster_id != assignments.cluster_id || !rids_equal) {
        std::cout << "  FAIL: cluster-id or RID-table sidecar did not round-trip" << std::endl;
        pass = false;
    }

    heap.close();
    ::unlink(heap_path.c_str());
    ::unlink(ivf_cluster_ids_path(prefix).c_str());
    ::unlink(ivf_rid_table_path(prefix).c_str());
    return report(pass);
}

bool test_assignment_error_paths(const std::string& prefix, const std::string& base_bin,
                                 const IVFMetadata& meta) {
    std::cout << "[Test] assignment rejects a mismatched heap, a mismatched base, and a used heap..."
              << std::endl;
    bool pass = true;
    std::string heap_path = ivf_raw_vectors_path(prefix) + ".err";
    ClusterAssignments assignments;
    RawVectorRIDTable rid_table;

    auto expect_throw = [&](const char* what, auto&& fn) {
        ::unlink(heap_path.c_str());
        try {
            fn();
            std::cout << "  FAIL: " << what << " did not throw" << std::endl;
            pass = false;
        } catch (const diskann::ANNException&) {
        }
    };

    expect_throw("heap elem_size != dim * sizeof(T)", [&] {
        RawVectorHeap heap;
        heap.open(heap_path, compute_raw_vector_heap_layout(4096, (meta.dim + 1) * sizeof(float)));
        assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table);
    });
    expect_throw("base dim != centroid dim", [&] {
        IVFMetadata other = meta;
        other.dim += 1;
        RawVectorHeap heap;
        heap.open(heap_path, compute_raw_vector_heap_layout(4096, other.dim * sizeof(float)));
        assign_ivf_clusters<float>(base_bin, other, heap, assignments, rid_table);
    });
    expect_throw("heap already has slots", [&] {
        RawVectorHeap heap;
        heap.open(heap_path, compute_raw_vector_heap_layout(4096, meta.dim * sizeof(float)));
        RawVectorFreeList free_list;
        heap.allocate_slot(free_list);
        assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table);
    });
    expect_throw("max_block_points == 0", [&] {
        RawVectorHeap heap;
        heap.open(heap_path, compute_raw_vector_heap_layout(4096, meta.dim * sizeof(float)));
        assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table, 0);
    });
    ::unlink(heap_path.c_str());
    return report(pass);
}

// ---------------------------------------------------------------------------
// Posting lists
// ---------------------------------------------------------------------------

// Random assignments over more clusters than get used, so partitions are
// interleaved (not contiguous ID runs) and several are empty.
bool test_posting_lists_invert_assignments(const std::string& prefix) {
    std::cout << "[Test] posting lists are the exact inverse of the assignments..." << std::endl;
    bool pass = true;
    const uint32_t nlist = 13;
    const uint32_t used_clusters = 10;  // 10..12 never appear; 3 is skipped too
    const size_t npts = 5000;

    ClusterAssignments assignments;
    std::mt19937 gen(99);
    std::uniform_int_distribution<uint32_t> pick(0, used_clusters - 1);
    for (size_t i = 0; i < npts; ++i) {
        uint32_t c = pick(gen);
        assignments.cluster_id.push_back(c == 3 ? 4 : c);
    }

    PostingLists lists = build_ivf_posting_lists(assignments, nlist);
    if (lists.offsets.size() != nlist + 1 || lists.offsets.front() != 0 ||
        lists.offsets.back() != npts || lists.ids.size() != npts ||
        !std::is_sorted(lists.offsets.begin(), lists.offsets.end())) {
        std::cout << "  FAIL: offsets are not a valid CSR row index" << std::endl;
        return report(false);
    }

    std::vector<uint32_t> seen(npts, 0);
    for (uint32_t c = 0; c < nlist; ++c) {
        uint32_t begin = lists.offsets[c], end = lists.offsets[c + 1];
        for (uint32_t k = begin; k < end; ++k) {
            uint32_t id = lists.ids[k];
            if (id >= npts || assignments.cluster_id[id] != c) {
                std::cout << "  FAIL: id " << id << " listed under cluster " << c << std::endl;
                return report(false);
            }
            if (k > begin && lists.ids[k - 1] >= id) {
                std::cout << "  FAIL: partition " << c << " is not strictly ascending" << std::endl;
                return report(false);
            }
            ++seen[id];
        }
    }
    if (std::count(seen.begin(), seen.end(), 1u) != static_cast<long>(npts)) {
        std::cout << "  FAIL: some vector ID is missing or listed more than once" << std::endl;
        pass = false;
    }
    for (uint32_t c : {3u, 10u, 11u, 12u}) {
        if (lists.offsets[c] != lists.offsets[c + 1]) {
            std::cout << "  FAIL: cluster " << c << " should be empty" << std::endl;
            pass = false;
        }
    }

    // Empty input still yields a well-formed (all-zero) row index.
    PostingLists none = build_ivf_posting_lists(ClusterAssignments{}, 5);
    if (none.offsets != std::vector<uint32_t>(6, 0) || !none.ids.empty()) {
        std::cout << "  FAIL: empty assignments did not give an empty CSR" << std::endl;
        pass = false;
    }

    // A cluster ID at or past nlist is a corrupt assignment, not an empty partition.
    bool threw = false;
    try {
        build_ivf_posting_lists(assignments, used_clusters - 1);
    } catch (const diskann::ANNException&) {
        threw = true;
    }
    if (!threw) {
        std::cout << "  FAIL: out-of-range cluster ID was accepted" << std::endl;
        pass = false;
    }

    // Sidecar round-trip, and the loader rejects offsets that do not match ids.
    save_ivf_posting_lists(prefix, lists);
    PostingLists loaded = load_ivf_posting_lists(prefix);
    if (loaded.offsets != lists.offsets || loaded.ids != lists.ids) {
        std::cout << "  FAIL: posting-list sidecars did not round-trip" << std::endl;
        pass = false;
    }
    PostingLists truncated = lists;
    truncated.ids.pop_back();
    save_ivf_posting_lists(prefix, truncated);
    threw = false;
    try {
        load_ivf_posting_lists(prefix);
    } catch (const diskann::ANNException&) {
        threw = true;
    }
    if (!threw) {
        std::cout << "  FAIL: loader accepted offsets whose total exceeds the ids file" << std::endl;
        pass = false;
    }
    ::unlink(ivf_posting_offsets_path(prefix).c_str());
    ::unlink(ivf_posting_ids_path(prefix).c_str());
    return report(pass);
}

}  // namespace

int main() {
    std::string prefix = "/tmp/ivf_pq_build_test_" + std::to_string((uint64_t) getpid());
    std::string base_f32 = prefix + "_base_f32.bin";
    std::string base_u8 = prefix + "_base_u8.bin";

    bool all_pass = true;
    try {
        std::vector<float> data_f32 = write_synthetic_base<float>(base_f32, BLOB_SPACING_F32);
        IVFMetadata meta = train_ivf_centroids<float>(base_f32, NUM_BLOBS, 1.0,
                                                      NUM_K_MEANS_ITERS, TRAIN_SEED);

        all_pass &= test_metadata_shape_and_padding(meta);
        all_pass &= test_centroids_recover_blobs(meta, BLOB_SPACING_F32);
        all_pass &= test_save_load_round_trip(prefix, meta);
        all_pass &= test_seeded_primitives_are_deterministic(base_f32);
        all_pass &= test_same_seed_reproduces_centroids(base_f32, meta);
        all_pass &= test_training_error_paths(base_f32);
        all_pass &= test_assign_clusters_and_load_heap<float>("float", prefix, base_f32, data_f32, meta);
        all_pass &= test_assignment_error_paths(prefix, base_f32, meta);
        all_pass &= test_posting_lists_invert_assignments(prefix);

        // uint8 base: the raw bytes stored in the heap are 1 byte per dim, and
        // training/assignment must convert without touching what is stored.
        std::vector<uint8_t> data_u8 = write_synthetic_base<uint8_t>(base_u8, BLOB_SPACING_U8);
        IVFMetadata meta_u8 = train_ivf_centroids<uint8_t>(base_u8, NUM_BLOBS, 1.0,
                                                           NUM_K_MEANS_ITERS, TRAIN_SEED);
        all_pass &= test_centroids_recover_blobs(meta_u8, BLOB_SPACING_U8);
        all_pass &= test_assign_clusters_and_load_heap<uint8_t>("uint8", prefix + "_u8", base_u8,
                                                                data_u8, meta_u8);
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }

    ::unlink(base_f32.c_str());
    ::unlink(base_u8.c_str());
    return all_pass ? 0 : 1;
}
