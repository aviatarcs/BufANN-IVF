// Tests for the IVF-PQ build steps on a synthetic-blob base: centroid
// training (shape, padding, blob recovery, seeding, save/load), cluster
// assignment + raw-vector bulk load (exact nearest-centroid agreement,
// RID/heap contents, multi-block streaming, sidecar round-trips, error paths),
// and posting-list construction (exact inverse of the assignments).

#include "bufann/ivf_pq_build.h"
#include "ivf_pq_test_util.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t NUM_BLOBS = 8;
const uint32_t POINTS_PER_BLOB = 500;
const uint32_t NUM_POINTS = NUM_BLOBS * POINTS_PER_BLOB;
const uint32_t BLOB_DIM = 12;  // not a multiple of 8, so padding is exercised
const uint32_t TRAIN_SEED = 12345;
const float BLOB_SPACING_F32 = 50.0f;
const float BLOB_SPACING_U8 = 30.0f;  // 8 blobs at 30 fit in [0, 255]

// Blob b is centered at b * spacing in every dimension and occupies rows
// [b*POINTS_PER_BLOB, (b+1)*POINTS_PER_BLOB) of the base file.
std::vector<float> blob_center(uint32_t b, float spacing) {
    return std::vector<float>(BLOB_DIM, float(b) * spacing);
}

template<typename T>
T clamp_to(float v) {
    float lo = float(std::numeric_limits<T>::lowest()), hi = float(std::numeric_limits<T>::max());
    return T(std::round(std::min(hi, std::max(lo, v))));
}
template<>
float clamp_to<float>(float v) { return v; }

template<typename T>
std::vector<T> write_synthetic_base(const std::string& path, float spacing) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::vector<T> data;
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
        float diff = point[d] - meta.centroids[size_t(c) * meta.aligned_dim + d];
        sum += diff * diff;
    }
    return sum;
}

uint32_t nearest_centroid(const float* point, const IVFMetadata& meta) {
    uint32_t best = 0;
    for (uint32_t c = 1; c < meta.nlist; ++c) {
        if (dist_to_centroid(point, meta, c) < dist_to_centroid(point, meta, best)) best = c;
    }
    return best;
}

IVFMetadata train(const std::string& base_bin, double sampling_rate = 1.0) {
    return train_ivf_centroids<float>(base_bin, NUM_BLOBS, sampling_rate, NUM_K_MEANS_ITERS, TRAIN_SEED);
}

// --- centroid training -----------------------------------------------------

bool test_metadata_shape_and_padding(const IVFMetadata& meta) {
    TestCase t("metadata shape and aligned-dim zero padding");
    t.check(meta.nlist == NUM_BLOBS && meta.dim == BLOB_DIM && meta.aligned_dim == 16,
            "unexpected nlist/dim/aligned_dim");
    t.check(meta.centroids.size() == size_t(meta.nlist) * meta.aligned_dim, "centroid buffer size");
    for (uint32_t c = 0; c < meta.nlist; ++c) {
        for (uint32_t d = meta.dim; d < meta.aligned_dim; ++d) {
            t.check(meta.centroids[size_t(c) * meta.aligned_dim + d] == 0.0f, "padding is not zero");
        }
    }
    return t.done();
}

bool test_centroids_recover_blobs(const IVFMetadata& meta, float spacing) {
    TestCase t("each blob has its own centroid within noise distance");
    std::vector<uint32_t> claimed;
    for (uint32_t b = 0; b < NUM_BLOBS; ++b) {
        std::vector<float> center = blob_center(b, spacing);
        uint32_t c = nearest_centroid(center.data(), meta);
        // The mean of 500 unit-variance samples sits ~0.05 from the true
        // center per dim; 25 squared over 12 dims is well above that and far
        // below the 12 * spacing^2 a wrong blob would give.
        t.check(dist_to_centroid(center.data(), meta, c) <= 25.0f,
                "blob " + std::to_string(b) + " has no nearby centroid");
        t.check(std::find(claimed.begin(), claimed.end(), c) == claimed.end(),
                "centroid " + std::to_string(c) + " claimed by two blobs");
        claimed.push_back(c);
    }
    return t.done();
}

bool test_centroid_save_load(const std::string& prefix, const IVFMetadata& meta) {
    TestCase t("centroid save/load round-trip");
    save_ivf_centroids(prefix, meta);
    IVFMetadata loaded = load_ivf_centroids(prefix, meta.dim);
    t.check(loaded.nlist == meta.nlist && loaded.dim == meta.dim &&
                loaded.aligned_dim == meta.aligned_dim && loaded.centroids == meta.centroids,
            "loaded centroids differ from what was saved");
    t.expect_throw("load with a dim whose padding differs",
                   [&] { load_ivf_centroids(prefix, meta.dim + 8); });
    ::unlink(ivf_centroids_path(prefix).c_str());
    return t.done();
}

// The seeded primitives are single-threaded, so they must be bit-exact.
bool test_seeded_primitives_are_deterministic(const std::string& base_bin) {
    TestCase t("seeded sampling and k-means++ init are bit-exact");
    auto sample = [&](uint32_t seed) {
        float* raw = nullptr;
        size_t n = 0, d = 0;
        gen_random_slice<float>(base_bin, 0.5, raw, n, d, seed);
        std::vector<float> out(raw, raw + n * d);
        delete[] raw;
        return out;
    };
    std::vector<float> s1 = sample(7), s2 = sample(7), s3 = sample(8);
    t.check(!s1.empty() && s1 == s2, "same seed drew different samples");
    t.check(s1 != s3, "different seeds drew the same sample");

    auto pivots = [&](uint32_t seed) {
        std::vector<float> p(NUM_BLOBS * BLOB_DIM);
        kmeans::kmeanspp_selecting_pivots(s1.data(), s1.size() / BLOB_DIM, BLOB_DIM, p.data(),
                                          NUM_BLOBS, seed);
        return p;
    };
    t.check(pivots(7) == pivots(7), "same seed picked different pivots");
    t.check(pivots(7) != pivots(8), "different seeds picked the same pivots");
    return t.done();
}

// Lloyd's sums cluster members in OpenMP arrival order, so two runs can differ
// by float rounding; the seed pins the sample and the init, hence the optimum.
bool test_same_seed_reproduces_centroids(const std::string& base_bin, const IVFMetadata& meta) {
    TestCase t("same seed reproduces the centroids to float rounding");
    IVFMetadata again = train(base_bin);
    float max_diff = 0.0f;
    for (size_t i = 0; i < meta.centroids.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(again.centroids[i] - meta.centroids[i]));
    }
    t.check(again.centroids.size() == meta.centroids.size() && max_diff <= 1e-3f,
            "second run differs by up to " + std::to_string(max_diff));
    return t.done();
}

bool test_training_error_paths(const std::string& base_bin) {
    TestCase t("training rejects nlist == 0, nlist > N, and too small a sample");
    t.expect_throw("nlist == 0", [&] { train_ivf_centroids<float>(base_bin, 0); });
    t.expect_throw("nlist > N", [&] { train_ivf_centroids<float>(base_bin, NUM_POINTS + 1); });
    t.expect_throw("sample of ~0 rows", [&] { train(base_bin, 1e-6); });
    return t.done();
}

// --- cluster assignment + bulk load ---------------------------------------

// Ground truth is a brute-force argmin over the same centroids. 333 does not
// divide NUM_POINTS, so the streaming loop runs full blocks and a short tail.
template<typename T>
bool test_assign_clusters_and_load_heap(const std::string& tag, const std::string& prefix,
                                        const std::string& base_bin, const std::vector<T>& base,
                                        const IVFMetadata& meta) {
    TestCase t(tag + ": assignment is the exact nearest centroid and the heap holds every vector");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, meta.dim * sizeof(T));
    std::string heap_path = ivf_raw_vectors_path(prefix);
    RawVectorHeap heap;
    heap.open(heap_path, layout);

    ClusterAssignments assignments;
    RawVectorRIDTable rid_table;
    assign_ivf_clusters<T>(base_bin, meta, heap, assignments, rid_table, 333);
    if (!t.check(assignments.cluster_id.size() == NUM_POINTS && rid_table.rid.size() == NUM_POINTS,
                 "wrong number of assignments or RIDs")) {
        return t.done();
    }

    size_t wrong_cluster = 0, wrong_rid = 0, wrong_bytes = 0;
    std::vector<float> point(meta.dim);
    std::vector<T> got(meta.dim);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        const T* row = base.data() + i * meta.dim;
        std::copy_n(row, meta.dim, point.begin());
        wrong_cluster += assignments.cluster_id[i] != nearest_centroid(point.data(), meta);

        RawVectorRID rid = rid_table.rid[i];
        wrong_rid += !rid_is_active(rid) || rid_flat_slot(rid) != i;
        heap.read_vector(rid_flat_slot(rid), got.data());
        wrong_bytes += std::memcmp(got.data(), row, meta.dim * sizeof(T)) != 0;
    }
    t.check(wrong_cluster == 0, std::to_string(wrong_cluster) + " vectors misassigned");
    t.check(wrong_rid == 0, std::to_string(wrong_rid) + " RIDs are not (active, slot i)");
    t.check(wrong_bytes == 0, std::to_string(wrong_bytes) + " raw vectors differ from the base");

    // Posting lists of these assignments are exactly the blobs' row ranges.
    PostingLists lists = build_ivf_posting_lists(assignments, meta.nlist);
    for (uint32_t c = 0; c < meta.nlist; ++c) {
        uint32_t begin = lists.offsets[c], end = lists.offsets[c + 1];
        bool whole_blob = end - begin == POINTS_PER_BLOB && lists.ids[begin] % POINTS_PER_BLOB == 0;
        for (uint32_t k = begin; whole_blob && k < end; ++k) {
            whole_blob = lists.ids[k] == lists.ids[begin] + (k - begin);
        }
        t.check(whole_blob, "partition " + std::to_string(c) + " is not one contiguous blob");
    }

    uint32_t pages = (NUM_POINTS + layout.slots_per_page - 1) / layout.slots_per_page;
    t.check(heap.next_flat_slot() == NUM_POINTS && heap.allocated_pages() == pages,
            "heap cursor is not at the end of the load");

    save_ivf_cluster_assignments(prefix, assignments);
    save_ivf_rid_table(prefix, rid_table);
    t.check(load_ivf_cluster_assignments(prefix).cluster_id == assignments.cluster_id,
            "cluster-id sidecar did not round-trip");
    RawVectorRIDTable loaded = load_ivf_rid_table(prefix);
    t.check(loaded.rid.size() == rid_table.rid.size() &&
                std::equal(rid_table.rid.begin(), rid_table.rid.end(), loaded.rid.begin(),
                           [](RawVectorRID a, RawVectorRID b) { return a.packed == b.packed; }),
            "RID-table sidecar did not round-trip");

    heap.close();
    ::unlink(heap_path.c_str());
    ::unlink(ivf_cluster_ids_path(prefix).c_str());
    ::unlink(ivf_rid_table_path(prefix).c_str());
    return t.done();
}

bool test_assignment_error_paths(const std::string& prefix, const std::string& base_bin,
                                 const IVFMetadata& meta) {
    TestCase t("assignment rejects a mismatched heap, a mismatched base, a used heap, zero block");
    std::string heap_path = ivf_raw_vectors_path(prefix) + ".err";
    ClusterAssignments assignments;
    RawVectorRIDTable rid_table;
    auto fresh_heap = [&](RawVectorHeap& heap, uint32_t elem_size) {
        ::unlink(heap_path.c_str());
        heap.open(heap_path, compute_raw_vector_heap_layout(4096, elem_size));
    };

    t.expect_throw("heap elem_size != dim * sizeof(T)", [&] {
        RawVectorHeap heap;
        fresh_heap(heap, (meta.dim + 1) * sizeof(float));
        assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table);
    });
    t.expect_throw("base dim != centroid dim", [&] {
        IVFMetadata other = meta;
        other.dim += 1;
        RawVectorHeap heap;
        fresh_heap(heap, other.dim * sizeof(float));
        assign_ivf_clusters<float>(base_bin, other, heap, assignments, rid_table);
    });
    t.expect_throw("heap already has a slot", [&] {
        RawVectorHeap heap;
        fresh_heap(heap, meta.dim * sizeof(float));
        RawVectorFreeList free_list;
        heap.allocate_slot(free_list);
        assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table);
    });
    t.expect_throw("max_block_points == 0", [&] {
        RawVectorHeap heap;
        fresh_heap(heap, meta.dim * sizeof(float));
        assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table, 0);
    });
    ::unlink(heap_path.c_str());
    return t.done();
}

// --- posting lists ---------------------------------------------------------

// Random assignments over more clusters than get used, so partitions are
// interleaved rather than contiguous ID runs, and several are empty.
bool test_posting_lists_invert_assignments(const std::string& prefix) {
    TestCase t("posting lists are the exact inverse of the assignments");
    const uint32_t nlist = 13;
    const size_t npts = 5000;
    ClusterAssignments assignments;
    std::mt19937 gen(99);
    std::uniform_int_distribution<uint32_t> pick(0, 9);
    for (size_t i = 0; i < npts; ++i) {
        uint32_t c = pick(gen);
        assignments.cluster_id.push_back(c == 3 ? 4 : c);  // 3 and 10..12 stay empty
    }

    PostingLists lists = build_ivf_posting_lists(assignments, nlist);
    if (!t.check(lists.offsets.size() == nlist + 1 && lists.offsets.front() == 0 &&
                     lists.offsets.back() == npts && lists.ids.size() == npts &&
                     std::is_sorted(lists.offsets.begin(), lists.offsets.end()),
                 "offsets are not a valid CSR row index")) {
        return t.done();
    }
    std::vector<uint32_t> times_listed(npts, 0);
    for (uint32_t c = 0; c < nlist; ++c) {
        for (uint32_t k = lists.offsets[c]; k < lists.offsets[c + 1]; ++k) {
            uint32_t id = lists.ids[k];
            t.check(id < npts && assignments.cluster_id[id] == c,
                    "id " + std::to_string(id) + " listed under cluster " + std::to_string(c));
            t.check(k == lists.offsets[c] || lists.ids[k - 1] < id,
                    "partition " + std::to_string(c) + " is not strictly ascending");
            if (id < npts) ++times_listed[id];
        }
    }
    t.check(std::all_of(times_listed.begin(), times_listed.end(), [](uint32_t n) { return n == 1; }),
            "some vector ID is missing or listed more than once");
    for (uint32_t c : {3u, 10u, 11u, 12u}) {
        t.check(lists.offsets[c] == lists.offsets[c + 1], "cluster " + std::to_string(c) + " not empty");
    }

    PostingLists none = build_ivf_posting_lists(ClusterAssignments{}, 5);
    t.check(none.offsets == std::vector<uint32_t>(6, 0) && none.ids.empty(),
            "empty assignments did not give an empty CSR");
    t.expect_throw("cluster ID >= nlist", [&] { build_ivf_posting_lists(assignments, 9); });

    save_ivf_posting_lists(prefix, lists);
    PostingLists loaded = load_ivf_posting_lists(prefix);
    t.check(loaded.offsets == lists.offsets && loaded.ids == lists.ids, "sidecars did not round-trip");
    PostingLists truncated = lists;
    truncated.ids.pop_back();
    save_ivf_posting_lists(prefix, truncated);
    t.expect_throw("offsets claiming more ids than the ids file holds",
                   [&] { load_ivf_posting_lists(prefix); });
    ::unlink(ivf_posting_offsets_path(prefix).c_str());
    ::unlink(ivf_posting_ids_path(prefix).c_str());
    return t.done();
}

}  // namespace

int main() {
    std::string prefix = temp_path("ivf_pq_build_test");
    std::string base_f32 = prefix + "_base_f32.bin", base_u8 = prefix + "_base_u8.bin";

    bool all_pass = true;
    try {
        std::vector<float> data_f32 = write_synthetic_base<float>(base_f32, BLOB_SPACING_F32);
        IVFMetadata meta = train(base_f32);
        all_pass &= test_metadata_shape_and_padding(meta);
        all_pass &= test_centroids_recover_blobs(meta, BLOB_SPACING_F32);
        all_pass &= test_centroid_save_load(prefix, meta);
        all_pass &= test_seeded_primitives_are_deterministic(base_f32);
        all_pass &= test_same_seed_reproduces_centroids(base_f32, meta);
        all_pass &= test_training_error_paths(base_f32);
        all_pass &= test_assign_clusters_and_load_heap<float>("float", prefix, base_f32, data_f32, meta);
        all_pass &= test_assignment_error_paths(prefix, base_f32, meta);
        all_pass &= test_posting_lists_invert_assignments(prefix);

        // uint8 base: the heap must hold the 1-byte-per-dim input, not a float conversion.
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
