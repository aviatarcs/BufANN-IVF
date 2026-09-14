// Tests for IVF centroid training and cluster assignment: metadata shape,
// padding, blob recovery on synthetic clusters, save/load round-trips, and
// assignment + raw-vector bulk load of the synthetic base.

#include "bufann/ivf_pq_build.h"
#include "ann_exception.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
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
const uint32_t BLOB_DIM = 12;  // not a multiple of 8, so padding is exercised
const float BLOB_SPACING = 50.0f;

// Blob b is centered at b * BLOB_SPACING in every dimension.
std::vector<float> blob_center(uint32_t b) {
    return std::vector<float>(BLOB_DIM, static_cast<float>(b) * BLOB_SPACING);
}

std::string write_synthetic_base(const std::string& path) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<float> data;
    data.reserve(static_cast<size_t>(NUM_BLOBS) * POINTS_PER_BLOB * BLOB_DIM);
    for (uint32_t b = 0; b < NUM_BLOBS; ++b) {
        std::vector<float> center = blob_center(b);
        for (uint32_t i = 0; i < POINTS_PER_BLOB; ++i) {
            for (uint32_t d = 0; d < BLOB_DIM; ++d) {
                data.push_back(center[d] + noise(gen));
            }
        }
    }
    diskann::save_bin<float>(path, data.data(),
                             static_cast<size_t>(NUM_BLOBS) * POINTS_PER_BLOB, BLOB_DIM);
    return path;
}

float dist_to_centroid(const std::vector<float>& point, const IVFMetadata& meta, uint32_t c) {
    float sum = 0.0f;
    for (uint32_t d = 0; d < meta.dim; ++d) {
        float diff = point[d] - meta.centroids[static_cast<size_t>(c) * meta.aligned_dim + d];
        sum += diff * diff;
    }
    return sum;
}

bool test_centroids_recover_blobs(const IVFMetadata& meta) {
    std::cout << "[Test] trained centroids recover the synthetic blobs..." << std::endl;
    bool pass = true;

    // Each blob should have its own nearby centroid.
    std::vector<uint32_t> claimed;
    for (uint32_t b = 0; b < NUM_BLOBS; ++b) {
        std::vector<float> center = blob_center(b);
        uint32_t best = 0;
        float best_dist = dist_to_centroid(center, meta, 0);
        for (uint32_t c = 1; c < meta.nlist; ++c) {
            float d = dist_to_centroid(center, meta, c);
            if (d < best_dist) {
                best_dist = d;
                best = c;
            }
        }
        if (best_dist > 25.0f) {
            std::cout << "  FAIL: blob " << b << " nearest centroid is " << best_dist
                      << " away (squared)" << std::endl;
            pass = false;
        }
        for (uint32_t prev : claimed) {
            if (prev == best) {
                std::cout << "  FAIL: centroid " << best << " claimed by two blobs" << std::endl;
                pass = false;
            }
        }
        claimed.push_back(best);
    }

    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

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

    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
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

    ::unlink(ivf_centroids_path(prefix).c_str());
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool test_same_seed_is_reproducible(const std::string& base_bin, const IVFMetadata& meta) {
    std::cout << "[Test] same seed reproduces identical centroids..." << std::endl;
    IVFMetadata again = train_ivf_centroids<float>(base_bin, NUM_BLOBS, 1.0,
                                                   NUM_K_MEANS_ITERS, TRAIN_SEED);
    bool pass = again.centroids == meta.centroids;
    if (!pass) {
        std::cout << "  FAIL: a second run with the same seed produced different centroids"
                  << std::endl;
    }
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// Each point must be assigned to the centroid nearest its blob, and each raw
// vector must come back byte-identical from the heap slot its RID names.
bool test_assign_clusters_and_load_heap(const std::string& prefix, const std::string& base_bin,
                                        const IVFMetadata& meta) {
    std::cout << "[Test] cluster assignment matches blobs and heap holds every raw vector..."
              << std::endl;
    bool pass = true;

    // Which centroid each blob trained to.
    std::vector<uint32_t> blob_to_centroid(NUM_BLOBS);
    for (uint32_t b = 0; b < NUM_BLOBS; ++b) {
        std::vector<float> center = blob_center(b);
        uint32_t best = 0;
        for (uint32_t c = 1; c < meta.nlist; ++c) {
            if (dist_to_centroid(center, meta, c) < dist_to_centroid(center, meta, best)) {
                best = c;
            }
        }
        blob_to_centroid[b] = best;
    }

    // 4 KB pages hold ~84 12-float slots, so 4000 points span ~48 pages with a
    // partial tail page. (Multi-flush behaviour is covered by the heap test.)
    RawVectorHeapLayout layout =
        compute_raw_vector_heap_layout(4096, meta.dim * sizeof(float));
    std::string heap_path = ivf_raw_vectors_path(prefix);
    ::unlink(heap_path.c_str());
    RawVectorHeap heap;
    heap.open(heap_path, layout);

    ClusterAssignments assignments;
    RawVectorRIDTable rid_table;
    assign_ivf_clusters<float>(base_bin, meta, heap, assignments, rid_table);

    const size_t npts = static_cast<size_t>(NUM_BLOBS) * POINTS_PER_BLOB;
    if (assignments.cluster_id.size() != npts || rid_table.rid.size() != npts) {
        std::cout << "  FAIL: expected " << npts << " assignments/RIDs, got "
                  << assignments.cluster_id.size() << "/" << rid_table.rid.size() << std::endl;
        pass = false;
    }

    float* raw = nullptr;
    size_t n = 0, d = 0;
    diskann::load_bin<float>(base_bin, raw, n, d);
    std::unique_ptr<float[]> base(raw);

    size_t wrong_cluster = 0, wrong_bytes = 0, wrong_rid = 0;
    std::vector<float> got(meta.dim);
    for (size_t i = 0; i < npts && pass; ++i) {
        uint32_t blob = static_cast<uint32_t>(i / POINTS_PER_BLOB);
        if (assignments.cluster_id[i] != blob_to_centroid[blob]) {
            ++wrong_cluster;
        }
        RawVectorRID rid = rid_table.rid[i];
        if (!rid_is_active(rid) || rid_flat_slot(rid) != i) {
            ++wrong_rid;
        }
        heap.read_vector(rid_flat_slot(rid), got.data());
        if (std::memcmp(got.data(), base.get() + i * meta.dim, meta.dim * sizeof(float)) != 0) {
            ++wrong_bytes;
        }
    }
    if (wrong_cluster || wrong_rid || wrong_bytes) {
        std::cout << "  FAIL: " << wrong_cluster << " misassigned, " << wrong_rid
                  << " bad RIDs, " << wrong_bytes << " raw-vector mismatches" << std::endl;
        pass = false;
    }
    if (heap.next_flat_slot() != npts) {
        std::cout << "  FAIL: heap cursor at " << heap.next_flat_slot() << ", expected " << npts
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
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

}  // namespace

int main() {
    std::string prefix = "/tmp/ivf_pq_build_test_" + std::to_string((uint64_t) getpid());
    std::string base_bin = prefix + "_base.bin";

    bool all_pass = true;
    try {
        write_synthetic_base(base_bin);
        IVFMetadata meta = train_ivf_centroids<float>(base_bin, NUM_BLOBS, 1.0,
                                                      NUM_K_MEANS_ITERS, TRAIN_SEED);

        all_pass &= test_metadata_shape_and_padding(meta);
        all_pass &= test_centroids_recover_blobs(meta);
        all_pass &= test_save_load_round_trip(prefix, meta);
        all_pass &= test_same_seed_is_reproducible(base_bin, meta);
        all_pass &= test_assign_clusters_and_load_heap(prefix, base_bin, meta);
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }

    ::unlink(base_bin.c_str());
    return all_pass ? 0 : 1;
}
