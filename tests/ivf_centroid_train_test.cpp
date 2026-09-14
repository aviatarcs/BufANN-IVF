// Tests for IVF centroid training: metadata shape, padding, blob recovery on
// synthetic clusters, and save/load round-trip.

#include "bufann/ivf_pq_build.h"
#include "ann_exception.h"
#include "utils.h"

#include <cmath>
#include <iostream>
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

}  // namespace

int main() {
    std::string prefix = "/tmp/ivf_centroid_train_test_" + std::to_string((uint64_t) getpid());
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
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }

    ::unlink(base_bin.c_str());
    return all_pass ? 0 : 1;
}
