// Build-time steps for the IVF-PQ index. Each step persists its own sidecar
// artifact; a later step stitches them into the combined index file described
// by IVFPQIndexFileHeader (see ivf_pq.h).

#pragma once

#include <cstdint>
#include <string>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

// Base vectors sampled per centroid when training the coarse quantizer.
// Below roughly 40 points per centroid k-means starts producing badly
// under-determined centers, so this leaves some headroom above that.
const uint32_t IVF_TRAIN_POINTS_PER_CENTROID = 64;

// Lloyd's iterations used to refine the sampled centroids.
const uint32_t IVF_MAX_KMEANS_REPS = 15;

std::string ivf_centroids_path(const std::string& index_prefix);

// Trains `nlist` coarse-quantizer centroids from the base vectors in
// `data_bin` and returns them padded to the aligned dimension.
//
// Draws a random sample of the base vectors into memory, seeds centers with
// k-means++, then refines with Lloyd's. The design doc also sketches a
// streaming mini-batch alternative for when a large enough sample no longer
// fits in memory; sampling is what the rest of this codebase already does at
// billion scale, so that is where this starts.
//
// sampling_rate - fraction of base vectors to train on. Pass 0 to derive it
//                 from nlist via IVF_TRAIN_POINTS_PER_CENTROID.
template<typename T>
IVFMetadata train_ivf_centroids(const std::string& data_bin,
                                uint32_t nlist,
                                double sampling_rate = 0.0,
                                uint32_t max_kmeans_reps = IVF_MAX_KMEANS_REPS);

// Persists centroids to ivf_centroids_path(index_prefix) as an
// [nlist x aligned_dim] float bin file.
void save_ivf_centroids(const std::string& index_prefix, const IVFMetadata& meta);

// Loads centroids previously written by save_ivf_centroids. `dim` is the
// unpadded vector dimensionality, which the bin file itself does not record.
IVFMetadata load_ivf_centroids(const std::string& index_prefix, uint32_t dim);

}  // namespace inplace
}  // namespace diskann
