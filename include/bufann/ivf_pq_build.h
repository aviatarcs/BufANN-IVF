// IVF-PQ build steps. Each persists a sidecar; a later step stitches them into
// the combined index file (IVFPQIndexFileHeader).

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "bufann/ivf_pq.h"
#include "defaults.h"

namespace diskann {
namespace inplace {

// Training points sampled per centroid; ~40 is the practical floor for k-means.
const uint32_t IVF_TRAIN_POINTS_PER_CENTROID = 64;

std::string ivf_centroids_path(const std::string& index_prefix);

// Trains `nlist` centroids on a random sample of `data_bin` (k-means++ seed,
// Lloyd's refinement) and returns them zero-padded to aligned_dim.
// sampling_rate 0 derives the sample size from IVF_TRAIN_POINTS_PER_CENTROID.
// A `seed` makes sampling and initialization deterministic.
template<typename T>
IVFMetadata train_ivf_centroids(const std::string& data_bin,
                                uint32_t nlist,
                                double sampling_rate = 0.0,
                                uint32_t max_kmeans_reps = NUM_K_MEANS_ITERS,
                                std::optional<uint32_t> seed = std::nullopt);

// Writes centroids as an [nlist x aligned_dim] float bin file.
void save_ivf_centroids(const std::string& index_prefix, const IVFMetadata& meta);

// `dim` is the unpadded dimensionality, which the bin file does not record.
IVFMetadata load_ivf_centroids(const std::string& index_prefix, uint32_t dim);

}  // namespace inplace
}  // namespace diskann
