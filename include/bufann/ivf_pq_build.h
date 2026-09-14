// IVF-PQ build steps. Each persists a sidecar; a later step stitches them into
// the combined index file (IVFPQIndexFileHeader).

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_raw_vector_heap.h"
#include "defaults.h"

namespace diskann {
namespace inplace {

// Training points sampled per centroid; ~40 is the practical floor for k-means.
const uint32_t IVF_TRAIN_POINTS_PER_CENTROID = 64;

// Cluster assignment streams the base file in blocks. A block is capped at
// IVF_ASSIGN_MAX_BLOCK_POINTS vectors and shrinks further so the per-block
// [points x nlist] float distance matrix stays under the byte budget.
const size_t IVF_ASSIGN_MAX_BLOCK_POINTS = size_t(1) << 20;
const size_t IVF_ASSIGN_DIST_MATRIX_BYTES = size_t(256) << 20;

// Pages the bulk loader buffers before each write to the raw-vector heap.
const uint32_t IVF_BULK_LOAD_PAGES_PER_FLUSH = 256;

std::string ivf_centroids_path(const std::string& index_prefix);
std::string ivf_cluster_ids_path(const std::string& index_prefix);
std::string ivf_rid_table_path(const std::string& index_prefix);
std::string ivf_raw_vectors_path(const std::string& index_prefix);

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

// Streams `data_bin` once: assigns every vector to its nearest centroid and
// bulk-loads the raw vectors into `heap`, which must be freshly opened with
// elem_size == meta.dim * sizeof(T). Vector i lands in flat slot i, so the
// RID table is the identity; it is still materialized because inserts later
// break that property. `max_block_points` bounds the streaming block.
template<typename T>
void assign_ivf_clusters(const std::string& data_bin, const IVFMetadata& meta,
                         RawVectorHeap& heap, ClusterAssignments& assignments,
                         RawVectorRIDTable& rid_table,
                         size_t max_block_points = IVF_ASSIGN_MAX_BLOCK_POINTS);

// Each is an [N x 1] uint32 bin file.
void save_ivf_cluster_assignments(const std::string& index_prefix,
                                  const ClusterAssignments& assignments);
ClusterAssignments load_ivf_cluster_assignments(const std::string& index_prefix);
void save_ivf_rid_table(const std::string& index_prefix, const RawVectorRIDTable& rid_table);
RawVectorRIDTable load_ivf_rid_table(const std::string& index_prefix);

}  // namespace inplace
}  // namespace diskann
