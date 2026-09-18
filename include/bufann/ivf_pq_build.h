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

// Cluster assignment and PQ encoding stream the base file in blocks of at
// most IVF_ASSIGN_MAX_BLOCK_POINTS vectors, shrunk further so the per-block
// [points x centers] float distance matrix stays under the byte budget.
const size_t IVF_ASSIGN_MAX_BLOCK_POINTS = size_t(1) << 20;
const size_t IVF_ASSIGN_DIST_MATRIX_BYTES = size_t(256) << 20;

const uint32_t IVF_BULK_LOAD_PAGES_PER_FLUSH = 256;

// Points sampled to train the PQ pivots (all chunks train on the same sample).
const size_t IVF_PQ_TRAIN_POINTS = size_t(1) << 17;

std::string ivf_centroids_path(const std::string& index_prefix);
std::string ivf_cluster_ids_path(const std::string& index_prefix);
std::string ivf_rid_table_path(const std::string& index_prefix);
std::string ivf_raw_vectors_path(const std::string& index_prefix);
std::string ivf_posting_offsets_path(const std::string& index_prefix);
std::string ivf_posting_ids_path(const std::string& index_prefix);
std::string ivf_pq_pivots_path(const std::string& index_prefix);
std::string ivf_pq_codes_path(const std::string& index_prefix);

// Step 1. Trains `nlist` centroids on a random sample of `data_bin` (k-means++
// init, Lloyd's refinement), zero-padded to aligned_dim. sampling_rate 0
// derives the sample size from IVF_TRAIN_POINTS_PER_CENTROID. `seed` makes
// the sample and the init deterministic.
template<typename T>
IVFMetadata train_ivf_centroids(const std::string& data_bin,
                                uint32_t nlist,
                                double sampling_rate = 0.0,
                                uint32_t max_kmeans_reps = NUM_K_MEANS_ITERS,
                                std::optional<uint32_t> seed = std::nullopt);

// Fills meta.centroid_l2sq from meta.centroids. Every producer of an
// IVFMetadata calls this; search relies on the norms matching the centroids.
void set_ivf_centroid_norms(IVFMetadata& meta);

// [nlist x aligned_dim] float bin. `dim` is the unpadded dimensionality,
// which the file does not record.
void save_ivf_centroids(const std::string& index_prefix, const IVFMetadata& meta);
IVFMetadata load_ivf_centroids(const std::string& index_prefix, uint32_t dim);

// Step 2. Streams `data_bin` once: assigns every vector to its nearest
// centroid and bulk-loads the raw vectors into `heap`, which must be freshly
// opened with elem_size == meta.dim * sizeof(T). Vector i lands in flat slot
// i; the RID table is materialized anyway because inserts later break that.
template<typename T>
void assign_ivf_clusters(const std::string& data_bin, const IVFMetadata& meta,
                         RawVectorHeap& heap, ClusterAssignments& assignments,
                         RawVectorRIDTable& rid_table,
                         size_t max_block_points = IVF_ASSIGN_MAX_BLOCK_POINTS);

// [N x 1] uint32 bins.
void save_ivf_cluster_assignments(const std::string& index_prefix,
                                  const ClusterAssignments& assignments);
ClusterAssignments load_ivf_cluster_assignments(const std::string& index_prefix);
void save_ivf_rid_table(const std::string& index_prefix, const RawVectorRIDTable& rid_table);
RawVectorRIDTable load_ivf_rid_table(const std::string& index_prefix);

// Step 3. Partition c is ids[offsets[c] : offsets[c+1]], ascending by vector
// ID; clusters with no members are empty ranges. Throws on cluster_id >= nlist.
PostingLists build_ivf_posting_lists(const ClusterAssignments& assignments, uint32_t nlist);

// [nlist+1 x 1] and [N x 1] uint32 bins.
void save_ivf_posting_lists(const std::string& index_prefix, const PostingLists& lists);
PostingLists load_ivf_posting_lists(const std::string& index_prefix);

// Steps 4 and 5 wrap upstream DiskANN's PQ (generate_pq_pivots /
// generate_pq_data_from_pivots) and its files, so a codebook built by
// benchmark/scripts/build_pq.sh for the graph index is usable as-is.
// dim must be a multiple of `chunks`: PQMetadata assumes a uniform chunk_dim.

// Step 4. Trains NUM_PQ_CENTERS pivots per chunk on a random sample of
// `data_bin` and writes ivf_pq_pivots_path(index_prefix). sampling_rate 0
// targets IVF_PQ_TRAIN_POINTS; `seed` as for centroids.
template<typename T>
void train_ivf_pq_pivots(const std::string& data_bin, const std::string& index_prefix,
                         uint32_t chunks, double sampling_rate = 0.0,
                         uint32_t max_kmeans_reps = NUM_K_MEANS_ITERS,
                         std::optional<uint32_t> seed = std::nullopt);

// Step 5. Streams `data_bin` and writes ivf_pq_codes_path(index_prefix), an
// [N x chunks] uint8 bin with codes[i][c] = argmin_j ||x_i[chunk c] - pivot[c][j]||^2.
template<typename T>
void encode_ivf_pq_codes(const std::string& data_bin, const std::string& index_prefix,
                         uint32_t chunks);

// Loads both files into PQMetadata. Upstream trains and encodes on x - mean;
// the mean is folded into the returned pivots, so distances to them are
// distances to the raw vectors.
PQMetadata load_ivf_pq(const std::string& index_prefix);

}  // namespace inplace
}  // namespace diskann
