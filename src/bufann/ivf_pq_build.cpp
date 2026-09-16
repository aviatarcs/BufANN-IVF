#include "bufann/ivf_pq_build.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "bufann/ivf_pq_require.h"
#include "cached_io.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

namespace diskann {
namespace inplace {

std::string ivf_centroids_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_centroids.bin";
}
std::string ivf_cluster_ids_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_cluster_ids.bin";
}
std::string ivf_rid_table_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_rid_table.bin";
}
std::string ivf_raw_vectors_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_raw_vectors.bin";
}
std::string ivf_posting_offsets_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_posting_offsets.bin";
}
std::string ivf_posting_ids_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_posting_ids.bin";
}
std::string ivf_pq_pivots_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_pq_pivots.bin";
}
std::string ivf_pq_codes_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_pq_codes.bin";
}

namespace {

const uint64_t BASE_FILE_READ_CACHE_BYTES = 64 * 1024 * 1024;
const uint32_t BIN_FILE_HEADER_BYTES = 2 * sizeof(uint32_t);  // npts, dim

// Copies `cols` values from each of `rows` rows between differently strided
// row-major buffers; used to add and strip the aligned_dim padding.
void copy_rows(const float* src, size_t src_stride, float* dst, size_t dst_stride,
               size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        std::copy_n(src + r * src_stride, cols, dst + r * dst_stride);
    }
}

// Points per streaming block: capped, and small enough that the
// [points x num_centers] float distance matrix stays within budget.
size_t streaming_block_size(size_t npts, size_t num_centers, size_t max_block_points) {
    size_t by_budget = IVF_ASSIGN_DIST_MATRIX_BYTES / (num_centers * sizeof(float));
    return std::max<size_t>(1, std::min({by_budget, max_block_points, npts}));
}

// Samples `data_bin` at sampling_rate (0 -> aim for target_points) into a
// float matrix; requires at least min_points rows.
template<typename T>
std::vector<float> sample_training_points(const std::string& data_bin, double sampling_rate,
                                          size_t target_points, size_t min_points,
                                          uint32_t seed, size_t& num_train) {
    size_t npts = 0, dim = 0;
    get_bin_metadata(data_bin, npts, dim);
    if (sampling_rate <= 0.0) {
        sampling_rate = double(std::min(target_points, npts)) / double(npts);
    }
    float* raw = nullptr;
    size_t sample_dim = 0;
    gen_random_slice<T>(data_bin, sampling_rate, raw, num_train, sample_dim, seed);
    std::unique_ptr<float[]> owned(raw);
    IVF_PQ_REQUIRE(num_train >= min_points,
                   "sampled " + std::to_string(num_train) + " training points, need at least " +
                       std::to_string(min_points) + "; raise sampling_rate");
    return std::vector<float>(owned.get(), owned.get() + num_train * dim);
}

void save_u32_column(const std::string& path, const uint32_t* data, size_t n) {
    diskann::save_bin<uint32_t>(path, const_cast<uint32_t*>(data), n, 1);
}

std::vector<uint32_t> load_u32_column(const std::string& path) {
    uint32_t* raw = nullptr;
    size_t n = 0, cols = 0;
    diskann::load_bin<uint32_t>(path, raw, n, cols);
    std::unique_ptr<uint32_t[]> owned(raw);
    IVF_PQ_REQUIRE(cols == 1, "expected a single-column uint32 bin file: " + path);
    return std::vector<uint32_t>(owned.get(), owned.get() + n);
}

}  // namespace

template<typename T>
IVFMetadata train_ivf_centroids(const std::string& data_bin, uint32_t nlist,
                                double sampling_rate, uint32_t max_kmeans_reps,
                                std::optional<uint32_t> seed) {
    IVF_PQ_REQUIRE(nlist > 0, "ivf_nlist must be greater than zero");
    size_t npts = 0, dim = 0;
    get_bin_metadata(data_bin, npts, dim);
    IVF_PQ_REQUIRE(npts >= nlist, "ivf_nlist exceeds the number of base vectors");

    uint32_t rng_seed = seed.has_value() ? *seed : std::random_device{}();
    size_t num_train = 0;
    std::vector<float> train_data = sample_training_points<T>(
        data_bin, sampling_rate, size_t(nlist) * IVF_TRAIN_POINTS_PER_CENTROID, nlist, rng_seed,
        num_train);
    diskann::cout << "Training " << nlist << " IVF centroids on " << num_train
                  << " sampled points (dim " << dim << ")" << std::endl;

    std::vector<float> centers(size_t(nlist) * dim);
    kmeans::kmeanspp_selecting_pivots(train_data.data(), num_train, dim, centers.data(), nlist,
                                      rng_seed);
    kmeans::run_lloyds(train_data.data(), num_train, dim, centers.data(), nlist, max_kmeans_reps,
                       NULL, NULL);

    IVFMetadata meta;
    meta.nlist = nlist;
    meta.dim = uint32_t(dim);
    meta.aligned_dim = align_dim(uint32_t(dim));
    meta.centroids.assign(size_t(nlist) * meta.aligned_dim, 0.0f);
    copy_rows(centers.data(), dim, meta.centroids.data(), meta.aligned_dim, nlist, dim);
    return meta;
}

void save_ivf_centroids(const std::string& index_prefix, const IVFMetadata& meta) {
    diskann::save_bin<float>(ivf_centroids_path(index_prefix),
                             const_cast<float*>(meta.centroids.data()), meta.nlist,
                             meta.aligned_dim);
}

IVFMetadata load_ivf_centroids(const std::string& index_prefix, uint32_t dim) {
    float* raw = nullptr;
    size_t nlist = 0, aligned_dim = 0;
    diskann::load_bin<float>(ivf_centroids_path(index_prefix), raw, nlist, aligned_dim);
    std::unique_ptr<float[]> owned(raw);
    IVF_PQ_REQUIRE(aligned_dim == align_dim(dim), "centroid file dimensionality does not match dim");

    IVFMetadata meta;
    meta.nlist = uint32_t(nlist);
    meta.dim = dim;
    meta.aligned_dim = uint32_t(aligned_dim);
    meta.centroids.assign(owned.get(), owned.get() + nlist * aligned_dim);
    return meta;
}

template<typename T>
void assign_ivf_clusters(const std::string& data_bin, const IVFMetadata& meta,
                         RawVectorHeap& heap, ClusterAssignments& assignments,
                         RawVectorRIDTable& rid_table, size_t max_block_points) {
    IVF_PQ_REQUIRE(max_block_points > 0, "max_block_points must be greater than zero");
    size_t npts = 0, file_dim = 0;
    get_bin_metadata(data_bin, npts, file_dim);
    IVF_PQ_REQUIRE(file_dim == meta.dim, "base file dimensionality does not match the centroids");
    IVF_PQ_REQUIRE(npts <= size_t(RAW_VECTOR_RID_SLOT_MASK) + 1,
                   "base file has more vectors than RawVectorRID can address");
    const size_t dim = meta.dim;
    const size_t elem_size = dim * sizeof(T);
    IVF_PQ_REQUIRE(heap.layout().elem_size == elem_size,
                   "raw-vector heap elem_size does not match dim * sizeof(T)");

    // compute_closest_centers works at `dim`, so strip the aligned_dim padding.
    std::vector<float> centroids(size_t(meta.nlist) * dim);
    copy_rows(meta.centroids.data(), meta.aligned_dim, centroids.data(), dim, meta.nlist, dim);

    size_t block_size = streaming_block_size(npts, meta.nlist, max_block_points);
    std::vector<T> block(block_size * dim);
    std::vector<float> block_float(block_size * dim);
    std::vector<uint32_t> block_closest(block_size);

    assignments.cluster_id.resize(npts);
    rid_table.rid.resize(npts);
    RawVectorHeapBulkWriter writer(heap, IVF_BULK_LOAD_PAGES_PER_FLUSH);

    diskann::cout << "Assigning " << npts << " vectors to " << meta.nlist
                  << " IVF centroids in blocks of " << block_size << std::endl;

    cached_ifstream reader(data_bin, BASE_FILE_READ_CACHE_BYTES, BIN_FILE_HEADER_BYTES);
    for (size_t start = 0; start < npts; start += block_size) {
        size_t cur = std::min(block_size, npts - start);
        reader.read(reinterpret_cast<char*>(block.data()), cur * elem_size);
        diskann::convert_types<T, float>(block.data(), block_float.data(), cur, dim);
        math_utils::compute_closest_centers(block_float.data(), cur, dim, centroids.data(),
                                            meta.nlist, 1, block_closest.data());
        std::copy_n(block_closest.data(), cur, assignments.cluster_id.begin() + start);

        for (size_t i = 0; i < cur; ++i) {
            uint32_t slot = writer.append(block.data() + i * dim);
            rid_table.rid[start + i] = make_raw_vector_rid(slot, true);
        }
    }
    writer.finish();
}

void save_ivf_cluster_assignments(const std::string& index_prefix,
                                  const ClusterAssignments& assignments) {
    save_u32_column(ivf_cluster_ids_path(index_prefix), assignments.cluster_id.data(),
                    assignments.cluster_id.size());
}

ClusterAssignments load_ivf_cluster_assignments(const std::string& index_prefix) {
    ClusterAssignments assignments;
    assignments.cluster_id = load_u32_column(ivf_cluster_ids_path(index_prefix));
    return assignments;
}

void save_ivf_rid_table(const std::string& index_prefix, const RawVectorRIDTable& rid_table) {
    static_assert(sizeof(RawVectorRID) == sizeof(uint32_t), "RID table is stored as uint32");
    save_u32_column(ivf_rid_table_path(index_prefix),
                    reinterpret_cast<const uint32_t*>(rid_table.rid.data()), rid_table.rid.size());
}

RawVectorRIDTable load_ivf_rid_table(const std::string& index_prefix) {
    RawVectorRIDTable rid_table;
    for (uint32_t packed : load_u32_column(ivf_rid_table_path(index_prefix))) {
        rid_table.rid.push_back(RawVectorRID{packed});
    }
    return rid_table;
}

PostingLists build_ivf_posting_lists(const ClusterAssignments& assignments, uint32_t nlist) {
    const std::vector<uint32_t>& cluster_id = assignments.cluster_id;
    const size_t npts = cluster_id.size();
    IVF_PQ_REQUIRE(npts <= std::numeric_limits<uint32_t>::max(),
                   "too many vectors for uint32 posting-list offsets");

    PostingLists lists;
    lists.offsets.assign(size_t(nlist) + 1, 0);
    for (size_t i = 0; i < npts; ++i) {
        IVF_PQ_REQUIRE(cluster_id[i] < nlist,
                       "vector " + std::to_string(i) + " is assigned to cluster " +
                           std::to_string(cluster_id[i]) + " but nlist is " + std::to_string(nlist));
        ++lists.offsets[cluster_id[i] + 1];
    }
    for (uint32_t c = 0; c < nlist; ++c) {
        lists.offsets[c + 1] += lists.offsets[c];
    }

    // Counting sort: walking i upward keeps each partition ascending.
    std::vector<uint32_t> next_free(lists.offsets.begin(), lists.offsets.end() - 1);
    lists.ids.resize(npts);
    for (size_t i = 0; i < npts; ++i) {
        lists.ids[next_free[cluster_id[i]]++] = uint32_t(i);
    }
    return lists;
}

void save_ivf_posting_lists(const std::string& index_prefix, const PostingLists& lists) {
    save_u32_column(ivf_posting_offsets_path(index_prefix), lists.offsets.data(),
                    lists.offsets.size());
    save_u32_column(ivf_posting_ids_path(index_prefix), lists.ids.data(), lists.ids.size());
}

PostingLists load_ivf_posting_lists(const std::string& index_prefix) {
    PostingLists lists;
    lists.offsets = load_u32_column(ivf_posting_offsets_path(index_prefix));
    lists.ids = load_u32_column(ivf_posting_ids_path(index_prefix));
    IVF_PQ_REQUIRE(!lists.offsets.empty() && lists.offsets.front() == 0 &&
                       lists.offsets.back() == lists.ids.size() &&
                       std::is_sorted(lists.offsets.begin(), lists.offsets.end()),
                   "posting-list offsets and ids sidecars are inconsistent");
    return lists;
}

template<typename T>
PQMetadata train_ivf_pq_pivots(const std::string& data_bin, uint32_t chunks, double sampling_rate,
                               uint32_t max_kmeans_reps, std::optional<uint32_t> seed) {
    IVF_PQ_REQUIRE(chunks > 0, "pq_chunks must be greater than zero");
    size_t npts = 0, dim = 0;
    get_bin_metadata(data_bin, npts, dim);
    IVF_PQ_REQUIRE(dim % chunks == 0, "dim " + std::to_string(dim) + " is not a multiple of pq_chunks " +
                                          std::to_string(chunks));
    const uint32_t k = NUM_PQ_CENTERS;
    const size_t chunk_dim = dim / chunks;

    uint32_t rng_seed = seed.has_value() ? *seed : std::random_device{}();
    size_t num_train = 0;
    std::vector<float> train_data = sample_training_points<T>(data_bin, sampling_rate,
                                                              IVF_PQ_TRAIN_POINTS, k, rng_seed,
                                                              num_train);
    diskann::cout << "Training " << chunks << " x " << k << " PQ pivots on " << num_train
                  << " sampled points (chunk_dim " << chunk_dim << ")" << std::endl;

    PQMetadata pq;
    pq.chunks = chunks;
    pq.chunk_dim = uint32_t(chunk_dim);
    pq.k = k;
    pq.pivots.resize(size_t(chunks) * k * chunk_dim);

    std::vector<float> chunk_data(num_train * chunk_dim);
    for (uint32_t c = 0; c < chunks; ++c) {
        copy_rows(train_data.data() + c * chunk_dim, dim, chunk_data.data(), chunk_dim, num_train,
                  chunk_dim);
        float* pivots = pq.pivots.data() + size_t(c) * k * chunk_dim;
        kmeans::kmeanspp_selecting_pivots(chunk_data.data(), num_train, chunk_dim, pivots, k,
                                          rng_seed + c);
        kmeans::run_lloyds(chunk_data.data(), num_train, chunk_dim, pivots, k, max_kmeans_reps,
                           NULL, NULL);
    }
    return pq;
}

template<typename T>
void encode_ivf_pq_codes(const std::string& data_bin, PQMetadata& pq, size_t max_block_points) {
    IVF_PQ_REQUIRE(max_block_points > 0, "max_block_points must be greater than zero");
    IVF_PQ_REQUIRE(pq.k > 0 && pq.k <= 256, "PQ codes are uint8, so k must be in [1, 256]");
    size_t npts = 0, dim = 0;
    get_bin_metadata(data_bin, npts, dim);
    IVF_PQ_REQUIRE(dim == size_t(pq.chunks) * pq.chunk_dim,
                   "base file dimensionality does not match chunks * chunk_dim");
    const size_t chunk_dim = pq.chunk_dim;

    size_t block_size = streaming_block_size(npts, pq.k, max_block_points);
    std::vector<T> block(block_size * dim);
    std::vector<float> block_float(block_size * dim);
    std::vector<float> chunk_data(block_size * chunk_dim);
    std::vector<uint32_t> closest(block_size);
    pq.codes.resize(npts * pq.chunks);

    diskann::cout << "Encoding " << npts << " vectors with " << pq.chunks << " PQ chunks in blocks of "
                  << block_size << std::endl;

    cached_ifstream reader(data_bin, BASE_FILE_READ_CACHE_BYTES, BIN_FILE_HEADER_BYTES);
    for (size_t start = 0; start < npts; start += block_size) {
        size_t cur = std::min(block_size, npts - start);
        reader.read(reinterpret_cast<char*>(block.data()), cur * dim * sizeof(T));
        diskann::convert_types<T, float>(block.data(), block_float.data(), cur, dim);

        for (uint32_t c = 0; c < pq.chunks; ++c) {
            copy_rows(block_float.data() + c * chunk_dim, dim, chunk_data.data(), chunk_dim, cur,
                      chunk_dim);
            math_utils::compute_closest_centers(chunk_data.data(), cur, chunk_dim,
                                                pq.pivots.data() + size_t(c) * pq.k * chunk_dim,
                                                pq.k, 1, closest.data());
            for (size_t i = 0; i < cur; ++i) {
                pq.codes[(start + i) * pq.chunks + c] = uint8_t(closest[i]);
            }
        }
    }
}

void save_ivf_pq(const std::string& index_prefix, const PQMetadata& pq) {
    diskann::save_bin<float>(ivf_pq_pivots_path(index_prefix), const_cast<float*>(pq.pivots.data()),
                             size_t(pq.chunks) * pq.k, pq.chunk_dim);
    diskann::save_bin<uint8_t>(ivf_pq_codes_path(index_prefix),
                               const_cast<uint8_t*>(pq.codes.data()), pq.codes.size() / pq.chunks,
                               pq.chunks);
}

PQMetadata load_ivf_pq(const std::string& index_prefix) {
    PQMetadata pq;
    pq.k = NUM_PQ_CENTERS;

    float* pivots = nullptr;
    size_t pivot_rows = 0, chunk_dim = 0;
    diskann::load_bin<float>(ivf_pq_pivots_path(index_prefix), pivots, pivot_rows, chunk_dim);
    std::unique_ptr<float[]> owned_pivots(pivots);
    IVF_PQ_REQUIRE(pivot_rows > 0 && pivot_rows % pq.k == 0,
                   "PQ pivot file rows are not a multiple of " + std::to_string(pq.k));
    pq.chunks = uint32_t(pivot_rows / pq.k);
    pq.chunk_dim = uint32_t(chunk_dim);
    pq.pivots.assign(owned_pivots.get(), owned_pivots.get() + pivot_rows * chunk_dim);

    uint8_t* codes = nullptr;
    size_t npts = 0, code_cols = 0;
    diskann::load_bin<uint8_t>(ivf_pq_codes_path(index_prefix), codes, npts, code_cols);
    std::unique_ptr<uint8_t[]> owned_codes(codes);
    IVF_PQ_REQUIRE(code_cols == pq.chunks, "PQ codes file does not have one column per chunk");
    pq.codes.assign(owned_codes.get(), owned_codes.get() + npts * code_cols);
    return pq;
}

template IVFMetadata train_ivf_centroids<float>(const std::string&, uint32_t, double, uint32_t,
                                                std::optional<uint32_t>);
template IVFMetadata train_ivf_centroids<uint8_t>(const std::string&, uint32_t, double, uint32_t,
                                                  std::optional<uint32_t>);
template IVFMetadata train_ivf_centroids<int8_t>(const std::string&, uint32_t, double, uint32_t,
                                                 std::optional<uint32_t>);

template void assign_ivf_clusters<float>(const std::string&, const IVFMetadata&, RawVectorHeap&,
                                         ClusterAssignments&, RawVectorRIDTable&, size_t);
template void assign_ivf_clusters<uint8_t>(const std::string&, const IVFMetadata&, RawVectorHeap&,
                                           ClusterAssignments&, RawVectorRIDTable&, size_t);
template void assign_ivf_clusters<int8_t>(const std::string&, const IVFMetadata&, RawVectorHeap&,
                                          ClusterAssignments&, RawVectorRIDTable&, size_t);

template PQMetadata train_ivf_pq_pivots<float>(const std::string&, uint32_t, double, uint32_t,
                                               std::optional<uint32_t>);
template PQMetadata train_ivf_pq_pivots<uint8_t>(const std::string&, uint32_t, double, uint32_t,
                                                 std::optional<uint32_t>);
template PQMetadata train_ivf_pq_pivots<int8_t>(const std::string&, uint32_t, double, uint32_t,
                                                std::optional<uint32_t>);

template void encode_ivf_pq_codes<float>(const std::string&, PQMetadata&, size_t);
template void encode_ivf_pq_codes<uint8_t>(const std::string&, PQMetadata&, size_t);
template void encode_ivf_pq_codes<int8_t>(const std::string&, PQMetadata&, size_t);

}  // namespace inplace
}  // namespace diskann
