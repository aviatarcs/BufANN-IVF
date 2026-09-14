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

    if (sampling_rate <= 0.0) {
        size_t target = std::min<size_t>(size_t(nlist) * IVF_TRAIN_POINTS_PER_CENTROID, npts);
        sampling_rate = double(target) / double(npts);
    }
    uint32_t rng_seed = seed.has_value() ? *seed : std::random_device{}();

    float* raw_sample = nullptr;
    size_t num_train = 0, sample_dim = 0;
    gen_random_slice<T>(data_bin, sampling_rate, raw_sample, num_train, sample_dim, rng_seed);
    std::unique_ptr<float[]> train_data(raw_sample);
    IVF_PQ_REQUIRE(num_train >= nlist,
                   "sampled " + std::to_string(num_train) + " training points for " +
                       std::to_string(nlist) + " centroids; raise sampling_rate or lower ivf_nlist");

    diskann::cout << "Training " << nlist << " IVF centroids on " << num_train
                  << " sampled points (dim " << dim << ")" << std::endl;

    std::vector<float> centers(size_t(nlist) * dim);
    kmeans::kmeanspp_selecting_pivots(train_data.get(), num_train, dim, centers.data(), nlist,
                                      rng_seed);
    kmeans::run_lloyds(train_data.get(), num_train, dim, centers.data(), nlist, max_kmeans_reps,
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

    size_t block_size = IVF_ASSIGN_DIST_MATRIX_BYTES / (size_t(meta.nlist) * sizeof(float));
    block_size = std::max<size_t>(1, std::min({block_size, max_block_points, npts}));
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

}  // namespace inplace
}  // namespace diskann
