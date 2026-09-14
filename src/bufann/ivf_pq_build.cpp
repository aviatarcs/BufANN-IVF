#include "bufann/ivf_pq_build.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "ann_exception.h"
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

// [N x 1] uint32 bin round-trips shared by the sidecars below.
void save_u32_column(const std::string& path, const uint32_t* data, size_t n) {
    diskann::save_bin<uint32_t>(path, const_cast<uint32_t*>(data), n, 1);
}

std::vector<uint32_t> load_u32_column(const std::string& path) {
    uint32_t* raw = nullptr;
    size_t n = 0, cols = 0;
    diskann::load_bin<uint32_t>(path, raw, n, cols);
    std::unique_ptr<uint32_t[]> owned(raw);
    if (cols != 1) {
        throw ANNException("expected a single-column uint32 bin file: " + path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    return std::vector<uint32_t>(owned.get(), owned.get() + n);
}

}  // namespace

template<typename T>
IVFMetadata train_ivf_centroids(const std::string& data_bin, uint32_t nlist,
                                double sampling_rate, uint32_t max_kmeans_reps,
                                std::optional<uint32_t> seed) {
    if (nlist == 0) {
        throw ANNException("ivf_nlist must be greater than zero", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    size_t npts = 0, file_dim = 0;
    get_bin_metadata(data_bin, npts, file_dim);
    if (npts < nlist) {
        throw ANNException("ivf_nlist exceeds the number of base vectors", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    if (sampling_rate <= 0.0) {
        size_t target = std::min<size_t>(
            static_cast<size_t>(nlist) * IVF_TRAIN_POINTS_PER_CENTROID, npts);
        sampling_rate = static_cast<double>(target) / static_cast<double>(npts);
    }

    // One seed drives both the sample and the k-means++ init, so a fixed seed
    // gives a fully reproducible training run.
    uint32_t rng_seed = seed.has_value() ? *seed : std::random_device{}();

    float* raw_sample = nullptr;
    size_t num_train = 0, train_dim = 0;
    gen_random_slice<T>(data_bin, sampling_rate, raw_sample, num_train, train_dim, rng_seed);
    std::unique_ptr<float[]> train_data(raw_sample);

    // gen_random_slice samples independently, so the draw can come up short.
    if (num_train < nlist) {
        throw ANNException(
            "sampled " + std::to_string(num_train) +
                " training points for " + std::to_string(nlist) +
                " centroids; raise sampling_rate or lower ivf_nlist",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Training " << nlist << " IVF centroids on " << num_train
                  << " sampled points (dim " << train_dim << ")" << std::endl;

    std::unique_ptr<float[]> centers(new float[static_cast<size_t>(nlist) * train_dim]);
    kmeans::kmeanspp_selecting_pivots(train_data.get(), num_train, train_dim,
                                      centers.get(), nlist, rng_seed);
    kmeans::run_lloyds(train_data.get(), num_train, train_dim, centers.get(),
                       nlist, max_kmeans_reps, NULL, NULL);

    IVFMetadata meta;
    meta.nlist = nlist;
    meta.dim = static_cast<uint32_t>(train_dim);
    meta.aligned_dim = align_dim(static_cast<uint32_t>(train_dim));

    // Zero-padded to aligned_dim so distance math runs at the aligned width.
    meta.centroids.assign(static_cast<size_t>(nlist) * meta.aligned_dim, 0.0f);
    for (uint32_t c = 0; c < nlist; ++c) {
        std::copy(centers.get() + static_cast<size_t>(c) * train_dim,
                  centers.get() + static_cast<size_t>(c + 1) * train_dim,
                  meta.centroids.begin() + static_cast<size_t>(c) * meta.aligned_dim);
    }
    return meta;
}

void save_ivf_centroids(const std::string& index_prefix, const IVFMetadata& meta) {
    diskann::save_bin<float>(ivf_centroids_path(index_prefix),
                             const_cast<float*>(meta.centroids.data()),
                             meta.nlist, meta.aligned_dim);
}

IVFMetadata load_ivf_centroids(const std::string& index_prefix, uint32_t dim) {
    float* raw = nullptr;
    size_t nlist = 0, aligned_dim = 0;
    diskann::load_bin<float>(ivf_centroids_path(index_prefix), raw, nlist, aligned_dim);
    std::unique_ptr<float[]> owned(raw);

    if (aligned_dim != align_dim(dim)) {
        throw ANNException("centroid file dimensionality does not match dim", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    IVFMetadata meta;
    meta.nlist = static_cast<uint32_t>(nlist);
    meta.dim = dim;
    meta.aligned_dim = static_cast<uint32_t>(aligned_dim);
    meta.centroids.assign(owned.get(), owned.get() + nlist * aligned_dim);
    return meta;
}

template<typename T>
void assign_ivf_clusters(const std::string& data_bin, const IVFMetadata& meta,
                         RawVectorHeap& heap, ClusterAssignments& assignments,
                         RawVectorRIDTable& rid_table, size_t max_block_points) {
    if (max_block_points == 0) {
        throw ANNException("max_block_points must be greater than zero", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t npts = 0, file_dim = 0;
    get_bin_metadata(data_bin, npts, file_dim);
    if (file_dim != meta.dim) {
        throw ANNException("base file dimensionality does not match the centroids", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t dim = meta.dim;
    const size_t elem_size = dim * sizeof(T);
    if (heap.layout().elem_size != elem_size) {
        throw ANNException("raw-vector heap elem_size does not match dim * sizeof(T)", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (npts > static_cast<size_t>(RAW_VECTOR_RID_SLOT_MASK) + 1) {
        throw ANNException("base file has more vectors than RawVectorRID can address", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    // compute_closest_centers works at `dim`, so strip the aligned_dim padding.
    std::vector<float> centroids(static_cast<size_t>(meta.nlist) * dim);
    for (uint32_t c = 0; c < meta.nlist; ++c) {
        std::copy(meta.centroids.begin() + static_cast<size_t>(c) * meta.aligned_dim,
                  meta.centroids.begin() + static_cast<size_t>(c) * meta.aligned_dim + dim,
                  centroids.begin() + static_cast<size_t>(c) * dim);
    }

    size_t block_size = IVF_ASSIGN_DIST_MATRIX_BYTES / (static_cast<size_t>(meta.nlist) * sizeof(float));
    block_size = std::min({block_size, max_block_points, npts});
    block_size = std::max<size_t>(1, block_size);

    std::unique_ptr<T[]> block_T(new T[block_size * dim]);
    std::unique_ptr<float[]> block_float(new float[block_size * dim]);
    std::unique_ptr<uint32_t[]> block_closest(new uint32_t[block_size]);

    assignments.cluster_id.resize(npts);
    rid_table.rid.resize(npts);
    RawVectorHeapBulkWriter writer(heap, IVF_BULK_LOAD_PAGES_PER_FLUSH);

    diskann::cout << "Assigning " << npts << " vectors to " << meta.nlist
                  << " IVF centroids in blocks of " << block_size << std::endl;

    cached_ifstream reader(data_bin, 64 * 1024 * 1024);
    uint32_t hdr[2];
    reader.read(reinterpret_cast<char*>(hdr), sizeof(hdr));

    for (size_t start = 0; start < npts; start += block_size) {
        size_t cur = std::min(block_size, npts - start);
        reader.read(reinterpret_cast<char*>(block_T.get()), cur * elem_size);
        diskann::convert_types<T, float>(block_T.get(), block_float.get(), cur, dim);

        math_utils::compute_closest_centers(block_float.get(), cur, dim, centroids.data(),
                                            meta.nlist, 1, block_closest.get());
        std::copy(block_closest.get(), block_closest.get() + cur,
                  assignments.cluster_id.begin() + start);

        for (size_t i = 0; i < cur; ++i) {
            uint32_t slot = writer.append(block_T.get() + i * dim);
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
    std::vector<uint32_t> packed = load_u32_column(ivf_rid_table_path(index_prefix));
    RawVectorRIDTable rid_table;
    rid_table.rid.resize(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        rid_table.rid[i].packed = packed[i];
    }
    return rid_table;
}

PostingLists build_ivf_posting_lists(const ClusterAssignments& assignments, uint32_t nlist) {
    const size_t npts = assignments.cluster_id.size();
    if (npts > std::numeric_limits<uint32_t>::max()) {
        throw ANNException("too many vectors for uint32 posting-list offsets", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    PostingLists lists;
    lists.offsets.assign(static_cast<size_t>(nlist) + 1, 0);
    for (size_t i = 0; i < npts; ++i) {
        uint32_t c = assignments.cluster_id[i];
        if (c >= nlist) {
            throw ANNException("vector " + std::to_string(i) + " is assigned to cluster " +
                                   std::to_string(c) + " but nlist is " + std::to_string(nlist),
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        ++lists.offsets[c + 1];
    }
    for (uint32_t c = 0; c < nlist; ++c) {
        lists.offsets[c + 1] += lists.offsets[c];
    }

    // Counting-sort placement; walking i upward keeps each partition ascending.
    std::vector<uint32_t> fill(lists.offsets.begin(), lists.offsets.end() - 1);
    lists.ids.resize(npts);
    for (size_t i = 0; i < npts; ++i) {
        lists.ids[fill[assignments.cluster_id[i]]++] = static_cast<uint32_t>(i);
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
    if (lists.offsets.empty() || lists.offsets.front() != 0 ||
        lists.offsets.back() != lists.ids.size() ||
        !std::is_sorted(lists.offsets.begin(), lists.offsets.end())) {
        throw ANNException("posting-list offsets and ids sidecars are inconsistent", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
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
