#include "bufann/ivf_pq_build.h"

#include <algorithm>
#include <memory>
#include <random>

#include "ann_exception.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

namespace diskann {
namespace inplace {

std::string ivf_centroids_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_centroids.bin";
}

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

template IVFMetadata train_ivf_centroids<float>(const std::string&, uint32_t, double, uint32_t,
                                                std::optional<uint32_t>);
template IVFMetadata train_ivf_centroids<uint8_t>(const std::string&, uint32_t, double, uint32_t,
                                                  std::optional<uint32_t>);
template IVFMetadata train_ivf_centroids<int8_t>(const std::string&, uint32_t, double, uint32_t,
                                                 std::optional<uint32_t>);

}  // namespace inplace
}  // namespace diskann
