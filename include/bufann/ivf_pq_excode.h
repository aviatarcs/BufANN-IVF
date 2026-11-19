// 9-bit extended-RaBitQ records of the base vectors, kept on SSD in a second
// heap file in posting-list order, for a re-rank that reads far fewer pages
// than the raw vectors.
//
// Extended RaBitQ (Gao et al., SIGMOD 2025; RaBitQ-SSD's on-SSD record). For
// a vector o in partition c, let r = P(o - c) be its residual under a random
// orthogonal P shared by the index. Each coordinate is coded in B = 9 bits as
// u_i in [0, 511]: the sign of r_i and 8 bits of |r_i| at the rescaling t
// that maximizes the cosine between x = u - 255.5 and r (x_i = sign(r_i) *
// (e_i + 0.5), e_i = floor(t |r_i|)). With alpha = ||r||^2 / <x, r>, the
// estimate of <r, y> is alpha <x, y>, exact along r and off only by
// ||r|| tan(theta) <e, y> for the unit e orthogonal to r in the plane of x,
// theta the angle between x and r. For a query q:
//
//   ||o - q||^2 = ||r||^2 + ||q - c||^2 - 2 <r, P(q - c)>
//              ~= f_add + ||q - c||^2 + f_rescale * (<u, Pq> - 255.5 sum(Pq))
//
// with f_add = ||r||^2 + 2 alpha <x, Pc> and f_rescale = -2 alpha, both
// stored per vector; the error is at most 2 ||r|| ||q - c|| tan(theta).
//
// A record is [f_add f32][f_rescale f32][sign bits, ceil(dim / 8) bytes,
// LSB first][low 8 bits of u, dim bytes]: 8 + 9 dim / 8 bytes, ~3.5x fewer
// than float32 at dim 128. Slot i of the store holds the vector at posting
// position i (PostingLists::ids[i]), so a partition's records are contiguous
// and the shortlist candidates of one partition share pages. The store is a
// RawVectorHeap (page headers, owner ids, buffer pool), written once by the
// build; vectors inserted later have no record and are re-ranked exactly.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_raw_vector_heap.h"

namespace diskann {
namespace inplace {

constexpr uint32_t IVF_EXCODE_BITS = 9;  // the only width the record layout supports
constexpr uint32_t IVF_EXCODE_FACTOR_BYTES = 2 * sizeof(float);

std::string ivf_excodes_path(const std::string& index_prefix);

inline uint32_t ivf_excode_record_bytes(uint32_t dim) { return IVF_EXCODE_FACTOR_BYTES + (dim + 7) / 8 + dim; }

// A random orthogonal [dim x dim] matrix: Gram-Schmidt over Gaussian rows,
// in double precision, deterministic in `seed`.
std::vector<float> make_ivf_random_rotation(uint32_t dim, uint32_t seed);

// Encodes a rotated residual `r` = P(o - c) ([dim]) into `record`
// (ivf_excode_record_bytes(dim)); `rc` is the rotated centroid P c.
void ivf_excode_encode(const float* r, const float* rc, uint32_t dim, uint8_t* record);

// A query's side of the estimate: Pq and sum(Pq).
struct IVFExQuery {
    std::vector<float> rotated;
    float sum = 0.0f;
};
void ivf_excode_prepare_query(const IVFExCodes& ex, uint32_t dim, const float* query, IVFExQuery& out);

// Estimated squared distance from the query to the record's vector, given
// the query's exact squared distance to the vector's centroid.
float ivf_excode_estimate(const uint8_t* record, uint32_t dim, const IVFExQuery& q, float query_centroid_l2sq);

// Builds the store for `index` (its centroids, posting lists and cluster
// ids) from `data_bin` into ivf_excodes_path(index_prefix), replacing any
// file there, and sets index.excodes. Streams the base file once per
// window of posting positions whose records fit in `buffer_bytes`.
template<typename T>
void build_ivf_excode_store(const std::string& data_bin, const std::string& index_prefix, IVFPQIndex& index,
                            uint32_t page_size, uint32_t seed = 1234, size_t buffer_bytes = size_t(4) << 30);

}  // namespace inplace
}  // namespace diskann
