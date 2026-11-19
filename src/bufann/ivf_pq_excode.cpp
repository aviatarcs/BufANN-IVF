#include "bufann/ivf_pq_excode.h"

#include <mkl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <random>

#include "bufann/ivf_pq_require.h"
#include "cached_io.h"
#include "utils.h"

namespace diskann {
namespace inplace {

std::string ivf_excodes_path(const std::string& index_prefix) { return index_prefix + "_ivf_excodes.bin"; }

namespace {

constexpr uint32_t EX_BITS = IVF_EXCODE_BITS - 1;  // magnitude bits; the ninth is the sign
constexpr int EX_MAX = (1 << EX_BITS) - 1;
constexpr float EX_CENTER = float((1 << IVF_EXCODE_BITS) - 1) / 2.0f;  // u - 255.5 = x

// The t maximizing the cosine between (floor(t o_i) + 0.5)_i and o, for
// o = |r| / ||r|| (all o_i >= 0), codes capped at EX_MAX: every t where
// some code steps up is visited in order from a start below the optimum,
// keeping the running numerator <x, o> and ||x||^2. RaBitQ-SSD's
// best_rescale_factor, whose start (0.81 of the cap) is its kTightStart
// for 8 extra bits.
double best_rescale_factor(const double* o, uint32_t dim) {
    constexpr double eps = 1e-5;
    constexpr int n_enum = 10;
    const double max_o = *std::max_element(o, o + dim);
    const double t_end = double(EX_MAX + n_enum) / max_o;
    const double t_start = t_end * 0.81;
    std::vector<int> code(dim);
    double sq = 0.25 * dim, num = 0.0;
    using Next = std::pair<double, uint32_t>;
    std::priority_queue<Next, std::vector<Next>, std::greater<>> next;
    for (uint32_t i = 0; i < dim; ++i) {
        code[i] = int(t_start * o[i] + eps);
        sq += double(code[i]) * code[i] + code[i];
        num += (code[i] + 0.5) * o[i];
        if (o[i] > 0.0 && code[i] < EX_MAX) next.emplace(double(code[i] + 1) / o[i], i);
    }
    double best_ip = num / std::sqrt(sq), best_t = t_start;
    while (!next.empty()) {
        const auto [t, i] = next.top();
        next.pop();
        const int c = ++code[i];
        sq += 2.0 * c;
        num += o[i];
        const double ip = num / std::sqrt(sq);
        if (ip > best_ip) best_ip = ip, best_t = t;
        if (c < EX_MAX) {
            const double t_next = double(c + 1) / o[i];
            if (t_next < t_end) next.emplace(t_next, i);
        }
    }
    return best_t;
}

}  // namespace

std::vector<float> make_ivf_random_rotation(uint32_t dim, uint32_t seed) {
    IVF_PQ_REQUIRE(dim > 0, "rotation needs dim > 0");
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss;
    std::vector<double> q(size_t(dim) * dim);
    for (uint32_t i = 0; i < dim; ++i) {
        double* v = q.data() + size_t(i) * dim;
        for (;;) {
            for (uint32_t d = 0; d < dim; ++d) v[d] = gauss(rng);
            // Two passes of Gram-Schmidt keep the rows orthogonal to double
            // precision even at dim ~1000.
            for (int pass = 0; pass < 2; ++pass) {
                for (uint32_t j = 0; j < i; ++j) {
                    const double* u = q.data() + size_t(j) * dim;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < dim; ++d) dot += v[d] * u[d];
                    for (uint32_t d = 0; d < dim; ++d) v[d] -= dot * u[d];
                }
            }
            double norm = 0.0;
            for (uint32_t d = 0; d < dim; ++d) norm += v[d] * v[d];
            norm = std::sqrt(norm);
            if (norm > 1e-6) {
                for (uint32_t d = 0; d < dim; ++d) v[d] /= norm;
                break;
            }
        }
    }
    return std::vector<float>(q.begin(), q.end());
}

void ivf_excode_encode(const float* r, const float* rc, uint32_t dim, uint8_t* record) {
    uint8_t* signs = record + IVF_EXCODE_FACTOR_BYTES;
    uint8_t* low = signs + (dim + 7) / 8;
    std::memset(signs, 0, (dim + 7) / 8);
    double norm2 = 0.0;
    for (uint32_t i = 0; i < dim; ++i) norm2 += double(r[i]) * r[i];
    float f_add = 0.0f, f_rescale = 0.0f;
    if (norm2 > 0.0) {
        const double norm = std::sqrt(norm2);
        std::vector<double> o(dim);
        for (uint32_t i = 0; i < dim; ++i) o[i] = std::fabs(double(r[i])) / norm;
        const double t = best_rescale_factor(o.data(), dim);
        double x_r = 0.0, x_rc = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            const int e = std::min(EX_MAX, int(t * o[i] + 1e-5));
            const bool pos = r[i] >= 0.0f;
            if (pos) signs[i / 8] |= uint8_t(1u << (i % 8));
            low[i] = uint8_t(pos ? e : EX_MAX - e);
            const double x = pos ? e + 0.5 : -(e + 0.5);
            x_r += x * r[i];
            x_rc += x * rc[i];
        }
        const double alpha = norm2 / x_r;  // x_r > 0: every x_i has r_i's sign
        f_add = float(norm2 + 2.0 * alpha * x_rc);
        f_rescale = float(-2.0 * alpha);
    } else {
        // o is its centroid: the estimate is ||q - c||^2 whatever the code.
        std::memset(low, 0, dim);
    }
    std::memcpy(record, &f_add, sizeof(float));
    std::memcpy(record + sizeof(float), &f_rescale, sizeof(float));
}

void ivf_excode_prepare_query(const IVFExCodes& ex, uint32_t dim, const float* query, IVFExQuery& out) {
    IVF_PQ_REQUIRE(ex.bits == IVF_EXCODE_BITS && ex.rotation.size() == size_t(dim) * dim,
                   "the index has no 9-bit ex-code store");
    out.rotated.resize(dim);
    // Called per query from the search threads: no MKL fan-out.
    const int mkl_threads = mkl_set_num_threads_local(1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<MKL_INT>(dim), static_cast<MKL_INT>(dim), 1.0f, ex.rotation.data(), static_cast<MKL_INT>(dim),
                query, 1, 0.0f, out.rotated.data(), 1);
    mkl_set_num_threads_local(mkl_threads);
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) s += out.rotated[i];
    out.sum = s;
}

float ivf_excode_estimate(const uint8_t* record, uint32_t dim, const IVFExQuery& q, float query_centroid_l2sq) {
    float f_add, f_rescale;
    std::memcpy(&f_add, record, sizeof(float));
    std::memcpy(&f_rescale, record + sizeof(float), sizeof(float));
    const uint8_t* signs = record + IVF_EXCODE_FACTOR_BYTES;
    const uint8_t* low = signs + (dim + 7) / 8;
    const float* y = q.rotated.data();
    float hi = 0.0f, lo = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        hi += ((signs[i / 8] >> (i % 8)) & 1u) ? y[i] : 0.0f;
        lo += float(low[i]) * y[i];
    }
    const float ip = 256.0f * hi + lo - EX_CENTER * q.sum;  // <u - 255.5, Pq>
    return f_add + query_centroid_l2sq + f_rescale * ip;
}

template<typename T>
void build_ivf_excode_store(const std::string& data_bin, const std::string& index_prefix, IVFPQIndex& index,
                            uint32_t page_size, uint32_t seed, size_t buffer_bytes) {
    IVF_PQ_REQUIRE(file_exists(data_bin), "file not found: " + data_bin);
    const IVFMetadata& meta = index.meta;
    const PostingLists& lists = index.lists;
    const uint32_t dim = meta.dim;
    const size_t n = ivf_pq_num_base(index);
    size_t npts = 0, file_dim = 0;
    get_bin_metadata(data_bin, npts, file_dim);
    IVF_PQ_REQUIRE(file_dim == dim && npts == n && lists.ids.size() == n && index.assignments.cluster_id.size() >= n,
                   "base file does not match the index's dim and posting lists");
    const uint32_t rec = ivf_excode_record_bytes(dim);
    IVF_PQ_REQUIRE(buffer_bytes >= rec, "ex-code build buffer is smaller than one record");

    IVFExCodes ex;
    ex.bits = IVF_EXCODE_BITS;
    ex.rotation = make_ivf_random_rotation(dim, seed);
    ex.layout = compute_raw_vector_heap_layout(page_size, rec);

    // Rotated centroids Pc, [nlist x dim].
    std::vector<float> rc(size_t(meta.nlist) * dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(meta.nlist), static_cast<MKL_INT>(dim), static_cast<MKL_INT>(dim), 1.0f,
                meta.centroids.data(), static_cast<MKL_INT>(meta.aligned_dim), ex.rotation.data(), static_cast<MKL_INT>(dim), 0.0f, rc.data(),
                static_cast<MKL_INT>(dim));
    std::vector<uint32_t> pos_of(n);
    for (size_t p = 0; p < n; ++p) pos_of[lists.ids[p]] = uint32_t(p);

    // A store is only ever written whole; the heap refuses to open over one.
    const std::string path = ivf_excodes_path(index_prefix);
    IVF_PQ_REQUIRE(!file_exists(path) || ::unlink(path.c_str()) == 0, "cannot replace " + path);
    RawVectorHeap heap;
    heap.open(path, ex.layout);
    RawVectorHeapBulkWriter writer(heap, 256);
    const size_t window = std::max<size_t>(1, buffer_bytes / rec);
    const size_t block_rows = size_t(1) << 16;
    std::vector<uint8_t> records;
    std::vector<T> block(block_rows * dim);
    std::vector<float> resid(block_rows * dim), rotated(block_rows * dim);
    for (size_t a = 0; a < n; a += window) {
        const size_t b = std::min(n, a + window);
        records.assign((b - a) * rec, 0);
        cached_ifstream reader(data_bin, size_t(64) << 20, 2 * sizeof(uint32_t));
        for (size_t start = 0; start < n; start += block_rows) {
            const size_t cur = std::min(block_rows, n - start);
            reader.read(reinterpret_cast<char*>(block.data()), cur * dim * sizeof(T));
            // Every block is rotated whole, whatever the window, so the
            // GEMM rounds each row the same and the store does not depend on
            // buffer_bytes; a window usually spans every position anyway.
            size_t in_window = 0;
            for (size_t i = 0; i < cur; ++i) {
                const uint32_t p = pos_of[start + i];
                in_window += p >= a && p < b;
                const float* c = meta.centroids.data() + size_t(index.assignments.cluster_id[start + i]) * meta.aligned_dim;
                for (uint32_t d = 0; d < dim; ++d) resid[i * dim + d] = float(block[i * dim + d]) - c[d];
            }
            if (in_window == 0) continue;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(cur), static_cast<MKL_INT>(dim),
                        static_cast<MKL_INT>(dim), 1.0f, resid.data(), static_cast<MKL_INT>(dim), ex.rotation.data(),
                        static_cast<MKL_INT>(dim), 0.0f, rotated.data(), static_cast<MKL_INT>(dim));
#pragma omp parallel for schedule(dynamic, 256)
            for (int64_t i = 0; i < int64_t(cur); ++i) {
                const uint32_t p = pos_of[start + size_t(i)];
                if (p < a || p >= b) continue;
                const uint32_t list = index.assignments.cluster_id[start + size_t(i)];
                ivf_excode_encode(rotated.data() + size_t(i) * dim, rc.data() + size_t(list) * dim, dim,
                                  records.data() + (p - a) * rec);
            }
        }
        for (size_t p = a; p < b; ++p) writer.append(lists.ids[p], records.data() + (p - a) * rec);
    }
    writer.finish();
    ex.pages = heap.allocated_pages();
    ex.next_slot = heap.next_flat_slot();
    heap.close();
    index.excodes = std::move(ex);
}

template void build_ivf_excode_store<float>(const std::string&, const std::string&, IVFPQIndex&, uint32_t, uint32_t,
                                            size_t);
template void build_ivf_excode_store<uint8_t>(const std::string&, const std::string&, IVFPQIndex&, uint32_t,
                                              uint32_t, size_t);
template void build_ivf_excode_store<int8_t>(const std::string&, const std::string&, IVFPQIndex&, uint32_t,
                                             uint32_t, size_t);

}  // namespace inplace
}  // namespace diskann
