// Shared by the self-checking IVF-PQ tests: the TestCase harness, exact
// distance and selection helpers, the synthetic fixtures, the oracles the
// search tests compare against, and a build-and-load of a small index.

#pragma once

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ann_exception.h"
#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_raw_vector_heap.h"
#include "bufann/ivf_pq_search.h"
#include "utils.h"

class TestCase {
public:
    explicit TestCase(const std::string& name) {
        std::cout << "[Test] " << name << "..." << std::endl;
    }

    bool check(bool cond, const std::string& msg) {
        if (!cond) {
            std::cout << "  FAIL: " << msg << std::endl;
            _pass = false;
        }
        return cond;
    }

    template<typename Fn>
    void expect_throw(const std::string& what, Fn&& fn) {
        try {
            fn();
            check(false, what + " did not throw");
        } catch (const diskann::ANNException&) {
        }
    }

    // For code paths that throw std::invalid_argument / std::runtime_error
    // rather than ANNException (the public API does).
    template<typename Fn>
    void expect_throw_any(const std::string& what, Fn&& fn) {
        try {
            fn();
            check(false, what + " did not throw");
        } catch (const std::exception&) {
        } catch (const diskann::ANNException&) {
        }
    }

    bool done() {
        std::cout << "  " << (_pass ? "PASS" : "FAIL") << std::endl;
        return _pass;
    }

private:
    bool _pass = true;
};

inline std::string temp_path(const std::string& stem) {
    return "/tmp/" + stem + "_" + std::to_string(static_cast<uint64_t>(getpid()));
}

// --- exact arithmetic ------------------------------------------------------

inline float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

inline float l2sq(const float* a, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += a[d] * a[d];
    return s;
}

// Indices of the `k` smallest values, ascending by value.
inline std::vector<uint32_t> top_k(const std::vector<float>& values, uint32_t k) {
    std::vector<uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), 0u);
    k = std::min<uint32_t>(k, uint32_t(order.size()));
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    order.resize(k);
    return order;
}

// The build is -Ofast, so a squared distance summed in another translation
// unit can differ by reassociation; well under this, far above nothing.
constexpr float DIST_REL_TOL = 1e-5f;

inline bool close(float a, float b) { return std::fabs(a - b) <= DIST_REL_TOL * std::max(std::fabs(a), std::fabs(b)); }

template<typename T>
T clamp_to(float v) {
    float lo = float(std::numeric_limits<T>::lowest()), hi = float(std::numeric_limits<T>::max());
    return T(std::round(std::min(hi, std::max(lo, v))));
}
template<>
inline float clamp_to<float>(float v) { return v; }

template<typename T>
std::vector<float> to_float(const T* v, size_t n) {
    return std::vector<float>(v, v + n);
}

// --- fixtures --------------------------------------------------------------

// `blobs` Gaussian blobs with centres drawn once (seed 99) in [40, 215] and
// noise of sd 6, so the same centres serve every draw of a fixture.
template<typename T>
std::vector<T> draw_blobs(uint32_t count, uint32_t seed, uint32_t dim, uint32_t blobs) {
    std::mt19937 gen(seed);
    std::mt19937 center_gen(99);
    std::uniform_real_distribution<float> spread(40.0f, 215.0f);
    std::vector<float> centers(size_t(blobs) * dim);
    for (float& v : centers) v = spread(center_gen);
    std::normal_distribution<float> noise(0.0f, 6.0f);
    std::vector<T> out(size_t(count) * dim);
    for (uint32_t i = 0; i < count; ++i) {
        const float* c = centers.data() + size_t(gen() % blobs) * dim;
        for (uint32_t d = 0; d < dim; ++d) out[size_t(i) * dim + d] = clamp_to<T>(c[d] + noise(gen));
    }
    return out;
}

// --- oracles ---------------------------------------------------------------

// Same distances position by position, and same id wherever the distance is
// not shared with a neighbouring position. `want` may carry one entry more
// than `got`: the runner-up, which only serves the tie test of the last
// position (uint8 data has integer distances, so ties there are common).
inline bool same_within_ties(const diskann::inplace::IVFPQSearchResult& got,
                             const diskann::inplace::IVFPQSearchResult& want) {
    if (want.ids.size() != got.ids.size() && want.ids.size() != got.ids.size() + 1) return false;
    for (size_t i = 0; i < got.ids.size(); ++i) {
        if (!close(got.dists[i], want.dists[i])) return false;
        bool tied = (i > 0 && close(want.dists[i], want.dists[i - 1])) ||
                    (i + 1 < want.ids.size() && close(want.dists[i], want.dists[i + 1]));
        if (!tied && got.ids[i] != want.ids[i]) return false;
    }
    return true;
}

// Exact top-k over the rows of `all` (row i is vector id i, `dim` floats)
// not marked in `dead`.
inline diskann::inplace::IVFPQSearchResult brute_force(const std::vector<float>& all, uint32_t dim,
                                                       const std::vector<uint8_t>& dead, const float* query,
                                                       uint32_t k) {
    const size_t n = all.size() / dim;
    std::vector<float> exact(n, std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < n; ++i) {
        if (!dead[i]) exact[i] = sq_dist(query, all.data() + size_t(i) * dim, dim);
    }
    diskann::inplace::IVFPQSearchResult want;
    for (uint32_t i : top_k(exact, k)) {
        if (dead[i]) break;
        want.ids.push_back(i), want.dists.push_back(exact[i]);
    }
    return want;
}

// --- a small index on disk -------------------------------------------------

struct Built {
    diskann::inplace::IVFPQIndex index;
    diskann::inplace::RawVectorHeap heap;
    diskann::inplace::IVFPQDelta delta;
};

inline void remove_index_files(const std::string& prefix) {
    using namespace diskann::inplace;
    for (const std::string& f : {ivf_pq_index_path(prefix), ivf_raw_vectors_path(prefix), ivf_pq_pivots_path(prefix),
                                 ivf_pq_codes_path(prefix)}) {
        ::unlink(f.c_str());
    }
}

// Runs build steps 1-6 on `base` (n x dim) under `prefix` with fixed seeds,
// then loads the index file and reopens the heap from it.
template<typename T>
std::unique_ptr<Built> build_index(const std::string& prefix, const std::vector<T>& base, uint32_t n, uint32_t dim,
                                   uint32_t nlist, uint32_t chunks, uint32_t seed) {
    using namespace diskann::inplace;
    const std::string base_bin = prefix + "_base.bin";
    diskann::save_bin<T>(base_bin, const_cast<T*>(base.data()), n, dim);
    {
        IVFPQIndex ix;
        ix.meta = train_ivf_centroids<T>(base_bin, nlist, 0.0, NUM_K_MEANS_ITERS, seed);
        ix.heap_layout = compute_raw_vector_heap_layout(4096, dim * sizeof(T));
        RawVectorHeap heap;
        heap.open(ivf_raw_vectors_path(prefix), ix.heap_layout);
        assign_ivf_clusters<T>(base_bin, ix.meta, heap, ix.assignments, ix.rid_table);
        ix.lists = build_ivf_posting_lists(ix.assignments, nlist);
        train_ivf_pq_pivots<T>(base_bin, prefix, chunks, 1.0, NUM_K_MEANS_ITERS, seed);
        encode_ivf_pq_codes<T>(base_bin, prefix, chunks);
        ix.pq = load_ivf_pq(prefix);
        ix.heap_pages = heap.allocated_pages();
        ix.heap_next_slot = heap.next_flat_slot();
        write_ivf_pq_index(prefix, ix);
    }
    auto built = std::make_unique<Built>();
    built->index = load_ivf_pq_index(prefix);
    built->heap.open_existing(ivf_raw_vectors_path(prefix), built->index.heap_layout, built->index.heap_next_slot,
                              built->index.heap_pages);
    ::unlink(base_bin.c_str());
    return built;
}
