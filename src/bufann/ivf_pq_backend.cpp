#include "bufann/ivf_pq_backend.h"

#include <algorithm>
#include <stdexcept>

#include "ann_exception.h"
#include "bufann/bufann_api.h"
#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"

namespace diskann {
namespace inplace {

namespace {

// The IVF-PQ internals throw ANNException, which is not a std::exception;
// the public API promises standard exceptions.
template<typename Fn>
auto as_std_exception(Fn&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const diskann::ANNException& e) {
        throw std::runtime_error(e.message());
    }
}

}  // namespace

void require_ivf_pq_config(const BufANNConfig& config, const char* where) {
    auto fail = [&](const std::string& msg) { throw std::invalid_argument(std::string(where) + ": " + msg); };
    if (config.dim == 0) fail("config.dim must be non-zero");
    if (config.ivf_nlist == 0) fail("config.ivf_nlist must be non-zero for an IVF-PQ index");
    if (config.ivf_pq_chunks == 0) fail("config.ivf_pq_chunks must be non-zero for an IVF-PQ index");
    if (config.dim % config.ivf_pq_chunks != 0) {
        fail("config.dim " + std::to_string(config.dim) + " is not a multiple of config.ivf_pq_chunks " +
             std::to_string(config.ivf_pq_chunks));
    }
}

template<typename T>
std::unique_ptr<IVFPQBackend> ivf_pq_backend_build(const std::string& data_bin, const std::string& index_prefix,
                                                   const BufANNConfig& config) {
    require_ivf_pq_config(config, "bufann_build");
    return as_std_exception([&] {
    IVFPQIndex ix;
    ix.meta = train_ivf_centroids<T>(data_bin, config.ivf_nlist);
    if (ix.meta.dim != config.dim) {
        throw std::runtime_error("bufann_build: data dim " + std::to_string(ix.meta.dim) + " != config.dim " +
                                 std::to_string(config.dim));
    }
    ix.heap_layout = compute_raw_vector_heap_layout(config.page_size, config.dim * sizeof(T));
    {
        RawVectorHeap heap;
        heap.open(ivf_raw_vectors_path(index_prefix), ix.heap_layout);
        assign_ivf_clusters<T>(data_bin, ix.meta, heap, ix.assignments, ix.rid_table);
        ix.heap_pages = heap.allocated_pages();
        ix.heap_next_slot = heap.next_flat_slot();
    }
    ix.lists = build_ivf_posting_lists(ix.assignments, config.ivf_nlist);
    train_ivf_pq_pivots<T>(data_bin, index_prefix, config.ivf_pq_chunks);
    encode_ivf_pq_codes<T>(data_bin, index_prefix, config.ivf_pq_chunks);
    ix.pq = load_ivf_pq(index_prefix);
    write_ivf_pq_index(index_prefix, ix);
    return ivf_pq_backend_load<T>(index_prefix, config);
    });
}

template<typename T>
std::unique_ptr<IVFPQBackend> ivf_pq_backend_load(const std::string& index_prefix, const BufANNConfig& config) {
    if (config.dim == 0) throw std::invalid_argument("bufann_load: config.dim must be non-zero");
    return as_std_exception([&] {
    auto backend = std::make_unique<IVFPQBackend>();
    backend->index_prefix = index_prefix;
    backend->index = load_ivf_pq_index(index_prefix);
    const IVFPQIndex& ix = backend->index;
    if (ix.meta.dim != config.dim) {
        throw std::runtime_error("bufann_load: index dim " + std::to_string(ix.meta.dim) + " != config.dim " +
                                 std::to_string(config.dim));
    }
    if (ix.heap_layout.elem_size != config.dim * sizeof(T)) {
        throw std::runtime_error("bufann_load: index stores " + std::to_string(ix.heap_layout.elem_size / config.dim) +
                                 "-byte elements, not the " + std::to_string(sizeof(T)) + "-byte T requested");
    }
    backend->heap.open_existing(ivf_raw_vectors_path(index_prefix), ix.heap_layout, ix.heap_next_slot, ix.heap_pages);
    return backend;
    });
}

template<typename T>
uint32_t ivf_pq_backend_query(IVFPQBackend& backend, const BufANNConfig& config, const T* query, uint32_t topK,
                              TagType* out_tags, uint32_t nprobe, uint32_t rerank_m, IVFPQSearchScratch& scratch) {
    if (nprobe == 0) nprobe = config.ivf_nprobe;
    if (nprobe == 0) throw std::invalid_argument("bufann_query: config.ivf_nprobe (or search_L) must be non-zero");
    if (rerank_m == 0) rerank_m = config.ivf_rerank_m;
    if (rerank_m == 0) rerank_m = std::max<uint32_t>(100, 10 * topK);
    rerank_m = std::max(rerank_m, topK);

    scratch.vector.resize(config.dim);
    for (uint32_t d = 0; d < config.dim; ++d) scratch.vector[d] = float(query[d]);
    // The scratch's query buffer is reused as the search's query, so copy it
    // out first: ivf_pq_search overwrites scratch.vector during the re-rank.
    std::vector<float> q(scratch.vector.begin(), scratch.vector.end());

    IVFPQSearchResult r = as_std_exception(
        [&] { return ivf_pq_search<T>(backend.index, backend.heap, q.data(), topK, nprobe, rerank_m, scratch); });
    const uint32_t n = uint32_t(std::min<size_t>(topK, r.ids.size()));
    if (out_tags != nullptr) {
        for (uint32_t i = 0; i < n; ++i) out_tags[i] = TagType(r.ids[i]);
    }
    return n;
}

#define IVF_PQ_INSTANTIATE_BACKEND(T)                                                                            \
    template std::unique_ptr<IVFPQBackend> ivf_pq_backend_build<T>(const std::string&, const std::string&,      \
                                                                   const BufANNConfig&);                        \
    template std::unique_ptr<IVFPQBackend> ivf_pq_backend_load<T>(const std::string&, const BufANNConfig&);     \
    template uint32_t ivf_pq_backend_query<T>(IVFPQBackend&, const BufANNConfig&, const T*, uint32_t, TagType*, \
                                              uint32_t, uint32_t, IVFPQSearchScratch&);
IVF_PQ_INSTANTIATE_BACKEND(float)
IVF_PQ_INSTANTIATE_BACKEND(uint8_t)
IVF_PQ_INSTANTIATE_BACKEND(int8_t)
#undef IVF_PQ_INSTANTIATE_BACKEND

}  // namespace inplace
}  // namespace diskann
