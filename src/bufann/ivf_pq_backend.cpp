#include "bufann/ivf_pq_backend.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

#include "ann_exception.h"
#include "bufann/bufann_api.h"
#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_index_file.h"
#include "bufann/ivf_pq_mutate.h"

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
    ivf_pq_recover_deletes(backend->index, backend->heap, backend->delta);
    return backend;
    });
}

namespace {

// Marks a tag in inserted_id whose insert is between reservation and publish.
constexpr uint32_t IVF_PQ_PENDING_ID = 0xFFFFFFFFu;

// Whether `tag` names an active vector or one being inserted. Caller holds
// delta.mtx.
bool tag_is_taken(const IVFPQBackend& backend, TagType tag) {
    auto inserted = backend.inserted_id.find(tag);
    if (inserted != backend.inserted_id.end()) {
        return inserted->second == IVF_PQ_PENDING_ID ||
               rid_is_active(load_rid(backend.index.rid_table.rid[inserted->second]));
    }
    return tag < ivf_pq_num_base(backend.index) && rid_is_active(load_rid(backend.index.rid_table.rid[tag]));
}

}  // namespace

// The tag is reserved before the heap work so two concurrent inserts of one
// tag cannot both get in, and the vector and its tag are published in one
// critical section so a search never returns an id without a tag.
template<typename T>
void ivf_pq_backend_insert(IVFPQBackend& backend, const BufANNConfig& config, TagType tag, const T* coords) {
    if (tag == INVALID_TAG) throw std::invalid_argument("bufann_insert: INVALID_TAG is reserved");
    if (coords == nullptr) throw std::invalid_argument("bufann_insert: coords is null");
    (void)config;  // dim is checked against the index by the prepare step
    {
        std::unique_lock<std::shared_mutex> lock(backend.delta.mtx);
        if (tag_is_taken(backend, tag)) {
            throw std::invalid_argument("bufann_insert: tag " + std::to_string(tag) + " is already active");
        }
        backend.inserted_id[tag] = IVF_PQ_PENDING_ID;
    }
    auto release_tag = [&] {
        std::unique_lock<std::shared_mutex> lock(backend.delta.mtx);
        backend.inserted_id.erase(tag);
    };
    IVFPQPreparedInsert prepared;
    try {
        prepared = as_std_exception(
            [&] { return ivf_pq_prepare_insert<T>(backend.index, backend.heap, backend.delta, coords); });
    } catch (...) {
        release_tag();
        throw;
    }
    const uint32_t slot = prepared.slot;
    std::unique_lock<std::shared_mutex> lock(backend.delta.mtx);
    uint32_t id;
    try {
        id = as_std_exception([&] { return ivf_pq_publish_insert(backend.index, backend.delta, std::move(prepared)); });
    } catch (...) {
        backend.inserted_id.erase(tag);
        lock.unlock();
        backend.heap.free_slot(slot, backend.delta.free_list);  // never published, so no reader can hold it
        throw;
    }
    const uint32_t num_base = ivf_pq_num_base(backend.index);
    backend.inserted_tag.resize(size_t(id) - num_base + 1, INVALID_TAG);
    backend.inserted_tag[size_t(id) - num_base] = tag;
    backend.inserted_id[tag] = id;
}

// The tag is resolved and the vector retracted in one critical section, so a
// concurrent insert of the same tag sees it free only once the delete is
// visible to searches; the slot is freed after the lock is released, as
// free_slot does heap I/O.
void ivf_pq_backend_delete(IVFPQBackend& backend, TagType tag) {
    auto not_active = [&] { return std::invalid_argument("bufann_delete: tag " + std::to_string(tag) + " is not active"); };
    uint32_t slot;
    {
        std::unique_lock<std::shared_mutex> lock(backend.delta.mtx);
        const uint32_t num_base = ivf_pq_num_base(backend.index);
        auto inserted = backend.inserted_id.find(tag);
        uint32_t id;
        if (inserted != backend.inserted_id.end()) {
            if (inserted->second == IVF_PQ_PENDING_ID) throw not_active();
            id = inserted->second;
        } else if (tag < num_base) {
            id = tag;
        } else {
            throw not_active();
        }
        if (!rid_is_active(load_rid(backend.index.rid_table.rid[id]))) throw not_active();
        slot = as_std_exception([&] { return ivf_pq_retract_delete(backend.index, backend.delta, id); });
        if (inserted != backend.inserted_id.end()) {
            backend.inserted_tag[id - num_base] = INVALID_TAG;
            backend.inserted_id.erase(inserted);
        }
    }
    as_std_exception([&] { backend.heap.free_slot(slot, backend.delta.free_list); });
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

    IVFPQSearchResult r = as_std_exception([&] {
        return ivf_pq_search<T>(backend.index, backend.heap, q.data(), topK, nprobe, rerank_m, scratch, &backend.delta);
    });
    const uint32_t n = uint32_t(std::min<size_t>(topK, r.ids.size()));
    if (out_tags != nullptr) {
        const uint32_t num_base = ivf_pq_num_base(backend.index);
        std::shared_lock<std::shared_mutex> lock(backend.delta.mtx);
        for (uint32_t i = 0; i < n; ++i) {
            out_tags[i] = r.ids[i] < num_base ? TagType(r.ids[i]) : backend.inserted_tag[r.ids[i] - num_base];
        }
    }
    return n;
}

#define IVF_PQ_INSTANTIATE_BACKEND(T)                                                                            \
    template std::unique_ptr<IVFPQBackend> ivf_pq_backend_build<T>(const std::string&, const std::string&,      \
                                                                   const BufANNConfig&);                        \
    template std::unique_ptr<IVFPQBackend> ivf_pq_backend_load<T>(const std::string&, const BufANNConfig&);     \
    template void ivf_pq_backend_insert<T>(IVFPQBackend&, const BufANNConfig&, TagType, const T*);              \
    template uint32_t ivf_pq_backend_query<T>(IVFPQBackend&, const BufANNConfig&, const T*, uint32_t, TagType*, \
                                              uint32_t, uint32_t, IVFPQSearchScratch&);
IVF_PQ_INSTANTIATE_BACKEND(float)
IVF_PQ_INSTANTIATE_BACKEND(uint8_t)
IVF_PQ_INSTANTIATE_BACKEND(int8_t)
#undef IVF_PQ_INSTANTIATE_BACKEND

}  // namespace inplace
}  // namespace diskann
