// The IVF-PQ implementation behind BufANNIndex<T> when
// BufANNConfig::index_type == IndexType::IvfPq. bufann_api.cpp dispatches
// here; the graph paths are untouched.
//
// A base vector's tag is its row in the base file (the RID table is the
// identity at build time); an inserted vector's tag is whatever the caller
// gave bufann_insert, recorded in the tag maps below. Inserts live in
// `delta` and the heap only: they are not in the index file until the
// posting-list rebuild (PLAN), so a reload drops them.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tsl/robin_map.h"

#include "bufann/inplace_backend.h"  // TagType
#include "bufann/ivf_pq.h"
#include "bufann/ivf_pq_raw_vector_heap.h"
#include "bufann/ivf_pq_search.h"

namespace diskann {
namespace inplace {

struct BufANNConfig;

struct IVFPQBackend {
    IVFPQIndex index;
    RawVectorHeap heap;
    std::string index_prefix;
    IVFPQDelta delta;
    // Tags of inserted vectors, both ways; base vectors are the identity.
    // Grown by inserts and read by queries under delta.mtx like the delta.
    std::vector<TagType> inserted_tag;             // [id - num_base] -> tag
    tsl::robin_map<TagType, uint32_t> inserted_id;  // tag -> id
};

// Validates the IVF-PQ fields of `config` (nlist, pq_chunks > 0; dim a
// multiple of pq_chunks). Throws std::invalid_argument.
void require_ivf_pq_config(const BufANNConfig& config, const char* where);

// Runs build steps 1-6 on `data_bin`, writing the index file, the raw-vector
// heap and the PQ files under `index_prefix`, and returns the loaded backend.
template<typename T>
std::unique_ptr<IVFPQBackend> ivf_pq_backend_build(const std::string& data_bin, const std::string& index_prefix,
                                                   const BufANNConfig& config);

// Loads a backend written by ivf_pq_backend_build; config.dim must match.
template<typename T>
std::unique_ptr<IVFPQBackend> ivf_pq_backend_load(const std::string& index_prefix, const BufANNConfig& config);

// Inserts `coords` (config.dim elements) under `tag`, which must not be the
// tag of an active vector. Throws std::invalid_argument on a bad tag,
// std::runtime_error on an internal failure.
template<typename T>
void ivf_pq_backend_insert(IVFPQBackend& backend, const BufANNConfig& config, TagType tag, const T* coords);

// Top-k search. `nprobe` 0 means config.ivf_nprobe; `rerank_m` 0 means
// config.ivf_rerank_m, itself 0 meaning max(100, 10 * topK). Writes at most
// topK tags (ascending by distance) and returns how many.
template<typename T>
uint32_t ivf_pq_backend_query(IVFPQBackend& backend, const BufANNConfig& config, const T* query, uint32_t topK,
                              TagType* out_tags, uint32_t nprobe, uint32_t rerank_m, IVFPQSearchScratch& scratch);

}  // namespace inplace
}  // namespace diskann
