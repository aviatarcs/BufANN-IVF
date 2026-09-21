#include "bufann/ivf_pq_mutate.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

namespace {

float sq_dist(const float* a, const float* b, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

}  // namespace

uint32_t ivf_nearest_centroid(const IVFMetadata& meta, const float* x) {
    IVF_PQ_REQUIRE(meta.nlist > 0 && meta.centroids.size() == size_t(meta.nlist) * meta.aligned_dim,
                   "IVFMetadata is inconsistent");
    uint32_t best = 0;
    float best_dist = sq_dist(x, meta.centroids.data(), meta.dim);
    for (uint32_t c = 1; c < meta.nlist; ++c) {
        const float d = sq_dist(x, meta.centroids.data() + size_t(c) * meta.aligned_dim, meta.dim);
        if (d < best_dist) best = c, best_dist = d;
    }
    return best;
}

void ivf_pq_encode(const PQMetadata& pq, const float* x, uint8_t* code) {
    IVF_PQ_REQUIRE(pq.chunks > 0 && pq.k > 0 && pq.k <= 256 &&
                       pq.pivots.size() == size_t(pq.chunks) * pq.k * pq.chunk_dim,
                   "PQMetadata is inconsistent");
    for (uint32_t c = 0; c < pq.chunks; ++c) {
        const float* sub = x + size_t(c) * pq.chunk_dim;
        const float* pivots = pq.pivots.data() + size_t(c) * pq.k * pq.chunk_dim;
        uint32_t best = 0;
        float best_dist = sq_dist(sub, pivots, pq.chunk_dim);
        for (uint32_t j = 1; j < pq.k; ++j) {
            const float d = sq_dist(sub, pivots + size_t(j) * pq.chunk_dim, pq.chunk_dim);
            if (d < best_dist) best = j, best_dist = d;
        }
        code[c] = uint8_t(best);
    }
}

template<typename T>
IVFPQPreparedInsert ivf_pq_prepare_insert(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, const T* vec) {
    IVF_PQ_REQUIRE(vec != nullptr, "vector is null");
    const uint32_t dim = ix.meta.dim;
    IVF_PQ_REQUIRE(heap.layout().elem_size == dim * sizeof(T),
                   "raw-vector heap elem_size does not match dim * sizeof(T)");
    IVF_PQ_REQUIRE(ix.pq.chunks * ix.pq.chunk_dim == dim, "PQMetadata does not cover dim");

    std::vector<float> x(dim);
    for (uint32_t d = 0; d < dim; ++d) x[d] = float(vec[d]);
    IVFPQPreparedInsert p;
    p.cluster = ivf_nearest_centroid(ix.meta, x.data());
    p.code.resize(ix.pq.chunks);
    ivf_pq_encode(ix.pq, x.data(), p.code.data());

    p.slot = heap.allocate_slot(delta.free_list);
    try {
        {
            std::unique_lock<WriterPreferringSharedMutex> lock(delta.mtx);
            IVF_PQ_REQUIRE(ix.rid_table.rid.size() == ix.assignments.cluster_id.size(),
                           "RID table and cluster assignments disagree on the vector count");
            IVF_PQ_REQUIRE(ix.rid_table.rid.size() < 0xFFFFFFFFu, "no unused vector id is left");
            p.id = uint32_t(ix.rid_table.rid.size());
            ix.assignments.cluster_id.push_back(p.cluster);
            ix.rid_table.rid.push_back(make_raw_vector_rid(p.slot, false));
        }
        heap.write_vector(p.slot, p.id, vec);
    } catch (...) {
        heap.free_slot(p.slot, delta.free_list);  // never published, so no reader can hold it
        throw;
    }
    return p;
}

uint32_t ivf_pq_publish_insert(IVFPQIndex& ix, IVFPQDelta& delta, IVFPQPreparedInsert&& p) {
    IVF_PQ_REQUIRE(p.code.size() == ix.pq.chunks && p.cluster < ix.meta.nlist && p.id < ix.rid_table.rid.size() &&
                       p.id < ix.assignments.cluster_id.size() && ix.assignments.cluster_id[p.id] == p.cluster &&
                       load_rid(ix.rid_table.rid[p.id]).packed == make_raw_vector_rid(p.slot, false).packed,
                   "prepared insert does not fit the index, or was not prepared by ivf_pq_prepare_insert");
    delta.codes.codes.emplace(p.id, std::move(p.code));
    delta.lists.pending_inserts[p.cluster].push_back(p.id);
    store_rid(ix.rid_table.rid[p.id], make_raw_vector_rid(p.slot, true));
    return p.id;
}

template<typename T>
uint32_t ivf_pq_insert(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, const T* vec) {
    IVFPQPreparedInsert prepared = ivf_pq_prepare_insert<T>(ix, heap, delta, vec);
    std::unique_lock<WriterPreferringSharedMutex> lock(delta.mtx);
    return ivf_pq_publish_insert(ix, delta, std::move(prepared));
}

uint32_t ivf_pq_retract_delete(IVFPQIndex& ix, IVFPQDelta& delta, uint32_t id) {
    IVF_PQ_REQUIRE(id < ix.rid_table.rid.size(), "vector " + std::to_string(id) + " does not exist");
    const RawVectorRID rid = load_rid(ix.rid_table.rid[id]);
    IVF_PQ_REQUIRE(rid_is_active(rid), "vector " + std::to_string(id) + " is not active");
    // Cleared first: from here on no search picks the slot up, and the
    // heap's grace period covers any that already did.
    store_rid(ix.rid_table.rid[id], make_raw_vector_rid(rid_flat_slot(rid), false));

    if (id < ivf_pq_num_base(ix)) {
        delta.lists.tombstones.insert(id);
    } else {
        const std::string unlisted = "inserted vector " + std::to_string(id) + " is not in its delta list";
        auto listed = delta.lists.pending_inserts.find(ix.assignments.cluster_id[id]);
        IVF_PQ_REQUIRE(listed != delta.lists.pending_inserts.end(), unlisted);
        auto it = std::find(listed->second.begin(), listed->second.end(), id);
        IVF_PQ_REQUIRE(it != listed->second.end(), unlisted);
        listed->second.erase(it);
        delta.codes.codes.erase(id);
    }
    return rid_flat_slot(rid);
}

void ivf_pq_delete(IVFPQIndex& ix, RawVectorHeap& heap, IVFPQDelta& delta, uint32_t id) {
    uint32_t slot;
    {
        std::unique_lock<WriterPreferringSharedMutex> lock(delta.mtx);
        slot = ivf_pq_retract_delete(ix, delta, id);
    }
    heap.free_slot(slot, delta.free_list);
}

size_t ivf_pq_recover_deletes(IVFPQIndex& ix, const RawVectorHeap& heap, IVFPQDelta& delta) {
    IVF_PQ_REQUIRE(ix.rid_table.rid.size() == ivf_pq_num_base(ix) && delta.lists.tombstones.empty() &&
                       delta.lists.pending_inserts.empty() && delta.free_list.deferred.empty() &&
                       delta.free_list.free_slots.empty(),
                   "ivf_pq_recover_deletes needs a freshly loaded index and an empty delta");
    const RawVectorHeapLayout& layout = heap.layout();
    const uint32_t pages = heap.allocated_pages();
    const uint32_t cursor = heap.next_flat_slot();
    std::vector<uint8_t> occupied(size_t(pages) * layout.bitmap_bytes);
    std::vector<uint32_t> owner(size_t(pages) * layout.slots_per_page);
    for (uint32_t p = 0; p < pages; ++p) {
        heap.read_page_directory(p, occupied.data() + size_t(p) * layout.bitmap_bytes,
                                 owner.data() + size_t(p) * layout.slots_per_page);
    }
    auto holds = [&](uint32_t flat, uint32_t id) {
        const bool bit = (occupied[size_t(layout.page_of(flat)) * layout.bitmap_bytes + layout.index_in_page(flat) / 8] &
                          layout.bitmap_mask(layout.index_in_page(flat))) != 0;
        return bit && owner[flat] == id;
    };

    size_t retracted = 0;
    std::vector<uint8_t> addressed(cursor, 0);
    for (uint32_t id = 0; id < ix.rid_table.rid.size(); ++id) {
        const RawVectorRID rid = ix.rid_table.rid[id];
        const uint32_t slot = rid_flat_slot(rid);
        IVF_PQ_REQUIRE(slot < cursor, "RID of vector " + std::to_string(id) + " lies at or past the heap's slot cursor");
        if (!rid_is_active(rid)) continue;
        if (holds(slot, id)) {
            addressed[slot] = 1;
        } else {
            ix.rid_table.rid[id] = make_raw_vector_rid(slot, false);
            delta.lists.tombstones.insert(id);
            ++retracted;
        }
    }
    for (uint32_t slot = 0; slot < cursor; ++slot) {
        if (!addressed[slot]) delta.free_list.free_slots.push_back(slot);
    }
    return retracted;
}

#define IVF_PQ_INSTANTIATE_INSERT(T)                                                                          \
    template IVFPQPreparedInsert ivf_pq_prepare_insert<T>(IVFPQIndex&, RawVectorHeap&, IVFPQDelta&, const T*); \
    template uint32_t ivf_pq_insert<T>(IVFPQIndex&, RawVectorHeap&, IVFPQDelta&, const T*);
IVF_PQ_INSTANTIATE_INSERT(float)
IVF_PQ_INSTANTIATE_INSERT(uint8_t)
IVF_PQ_INSTANTIATE_INSERT(int8_t)
#undef IVF_PQ_INSTANTIATE_INSERT

}  // namespace inplace
}  // namespace diskann
