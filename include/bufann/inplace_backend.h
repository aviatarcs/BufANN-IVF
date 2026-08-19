// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <limits>
#include <stdexcept>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <parallel_hashmap/phmap.h>

#include "distance.h"
#include "neighbor.h"
#include "pq_table.h"
#include "utils.h"
#include "tsl/robin_set.h"

class AlignedFileReader;

namespace diskann {
  template<typename T, typename TagT>
  class Index;
}

namespace diskann {
namespace inplace {

static constexpr uint32_t INVALID_PAGE = 0xFFFFFFFFu;
static constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;
using TagType = uint32_t;
static constexpr TagType INVALID_TAG = 0xFFFFFFFFu;
static constexpr uint64_t LIVE_DELETE_GENERATION = std::numeric_limits<uint64_t>::max();

uint64_t thread_page_access_count();
uint64_t thread_physical_reads_count();

enum class NodeState : uint8_t {
    Free = 0,
    Allocating = 1,
    Active = 2,
    DeletePending = 3,
};

enum class DistanceScope : uint8_t { SEARCH = 0, INSERT = 1, MAINTENANCE = 2 };
enum class TraversalScope : uint8_t { SEARCH = 0, INSERT = 1 };

// ---------------------------------------------------------------------------
// InPlaceIOStats
// ---------------------------------------------------------------------------
struct InPlaceIOStats {
    std::atomic<uint64_t> sequential_read_ios{0};
    std::atomic<uint64_t> sequential_read_bytes{0};
    std::atomic<uint64_t> buffer_pool_write_ios{0};

    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> dirty_evictions{0};
    std::atomic<uint64_t> page_fault_total{0};
    std::atomic<uint64_t> page_fault_with_evict{0};
    std::atomic<uint64_t> page_fault_no_evict{0};
    std::atomic<uint64_t> page_fault_total_ns{0};
    std::atomic<uint64_t> page_in_ns{0};
    std::atomic<uint64_t> eviction_total_ns{0};
    std::atomic<uint64_t> eviction_scan_ns{0};
    std::atomic<uint64_t> eviction_wait_pins_ns{0};
    std::atomic<uint64_t> eviction_flush_ns{0};
    std::atomic<uint64_t> page_fault_page_table_ns{0};
    std::atomic<uint64_t> page_fault_frame_setup_ns{0};

    // Batched pin via libaio. Tracked separately to keep batch I/O distinguishable
    // in-process; the driver folds these into the simple page_fault_* / page_in_ns
    // counters before emitting JSON.
    std::atomic<uint64_t> batch_pin_calls{0};
    std::atomic<uint64_t> batch_misses_total{0};
    std::atomic<uint64_t> batch_with_evict{0};
    std::atomic<uint64_t> batch_no_evict{0};
    std::atomic<uint64_t> batch_io_ns_total{0};

    std::atomic<uint64_t> pages_flushed{0};
    std::atomic<uint64_t> dirty_page_bytes_flushed{0};

    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_deletes{0};

    std::atomic<uint64_t> repair_queue_length{0};

    std::atomic<uint64_t> query_comparable_cpu_ns{0};
    std::atomic<uint64_t> query_comparable_cmps{0};
    std::atomic<uint64_t> query_io_ns{0};
    std::atomic<uint64_t> update_io_ns{0};
    std::atomic<uint64_t> total_io_ns{0};

    void   reset();
    std::string to_json() const;
    void record_io_ns(bool is_query, uint64_t ns) {
        total_io_ns.fetch_add(ns, std::memory_order_relaxed);
        if (is_query) {
            query_io_ns.fetch_add(ns, std::memory_order_relaxed);
        } else {
            update_io_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }
};

// ---------------------------------------------------------------------------
// PackedSlotHeader -- 8 bytes, on-disk
// ---------------------------------------------------------------------------
struct PackedSlotHeader {
    uint16_t degree;
    uint8_t  flags;
    uint8_t  _pad;
    uint32_t node_id;
};
static_assert(sizeof(PackedSlotHeader) == 8, "slot header must be 8 bytes");

// ---------------------------------------------------------------------------
// AccessMode / Frame Guards
// ---------------------------------------------------------------------------
enum AccessMode : uint8_t { READ = 0, WRITE = 1 };

class InPlaceGraphStore;
class PinnedFrame;
class FrameReadGuard;
class FrameWriteGuard;
class ConstNodeRef;
class MutableNodeRef;
enum class FrameRegion : uint8_t;

struct NodeRID {
    uint32_t page_id = INVALID_PAGE;
    uint16_t slot_idx = 0;
    std::atomic<uint8_t> active{0};
    std::atomic<uint8_t> pq_ready{0};

    // boilerplate code to overcome the copy restriction on atomic<> variables.
    NodeRID() = default;
    NodeRID(uint32_t page, uint16_t slot) : page_id(page), slot_idx(slot) {}
    NodeRID(uint32_t page, uint16_t slot, uint8_t is_active, uint8_t is_pq_ready)
        : page_id(page), slot_idx(slot), active(is_active), pq_ready(is_pq_ready) {}
    NodeRID(const NodeRID& other)
        : page_id(other.page_id),
          slot_idx(other.slot_idx),
          active(other.active.load(std::memory_order_relaxed)),
          pq_ready(other.pq_ready.load(std::memory_order_relaxed)) {}
    NodeRID& operator=(const NodeRID& other) {
        if (this == &other) return *this;
        page_id = other.page_id;
        slot_idx = other.slot_idx;
        active.store(other.active.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
        pq_ready.store(other.pq_ready.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
        return *this;
    }
};
static_assert(sizeof(NodeRID) == 8, "NodeRID should fit in existing padding");

struct PageFrame;

class ConstNodeRef {
 public:
    ConstNodeRef() = default;

    bool valid() const { return _slot_ptr != nullptr && _store != nullptr; }
    uint16_t degree() const;
    uint32_t node_id() const;
    const uint32_t* neighbors() const;
    const char* coords_bytes() const;
    template <typename T>
    const T* coords() const;

 protected:
    friend class FrameReadGuard;
    friend class FrameWriteGuard;
    friend class MutableNodeRef;
    ConstNodeRef(const InPlaceGraphStore* store, const char* slot_ptr);

    const InPlaceGraphStore* _store = nullptr;
    const char* _slot_ptr = nullptr;
};

class MutableNodeRef {
 public:
    MutableNodeRef() = default;

    bool valid() const { return _slot_ptr != nullptr && _store != nullptr; }
    uint16_t degree() const;
    uint32_t node_id() const;
    uint32_t* neighbors();
    const uint32_t* neighbors() const;
    char* coords_bytes();
    const char* coords_bytes() const;
    template <typename T>
    T* coords();
    template <typename T>
    const T* coords() const;
    void set_degree(uint16_t degree);
    void set_node_id(uint32_t node_id);
    void set_neighbors(const uint32_t* neighbors, size_t count);
    template <typename T>
    void set_coords(const T* coords);

 private:
    friend class FrameWriteGuard;
    MutableNodeRef(InPlaceGraphStore* store, char* slot_ptr,
                   FrameWriteGuard* guard);
    void mark_dirty();

    InPlaceGraphStore* _store = nullptr;
    char* _slot_ptr = nullptr;
    FrameWriteGuard* _guard = nullptr;
};

class PinnedFrame {
 public:
    PinnedFrame() = default;
    PinnedFrame(const PinnedFrame&) = delete;
    PinnedFrame& operator=(const PinnedFrame&) = delete;
    PinnedFrame(PinnedFrame&& other) noexcept;
    PinnedFrame& operator=(PinnedFrame&& other) noexcept;
    ~PinnedFrame();

    bool valid() const { return _store != nullptr && _frame != nullptr; }
    uint32_t page_id() const { return _page_id; }
    uint64_t version() const;
    uint16_t slots_per_page() const;

    FrameReadGuard read_guard();
    FrameWriteGuard write_guard();

 private:
    friend class InPlaceGraphStore;
    friend class FrameReadGuard;
    friend class FrameWriteGuard;
    PinnedFrame(InPlaceGraphStore* store, PageFrame* frame, uint32_t page_id,
                FrameRegion hint);
    void release();
    void mark_dirty();
    char* slot_ptr(uint16_t slot_idx);
    const char* slot_ptr(uint16_t slot_idx) const;
    bool slot_occupied(uint16_t slot_idx) const;
    uint32_t header_bytes() const;
    uint32_t bitmap_bytes() const;

    InPlaceGraphStore* _store = nullptr;
    PageFrame* _frame = nullptr;
    uint32_t _page_id = INVALID_PAGE;
    FrameRegion _hint{};
    bool _dirty = false;
    bool _discard = false;
};

class FrameReadGuard {
 public:
    FrameReadGuard() = default;
    FrameReadGuard(const FrameReadGuard&) = delete;
    FrameReadGuard& operator=(const FrameReadGuard&) = delete;
    FrameReadGuard(FrameReadGuard&&) noexcept = default;
    FrameReadGuard& operator=(FrameReadGuard&&) noexcept = default;
    ~FrameReadGuard() = default;

    ConstNodeRef read_node(uint16_t slot_idx) const;
    ConstNodeRef read_node(const NodeRID& rid) const;

 private:
    friend class PinnedFrame;
    explicit FrameReadGuard(PinnedFrame* frame);
    PinnedFrame* _frame = nullptr;
    std::shared_lock<std::shared_mutex> _lock;
};

class FrameWriteGuard {
 public:
    FrameWriteGuard() = default;
    FrameWriteGuard(const FrameWriteGuard&) = delete;
    FrameWriteGuard& operator=(const FrameWriteGuard&) = delete;
    FrameWriteGuard(FrameWriteGuard&&) noexcept = default;
    FrameWriteGuard& operator=(FrameWriteGuard&&) noexcept = default;
    ~FrameWriteGuard();

    ConstNodeRef read_node(uint16_t slot_idx) const;
    ConstNodeRef read_node(const NodeRID& rid) const;
    MutableNodeRef write_node(uint16_t slot_idx);
    MutableNodeRef write_node(const NodeRID& rid);
    void commit();

 private:
    friend class PinnedFrame;
    friend class MutableNodeRef;
    explicit FrameWriteGuard(PinnedFrame* frame);
    void mark_dirty();

    PinnedFrame* _frame = nullptr;
    std::unique_lock<std::shared_mutex> _lock;
    bool _dirty = false;
    bool _committed = false;
};

struct QueryBatchItem {
    uint32_t page_id = INVALID_PAGE;
    uint16_t slot_idx = 0;
    uint32_t node_id = INVALID_NODE;
    size_t out_idx = 0;
    const std::atomic<NodeState>* state_ptr = nullptr;
};

// ---------------------------------------------------------------------------
// InPlaceSearchScratch -- per-thread scratch buffers
// ---------------------------------------------------------------------------
struct InPlaceSearchScratch {
    float*   pq_dists         = nullptr;
    float*   dist_scratch     = nullptr;
    uint8_t* pq_coord_scratch = nullptr;
    std::vector<Neighbor> traversal_results;
    std::vector<Neighbor> traversal_retset;
    std::vector<Neighbor> frontier_nodes;
    std::vector<unsigned> frontier_ids;
    tsl::robin_set<unsigned> visited;
    std::vector<unsigned> frontier_neighbor_ids;
    std::vector<uint32_t> frontier_candidate_ids;
    std::vector<float> frontier_candidate_dists;
    std::vector<QueryBatchItem> frontier_batch_items;
    std::vector<uint8_t>  frontier_found;
    uint64_t local_query_comparable_cpu_ns = 0;
    uint64_t local_query_comparable_cmps = 0;

    static constexpr uint32_t MAX_SCRATCH_NODES = 16384;

    void init(uint32_t aligned_dim, uint32_t n_chunks, uint32_t elem_size,
              uint32_t search_list_size, uint32_t max_degree,
              uint32_t beamwidth);
    void reset() {
        local_query_comparable_cpu_ns = 0;
        local_query_comparable_cmps = 0;
        traversal_results.clear();
        traversal_retset.clear();
        frontier_nodes.clear();
        frontier_ids.clear();
        visited.clear();
        frontier_neighbor_ids.clear();
        frontier_candidate_ids.clear();
        frontier_candidate_dists.clear();
        frontier_batch_items.clear();
        frontier_found.clear();
    }
    inline void record_comparable_distance(DistanceScope scope, uint64_t ns, uint64_t ops) {
        if (scope != DistanceScope::SEARCH) return;
        local_query_comparable_cpu_ns += ns;
        local_query_comparable_cmps += ops;
    }
    void flush_distance_stats(InPlaceIOStats& stats);
    ~InPlaceSearchScratch();
};

// ---------------------------------------------------------------------------
// SpinLock — TTAS spinlock; satisfies BasicLockable (std::lock_guard compatible)
// ---------------------------------------------------------------------------
struct SpinLock {
    std::atomic<bool> _locked{false};

    void lock() noexcept {
        for (;;) {
            // Test-and-Test-and-Set: relaxed load avoids cache-line invalidation
            if (!_locked.load(std::memory_order_relaxed) &&
                !_locked.exchange(true, std::memory_order_acquire))
                return;
            _mm_pause();
        }
    }

    void unlock() noexcept {
        _locked.store(false, std::memory_order_release);
    }

    bool try_lock() noexcept {
        return !_locked.load(std::memory_order_relaxed) &&
               !_locked.exchange(true, std::memory_order_acquire);
    }
};

// ---------------------------------------------------------------------------
// PageFrame + FrameRegion
// ---------------------------------------------------------------------------
enum class FrameRegion : uint8_t { QUERY = 0, UPDATE = 1, SHARED = 2 };
enum class FrameState : uint8_t { FREE = 0, LOADING = 1, READY = 2, FLUSHING = 3, EVICTING = 4 };

struct PageFrame {
    std::atomic<uint32_t> page_id{INVALID_PAGE};
    std::atomic<uint64_t> generation{1};
    std::atomic<uint8_t> state{static_cast<uint8_t>(FrameState::FREE)};
    char*       data      = nullptr;
    std::atomic<bool> dirty{false};
    std::atomic<uint32_t> pin_count{0};
    std::atomic<uint32_t> update_pin_count{0};
    std::atomic<uint8_t> ref_bit{0};
    mutable std::shared_mutex data_lock;
    uint32_t frame_index = INVALID_PAGE;
};

// ---------------------------------------------------------------------------
// BufferPool -- global CLOCK eviction and background flush
// ---------------------------------------------------------------------------
class BufferPool {
 public:
    BufferPool();
    ~BufferPool();

    void init(uint32_t page_size, uint32_t num_frames,
              const std::string& heap_path, InPlaceIOStats* stats,
              uint32_t flush_budget_pages_per_cycle = 32,
              uint32_t flush_wakeup_ms = 100,
              bool truncate_heap = true);

    PageFrame& pin(uint32_t page_id,
                   FrameRegion hint = FrameRegion::QUERY);

    // Blocking batched pin via libaio. On return, every entry of out_frames is
    // pinned and READY. hint applies to all ids; duplicate ids resolve to the
    // same PageFrame*. size() must be <= kPinBatchMax. Caller unpins each result.
    static constexpr uint32_t kPinBatchMax = 256;
    void pin_batch(const uint32_t* page_ids, PageFrame** out_frames,
                   size_t n, FrameRegion hint = FrameRegion::QUERY,
                   bool account_sequential_read = false);
    PageFrame*  try_pin_resident(uint32_t page_id,
                                 FrameRegion hint = FrameRegion::QUERY);
    void       unpin_frame(PageFrame* frame, bool dirty = false,
                           FrameRegion hint = FrameRegion::QUERY);
    void       discard_frame(PageFrame* frame);
    void       unpin(uint32_t page_id, bool dirty = false,
                     FrameRegion hint = FrameRegion::QUERY);
    void       protect_page(uint32_t page_id, uint8_t credit,
                            FrameRegion hint = FrameRegion::QUERY);
    void       mark_dirty(uint32_t page_id);
    void       flush_all_dirty();
    uint32_t   flush_dirty_budget(uint32_t max_pages);
    // Grow the heap file by batch_size pages (called under InPlaceGraphStore::_pages_mtx).
    // Returns the first new page_id; updates _total_pages.
    uint32_t   grow_heap(uint32_t batch_size);
    void       set_total_pages(uint32_t used_pages);

    void start_bg_flush(float high_wm = 0.70f, float low_wm = 0.30f);
    void stop_bg_flush();
    void reset();
    uint32_t dirty_page_count() const;
    double dirty_ratio() const;
    void set_flush_budget_pages_per_cycle(uint32_t flush_budget_pages_per_cycle);

    uint32_t page_size() const { return _page_size; }
    uint32_t num_frames() const { return _num_frames; }
    uint32_t total_pages() const { return _total_pages.load(std::memory_order_acquire); }
    uint32_t pin_count_of(uint32_t page_id) const;

 private:
    struct ResidencySegment;
    struct FrameWaitState {
        std::mutex mtx;
        std::condition_variable cv;
    };

    struct ShardState {
        // Page table: phmap with std::shared_mutex per sub-map.
        // if_contains (read) takes a shared lock; lazy_emplace_l / erase take an exclusive lock.
        // 2^8 = 256 sub-maps per shard for fine-grained concurrency.
        phmap::parallel_flat_hash_map<
            uint32_t, uint32_t,
            phmap::priv::hash_default_hash<uint32_t>,
            phmap::priv::hash_default_eq<uint32_t>,
            std::allocator<std::pair<const uint32_t, uint32_t>>,
            8,
            std::shared_mutex
        > page_table;

        // Cursor state: SpinLock (TTAS), never held during I/O or pin-count waits.
        mutable SpinLock cursor_lock;
        uint32_t clock_hand   = 0;
        uint32_t flush_cursor = 0;

        // Read-only after init — no lock needed.
        uint32_t frame_begin   = 0;
        uint32_t frame_end     = 0;
    };

    uint32_t shard_index(uint32_t page_id) const;
    ShardState& shard_for_page(uint32_t page_id);
    const ShardState& shard_for_page(uint32_t page_id) const;
    uint32_t residency_segment_index(uint32_t page_id) const;
    uint32_t residency_segment_offset(uint32_t page_id) const;
    ResidencySegment* residency_segment(uint32_t page_id) const;
    ResidencySegment* ensure_residency_segment(uint32_t page_id);
    uint32_t lookup_resident_frame(uint32_t page_id) const;
    void publish_resident_frame(uint32_t page_id, uint32_t frame_idx);
    void clear_resident_frame(uint32_t page_id, uint32_t frame_idx);
    PageFrame* try_pin_query_fast(uint32_t page_id);
    void record_buffer_pool_write();
    void mark_frame_dirty(PageFrame& frame);
    void reset_residency_directory();
    uint32_t evict_one(ShardState& shard, FrameRegion preferred,
                       bool* used_eviction_out = nullptr);
    std::vector<std::pair<uint32_t, std::vector<char>>> collect_dirty_frames(ShardState& shard, uint32_t max_frames, bool flush_all);
    void     read_page(uint32_t page_id, char* buf);
    void     write_page(uint32_t page_id, const char* buf);
    void     bg_flush_loop(float high_wm, float low_wm);

    static constexpr uint32_t kResidencySegmentBits = 14;
    static constexpr uint32_t kResidencySegmentSize = 1u << kResidencySegmentBits;
    static constexpr uint32_t kMaxResidencySegments = 1u << (32 - kResidencySegmentBits);

    uint32_t         _page_size   = 0;
    uint32_t         _num_frames  = 0;
    std::deque<PageFrame> _frames;
    std::vector<std::unique_ptr<FrameWaitState>> _frame_waiters;
    std::vector<std::unique_ptr<ShardState>> _shards;
    std::unique_ptr<std::atomic<ResidencySegment*>[]> _resident_page_frames;
    int              _heap_fd     = -1;
    // Shared with _heap_fd via use_external_fd(); does not own the fd.
    std::unique_ptr<AlignedFileReader> _aio_reader;
    std::atomic<uint32_t> _total_pages{0};
    uint32_t         _num_shards = 1;
    uint32_t         _flush_budget_pages_per_cycle = 32;
    uint32_t         _flush_wakeup_ms = 100;
    std::atomic<uint32_t> _dirty_frame_count{0};

    std::thread         _flush_thread;
    std::atomic<bool>   _flush_running{false};
    InPlaceIOStats*     _stats = nullptr;
};

// ---------------------------------------------------------------------------
// InPlaceGraphStore
// ---------------------------------------------------------------------------
class InPlaceGraphStore {
 public:
    InPlaceGraphStore() {
        _entry_pool.fill(INVALID_NODE);
        _node_reservoir.fill(INVALID_NODE);
    }
    ~InPlaceGraphStore();

    void init(uint32_t dim, uint32_t Mmax, uint32_t elem_size_bytes,
              uint32_t page_size, uint32_t buffer_pool_frames,
              const std::string& heap_path,
              uint32_t flush_budget_pages_per_cycle = 32,
              uint32_t flush_wakeup_ms = 100,
              bool truncate_heap = true,
              uint32_t max_nodes = 0);

    // PQ support
    void load_pq_codes_from_disk_index(const std::string& pq_prefix,
                                       uint32_t num_chunks);
    void compute_pq_dists_query(const unsigned* ids, uint64_t n_ids,
                                const float* pq_dists, float* dists_out);
    template<typename T>
    void compute_pq_dists_src(uint32_t src, const unsigned* ids,
                              float* dists_out, uint32_t count,
                              uint8_t* scratch,
                              FixedChunkPQTable<T>& pq_table);
    // Gather PQ codes for `n` node IDs into caller-supplied flat buffer
    // (must have capacity >= n * n_chunks bytes; missing nodes get zeros).
    void gather_pq_codes(const unsigned* ids, uint64_t n, uint8_t* out) const;
    // Raw pointer to the PQ codes of a single node, or nullptr if absent.
    const uint8_t* pq_codes_for_node(uint32_t node_id) const;
    uint32_t n_chunks() const { return _n_chunks; }

    // Frame-level pin/unpin with guard-based node access
    PinnedFrame pin_page(uint32_t page_id, AccessMode mode);
    PinnedFrame pin_page_transient(uint32_t page_id, AccessMode mode);
    PinnedFrame pin_page_for_node(uint32_t node_id, AccessMode mode, NodeRID* rid_out = nullptr);
    void pin_pages_batch(const uint32_t* page_ids, size_t n,
                         AccessMode mode, PinnedFrame* out_frames,
                         bool account_sequential_read = false);
    void pin_pages_batch_transient(const uint32_t* page_ids, size_t n,
                                   AccessMode mode, PinnedFrame* out_frames,
                                   bool account_sequential_read = false);

    // Batched analogue of pin_page_for_node. For each node_ids[i], writes the
    // pinned PinnedFrame into out_frames[i] and the resolved RID into out_rids[i].
    // Nodes whose lookup fails get a default-constructed (invalid) PinnedFrame
    // and a default NodeRID. n must be <= BufferPool::kPinBatchMax.
    void pin_pages_for_nodes_batch(const uint32_t* node_ids, size_t n,
                                   AccessMode mode,
                                   PinnedFrame* out_frames,
                                   NodeRID*     out_rids);
    void     protect_seed_pages(const std::vector<uint32_t>& node_ids,
                                uint8_t credit = 2);
    void     batch_fetch_coords(const std::vector<uint32_t>& ids,
                                char* out_coords,
                                std::vector<uint8_t>& found,
                                bool transient = false);
    template<typename T>
    void     batch_compute_dists(const std::vector<uint32_t>& ids,
                                 const T* query,
                                 diskann::Distance<T>* dist,
                                 std::vector<float>& out_dists,
                                 std::vector<uint8_t>& found,
                                 InPlaceSearchScratch* scratch = nullptr);
    template<typename T>
    void     batch_compute_dists_latched(const std::vector<uint32_t>& ids,
                                         const T* query,
                                         diskann::Distance<T>* dist,
                                         std::vector<float>& out_dists,
                                         std::vector<uint8_t>& found,
                                         InPlaceSearchScratch* scratch = nullptr);
    void     preload_query_hot_pages(const std::vector<uint32_t>& seed_nodes,
                                     uint32_t page_budget,
                                     uint32_t node_budget,
                                     uint8_t protect_credit = 2);
    void     preload_pages(uint32_t max_pages, uint32_t num_threads);

    // Allocation
    uint32_t allocate_node();                  // reserve a fresh id + materialize its slot
    void     allocate_node(uint32_t node_id);  // materialize a caller-supplied id
    void     allocate_nodes_batch(const uint32_t* node_ids, size_t count);
    void     publish_node(uint32_t node_id);
    void     publish_nodes_batch(const uint32_t* node_ids, size_t count);
    bool     is_active(uint32_t node_id) const;
    bool     is_delete_pending(uint32_t node_id) const;
    bool     set_node_tag(uint32_t node_id, TagType tag);
    TagType  node_tag(uint32_t node_id) const;
    bool     mark_tag_deleted(TagType tag);
    bool     is_tag_deleted(TagType tag) const;
    size_t   deleted_tag_count() const;
    std::vector<uint32_t> drain_deleted_tags_to_delete_pending();
    uint32_t get_page_id(uint32_t node_id) const;
    void     reserve_meta_capacity(size_t needed);

    // Page-level pin count query for safe reclamation
    uint32_t page_pin_count(uint32_t page_id) const;

    // PQ encode for new inserts
    template<typename T>
    void encode_pq(uint32_t node_id, const T* coords,
                   FixedChunkPQTable<T>& pq_table);

    uint32_t dirty_page_count() const;
    uint32_t remove_deleted_edges_full_scan(uint32_t num_threads = 1);

    // Synchronous per-call delete repair. Processes exactly `del_id` —
    // beam-search in-neighbor discovery, c_replace edge repair, bridge
    // propagation. Final slot/internal-ID reclaim is performed by maintenance.
    template<typename T>
    uint32_t repair_deleted_single(uint32_t del_id,
                                   unsigned R, unsigned C,
                                   float alpha, unsigned aligned_dim,
                                   diskann::Distance<T>* dist,
                                   uint32_t c_replace = 3,
                                   uint32_t delete_repair_L = 0,
                                   uint32_t beamwidth = 4,
                                   FixedChunkPQTable<T>* pq_table = nullptr);

    // Synchronous repair over an explicit caller-provided batch (e.g. a
    // per-thread micro-batch).
    template<typename T>
    uint32_t repair_deleted_explicit(const std::vector<uint32_t>& delete_batch,
                                     unsigned R, unsigned C,
                                     float alpha, unsigned aligned_dim,
                                     diskann::Distance<T>* dist,
                                     uint32_t c_replace = 3,
                                     uint32_t delete_repair_L = 0,
                                     uint32_t beamwidth = 4,
                                     FixedChunkPQTable<T>* pq_table = nullptr);

    // Foreground delete lifecycle transition. Returns true if the node
    // transitioned from Active to DeletePending.
    bool transition_active_to_delete_pending(uint32_t node_id);

    // Bulk load
    template<typename T, typename TagT>
    void bulk_load_from_index(diskann::Index<T, TagT>& mem_index,
                              uint32_t n);

    // Warmup + lifecycle
    void warmup_bfs(uint32_t entry_point, uint32_t num_nodes);
    // Node-based level-synchronous BFS warmup from entry_point. The frontier
    // holds graph node IDs; each visited node's host page is pinned as a
    // side effect (via pin_page_for_node), and only that node's neighbors
    // expand the frontier — so the warmed set corresponds to the first few
    // graph hops around entry_point rather than the full slot population of
    // each touched page. Hops are fetched in parallel with num_threads OMP
    // threads. Budget is on *pages*: when a hop's frontier exceeds the
    // remaining page budget, we parallel-pin a `remaining`-node prefix
    // (likely undershoots due to node->page sharing), then tail-pin the rest
    // of the hop's frontier sequentially until visited_pages == num_pages.
    // If the frontier is exhausted before the budget is filled, terminates
    // undershot.
    void warmup_bfs_pages(uint32_t entry_point, uint32_t num_pages,
                          uint32_t num_threads);
    // Drain every thread's TLS hit buffer, then zero globals. Prefer over
    // bare stats().reset(), which leaves TLS to leak into the next phase.
    void reset_stats();
    void start_bg_flush(float high_wm = 0.70f, float low_wm = 0.30f);
    void stop_bg_flush();
    void flush();
    uint32_t flush_dirty_budget(uint32_t max_pages);
    double dirty_ratio() const;
    void set_bg_flush_budget_pages_per_cycle(uint32_t flush_budget_pages_per_cycle);
    void set_entry_point(uint32_t entry_point);
    void seed_entry_pool_from_reservoir();
    void get_entry_points(std::vector<uint32_t>& out);
    void save_snapshot(const std::string& meta_path);
    void load_snapshot(const std::string& meta_path);

    // Stats + info
    InPlaceIOStats& stats();
    const InPlaceIOStats& stats() const { return _stats; }
    uint32_t entry_point() {
        std::vector<uint32_t> pts;
        get_entry_points(pts);
        return pts.empty() ? INVALID_NODE : pts[0];
    }
    uint32_t entry_point() const {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        return _entry_pool[0];
    }
    uint32_t num_active() const;
    uint32_t aligned_dim() const { return _aligned_dim; }
    uint32_t slot_size() const { return _slot_size; }
    uint32_t page_size() const { return _page_size; }
    uint32_t max_degree() const { return _Mmax; }
    uint32_t total_pages() const { return _bp.total_pages(); }
    size_t   memory_usage_bytes() const;
    void set_page_data_locks_enabled(bool enabled) { _page_data_locks_enabled = enabled; }
    bool page_data_locks_enabled() const { return _page_data_locks_enabled; }

    // Memory breakdown for metrics
    size_t buffer_pool_bytes() const;
    size_t locator_bytes() const;
    size_t pq_data_bytes() const;

 private:
    using RID = NodeRID;

    struct PageDir {
        uint16_t slots_per_page;
        uint16_t num_occupied;
    };

    // Shared implementation for repair_deleted_single / repair_deleted_explicit.
    template<typename T>
    uint32_t repair_deleted_ids_impl(const std::vector<uint32_t>& delete_batch,
                                     unsigned R, unsigned C, float alpha,
                                     unsigned aligned_dim,
                                     diskann::Distance<T>* dist,
                                     uint32_t c_replace,
                                     uint32_t delete_repair_L,
                                     uint32_t beamwidth,
                                     FixedChunkPQTable<T>* pq_table);

    uint32_t claim_page();           // Phase 1: reserve a slot on a page (under _pages_mtx)
    RID allocate_slot_on_page(uint32_t pid);  // Phase 2: write bitmap + header (frame lock)
    bool node_present(uint32_t node_id) const;
    bool lookup_node_rid(uint32_t node_id, RID& rid) const;
    bool invalidate_node_rid_if_page(uint32_t node_id, uint32_t page_id);
    uint32_t reserve_internal_id();
    void materialize_node(uint32_t node_id, const RID& rid);
    void reset_metadata(size_t expected_entries = 0);
    void ensure_node_id_in_capacity(uint32_t node_id, const char* caller) const;
    void reset_node_slot(uint32_t node_id);
    void ensure_pq_storage();
    const uint8_t* pending_pq_codes(uint32_t node_id) const;
    void clear_pending_pq_codes();
    std::vector<uint32_t> reclaim_delete_pending_ids(
        const std::vector<uint32_t>& node_ids);

    static constexpr size_t kDefaultNodeCapacity = 1u << 20;
    static constexpr size_t kEntryPoolSize = 10;
    static constexpr size_t kReservoirSize = 64;
    size_t _max_nodes = 0;
    std::unique_ptr<RID[]> _node_rids;
    std::unique_ptr<TagType[]> _node_tags;
    std::unique_ptr<std::atomic<NodeState>[]> _node_states;

    uint32_t _dim         = 0;
    uint32_t _aligned_dim = 0;
    uint32_t _Mmax        = 0;
    uint32_t _elem_size   = 0;
    uint32_t _page_size   = 0;
    uint32_t _slot_size   = 0;
    uint32_t _slots_per_page = 0;

    BufferPool     _bp;
    InPlaceIOStats _stats;
    std::mutex _delete_pending_since_maintenance_mtx;
    std::vector<uint32_t> _delete_pending_since_maintenance;
    using DeletedTagSet = phmap::parallel_flat_hash_set_m<TagType>;
    mutable std::shared_mutex _deleted_tags_swap_mtx;
    DeletedTagSet _deleted_tags_current;
    DeletedTagSet _deleted_tags_draining;
    std::mutex _freelist_mtx;
    std::vector<uint32_t> _freelist;
    std::atomic<uint32_t> _next_internal_id{0};

    // PQ
    const uint8_t* _pending_pq_codes = nullptr;
    size_t _pending_pq_code_count = 0;
    void* _pending_pq_mmap_base = nullptr;
    size_t _pending_pq_mmap_bytes = 0;
    std::unique_ptr<uint8_t[]> _pq_codes;
    uint32_t _n_chunks   = 0;
    std::atomic<uint32_t> _num_active{0};

    // Page directory — protected by _pages_mtx
    mutable std::mutex    _pages_mtx;
    std::vector<PageDir>  _page_dir;
    std::set<uint32_t>    _pages_with_space;
    uint32_t              _alloc_page_rr_cursor = 0;

    static constexpr uint32_t kPageAllocBatchSize = 256;

    bool _page_data_locks_enabled = false;
    mutable std::mutex _entry_mtx;
    std::array<uint32_t, kEntryPoolSize> _entry_pool;
    std::array<uint32_t, kReservoirSize> _node_reservoir;
    size_t _reservoir_cursor = 0;

    friend class ConstNodeRef;
    friend class MutableNodeRef;
    friend class PinnedFrame;
    friend class FrameReadGuard;
    friend class FrameWriteGuard;
};

template<typename T>
void InPlaceGraphStore::batch_compute_dists(const std::vector<uint32_t>& ids,
                                            const T* query,
                                            diskann::Distance<T>* dist,
                                            std::vector<float>& out_dists,
                                            std::vector<uint8_t>& found,
                                            InPlaceSearchScratch* scratch) {
    found.assign(ids.size(), 0);
    out_dists.resize(ids.size());
    if (ids.empty() || query == nullptr || dist == nullptr) return;

    std::vector<QueryBatchItem> local_items;
    std::vector<QueryBatchItem>& items =
        scratch ? scratch->frontier_batch_items : local_items;
    items.clear();
    if (items.capacity() < ids.size()) {
        items.reserve(ids.size());
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t node_id = ids[i];
        RID rid;
        if (!lookup_node_rid(node_id, rid)) continue;
        if (rid.page_id == INVALID_PAGE) continue;
        items.push_back(QueryBatchItem{
            rid.page_id, rid.slot_idx, node_id, i, &_node_states[node_id]});
    }
    std::sort(items.begin(), items.end(), [](const QueryBatchItem& a, const QueryBatchItem& b) {
        if (a.page_id != b.page_id) return a.page_id < b.page_id;
        return a.slot_idx < b.slot_idx;
    });

    size_t cursor = 0;
    while (cursor < items.size()) {
        uint32_t page_id = items[cursor].page_id;
        PinnedFrame frame = pin_page(page_id, READ);
        if (!frame.valid()) {
            while (cursor < items.size() && items[cursor].page_id == page_id) ++cursor;
            continue;
        }
        auto guard = frame.read_guard();
        while (cursor < items.size() && items[cursor].page_id == page_id) {
            const auto& item = items[cursor];
            ConstNodeRef node = guard.read_node(item.slot_idx);
            if (node.valid() &&
                node.node_id() == item.node_id &&
                item.state_ptr != nullptr &&
                item.state_ptr->load(std::memory_order_acquire) ==
                    NodeState::Active) {
                const T* coords_ptr = node.coords<T>();
                out_dists[item.out_idx] = dist->compare(query, coords_ptr, _aligned_dim);
                found[item.out_idx] = 1;
            }
            ++cursor;
        }
    }
}

template<typename T>
void InPlaceGraphStore::batch_compute_dists_latched(
                                            const std::vector<uint32_t>& ids,
                                            const T* query,
                                            diskann::Distance<T>* dist,
                                            std::vector<float>& out_dists,
                                            std::vector<uint8_t>& found,
                                            InPlaceSearchScratch* scratch) {
    batch_compute_dists<T>(ids, query, dist, out_dists, found, scratch);
}

template <typename T>
const T* ConstNodeRef::coords() const {
    if (!valid()) return nullptr;
    if (_store->_elem_size != sizeof(T)) {
        throw std::runtime_error("ConstNodeRef::coords type size mismatch");
    }
    return reinterpret_cast<const T*>(_slot_ptr + sizeof(PackedSlotHeader));
}

template <typename T>
T* MutableNodeRef::coords() {
    if (!valid()) return nullptr;
    if (_store->_elem_size != sizeof(T)) {
        throw std::runtime_error("MutableNodeRef::coords type size mismatch");
    }
    return reinterpret_cast<T*>(_slot_ptr + sizeof(PackedSlotHeader));
}

template <typename T>
const T* MutableNodeRef::coords() const {
    if (!valid()) return nullptr;
    if (_store->_elem_size != sizeof(T)) {
        throw std::runtime_error("MutableNodeRef::coords type size mismatch");
    }
    return reinterpret_cast<const T*>(_slot_ptr + sizeof(PackedSlotHeader));
}

template <typename T>
void MutableNodeRef::set_coords(const T* coords_src) {
    if (!valid() || coords_src == nullptr) return;
    T* dst = coords<T>();
    std::memcpy(dst, coords_src, static_cast<size_t>(_store->_aligned_dim) * sizeof(T));
    mark_dirty();
}

}  // namespace inplace
}  // namespace diskann
