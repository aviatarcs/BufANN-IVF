// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bufann/inplace_backend.h"
#include "index.h"
#include "linux_aligned_file_reader.h"
#include "logger.h"

#include <cerrno>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>

namespace diskann {
namespace inplace {

namespace {

void* checked_aligned_alloc(size_t alignment, size_t size, const char* what) {
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, alignment, size);
    if (rc != 0) {
        throw std::runtime_error(std::string("posix_memalign failed for ") + what +
                                 " rc=" + std::to_string(rc));
    }
    return ptr;
}

// Per-thread batched query-hit counters. Relaxed atomics so a foreign
// thread (TlsRegistry::drain_into, called from reset_stats) can safely
// exchange-to-zero while the owner thread is mid-increment.
struct QueryHotStatsBuffer {
    std::atomic<uint64_t> cache_hits{0};
    QueryHotStatsBuffer();
    ~QueryHotStatsBuffer();
};

// Global registry of every live TLS buffer. Each thread self-registers on
// first access (TLS ctor) and deregisters on thread death (TLS dtor), so
// drain_into reaches every thread that has ever touched the counters —
// OMP workers, std::thread, anything — with no thread-count plumbing.
class TlsRegistry {
public:
    void add(QueryHotStatsBuffer* b) {
        std::lock_guard<std::mutex> lk(_mtx);
        _bufs.push_back(b);
    }
    void remove(QueryHotStatsBuffer* b) {
        std::lock_guard<std::mutex> lk(_mtx);
        _bufs.erase(std::remove(_bufs.begin(), _bufs.end(), b), _bufs.end());
    }
    void drain_into(InPlaceIOStats& s) {
        std::lock_guard<std::mutex> lk(_mtx);
        for (auto* b : _bufs) {
            uint64_t h = b->cache_hits.exchange(0, std::memory_order_relaxed);
            if (h) s.cache_hits.fetch_add(h, std::memory_order_relaxed);
        }
    }
private:
    std::mutex _mtx;
    std::vector<QueryHotStatsBuffer*> _bufs;
};

static TlsRegistry& tls_registry() {
    static TlsRegistry r;
    return r;
}

QueryHotStatsBuffer::QueryHotStatsBuffer()  { tls_registry().add(this); }
QueryHotStatsBuffer::~QueryHotStatsBuffer() { tls_registry().remove(this); }

thread_local QueryHotStatsBuffer tls_query_hot_stats;

inline void record_query_cache_hit(uint64_t n = 1) {
    tls_query_hot_stats.cache_hits.fetch_add(n, std::memory_order_relaxed);
}

inline void flush_query_hot_stats(InPlaceIOStats& stats) {
    uint64_t h = tls_query_hot_stats.cache_hits.exchange(0, std::memory_order_relaxed);
    if (h) stats.cache_hits.fetch_add(h, std::memory_order_relaxed);
}

#if defined(__GNUC__) || defined(__clang__)
struct PackedSlotHeaderView {
    uint16_t degree;
    uint8_t flags;
    uint8_t _pad;
    uint32_t node_id;
} __attribute__((packed, may_alias));
static_assert(sizeof(PackedSlotHeaderView) == sizeof(PackedSlotHeader),
              "packed slot view must match header size");

inline PackedSlotHeader load_slot_header(const char* slot_ptr) {
    const auto* view = reinterpret_cast<const PackedSlotHeaderView*>(slot_ptr);
    PackedSlotHeader hdr;
    hdr.degree = view->degree;
    hdr.flags = view->flags;
    hdr._pad = view->_pad;
    hdr.node_id = view->node_id;
    return hdr;
}
#else
inline PackedSlotHeader load_slot_header(const char* slot_ptr) {
    PackedSlotHeader hdr;
    std::memcpy(&hdr, slot_ptr, sizeof(hdr));
    return hdr;
}
#endif

inline FrameState load_frame_state(const PageFrame& frame) {
    return static_cast<FrameState>(frame.state.load(std::memory_order_acquire));
}

inline void store_frame_state(PageFrame& frame, FrameState state) {
    frame.state.store(static_cast<uint8_t>(state), std::memory_order_release);
}

inline bool frame_state_allows_pin(const PageFrame& frame, FrameState state,
                                   FrameRegion hint) {
    if (state == FrameState::READY) return true;
    if (hint == FrameRegion::QUERY && state == FrameState::FLUSHING) {
        return frame.update_pin_count.load(std::memory_order_acquire) == 0;
    }
    return false;
}

inline uint64_t elapsed_ns(
    const std::chrono::steady_clock::time_point& start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

}  // namespace

// ===========================================================================
// InPlaceIOStats
// ===========================================================================
void InPlaceIOStats::reset() {
    sequential_read_ios.store(0);
    sequential_read_bytes.store(0);
    buffer_pool_write_ios.store(0);
    cache_hits.store(0);
    cache_misses.store(0);
    evictions.store(0);
    dirty_evictions.store(0);
    page_fault_total.store(0);
    page_fault_with_evict.store(0);
    page_fault_no_evict.store(0);
    page_fault_total_ns.store(0);
    page_in_ns.store(0);
    eviction_total_ns.store(0);
    eviction_scan_ns.store(0);
    eviction_wait_pins_ns.store(0);
    eviction_flush_ns.store(0);
    page_fault_page_table_ns.store(0);
    page_fault_frame_setup_ns.store(0);
    batch_pin_calls.store(0);
    batch_misses_total.store(0);
    batch_with_evict.store(0);
    batch_no_evict.store(0);
    batch_io_ns_total.store(0);
    pages_flushed.store(0);
    dirty_page_bytes_flushed.store(0);
    total_inserts.store(0);
    total_deletes.store(0);
    repair_queue_length.store(0);
    query_comparable_cpu_ns.store(0);
    query_comparable_cmps.store(0);
    query_io_ns.store(0);
    update_io_ns.store(0);
    total_io_ns.store(0);
}

std::string InPlaceIOStats::to_json() const {
    std::ostringstream ss;
    ss << "{"
       << "\"sequential_read_ios\":" << sequential_read_ios.load() << ","
       << "\"sequential_read_bytes\":" << sequential_read_bytes.load() << ","
       << "\"buffer_pool_write_ios\":" << buffer_pool_write_ios.load() << ","
       << "\"cache_hits\":" << cache_hits.load() << ","
       << "\"cache_misses\":" << cache_misses.load() << ","
       << "\"evictions\":" << evictions.load() << ","
       << "\"dirty_evictions\":" << dirty_evictions.load() << ","
       << "\"page_fault_total\":" << page_fault_total.load() << ","
       << "\"page_fault_with_evict\":" << page_fault_with_evict.load() << ","
       << "\"page_fault_no_evict\":" << page_fault_no_evict.load() << ","
       << "\"page_fault_total_ns\":" << page_fault_total_ns.load() << ","
       << "\"page_in_ns\":" << page_in_ns.load() << ","
       << "\"eviction_total_ns\":" << eviction_total_ns.load() << ","
       << "\"eviction_scan_ns\":" << eviction_scan_ns.load() << ","
       << "\"eviction_wait_pins_ns\":" << eviction_wait_pins_ns.load() << ","
       << "\"eviction_flush_ns\":" << eviction_flush_ns.load() << ","
       << "\"page_fault_page_table_ns\":" << page_fault_page_table_ns.load() << ","
       << "\"page_fault_frame_setup_ns\":" << page_fault_frame_setup_ns.load() << ","
       << "\"pages_flushed\":" << pages_flushed.load() << ","
       << "\"dirty_page_bytes_flushed\":" << dirty_page_bytes_flushed.load()
       << ","
       << "\"total_inserts\":" << total_inserts.load() << ","
       << "\"total_deletes\":" << total_deletes.load() << ","
       << "\"query_comparable_cpu_ns\":" << query_comparable_cpu_ns.load() << ","
       << "\"query_comparable_cmps\":" << query_comparable_cmps.load() << ","
       << "\"query_io_ns\":" << query_io_ns.load() << ","
       << "\"update_io_ns\":" << update_io_ns.load() << ","
       << "\"total_io_ns\":" << total_io_ns.load()
       << "}";
    return ss.str();
}

// ===========================================================================
// InPlaceSearchScratch
// ===========================================================================
void InPlaceSearchScratch::init(uint32_t aligned_dim, uint32_t n_chunks,
                                uint32_t elem_size,
                                uint32_t search_list_size,
                                uint32_t max_degree,
                                uint32_t beamwidth) {
    (void)aligned_dim;
    (void)elem_size;

    size_t dist_bytes = (size_t)MAX_SCRATCH_NODES * sizeof(float);
    dist_scratch = static_cast<float*>(checked_aligned_alloc(32, dist_bytes, "dist_scratch"));
    memset(dist_scratch, 0, dist_bytes);

    if (n_chunks > 0) {
        size_t pq_dist_bytes = (size_t)n_chunks * 256 * sizeof(float);
        pq_dists = static_cast<float*>(checked_aligned_alloc(32, pq_dist_bytes, "pq_dists"));
        memset(pq_dists, 0, pq_dist_bytes);

        size_t pq_coord_bytes = (size_t)n_chunks * MAX_SCRATCH_NODES;
        pq_coord_scratch = static_cast<uint8_t*>(
            checked_aligned_alloc(32, pq_coord_bytes, "pq_coord_scratch"));
        memset(pq_coord_scratch, 0, pq_coord_bytes);
    }

    const size_t l_cap = std::max<size_t>(1, search_list_size);
    const size_t degree_cap = std::max<size_t>(1, max_degree);
    const size_t beam_cap = std::max<size_t>(1, beamwidth);
    const size_t frontier_cap = std::max<size_t>(degree_cap * 2, l_cap * 2);
    const size_t visited_cap = std::min<size_t>(
        MAX_SCRATCH_NODES, std::max<size_t>(10 * l_cap, l_cap * degree_cap));

    traversal_results.reserve(10 * l_cap);
    traversal_retset.reserve(l_cap + 1);
    frontier_nodes.reserve(beam_cap);
    frontier_ids.reserve(beam_cap);
    visited.reserve(visited_cap);
    frontier_neighbor_ids.reserve(frontier_cap);
    frontier_candidate_ids.reserve(frontier_cap);
    frontier_candidate_dists.reserve(frontier_cap);
    frontier_batch_items.reserve(frontier_cap);
    frontier_found.reserve(frontier_cap);
}

InPlaceSearchScratch::~InPlaceSearchScratch() {
    if (pq_dists)              free(pq_dists);
    if (dist_scratch)          free(dist_scratch);
    if (pq_coord_scratch)      free(pq_coord_scratch);
}

void InPlaceSearchScratch::flush_distance_stats(InPlaceIOStats& stats) {
    if (local_query_comparable_cpu_ns || local_query_comparable_cmps) {
        stats.query_comparable_cpu_ns.fetch_add(local_query_comparable_cpu_ns,
                                                std::memory_order_relaxed);
        stats.query_comparable_cmps.fetch_add(local_query_comparable_cmps,
                                              std::memory_order_relaxed);
    }
    local_query_comparable_cpu_ns = 0;
    local_query_comparable_cmps = 0;
    flush_query_hot_stats(stats);
}

// ===========================================================================
// BufferPool
// ===========================================================================
struct BufferPool::ResidencySegment {
    std::unique_ptr<std::atomic<uint32_t>[]> frame_indices;

    ResidencySegment()
        : frame_indices(new std::atomic<uint32_t>[kResidencySegmentSize]) {
        for (uint32_t i = 0; i < kResidencySegmentSize; ++i) {
            frame_indices[i].store(INVALID_PAGE, std::memory_order_relaxed);
        }
    }
};

BufferPool::BufferPool() = default;

BufferPool::~BufferPool() {
    stop_bg_flush();
    reset_residency_directory();
    if (_aio_reader) {
        _aio_reader->deregister_all_threads();
        _aio_reader.reset();
    }
    for (auto& f : _frames) {
        if (f.data) {
            free(f.data);
            f.data = nullptr;
        }
    }
    if (_heap_fd >= 0) {
        ::close(_heap_fd);
        _heap_fd = -1;
    }
}

void BufferPool::init(uint32_t page_size, uint32_t num_frames,
                      const std::string& heap_path, InPlaceIOStats* stats,
                      uint32_t flush_budget_pages_per_cycle,
                      uint32_t flush_wakeup_ms,
                      bool truncate_heap) {
    _page_size  = page_size;
    _num_frames = num_frames;
    _stats      = stats;
    _flush_budget_pages_per_cycle = std::max<uint32_t>(1, flush_budget_pages_per_cycle);
    _flush_wakeup_ms = std::max<uint32_t>(1, flush_wakeup_ms);
    _dirty_frame_count.store(0, std::memory_order_relaxed);

    _frames.clear();
    _frame_waiters.clear();
    reset_residency_directory();
    for (uint32_t i = 0; i < num_frames; ++i) {
        _frames.emplace_back();
        auto& f = _frames.back();
        f.frame_index = i;
        f.data = static_cast<char*>(checked_aligned_alloc(4096, page_size, "buffer frame"));
        memset(f.data, 0, page_size);
        _frame_waiters.emplace_back(std::make_unique<FrameWaitState>());
    }

    uint32_t desired_shards =
        std::min<uint32_t>(16, std::max<uint32_t>(4, num_frames / 64));
    _num_shards = std::max<uint32_t>(
        1, std::min<uint32_t>(num_frames == 0 ? 1 : num_frames, desired_shards));
    _shards.clear();
    _shards.reserve(_num_shards);
    uint32_t frame_begin = 0;
    for (uint32_t shard_idx = 0; shard_idx < _num_shards; ++shard_idx) {
        auto shard = std::make_unique<ShardState>();
        uint32_t frame_end = ((shard_idx + 1) * num_frames) / _num_shards;
        shard->frame_begin = frame_begin;
        shard->frame_end = frame_end;
        shard->clock_hand = frame_begin;
        shard->flush_cursor = frame_begin;
        _shards.emplace_back(std::move(shard));
        frame_begin = frame_end;
    }

    int open_flags = O_RDWR | O_CREAT | O_DIRECT | O_LARGEFILE;
    if (truncate_heap) open_flags |= O_TRUNC;
    _heap_fd = ::open(heap_path.c_str(), open_flags, 0644);
    // if (_heap_fd < 0) {
    //     open_flags = O_RDWR | O_CREAT;
    //     if (truncate_heap) open_flags |= O_TRUNC;
    //     _heap_fd = ::open(heap_path.c_str(), open_flags, 0644);
    // }
    if (_heap_fd < 0) {
        throw ANNException("Failed to open heap file: " + heap_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    auto linux_reader = std::make_unique<LinuxAlignedFileReader>();
    linux_reader->use_external_fd(_heap_fd);
    _aio_reader = std::move(linux_reader);
    if (truncate_heap) {
        _total_pages.store(0, std::memory_order_release);
    } else {
        struct stat st;
        if (::fstat(_heap_fd, &st) != 0) {
            throw ANNException("Failed to stat heap file: " + heap_path, -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        if (st.st_size < 0) {
            throw ANNException("Heap file size is negative: " + heap_path, -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
        uint64_t file_pages_u64 =
            (file_bytes + static_cast<uint64_t>(_page_size) - 1) / static_cast<uint64_t>(_page_size);
        if (file_pages_u64 > std::numeric_limits<uint32_t>::max()) {
            throw ANNException("Heap file page count exceeds uint32_t range: " + heap_path, -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        _total_pages.store(static_cast<uint32_t>(file_pages_u64), std::memory_order_release);
    }
}

uint32_t BufferPool::shard_index(uint32_t page_id) const {
    return _num_shards == 0 ? 0 : (page_id % _num_shards);
}

BufferPool::ShardState& BufferPool::shard_for_page(uint32_t page_id) {
    return *_shards[shard_index(page_id)];
}

const BufferPool::ShardState& BufferPool::shard_for_page(uint32_t page_id) const {
    return *_shards[shard_index(page_id)];
}

uint32_t BufferPool::residency_segment_index(uint32_t page_id) const {
    return page_id >> kResidencySegmentBits;
}

uint32_t BufferPool::residency_segment_offset(uint32_t page_id) const {
    return page_id & (kResidencySegmentSize - 1);
}

BufferPool::ResidencySegment* BufferPool::residency_segment(uint32_t page_id) const {
    if (!_resident_page_frames) return nullptr;
    return _resident_page_frames[residency_segment_index(page_id)].load(std::memory_order_acquire);
}

BufferPool::ResidencySegment* BufferPool::ensure_residency_segment(uint32_t page_id) {
    if (!_resident_page_frames) {
        _resident_page_frames.reset(new std::atomic<ResidencySegment*>[kMaxResidencySegments]);
        for (uint32_t i = 0; i < kMaxResidencySegments; ++i) {
            _resident_page_frames[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    uint32_t seg_idx = residency_segment_index(page_id);
    ResidencySegment* segment = _resident_page_frames[seg_idx].load(std::memory_order_acquire);
    if (segment != nullptr) return segment;

    auto* fresh = new ResidencySegment();
    ResidencySegment* expected = nullptr;
    if (_resident_page_frames[seg_idx].compare_exchange_strong(
            expected, fresh, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return fresh;
    }
    delete fresh;
    return expected;
}

uint32_t BufferPool::lookup_resident_frame(uint32_t page_id) const {
    ResidencySegment* segment = residency_segment(page_id);
    if (segment == nullptr) return INVALID_PAGE;
    return segment->frame_indices[residency_segment_offset(page_id)].load(std::memory_order_acquire);
}

void BufferPool::publish_resident_frame(uint32_t page_id, uint32_t frame_idx) {
    ResidencySegment* segment = ensure_residency_segment(page_id);
    segment->frame_indices[residency_segment_offset(page_id)].store(frame_idx, std::memory_order_release);
}

void BufferPool::clear_resident_frame(uint32_t page_id, uint32_t frame_idx) {
    ResidencySegment* segment = residency_segment(page_id);
    if (segment == nullptr) return;
    std::atomic<uint32_t>& slot = segment->frame_indices[residency_segment_offset(page_id)];
    uint32_t expected = frame_idx;
    slot.compare_exchange_strong(expected, INVALID_PAGE,
                                 std::memory_order_acq_rel,
                                 std::memory_order_acquire);
}

PageFrame* BufferPool::try_pin_query_fast(uint32_t page_id) {
    uint32_t frame_idx = lookup_resident_frame(page_id);
    if (frame_idx == INVALID_PAGE || frame_idx >= _frames.size()) return nullptr;

    auto& frame = _frames[frame_idx];
    uint64_t generation = frame.generation.load(std::memory_order_acquire);
    frame.pin_count.fetch_add(1, std::memory_order_acq_rel);

    bool valid = frame.page_id.load(std::memory_order_acquire) == page_id &&
                 frame.generation.load(std::memory_order_acquire) == generation &&
                 frame_state_allows_pin(frame, load_frame_state(frame), FrameRegion::QUERY);
    if (!valid) {
        unpin_frame(&frame, false);
        return nullptr;
    }

    frame.ref_bit.store(1, std::memory_order_release);
    if (_stats) {
        record_query_cache_hit();
    }
    return &frame;
}

void BufferPool::reset_residency_directory() {
    if (!_resident_page_frames) {
        _resident_page_frames.reset(new std::atomic<ResidencySegment*>[kMaxResidencySegments]);
        for (uint32_t i = 0; i < kMaxResidencySegments; ++i) {
            _resident_page_frames[i].store(nullptr, std::memory_order_relaxed);
        }
        return;
    }

    for (uint32_t i = 0; i < kMaxResidencySegments; ++i) {
        ResidencySegment* segment = _resident_page_frames[i].exchange(nullptr, std::memory_order_acq_rel);
        delete segment;
    }
}

namespace {
thread_local uint64_t t_page_accesses = 0;
thread_local uint64_t t_phys_reads = 0;
}

uint64_t thread_page_access_count() { return t_page_accesses; }
uint64_t thread_physical_reads_count() { return t_phys_reads; }

void BufferPool::read_page(uint32_t page_id, char* buf) {
    ssize_t n = ::pread(_heap_fd, buf, _page_size,
                        (off_t)page_id * _page_size);
    if (n < 0) n = 0;
    if ((size_t)n < _page_size) {
        memset(buf + n, 0, _page_size - n);
    }
    ++t_phys_reads;
}

void BufferPool::write_page(uint32_t page_id, const char* buf) {
    ssize_t n = ::pwrite(_heap_fd, buf, _page_size,
                         (off_t)page_id * _page_size);
    (void)n;
}

uint32_t BufferPool::evict_one(ShardState& shard, FrameRegion preferred,
                               bool* used_eviction_out) {
    const auto evict_begin = std::chrono::steady_clock::now();
    uint64_t local_scan_ns = 0;
    uint64_t local_wait_pins_ns = 0;
    uint64_t local_flush_ns = 0;
    auto finalize_evict = [&](uint32_t frame_idx, bool did_evict) -> uint32_t {
        if (used_eviction_out != nullptr) {
            *used_eviction_out = did_evict;
        }
        if (did_evict && _stats != nullptr) {
            _stats->eviction_total_ns.fetch_add(elapsed_ns(evict_begin), std::memory_order_relaxed);
            _stats->eviction_scan_ns.fetch_add(local_scan_ns, std::memory_order_relaxed);
            _stats->eviction_wait_pins_ns.fetch_add(local_wait_pins_ns, std::memory_order_relaxed);
            _stats->eviction_flush_ns.fetch_add(local_flush_ns, std::memory_order_relaxed);
        }
        return frame_idx;
    };

    // Finish evicting a frame that has already been claimed (state==EVICTING).
    // Called with NO lock held. old_page_id already erased from page_table.
    auto finish_evict = [&](uint32_t idx, uint32_t old_page_id) -> uint32_t {
        auto& f = _frames[idx];
        // Wait for existing pinners to release — no lock held, bounded by unpin_frame()
        const auto wait_begin = std::chrono::steady_clock::now();
        while (f.pin_count.load(std::memory_order_acquire) > 0) {
            _mm_pause();
        }
        local_wait_pins_ns += elapsed_ns(wait_begin);
        const bool was_dirty = f.dirty.exchange(false, std::memory_order_acq_rel);
        if (was_dirty) {
            _dirty_frame_count.fetch_sub(1, std::memory_order_relaxed);
        }
        std::vector<char> flush_copy;
        if (was_dirty) flush_copy.assign(f.data, f.data + _page_size);
        if (_stats) {
            _stats->evictions.fetch_add(1);
            if (was_dirty) _stats->dirty_evictions.fetch_add(1);
        }
        f.page_id.store(INVALID_PAGE, std::memory_order_release);
        f.generation.fetch_add(1, std::memory_order_acq_rel);
        f.pin_count.store(0, std::memory_order_release);
        f.update_pin_count.store(0, std::memory_order_release);
        f.ref_bit.store(0, std::memory_order_release);
        store_frame_state(f, FrameState::LOADING);
        if (was_dirty) {
            const auto flush_begin = std::chrono::steady_clock::now();
            write_page(old_page_id, flush_copy.data());
            local_flush_ns += elapsed_ns(flush_begin);
            if (_stats) {
                _stats->pages_flushed.fetch_add(1);
                _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
            }
        }
        return idx;
    };

    // New policy: start from CLOCK hand immediately on a miss. We still keep the
    // previous region-priority path below for easy fallback if needed.
    struct ClockClaim {
        uint32_t idx;
        uint32_t old_page_id;
        bool from_free_frame;
    };

    auto claim_clock_frame = [&]() -> ClockClaim {
        const auto scan_begin = std::chrono::steady_clock::now();
        std::unique_lock<SpinLock> clk(shard.cursor_lock);
        const uint32_t shard_frames = shard.frame_end - shard.frame_begin;
        const uint32_t budget = std::max<uint32_t>(1, shard_frames) * 2;

        auto next_clock_idx = [&]() -> uint32_t {
            uint32_t idx = shard.clock_hand;
            if (++shard.clock_hand >= shard.frame_end) {
                shard.clock_hand = shard.frame_begin;
            }
            return idx;
        };

        auto claim_free_frame = [&](uint32_t idx, PageFrame& f) -> ClockClaim {
            uint8_t expected_free = static_cast<uint8_t>(FrameState::FREE);
            if (f.state.compare_exchange_strong(expected_free,
                                                static_cast<uint8_t>(FrameState::LOADING),
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed)) {
                f.generation.fetch_add(1, std::memory_order_acq_rel);
                local_scan_ns += elapsed_ns(scan_begin);
                return {idx, INVALID_PAGE, true};
            }
            return {INVALID_PAGE, INVALID_PAGE, false};
        };

        auto claim_ready_victim = [&](uint32_t idx, uint32_t pid, PageFrame& f,
                                      bool allow_dirty) -> ClockClaim {
            if (load_frame_state(f) != FrameState::READY) return {INVALID_PAGE, INVALID_PAGE, false};
            if (f.pin_count.load(std::memory_order_acquire) > 0) return {INVALID_PAGE, INVALID_PAGE, false};
            if (f.ref_bit.exchange(0, std::memory_order_acq_rel)) return {INVALID_PAGE, INVALID_PAGE, false};
            if (!allow_dirty && f.dirty.load(std::memory_order_acquire)) {
                return {INVALID_PAGE, INVALID_PAGE, false};
            }

            uint8_t expected_ready = static_cast<uint8_t>(FrameState::READY);
            if (!f.state.compare_exchange_strong(expected_ready,
                                                 static_cast<uint8_t>(FrameState::EVICTING),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed)) {
                return {INVALID_PAGE, INVALID_PAGE, false};
            }

            shard.page_table.erase(pid);
            clear_resident_frame(pid, idx);
            local_scan_ns += elapsed_ns(scan_begin);
            return {idx, pid, false};
        };

        for (uint32_t scanned = 0; scanned < budget; ++scanned) {
            const uint32_t idx = next_clock_idx();
            auto& f = _frames[idx];
            const uint32_t pid = f.page_id.load(std::memory_order_acquire);

            if (pid == INVALID_PAGE) {
                ClockClaim claim = claim_free_frame(idx, f);
                if (claim.idx != INVALID_PAGE) return claim;
                continue;
            }

            ClockClaim claim = claim_ready_victim(idx, pid, f, true);
            if (claim.idx != INVALID_PAGE) return claim;
        }
        local_scan_ns += elapsed_ns(scan_begin);
        return {INVALID_PAGE, INVALID_PAGE, false};
    };

    const ClockClaim claim = claim_clock_frame();
    if (claim.idx == INVALID_PAGE) {
        throw ANNException("BufferPool: no reusable frame found by CLOCK scan", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (claim.from_free_frame) {
        return finalize_evict(claim.idx, false);
    }
    return finalize_evict(finish_evict(claim.idx, claim.old_page_id), true);
}

PageFrame& BufferPool::pin(uint32_t page_id, FrameRegion hint) {
    if (hint == FrameRegion::QUERY) {
        if (PageFrame* fast = try_pin_query_fast(page_id)) {
            return *fast;
        }
    }

    ShardState& shard = shard_for_page(page_id);
    while (true) {
        // ---------------------------------------------------------------
        // Fast path: page already resident — phmap shared lock, no mutex
        // ---------------------------------------------------------------
        PageFrame* hit_frame = nullptr;
        uint32_t   loading_idx = INVALID_PAGE;

        shard.page_table.if_contains(page_id, [&](const auto& kv) {
            auto& f = _frames[kv.second];
            // Optimistic pin: increment first, validate state after.
            // QUERY may read FLUSHING frames once pre-flush writers have drained;
            // UPDATE pins still require READY.
            f.pin_count.fetch_add(1, std::memory_order_acq_rel);
            if (hint != FrameRegion::QUERY) {
                f.update_pin_count.fetch_add(1, std::memory_order_acq_rel);
            }
            FrameState st = load_frame_state(f);
            if (frame_state_allows_pin(f, st, hint) &&
                f.page_id.load(std::memory_order_acquire) == page_id) {
                f.ref_bit.store(1, std::memory_order_release);
                if (hint == FrameRegion::QUERY) {
                    publish_resident_frame(page_id, kv.second);
                }
                if (_stats) {
                    if (hint == FrameRegion::QUERY) {
                        record_query_cache_hit();
                    } else {
                        _stats->cache_hits.fetch_add(1);
                    }
                }
                hit_frame = &f;
            } else {
                // Frame is not pinnable for this access mode — undo the pin.
                if (hint != FrameRegion::QUERY) {
                    f.update_pin_count.fetch_sub(1, std::memory_order_acq_rel);
                }
                f.pin_count.fetch_sub(1, std::memory_order_acq_rel);
                loading_idx = kv.second;
            }
        });

        if (hit_frame) return *hit_frame;

        if (loading_idx != INVALID_PAGE) {
            // Another thread is loading or flushing this frame; wait outside all locks.
            FrameWaitState* wait_state = _frame_waiters[loading_idx].get();
            auto wait_begin = std::chrono::high_resolution_clock::now();
            std::unique_lock<std::mutex> wait_lk(wait_state->mtx);
            wait_state->cv.wait(wait_lk, [&f_ref = _frames[loading_idx], hint] {
                FrameState st = load_frame_state(f_ref);
                return st != FrameState::LOADING &&
                       (st != FrameState::FLUSHING ||
                        frame_state_allows_pin(f_ref, st, hint));
            });
            auto wait_end = std::chrono::high_resolution_clock::now();
            if (_stats) {
                _stats->record_io_ns(
                    hint == FrameRegion::QUERY,
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_begin).count()));
            }
            continue;
        }

        // ---------------------------------------------------------------
        // Slow path: cache miss
        // ---------------------------------------------------------------
        // Track both logical buffer-pool misses and physical page faults.
        // They can diverge when batched requests are deduped.
        const auto page_fault_begin = std::chrono::steady_clock::now();
        bool used_eviction = false;
        uint64_t local_page_table_ns = 0;
        uint64_t local_frame_setup_ns = 0;
        // Step 1: claim a frame.
        uint32_t frame_idx = INVALID_PAGE;
        if (frame_idx == INVALID_PAGE) {
            bool evicted = false;
            frame_idx = evict_one(shard, hint, &evicted);
            used_eviction = evicted;
        }

        // Step 2: atomically insert into page_table.
        // If another thread beat us to this page_id, release our frame and retry.
        const auto page_table_begin = std::chrono::steady_clock::now();
        bool we_won = shard.page_table.lazy_emplace_l(page_id,
            [&](auto& kv) {
                // Key already exists — another thread is loading page_id.
                auto& claimed_f = _frames[frame_idx];
                claimed_f.page_id.store(INVALID_PAGE, std::memory_order_release);
                store_frame_state(claimed_f, FrameState::FREE);
                loading_idx = kv.second;
            },
            [&](auto&& ctor) {
                // Key not found — insert our (page_id, frame_idx) entry.
                ctor(std::piecewise_construct,
                     std::make_tuple(page_id),
                     std::make_tuple(frame_idx));
            });
        local_page_table_ns += elapsed_ns(page_table_begin);
        // lazy_emplace_l returns true when key was NOT present (we inserted it).
        if (!we_won) continue;

        // Step 3: set frame metadata — frame is ours (LOADING state, not visible to others yet).
        const auto frame_setup_begin = std::chrono::steady_clock::now();
        auto& f = _frames[frame_idx];
        f.page_id.store(page_id, std::memory_order_release);
        f.pin_count.store(1, std::memory_order_release);
        f.update_pin_count.store(hint == FrameRegion::QUERY ? 0u : 1u,
                                 std::memory_order_release);
        f.dirty.store(false, std::memory_order_release);
        f.ref_bit.store(1, std::memory_order_release);
        local_frame_setup_ns += elapsed_ns(frame_setup_begin);

        // Step 4: read page from disk — no lock held.
        auto io_begin = std::chrono::high_resolution_clock::now();
        read_page(page_id, f.data);
        auto io_end = std::chrono::high_resolution_clock::now();
        if (_stats) {
            const uint64_t page_in_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(io_end - io_begin).count());
            _stats->page_in_ns.fetch_add(page_in_ns, std::memory_order_relaxed);
            _stats->record_io_ns(
                hint == FrameRegion::QUERY,
                page_in_ns);
        }

        // Step 5: mark READY and wake any waiters.
        store_frame_state(f, FrameState::READY);
        {
            std::lock_guard<std::mutex> wait_lk(_frame_waiters[frame_idx]->mtx);
        }
        _frame_waiters[frame_idx]->cv.notify_all();
        if (hint == FrameRegion::QUERY) {
            publish_resident_frame(page_id, frame_idx);
        }
        if (_stats) {
            _stats->page_fault_total.fetch_add(1, std::memory_order_relaxed);
            _stats->cache_misses.fetch_add(1, std::memory_order_relaxed);
            if (used_eviction) {
                _stats->page_fault_with_evict.fetch_add(1, std::memory_order_relaxed);
            } else {
                _stats->page_fault_no_evict.fetch_add(1, std::memory_order_relaxed);
            }
            _stats->page_fault_total_ns.fetch_add(
                elapsed_ns(page_fault_begin), std::memory_order_relaxed);
            _stats->page_fault_page_table_ns.fetch_add(local_page_table_ns, std::memory_order_relaxed);
            _stats->page_fault_frame_setup_ns.fetch_add(local_frame_setup_ns, std::memory_order_relaxed);
        }
        return f;
    }
}

static_assert(BufferPool::kPinBatchMax <= 256,
              "kPinBatchMax must fit in MAX_EVENTS (256) of linux_aligned_file_reader");

void BufferPool::pin_batch(const uint32_t* page_ids, PageFrame** out_frames,
                           size_t n, FrameRegion hint,
                           bool account_sequential_read) {
    if (n == 0) return;
    if (n > kPinBatchMax) {
        throw std::invalid_argument(
            "BufferPool::pin_batch: size exceeds kPinBatchMax (256)");
    }
    assert(_aio_reader && "BufferPool::pin_batch called before init()");

    // Lazy thread registration (PipeANN pattern). Hard-fail if io_setup fails.
    IOContext bad = (IOContext)(-1);
    IOContext ctx = _aio_reader->get_ctx();
    if (ctx == bad) {
        _aio_reader->register_thread();
        ctx = _aio_reader->get_ctx();
    }
    if (ctx == bad) {
        throw std::runtime_error(
            "BufferPool::pin_batch: io_setup failed for this thread");
    }

    // --- Step A: dedupe page_ids ---
    uint32_t unique_ids[kPinBatchMax];
    uint16_t slot_to_unique[kPinBatchMax];
    size_t n_unique = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t pid = page_ids[i];
        size_t u = 0;
        for (; u < n_unique; ++u) {
            if (unique_ids[u] == pid) break;
        }
        slot_to_unique[i] = static_cast<uint16_t>(u);
        if (u == n_unique) {
            unique_ids[n_unique++] = pid;
        }
    }

    enum Kind : uint8_t { K_HIT = 0, K_WAIT_OTHER = 1, K_MISS_OURS = 2 };
    struct Entry {
        PageFrame* frame;
        uint32_t   frame_idx;     // valid for WAIT_OTHER, MISS_OURS
        Kind       kind;
        bool       used_eviction; // MISS_OURS only
    };
    Entry entries[kPinBatchMax];
    std::vector<AlignedRead> reqs;
    reqs.reserve(n_unique);

    // --- Step B: per-unique classification (one pin per unique frame) ---
    for (size_t u = 0; u < n_unique; ++u) {
        const uint32_t page_id = unique_ids[u];
        ShardState& shard = shard_for_page(page_id);
        while (true) {
            PageFrame* hit_frame = nullptr;
            uint32_t   loading_idx = INVALID_PAGE;
            shard.page_table.if_contains(page_id, [&](const auto& kv) {
                auto& f = _frames[kv.second];
                f.pin_count.fetch_add(1, std::memory_order_acq_rel);
                if (hint != FrameRegion::QUERY) {
                    f.update_pin_count.fetch_add(1, std::memory_order_acq_rel);
                }
                FrameState st = load_frame_state(f);
                if (frame_state_allows_pin(f, st, hint) &&
                    f.page_id.load(std::memory_order_acquire) == page_id) {
                    f.ref_bit.store(1, std::memory_order_release);
                    if (hint == FrameRegion::QUERY) {
                        publish_resident_frame(page_id, kv.second);
                    }
                    hit_frame = &f;
                } else {
                    if (hint != FrameRegion::QUERY) {
                        f.update_pin_count.fetch_sub(1, std::memory_order_acq_rel);
                    }
                    f.pin_count.fetch_sub(1, std::memory_order_acq_rel);
                    loading_idx = kv.second;
                }
            });
            if (hit_frame) {
                entries[u] = Entry{ hit_frame, INVALID_PAGE, K_HIT, false };
                break;
            }
            if (loading_idx != INVALID_PAGE) {
                // Another thread is loading this page; take a pin and wait later.
                auto& f = _frames[loading_idx];
                f.pin_count.fetch_add(1, std::memory_order_acq_rel);
                if (hint != FrameRegion::QUERY) {
                    f.update_pin_count.fetch_add(1, std::memory_order_acq_rel);
                }
                entries[u] = Entry{ &f, loading_idx, K_WAIT_OTHER, false };
                break;
            }
            // Slow path: claim a frame, race to insert.
            bool used_eviction = false;
            uint32_t frame_idx = evict_one(shard, hint, &used_eviction);
            bool we_won = shard.page_table.lazy_emplace_l(page_id,
                [&](auto& /*kv*/) {
                    auto& claimed_f = _frames[frame_idx];
                    claimed_f.page_id.store(INVALID_PAGE, std::memory_order_release);
                    store_frame_state(claimed_f, FrameState::FREE);
                },
                [&](auto&& ctor) {
                    ctor(std::piecewise_construct,
                         std::make_tuple(page_id),
                         std::make_tuple(frame_idx));
                });
            if (!we_won) continue;
            auto& f = _frames[frame_idx];
            f.page_id.store(page_id, std::memory_order_release);
            f.pin_count.store(1, std::memory_order_release);
            f.update_pin_count.store(hint == FrameRegion::QUERY ? 0u : 1u,
                                     std::memory_order_release);
            f.dirty.store(false, std::memory_order_release);
            f.ref_bit.store(1, std::memory_order_release);
            // State is LOADING (set by evict_one's finish_evict path).
            entries[u] = Entry{ &f, frame_idx, K_MISS_OURS, used_eviction };
            reqs.push_back(AlignedRead{
                static_cast<uint64_t>(page_id) * static_cast<uint64_t>(_page_size),
                static_cast<uint64_t>(_page_size),
                f.data
            });
            break;
        }
    }

    // --- Step C: batched libaio read for MISS_OURS ---
    uint64_t n_miss = 0, n_with_evict = 0, n_no_evict = 0;
    uint64_t batch_ns = 0;
    if (!reqs.empty()) {
        const auto io_begin = std::chrono::high_resolution_clock::now();
        _aio_reader->read(reqs, ctx);
        const auto io_end = std::chrono::high_resolution_clock::now();
        batch_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(io_end - io_begin).count());

        // --- Step D: transition MISS_OURS frames to READY, notify waiters ---
        for (size_t u = 0; u < n_unique; ++u) {
            if (entries[u].kind != K_MISS_OURS) continue;
            auto& f = *entries[u].frame;
            store_frame_state(f, FrameState::READY);
            {
                std::lock_guard<std::mutex> wait_lk(_frame_waiters[entries[u].frame_idx]->mtx);
            }
            _frame_waiters[entries[u].frame_idx]->cv.notify_all();
            if (hint == FrameRegion::QUERY) {
                publish_resident_frame(unique_ids[u], entries[u].frame_idx);
            }
            ++n_miss;
            if (entries[u].used_eviction) ++n_with_evict;
            else                          ++n_no_evict;
        }
    }

    // --- Step E: CV-wait WAIT_OTHER entries ---
    for (size_t u = 0; u < n_unique; ++u) {
        if (entries[u].kind != K_WAIT_OTHER) continue;
        FrameWaitState* wait_state = _frame_waiters[entries[u].frame_idx].get();
        auto& f_ref = *entries[u].frame;
        std::unique_lock<std::mutex> wait_lk(wait_state->mtx);
        wait_state->cv.wait(wait_lk, [&f_ref, hint] {
            FrameState st = load_frame_state(f_ref);
            return st != FrameState::LOADING &&
                   (st != FrameState::FLUSHING ||
                    frame_state_allows_pin(f_ref, st, hint));
        });
    }

    // --- Step F: extra pins for duplicate slots ---
    uint16_t pin_count_per_unique[kPinBatchMax] = {0};
    for (size_t i = 0; i < n; ++i) ++pin_count_per_unique[slot_to_unique[i]];
    uint64_t logical_hits = 0;
    uint64_t logical_misses = 0;
    for (size_t u = 0; u < n_unique; ++u) {
        if (entries[u].kind != K_MISS_OURS) {
            logical_hits += pin_count_per_unique[u];
        } else {
            logical_misses += pin_count_per_unique[u];
        }
        if (pin_count_per_unique[u] <= 1) continue;
        const uint32_t extra = pin_count_per_unique[u] - 1u;
        auto& f = *entries[u].frame;
        f.pin_count.fetch_add(extra, std::memory_order_acq_rel);
        if (hint != FrameRegion::QUERY) {
            f.update_pin_count.fetch_add(extra, std::memory_order_acq_rel);
        }
    }

    // --- Step G: fill output ---
    for (size_t i = 0; i < n; ++i) {
        out_frames[i] = entries[slot_to_unique[i]].frame;
    }

    // Keep phase-local diagnostics aligned with synchronous read_page().
    // Insert/delete phase scopes sample this thread-local counter directly.
    if (n_miss > 0) {
        t_phys_reads += n_miss;
    }

    // --- Stats ---
    // Logical cache misses are counted per request; batch_misses_total counts
    // only the unique pages this batch physically read.
    if (_stats) {
        _stats->batch_pin_calls.fetch_add(1, std::memory_order_relaxed);
        if (logical_hits > 0) {
            if (hint == FrameRegion::QUERY) {
                record_query_cache_hit(logical_hits);
            } else {
                _stats->cache_hits.fetch_add(logical_hits, std::memory_order_relaxed);
            }
        }
        if (logical_misses > 0) {
            _stats->cache_misses.fetch_add(logical_misses, std::memory_order_relaxed);
        }
        if (n_miss > 0) {
            const uint64_t read_bytes =
                n_miss * static_cast<uint64_t>(_page_size);
            _stats->batch_misses_total.fetch_add(n_miss, std::memory_order_relaxed);
            _stats->batch_with_evict.fetch_add(n_with_evict, std::memory_order_relaxed);
            _stats->batch_no_evict.fetch_add(n_no_evict, std::memory_order_relaxed);
            _stats->batch_io_ns_total.fetch_add(batch_ns, std::memory_order_relaxed);
            if (account_sequential_read) {
                _stats->sequential_read_ios.fetch_add(n_miss, std::memory_order_relaxed);
                _stats->sequential_read_bytes.fetch_add(read_bytes, std::memory_order_relaxed);
            }
        }
    }
}

PageFrame* BufferPool::try_pin_resident(uint32_t page_id, FrameRegion hint) {
    ShardState& shard = shard_for_page(page_id);
    PageFrame* result = nullptr;
    shard.page_table.if_contains(page_id, [&](const auto& kv) {
        auto& f = _frames[kv.second];
        f.pin_count.fetch_add(1, std::memory_order_acq_rel);
        if (hint != FrameRegion::QUERY) {
            f.update_pin_count.fetch_add(1, std::memory_order_acq_rel);
        }
        if (frame_state_allows_pin(f, load_frame_state(f), hint) &&
            f.page_id.load(std::memory_order_acquire) == page_id) {
            f.ref_bit.store(1, std::memory_order_release);
            if (_stats) {
                if (hint == FrameRegion::QUERY) {
                    record_query_cache_hit();
                } else {
                    _stats->cache_hits.fetch_add(1);
                }
            }
            result = &f;
        } else {
            if (hint != FrameRegion::QUERY) {
                f.update_pin_count.fetch_sub(1, std::memory_order_acq_rel);
            }
            f.pin_count.fetch_sub(1, std::memory_order_acq_rel);
        }
    });
    return result;
}

void BufferPool::protect_page(uint32_t page_id, uint8_t credit, FrameRegion hint) {
    if (credit == 0) return;
    auto& f = pin(page_id, hint);
    f.ref_bit.store(1, std::memory_order_release);
    unpin_frame(&f, false, hint);
}

void BufferPool::unpin_frame(PageFrame* frame, bool dirty, FrameRegion hint) {
    if (frame == nullptr) return;
    if (dirty) {
        record_buffer_pool_write();
        mark_frame_dirty(*frame);
    }
    bool notify_flush_waiters = false;
    if (hint != FrameRegion::QUERY) {
        uint32_t prev_updates = frame->update_pin_count.fetch_sub(1, std::memory_order_acq_rel);
#ifndef NDEBUG
        assert(prev_updates > 0);
#endif
        notify_flush_waiters =
            prev_updates == 1 && load_frame_state(*frame) == FrameState::FLUSHING;
    }
#ifndef NDEBUG
    uint32_t prev = frame->pin_count.fetch_sub(1, std::memory_order_acq_rel);
    assert(prev > 0);
#else
    frame->pin_count.fetch_sub(1, std::memory_order_acq_rel);
#endif
    if (notify_flush_waiters) {
        // Wake QUERY pins waiting for pre-flush UPDATE pins to drain.
        uint32_t frame_idx = frame->frame_index;
        if (frame_idx < _frame_waiters.size()) {
            std::lock_guard<std::mutex> wait_lk(_frame_waiters[frame_idx]->mtx);
            _frame_waiters[frame_idx]->cv.notify_all();
        }
    }
}

void BufferPool::discard_frame(PageFrame* frame) {
    if (frame == nullptr) return;
    if (frame->pin_count.load(std::memory_order_acquire) > 0) return;
    const uint32_t page_id = frame->page_id.load(std::memory_order_acquire);
    if (page_id == INVALID_PAGE) return;

    uint8_t expected = static_cast<uint8_t>(FrameState::READY);
    if (!frame->state.compare_exchange_strong(expected,
                                              static_cast<uint8_t>(FrameState::EVICTING),
                                              std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
        return;
    }

    ShardState& shard = shard_for_page(page_id);
    shard.page_table.erase(page_id);
    clear_resident_frame(page_id, frame->frame_index);

    while (frame->pin_count.load(std::memory_order_acquire) > 0) {
        _mm_pause();
    }

    const bool was_dirty = frame->dirty.exchange(false, std::memory_order_acq_rel);
    if (was_dirty) {
        _dirty_frame_count.fetch_sub(1, std::memory_order_relaxed);
        write_page(page_id, frame->data);
        if (_stats) _stats->pages_flushed.fetch_add(1);
    }
    if (_stats) _stats->evictions.fetch_add(1);

    frame->page_id.store(INVALID_PAGE, std::memory_order_release);
    frame->generation.fetch_add(1, std::memory_order_acq_rel);
    frame->ref_bit.store(0, std::memory_order_release);
    store_frame_state(*frame, FrameState::FREE);
}

void BufferPool::unpin(uint32_t page_id, bool dirty, FrameRegion hint) {
    ShardState& shard = shard_for_page(page_id);
    shard.page_table.if_contains(page_id, [&](const auto& kv) {
        unpin_frame(&_frames[kv.second], dirty, hint);
    });
}

void BufferPool::mark_dirty(uint32_t page_id) {
    ShardState& shard = shard_for_page(page_id);
    shard.page_table.if_contains(page_id, [&](const auto& kv) {
        record_buffer_pool_write();
        mark_frame_dirty(_frames[kv.second]);
    });
}

void BufferPool::record_buffer_pool_write() {
    if (_stats == nullptr) return;
    _stats->buffer_pool_write_ios.fetch_add(1, std::memory_order_relaxed);
}

void BufferPool::mark_frame_dirty(PageFrame& frame) {
    if (!frame.dirty.exchange(true, std::memory_order_acq_rel)) {
        _dirty_frame_count.fetch_add(1, std::memory_order_relaxed);
    }
}

uint32_t BufferPool::flush_dirty_budget(uint32_t max_pages) {
    uint32_t flushed = 0;
    uint32_t remaining =
        max_pages == 0 ? std::numeric_limits<uint32_t>::max() : max_pages;
    for (auto& shard_ptr : _shards) {
        if (remaining == 0) break;
        auto flushes = collect_dirty_frames(*shard_ptr, remaining, max_pages == 0);
        for (auto& flush_entry : flushes) {
            write_page(flush_entry.first, flush_entry.second.data());
            if (_stats) {
                _stats->pages_flushed.fetch_add(1);
                _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
            }
            // Frame is still in FLUSHING state (only we can transition it).
            // Use if_contains to find the frame and CAS FLUSHING → READY.
            ShardState& shard = shard_for_page(flush_entry.first);
            shard.page_table.if_contains(flush_entry.first, [&](const auto& kv) {
                auto& f = _frames[kv.second];
                uint8_t expected = static_cast<uint8_t>(FrameState::FLUSHING);
                if (f.state.compare_exchange_strong(expected,
                                                    static_cast<uint8_t>(FrameState::READY),
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> wait_lk(_frame_waiters[kv.second]->mtx);
                    _frame_waiters[kv.second]->cv.notify_all();
                }
            });
        }
        flushed += static_cast<uint32_t>(flushes.size());
        if (max_pages != 0) {
            remaining -= static_cast<uint32_t>(flushes.size());
        }
    }
    return flushed;
}

void BufferPool::flush_all_dirty() {
    (void)flush_dirty_budget(0);
}

uint32_t BufferPool::dirty_page_count() const {
    return _dirty_frame_count.load(std::memory_order_relaxed);
}

double BufferPool::dirty_ratio() const {
    if (_num_frames == 0) return 0.0;
    return static_cast<double>(dirty_page_count()) / static_cast<double>(_num_frames);
}

void BufferPool::set_flush_budget_pages_per_cycle(uint32_t flush_budget_pages_per_cycle) {
    _flush_budget_pages_per_cycle = std::max<uint32_t>(1, flush_budget_pages_per_cycle);
}

uint32_t BufferPool::grow_heap(uint32_t batch_size) {
    uint32_t base = _total_pages.load(std::memory_order_relaxed);
    uint64_t max_pages = static_cast<uint64_t>(std::numeric_limits<off_t>::max()) /
                         static_cast<uint64_t>(_page_size);
    if (static_cast<uint64_t>(base) + batch_size > max_pages) {
        throw ANNException("BufferPool grow_heap: heap size limit exceeded", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    uint32_t new_total = base + batch_size;
    off_t new_size = static_cast<off_t>(new_total) * static_cast<off_t>(_page_size);
    if (::ftruncate(_heap_fd, new_size) != 0) {
        throw ANNException("BufferPool grow_heap ftruncate failed: errno=" +
                               std::to_string(errno) + " (" + std::string(std::strerror(errno)) + ")",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    _total_pages.store(new_total, std::memory_order_release);
    return base;
}

void BufferPool::set_total_pages(uint32_t used_pages) {
    uint32_t current = _total_pages.load(std::memory_order_acquire);
    if (used_pages > current) {
        throw ANNException("set_total_pages: used_pages=" + std::to_string(used_pages) +
                               " exceeds heap capacity=" + std::to_string(current),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    _total_pages.store(used_pages, std::memory_order_release);
}

std::vector<std::pair<uint32_t, std::vector<char>>> BufferPool::collect_dirty_frames(
    ShardState& shard, uint32_t max_frames, bool flush_all) {
    std::vector<std::pair<uint32_t, std::vector<char>>> flushes;
    if (shard.frame_begin >= shard.frame_end) return flushes;
    uint32_t shard_frames = shard.frame_end - shard.frame_begin;
    uint32_t budget = flush_all ? shard_frames
                                : std::max<uint32_t>(1, std::min<uint32_t>(max_frames, shard_frames));
    uint32_t scanned = 0;
    while (scanned < shard_frames && flushes.size() < budget) {
        // Advance flush_cursor under cursor_lock (brief).
        uint32_t idx;
        {
            std::lock_guard<SpinLock> clk(shard.cursor_lock);
            idx = shard.flush_cursor;
            if (++shard.flush_cursor >= shard.frame_end) shard.flush_cursor = shard.frame_begin;
        }
        scanned++;

        auto& f = _frames[idx];
        uint32_t page_id = f.page_id.load(std::memory_order_acquire);
        if (page_id == INVALID_PAGE) continue;
        if (load_frame_state(f) != FrameState::READY) continue;
        if (f.update_pin_count.load(std::memory_order_acquire) > 0) continue;
        if (!f.dirty.load(std::memory_order_acquire)) continue;

        // Atomically claim for flushing: READY → FLUSHING.
        // After this CAS, UPDATE pins decline the frame, QUERY pins may use it
        // once any pre-flush UPDATE pins have drained, and eviction skips it.
        uint8_t expected_ready = static_cast<uint8_t>(FrameState::READY);
        if (!f.state.compare_exchange_strong(expected_ready,
                                             static_cast<uint8_t>(FrameState::FLUSHING),
                                             std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
            continue;  // another thread claimed it first
        }

        while (f.update_pin_count.load(std::memory_order_acquire) > 0) {
            _mm_pause();
        }

        const bool was_dirty = f.dirty.exchange(false, std::memory_order_acq_rel);
        if (!was_dirty) {
            store_frame_state(f, FrameState::READY);
            {
                std::lock_guard<std::mutex> wait_lk(_frame_waiters[idx]->mtx);
            }
            _frame_waiters[idx]->cv.notify_all();
            continue;
        }
        _dirty_frame_count.fetch_sub(1, std::memory_order_relaxed);
        std::vector<char> copy(f.data, f.data + _page_size);
        flushes.emplace_back(page_id, std::move(copy));
    }
    return flushes;
}

void BufferPool::bg_flush_loop(float high_wm, float low_wm) {
    while (_flush_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(_flush_wakeup_ms));
        uint32_t dirty_count = dirty_page_count();
        float ratio = (float)dirty_count / (float)(_num_frames ? _num_frames : 1);
        if (ratio <= high_wm) continue;
        for (auto& shard_ptr : _shards) {
            if (dirty_count <= static_cast<uint32_t>(low_wm * _num_frames)) break;
            auto flushes = collect_dirty_frames(*shard_ptr, _flush_budget_pages_per_cycle, false);
            for (auto& flush_entry : flushes) {
                write_page(flush_entry.first, flush_entry.second.data());
                if (_stats) {
                    _stats->pages_flushed.fetch_add(1);
                    _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
                }
                dirty_count = dirty_page_count();
                ShardState& shard = shard_for_page(flush_entry.first);
                shard.page_table.if_contains(flush_entry.first, [&](const auto& kv) {
                    auto& f = _frames[kv.second];
                    uint8_t expected = static_cast<uint8_t>(FrameState::FLUSHING);
                    if (f.state.compare_exchange_strong(expected,
                                                        static_cast<uint8_t>(FrameState::READY),
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_relaxed)) {
                        std::lock_guard<std::mutex> wait_lk(_frame_waiters[kv.second]->mtx);
                        _frame_waiters[kv.second]->cv.notify_all();
                    }
                });
            }
        }
    }
}

void BufferPool::start_bg_flush(float high_wm, float low_wm) {
    if (_flush_running.load()) return;
    _flush_running.store(true);
    _flush_thread = std::thread(&BufferPool::bg_flush_loop, this,
                                high_wm, low_wm);
}

void BufferPool::stop_bg_flush() {
    _flush_running.store(false);
    if (_flush_thread.joinable()) _flush_thread.join();
}

uint32_t BufferPool::pin_count_of(uint32_t page_id) const {
    const ShardState& shard = shard_for_page(page_id);
    uint32_t count = 0;
    shard.page_table.if_contains(page_id, [&](const auto& kv) {
        count = _frames[kv.second].pin_count.load(std::memory_order_acquire);
    });
    return count;
}

void BufferPool::reset() {
    stop_bg_flush();
    reset_residency_directory();
    for (auto& f : _frames) {
        f.page_id.store(INVALID_PAGE, std::memory_order_release);
        f.generation.fetch_add(1, std::memory_order_acq_rel);
        f.dirty.store(false, std::memory_order_release);
        store_frame_state(f, FrameState::FREE);
        f.pin_count.store(0, std::memory_order_release);
        f.update_pin_count.store(0, std::memory_order_release);
        f.ref_bit.store(0, std::memory_order_release);
    }
    _dirty_frame_count.store(0, std::memory_order_relaxed);
    for (auto& shard_ptr : _shards) {
        shard_ptr->page_table.clear();
        std::lock_guard<SpinLock> clk(shard_ptr->cursor_lock);
        shard_ptr->clock_hand = shard_ptr->frame_begin;
        shard_ptr->flush_cursor = shard_ptr->frame_begin;
    }
}

// ===========================================================================
// Standalone PQ helpers (copied from pq_flash_index.cpp anonymous namespace)
// ===========================================================================
static void pq_dist_lookup(const _u8* pq_ids, const _u64 n_pts,
                            const _u64 pq_nchunks, const float* pq_dists,
                            float* dists_out) {
    _mm_prefetch((char*)dists_out, _MM_HINT_T0);
    _mm_prefetch((char*)pq_ids, _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const float* chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
            _mm_prefetch((char*)(chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
            _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
            dists_out[idx] += chunk_dists[pq_centerid];
        }
    }
}

// ===========================================================================
// InPlaceGraphStore
// ===========================================================================
InPlaceGraphStore::~InPlaceGraphStore() {
    stop_bg_flush();
    clear_pending_pq_codes();
    _node_rids.reset();
    _node_tags.reset();
    _node_states.reset();
    _pq_codes.reset();
}


void InPlaceGraphStore::ensure_node_id_in_capacity(uint32_t node_id,
                                                   const char* caller) const {
    if (node_id < _max_nodes) return;
    throw ANNException(std::string(caller) + ": internal node_id " +
                           std::to_string(node_id) +
                           " exceeds max_dataset_size " +
                           std::to_string(_max_nodes),
                       -1, __FUNCSIG__, __FILE__, __LINE__);
}

void InPlaceGraphStore::ensure_pq_storage() {
    if (_n_chunks == 0 || _pq_codes != nullptr) return;
    if (_max_nodes == 0) {
        throw ANNException("ensure_pq_storage: metadata arrays are not initialized",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_max_nodes > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(_n_chunks)) {
        throw ANNException("ensure_pq_storage: PQ code array too large", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t bytes = _max_nodes * static_cast<size_t>(_n_chunks);
    _pq_codes.reset(new uint8_t[bytes]());
}

const uint8_t* InPlaceGraphStore::pending_pq_codes(uint32_t node_id) const {
    if (_pending_pq_codes == nullptr || _n_chunks == 0 ||
        node_id >= _pending_pq_code_count) {
        return nullptr;
    }
    return _pending_pq_codes + static_cast<size_t>(node_id) * _n_chunks;
}

void InPlaceGraphStore::clear_pending_pq_codes() {
    if (_pending_pq_mmap_base != nullptr) {
        munmap(_pending_pq_mmap_base, _pending_pq_mmap_bytes);
    }
    _pending_pq_codes = nullptr;
    _pending_pq_code_count = 0;
    _pending_pq_mmap_base = nullptr;
    _pending_pq_mmap_bytes = 0;
}

uint32_t InPlaceGraphStore::reserve_internal_id() {
    {
        std::lock_guard<std::mutex> lk(_freelist_mtx);
        while (!_freelist.empty()) {
            const uint32_t node_id = _freelist.back();
            _freelist.pop_back();
            if (node_id >= _max_nodes ||
                _node_states[node_id].load(std::memory_order_acquire) ==
                    NodeState::Free) {
                ensure_node_id_in_capacity(node_id, "reserve_internal_id");
                return node_id;
            }
        }
    }
    const uint32_t node_id =
        _next_internal_id.fetch_add(1, std::memory_order_relaxed);
    ensure_node_id_in_capacity(node_id, "reserve_internal_id");
    return node_id;
}

void InPlaceGraphStore::reserve_meta_capacity(size_t needed) {
    if (needed <= _max_nodes) return;
    throw ANNException("reserve_meta_capacity: requested " +
                           std::to_string(needed) +
                           " slots but max_dataset_size is " +
                           std::to_string(_max_nodes),
                       -1, __FUNCSIG__, __FILE__, __LINE__);
}

void InPlaceGraphStore::reset_metadata(size_t expected_entries) {
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        _delete_pending_since_maintenance.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        _deleted_tags_current.clear();
        _deleted_tags_draining.clear();
    }
    {
        std::lock_guard<std::mutex> lk(_freelist_mtx);
        _freelist.clear();
    }
    _num_active.store(0, std::memory_order_relaxed);
    _next_internal_id.store(0, std::memory_order_relaxed);
    reserve_meta_capacity(expected_entries);
    for (size_t i = 0; i < _max_nodes; ++i) {
        reset_node_slot(static_cast<uint32_t>(i));
    }
}

void InPlaceGraphStore::reset_node_slot(uint32_t node_id) {
    if (node_id >= _max_nodes) return;
    _node_rids[node_id] = RID{INVALID_PAGE, 0};
    _node_tags[node_id] = INVALID_TAG;
    _node_states[node_id].store(NodeState::Free, std::memory_order_relaxed);
    if (_n_chunks > 0 && _pq_codes != nullptr) {
        memset(_pq_codes.get() + static_cast<size_t>(node_id) * _n_chunks, 0,
               _n_chunks);
    }
}

bool InPlaceGraphStore::node_present(uint32_t node_id) const {
    return node_id < _max_nodes && _node_rids[node_id].page_id != INVALID_PAGE;
}

bool InPlaceGraphStore::lookup_node_rid(uint32_t node_id, RID& rid) const {
    if (!node_present(node_id)) return false;
    rid = _node_rids[node_id];
    return rid.page_id != INVALID_PAGE;
}

bool InPlaceGraphStore::invalidate_node_rid_if_page(uint32_t node_id, uint32_t page_id) {
    if (!node_present(node_id) || _node_rids[node_id].page_id != page_id) return false;
    _node_rids[node_id].page_id = INVALID_PAGE;
    return true;
}

void InPlaceGraphStore::materialize_node(uint32_t node_id, const RID& rid) {
    ensure_node_id_in_capacity(node_id, "materialize_node");
    _node_rids[node_id] = rid;
    _node_tags[node_id] = static_cast<TagType>(node_id);
    _node_states[node_id].store(NodeState::Allocating, std::memory_order_relaxed);
    _node_rids[node_id].active.store(0, std::memory_order_relaxed);
    _node_rids[node_id].pq_ready.store(0, std::memory_order_relaxed);
    if (_n_chunks > 0) {
        ensure_pq_storage();
        uint8_t* dst = _pq_codes.get() + static_cast<size_t>(node_id) * _n_chunks;
        memset(dst, 0, _n_chunks);
        const uint8_t* src = pending_pq_codes(node_id);
        if (src != nullptr) {
            memcpy(dst, src, _n_chunks);
            _node_rids[node_id].pq_ready.store(1, std::memory_order_release);
        }
    }
    uint32_t observed = _next_internal_id.load(std::memory_order_relaxed);
    while (observed <= node_id &&
           !_next_internal_id.compare_exchange_weak(
               observed, node_id + 1, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

ConstNodeRef::ConstNodeRef(const InPlaceGraphStore* store, const char* slot_ptr)
    : _store(store), _slot_ptr(slot_ptr) {}

uint16_t ConstNodeRef::degree() const {
    if (!valid()) return 0;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.degree;
}

uint32_t ConstNodeRef::node_id() const {
    if (!valid()) return INVALID_NODE;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.node_id;
}

const uint32_t* ConstNodeRef::neighbors() const {
    if (!valid()) return nullptr;
    return reinterpret_cast<const uint32_t*>(
        _slot_ptr + sizeof(PackedSlotHeader) +
        static_cast<size_t>(_store->_aligned_dim) * _store->_elem_size);
}

const char* ConstNodeRef::coords_bytes() const {
    if (!valid()) return nullptr;
    return _slot_ptr + sizeof(PackedSlotHeader);
}

MutableNodeRef::MutableNodeRef(InPlaceGraphStore* store, char* slot_ptr,
                               FrameWriteGuard* guard)
    : _store(store), _slot_ptr(slot_ptr), _guard(guard) {}

void MutableNodeRef::mark_dirty() {
    if (_guard != nullptr) _guard->mark_dirty();
}

uint16_t MutableNodeRef::degree() const {
    if (!valid()) return 0;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.degree;
}

uint32_t MutableNodeRef::node_id() const {
    if (!valid()) return INVALID_NODE;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.node_id;
}

uint32_t* MutableNodeRef::neighbors() {
    if (!valid()) return nullptr;
    return reinterpret_cast<uint32_t*>(
        _slot_ptr + sizeof(PackedSlotHeader) +
        static_cast<size_t>(_store->_aligned_dim) * _store->_elem_size);
}

const uint32_t* MutableNodeRef::neighbors() const {
    if (!valid()) return nullptr;
    return reinterpret_cast<const uint32_t*>(
        _slot_ptr + sizeof(PackedSlotHeader) +
        static_cast<size_t>(_store->_aligned_dim) * _store->_elem_size);
}

char* MutableNodeRef::coords_bytes() {
    if (!valid()) return nullptr;
    return _slot_ptr + sizeof(PackedSlotHeader);
}

const char* MutableNodeRef::coords_bytes() const {
    if (!valid()) return nullptr;
    return _slot_ptr + sizeof(PackedSlotHeader);
}

void MutableNodeRef::set_degree(uint16_t degree_value) {
    if (!valid()) return;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    hdr.degree = static_cast<uint16_t>(
        std::min<size_t>(degree_value, _store->_Mmax));
    std::memcpy(_slot_ptr, &hdr, sizeof(hdr));
    mark_dirty();
}

void MutableNodeRef::set_node_id(uint32_t node_id_value) {
    if (!valid()) return;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    hdr.node_id = node_id_value;
    std::memcpy(_slot_ptr, &hdr, sizeof(hdr));
    mark_dirty();
}

void MutableNodeRef::set_neighbors(const uint32_t* neighbors_src, size_t count) {
    if (!valid() || (count != 0 && neighbors_src == nullptr)) return;
    const size_t n = std::min<size_t>(count, _store->_Mmax);
    if (n != 0) {
        std::memcpy(neighbors(), neighbors_src, n * sizeof(uint32_t));
    }
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    hdr.degree = static_cast<uint16_t>(n);
    std::memcpy(_slot_ptr, &hdr, sizeof(hdr));
    mark_dirty();
}

PinnedFrame::PinnedFrame(InPlaceGraphStore* store, PageFrame* frame, uint32_t page_id,
                         FrameRegion hint)
    : _store(store), _frame(frame), _page_id(page_id), _hint(hint) {}

PinnedFrame::PinnedFrame(PinnedFrame&& other) noexcept {
    *this = std::move(other);
}

PinnedFrame& PinnedFrame::operator=(PinnedFrame&& other) noexcept {
    if (this == &other) return *this;
    release();
    _store = other._store;
    _frame = other._frame;
    _page_id = other._page_id;
    _hint = other._hint;
    _dirty = other._dirty;
    _discard = other._discard;
    other._store = nullptr;
    other._frame = nullptr;
    other._page_id = INVALID_PAGE;
    other._dirty = false;
    other._discard = false;
    return *this;
}

PinnedFrame::~PinnedFrame() { release(); }

uint64_t PinnedFrame::version() const {
    return 0;
}

uint16_t PinnedFrame::slots_per_page() const {
    return _store == nullptr ? 0 : static_cast<uint16_t>(_store->_slots_per_page);
}

FrameReadGuard PinnedFrame::read_guard() { return FrameReadGuard(this); }

FrameWriteGuard PinnedFrame::write_guard() { return FrameWriteGuard(this); }

void PinnedFrame::release() {
    if (_store == nullptr || _frame == nullptr) return;
    PageFrame* frame = _frame;
    const bool discard = _discard;
    _store->_bp.unpin_frame(frame, _dirty, _hint);
    if (discard) _store->_bp.discard_frame(frame);
    _store = nullptr;
    _frame = nullptr;
    _page_id = INVALID_PAGE;
    _dirty = false;
    _discard = false;
}

void PinnedFrame::mark_dirty() { _dirty = true; }

uint32_t PinnedFrame::bitmap_bytes() const {
    return _store == nullptr ? 0 : (_store->_slots_per_page + 7) / 8;
}

uint32_t PinnedFrame::header_bytes() const {
    return 8 + bitmap_bytes();
}

char* PinnedFrame::slot_ptr(uint16_t slot_idx) {
    if (!valid() || slot_idx >= _store->_slots_per_page) return nullptr;
    return _frame->data + header_bytes() +
           static_cast<size_t>(slot_idx) * _store->_slot_size;
}

const char* PinnedFrame::slot_ptr(uint16_t slot_idx) const {
    if (!valid() || slot_idx >= _store->_slots_per_page) return nullptr;
    return _frame->data + header_bytes() +
           static_cast<size_t>(slot_idx) * _store->_slot_size;
}

bool PinnedFrame::slot_occupied(uint16_t slot_idx) const {
    if (!valid() || slot_idx >= _store->_slots_per_page) return false;
    const uint8_t* bmap = reinterpret_cast<const uint8_t*>(_frame->data + 8);
    return ((bmap[slot_idx / 8] >> (slot_idx % 8)) & 1u) != 0;
}

FrameReadGuard::FrameReadGuard(PinnedFrame* frame)
    : _frame(frame), _lock() {
    if (_frame != nullptr && _frame->_frame != nullptr &&
        _frame->_store != nullptr &&
        _frame->_store->page_data_locks_enabled()) {
        _lock = std::shared_lock<std::shared_mutex>(_frame->_frame->data_lock);
    }
}

ConstNodeRef FrameReadGuard::read_node(uint16_t slot_idx) const {
    if (_frame == nullptr || !_frame->valid()) return ConstNodeRef();
    if (!_frame->slot_occupied(slot_idx)) return ConstNodeRef();
    return ConstNodeRef(_frame->_store, _frame->slot_ptr(slot_idx));
}

ConstNodeRef FrameReadGuard::read_node(const NodeRID& rid) const {
    if (_frame == nullptr || !_frame->valid()) return ConstNodeRef();
    if (rid.page_id != _frame->page_id()) return ConstNodeRef();
    return read_node(rid.slot_idx);
}

FrameWriteGuard::FrameWriteGuard(PinnedFrame* frame)
    : _frame(frame), _lock() {
    if (_frame != nullptr && _frame->_frame != nullptr &&
        _frame->_store != nullptr &&
        _frame->_store->page_data_locks_enabled()) {
        _lock = std::unique_lock<std::shared_mutex>(_frame->_frame->data_lock);
    }
}

FrameWriteGuard::~FrameWriteGuard() { commit(); }

void FrameWriteGuard::mark_dirty() { _dirty = true; }

ConstNodeRef FrameWriteGuard::read_node(uint16_t slot_idx) const {
    if (_frame == nullptr || !_frame->valid()) return ConstNodeRef();
    if (!_frame->slot_occupied(slot_idx)) return ConstNodeRef();
    return ConstNodeRef(_frame->_store, _frame->slot_ptr(slot_idx));
}

ConstNodeRef FrameWriteGuard::read_node(const NodeRID& rid) const {
    if (_frame == nullptr || !_frame->valid()) return ConstNodeRef();
    if (rid.page_id != _frame->page_id()) return ConstNodeRef();
    return read_node(rid.slot_idx);
}

MutableNodeRef FrameWriteGuard::write_node(uint16_t slot_idx) {
    if (_frame == nullptr || !_frame->valid()) return MutableNodeRef();
    if (!_frame->slot_occupied(slot_idx)) return MutableNodeRef();
    return MutableNodeRef(_frame->_store, _frame->slot_ptr(slot_idx), this);
}

MutableNodeRef FrameWriteGuard::write_node(const NodeRID& rid) {
    if (_frame == nullptr || !_frame->valid()) return MutableNodeRef();
    if (rid.page_id != _frame->page_id()) return MutableNodeRef();
    return write_node(rid.slot_idx);
}

void FrameWriteGuard::commit() {
    if (_committed || _frame == nullptr || !_frame->valid()) return;
    if (_dirty) {
        _frame->mark_dirty();
    }
    _committed = true;
}

PinnedFrame InPlaceGraphStore::pin_page(uint32_t page_id, AccessMode mode) {
    FrameRegion hint = (mode == WRITE) ? FrameRegion::UPDATE : FrameRegion::QUERY;
    PageFrame& frame = _bp.pin(page_id, hint);
    ++t_page_accesses;
    return PinnedFrame(this, &frame, page_id, hint);
}

PinnedFrame InPlaceGraphStore::pin_page_transient(uint32_t page_id, AccessMode mode) {
    const FrameRegion hint =
        (mode == WRITE) ? FrameRegion::UPDATE : FrameRegion::QUERY;
    PageFrame* resident = _bp.try_pin_resident(page_id, hint);
    if (resident != nullptr) {
        ++t_page_accesses;
        return PinnedFrame(this, resident, page_id, hint);
    }
    PinnedFrame frame = pin_page(page_id, mode);
    frame._discard = true;
    return frame;
}

PinnedFrame InPlaceGraphStore::pin_page_for_node(uint32_t node_id, AccessMode mode,
                                                  NodeRID* rid_out) {
    RID rid;
    if (!lookup_node_rid(node_id, rid) || rid.page_id == INVALID_PAGE) {
        return PinnedFrame();
    }
    if (rid_out != nullptr) {
        *rid_out = rid;
    }
    return pin_page(rid.page_id, mode);
}

void InPlaceGraphStore::pin_pages_batch(const uint32_t* page_ids, size_t n,
                                        AccessMode mode,
                                        PinnedFrame* out_frames,
                                        bool account_sequential_read) {
    assert(n <= BufferPool::kPinBatchMax);
    if (n == 0) return;

    const FrameRegion hint =
        (mode == WRITE) ? FrameRegion::UPDATE : FrameRegion::QUERY;

    PageFrame* frames[BufferPool::kPinBatchMax];
    _bp.pin_batch(page_ids, frames, n, hint, account_sequential_read);

    t_page_accesses += n;
    for (size_t i = 0; i < n; ++i) {
        out_frames[i] = PinnedFrame(this, frames[i], page_ids[i], hint);
    }
}

void InPlaceGraphStore::pin_pages_batch_transient(const uint32_t* page_ids,
                                                 size_t n, AccessMode mode,
                                                 PinnedFrame* out_frames,
                                                 bool account_sequential_read) {
    assert(n <= BufferPool::kPinBatchMax);
    if (n == 0) return;

    const FrameRegion hint =
        (mode == WRITE) ? FrameRegion::UPDATE : FrameRegion::QUERY;

    uint32_t miss_pages[BufferPool::kPinBatchMax];
    uint16_t back_index[BufferPool::kPinBatchMax];
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        PageFrame* resident = _bp.try_pin_resident(page_ids[i], hint);
        if (resident != nullptr) {
            ++t_page_accesses;
            out_frames[i] = PinnedFrame(this, resident, page_ids[i], hint);
            continue;
        }
        miss_pages[k] = page_ids[i];
        back_index[k] = static_cast<uint16_t>(i);
        ++k;
    }

    if (k == 0) return;

    PinnedFrame miss_frames[BufferPool::kPinBatchMax];
    pin_pages_batch(miss_pages, k, mode, miss_frames,
                    account_sequential_read);
    for (size_t j = 0; j < k; ++j) {
        miss_frames[j]._discard = true;
        out_frames[back_index[j]] = std::move(miss_frames[j]);
    }
}

void InPlaceGraphStore::pin_pages_for_nodes_batch(
    const uint32_t* node_ids, size_t n, AccessMode mode,
    PinnedFrame* out_frames, NodeRID* out_rids) {
    assert(n <= BufferPool::kPinBatchMax);
    if (n == 0) return;

    // Compact lookups: keep only nodes that resolve to a valid page.
    uint32_t  pages[BufferPool::kPinBatchMax];
    uint16_t  back_index[BufferPool::kPinBatchMax];
    size_t    k = 0;
    for (size_t i = 0; i < n; ++i) {
        RID rid;
        if (!lookup_node_rid(node_ids[i], rid) || rid.page_id == INVALID_PAGE) {
            out_frames[i] = PinnedFrame();
            out_rids[i]   = NodeRID();
            continue;
        }
        out_rids[i]      = rid;
        pages[k]         = rid.page_id;
        back_index[k]    = static_cast<uint16_t>(i);
        ++k;
    }

    if (k == 0) return;

    PinnedFrame frames[BufferPool::kPinBatchMax];
    pin_pages_batch(pages, k, mode, frames);
    for (size_t j = 0; j < k; ++j) {
        out_frames[back_index[j]] = std::move(frames[j]);
    }
}

void InPlaceGraphStore::init(uint32_t dim, uint32_t Mmax,
                             uint32_t elem_size_bytes, uint32_t page_size,
                             uint32_t buffer_pool_frames,
                             const std::string& heap_path,
                             uint32_t flush_budget_pages_per_cycle,
                             uint32_t flush_wakeup_ms,
                             bool truncate_heap,
                             uint32_t max_nodes) {
    _dim       = dim;
    _aligned_dim = (uint32_t)ROUND_UP(dim, 8);
    _Mmax      = Mmax;
    _elem_size = elem_size_bytes;
    _page_size = page_size;

    _slot_size = sizeof(PackedSlotHeader) +
                 _aligned_dim * _elem_size +
                 Mmax * sizeof(uint32_t);

    // Compute slots per page iteratively (page header + bitmap + slots)
    uint32_t avail = page_size - 8;  // page header is 8 bytes
    _slots_per_page = 0;
    for (uint32_t s = 1; s <= avail / _slot_size; s++) {
        uint32_t bmap = (s + 7) / 8;
        if (8 + bmap + s * _slot_size <= page_size) {
            _slots_per_page = s;
        }
    }
    if (_slots_per_page == 0) {
        throw ANNException("Slot size exceeds page size", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    _bp.init(page_size, buffer_pool_frames, heap_path, &_stats,
             flush_budget_pages_per_cycle, flush_wakeup_ms, truncate_heap);
    clear_pending_pq_codes();
    _n_chunks = 0;
    _max_nodes = max_nodes == 0 ? kDefaultNodeCapacity : max_nodes;
    _node_rids.reset(new RID[_max_nodes]);
    _node_tags.reset(new TagType[_max_nodes]);
    _node_states.reset(new std::atomic<NodeState>[_max_nodes]);
    _pq_codes.reset();
    reset_metadata();
    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        _entry_pool.fill(INVALID_NODE);
        _node_reservoir.fill(INVALID_NODE);
        _reservoir_cursor = 0;
    }
}

static inline uint32_t page_bitmap_bytes(uint32_t slots_per_page) {
    return (slots_per_page + 7) / 8;
}

static inline uint32_t page_header_bytes(uint32_t slots_per_page) {
    return 8 + page_bitmap_bytes(slots_per_page);
}

static inline void set_page_bitmap(char* page_data, uint16_t slot_idx) {
    uint8_t* bmap = reinterpret_cast<uint8_t*>(page_data + 8);
    bmap[slot_idx / 8] |= (1u << (slot_idx % 8));
}

static inline void clear_page_bitmap(char* page_data, uint16_t slot_idx) {
    uint8_t* bmap = reinterpret_cast<uint8_t*>(page_data + 8);
    bmap[slot_idx / 8] &= static_cast<uint8_t>(~(1u << (slot_idx % 8)));
}

static inline bool test_page_bitmap(const char* page_data, uint16_t slot_idx) {
    const uint8_t* bmap = reinterpret_cast<const uint8_t*>(page_data + 8);
    return ((bmap[slot_idx / 8] >> (slot_idx % 8)) & 1u) != 0;
}

// Find the first free slot in the page bitmap using CTZ (count-trailing-zeros).
// The bitmap is a uint8_t array at page offset +8, LSB = slot 0.
// Returns slots_per_page if no free slot found (invariant violation — caller crashes).
static uint16_t find_free_slot(const char* page_data, uint16_t slots_per_page) {
    const uint8_t* bmap = reinterpret_cast<const uint8_t*>(page_data + 8);
    uint16_t n_bytes = static_cast<uint16_t>((slots_per_page + 7) / 8);
    for (uint16_t b = 0; b < n_bytes; ++b) {
        uint8_t byte = bmap[b];
        if (byte != 0xFFu) {
            unsigned free_bits = static_cast<unsigned>(~byte) & 0xFFu;
            unsigned bit = static_cast<unsigned>(__builtin_ctz(free_bits));
            uint16_t slot = static_cast<uint16_t>(b * 8u + bit);
            if (slot < slots_per_page) return slot;
        }
    }
    return slots_per_page;
}

// Phase 1: claim a slot reservation on some page (round-robin across pages with space).
// Pre-increments _page_dir[pid].num_occupied to guarantee Phase 2 finds a free bitmap slot.
// Caller must hold _pages_mtx.
uint32_t InPlaceGraphStore::claim_page() {
    for (;;) {
        if (!_pages_with_space.empty()) {
            auto it = _pages_with_space.lower_bound(_alloc_page_rr_cursor);
            if (it == _pages_with_space.end()) it = _pages_with_space.begin();
            uint32_t pid = *it;
            PageDir& dir = _page_dir[pid];
            dir.num_occupied++;
            if (dir.num_occupied >= dir.slots_per_page) {
                _pages_with_space.erase(it);
            }
            _alloc_page_rr_cursor = pid + 1;
            return pid;
        }
        // Batch grow: _page_dir updated before _pages_with_space.
        uint32_t base = _bp.grow_heap(kPageAllocBatchSize);
        _page_dir.resize(base + kPageAllocBatchSize,
                         PageDir{static_cast<uint16_t>(_slots_per_page), 0});
        for (uint32_t i = base; i < base + kPageAllocBatchSize; ++i) {
            _pages_with_space.insert(_pages_with_space.end(), i);
        }
    }
}

// Phase 2: write the bitmap + slot header under the frame write lock.
// The slot is guaranteed to exist because Phase 1 pre-incremented num_occupied.
InPlaceGraphStore::RID InPlaceGraphStore::allocate_slot_on_page(uint32_t pid) {
    PinnedFrame frame = pin_page(pid, WRITE);
    auto guard = frame.write_guard();
    char* page_data = frame._frame->data;

    uint16_t slot = find_free_slot(page_data, static_cast<uint16_t>(_slots_per_page));
    if (slot >= static_cast<uint16_t>(_slots_per_page)) {
        throw ANNException("allocate_slot_on_page: bitmap invariant violated — no free slot on claimed page " +
                               std::to_string(pid),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    set_page_bitmap(page_data, slot);
    // Write slots_per_page to page header (idempotent; new pages are zero-filled by OS).
    uint16_t spp = static_cast<uint16_t>(_slots_per_page);
    memcpy(page_data, &spp, 2);
    frame.mark_dirty();
    return RID{pid, slot};
}

uint32_t InPlaceGraphStore::allocate_node() {
    const uint32_t node_id = reserve_internal_id();
    allocate_node(node_id);
    return node_id;
}

void InPlaceGraphStore::allocate_node(uint32_t node_id) {
    uint32_t pid;
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        pid = claim_page();
    }
    RID rid = allocate_slot_on_page(pid);
    materialize_node(node_id, rid);

    PinnedFrame frame = pin_page(rid.page_id, WRITE);
    auto guard = frame.write_guard();
    char* slot_ptr = frame._frame->data + page_header_bytes(_slots_per_page) +
                     rid.slot_idx * _slot_size;
    PackedSlotHeader hdr;
    hdr.degree  = 0;
    hdr.flags   = 0;
    hdr._pad    = 0;
    hdr.node_id = node_id;
    memcpy(slot_ptr, &hdr, sizeof(hdr));
    frame.mark_dirty();
}

void InPlaceGraphStore::allocate_nodes_batch(const uint32_t* node_ids, size_t count) {
    if (node_ids == nullptr || count == 0) return;
    struct AllocatedItem {
        uint32_t page_id;
        uint16_t slot_idx;
        uint32_t node_id;
    };
    // Phase 1: claim pages for all nodes under _pages_mtx.
    std::vector<std::pair<uint32_t, uint32_t>> page_claims(count); // {page_id, node_id}
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        for (size_t i = 0; i < count; ++i) {
            page_claims[i] = {claim_page(), node_ids[i]};
        }
    }
    // Phase 2: for each claim, find a free slot under the frame lock, then write metadata.
    std::vector<AllocatedItem> items;
    items.reserve(count);
    std::sort(page_claims.begin(), page_claims.end());
    size_t cursor = 0;
    while (cursor < page_claims.size()) {
        uint32_t page_id = page_claims[cursor].first;
        PinnedFrame frame = pin_page(page_id, WRITE);
        auto guard = frame.write_guard();
        char* page_data = frame._frame->data;
        while (cursor < page_claims.size() && page_claims[cursor].first == page_id) {
            uint32_t nid = page_claims[cursor].second;
            uint16_t slot = find_free_slot(page_data, static_cast<uint16_t>(_slots_per_page));
            if (slot >= static_cast<uint16_t>(_slots_per_page)) {
                throw ANNException("allocate_nodes_batch: bitmap invariant violated on page " +
                                       std::to_string(page_id),
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            set_page_bitmap(page_data, slot);
            uint16_t spp = static_cast<uint16_t>(_slots_per_page);
            memcpy(page_data, &spp, 2);
            items.push_back(AllocatedItem{page_id, slot, nid});
            ++cursor;
        }
        frame.mark_dirty();
    }
    // Register metadata (phmap handles concurrent inserts internally).
    for (const auto& item : items) {
        materialize_node(item.node_id, RID{item.page_id, item.slot_idx});
    }
    // Write slot headers sorted by page to batch I/O.
    cursor = 0;
    while (cursor < items.size()) {
        uint32_t page_id = items[cursor].page_id;
        PinnedFrame frame = pin_page(page_id, WRITE);
        auto guard = frame.write_guard();
        while (cursor < items.size() && items[cursor].page_id == page_id) {
            const auto& item = items[cursor];
            char* slot_ptr = frame._frame->data + page_header_bytes(_slots_per_page) +
                             item.slot_idx * _slot_size;
            PackedSlotHeader hdr;
            hdr.degree  = 0;
            hdr.flags   = 0;
            hdr._pad    = 0;
            hdr.node_id = item.node_id;
            memcpy(slot_ptr, &hdr, sizeof(hdr));
            ++cursor;
        }
        frame.mark_dirty();
    }
}

void InPlaceGraphStore::publish_node(uint32_t node_id) {
    if (node_present(node_id)) {
        NodeState prev =
            _node_states[node_id].exchange(NodeState::Active,
                                           std::memory_order_release);
        _node_rids[node_id].active.store(1, std::memory_order_release);
        if (prev != NodeState::Active) {
            _num_active.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::lock_guard<std::mutex> lk(_entry_mtx);
    _node_reservoir[_reservoir_cursor] = node_id;
    _reservoir_cursor = (_reservoir_cursor + 1) % kReservoirSize;
}

void InPlaceGraphStore::publish_nodes_batch(const uint32_t* node_ids, size_t count) {
    if (node_ids == nullptr || count == 0) return;
    for (size_t i = 0; i < count; ++i) {
        uint32_t node_id = node_ids[i];
        if (node_present(node_id)) {
            NodeState prev =
                _node_states[node_id].exchange(NodeState::Active,
                                               std::memory_order_release);
            _node_rids[node_id].active.store(1, std::memory_order_release);
            if (prev != NodeState::Active) {
                _num_active.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    std::lock_guard<std::mutex> lk(_entry_mtx);
    for (size_t i = 0; i < count; ++i) {
        _node_reservoir[_reservoir_cursor] = node_ids[i];
        _reservoir_cursor = (_reservoir_cursor + 1) % kReservoirSize;
    }
}

void InPlaceGraphStore::protect_seed_pages(const std::vector<uint32_t>& node_ids, uint8_t credit) {
    if (credit == 0) return;
    tsl::robin_set<uint32_t> seen_pages;
    seen_pages.reserve(node_ids.size());
    for (uint32_t node_id : node_ids) {
        RID rid;
        if (!lookup_node_rid(node_id, rid) || rid.page_id == INVALID_PAGE) continue;
        if (!seen_pages.insert(rid.page_id).second) continue;
        _bp.protect_page(rid.page_id, credit, FrameRegion::QUERY);
    }
}

void InPlaceGraphStore::batch_fetch_coords(const std::vector<uint32_t>& ids,
                                           char* out_coords,
                                           std::vector<uint8_t>& found,
                                           bool transient) {
    found.assign(ids.size(), 0);
    if (ids.empty() || out_coords == nullptr) return;

    struct BatchItem {
        uint32_t page_id;
        uint16_t slot_idx;
        uint32_t node_id;
        size_t out_idx;
    };

    std::vector<BatchItem> items;
    items.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t node_id = ids[i];
        RID rid;
        if (!lookup_node_rid(node_id, rid) || rid.page_id == INVALID_PAGE) continue;
        items.push_back(BatchItem{rid.page_id, rid.slot_idx, node_id, i});
    }
    std::sort(items.begin(), items.end(), [](const BatchItem& a, const BatchItem& b) {
        if (a.page_id != b.page_id) return a.page_id < b.page_id;
        return a.slot_idx < b.slot_idx;
    });

    size_t cursor = 0;
    while (cursor < items.size()) {
        uint32_t page_id = items[cursor].page_id;
        PinnedFrame frame = transient ? pin_page_transient(page_id, READ)
                                      : pin_page(page_id, READ);
        if (!frame.valid()) {
            while (cursor < items.size() && items[cursor].page_id == page_id) {
                ++cursor;
            }
            continue;
        }
        auto guard = frame.read_guard();

        while (cursor < items.size() && items[cursor].page_id == page_id) {
            const auto& item = items[cursor];
            ConstNodeRef node = guard.read_node(item.slot_idx);
            if (node.valid() &&
                node.node_id() == item.node_id &&
                is_active(item.node_id)) {
                const char* coords_ptr = node.coords_bytes();
                std::memcpy(
                    out_coords + item.out_idx * static_cast<size_t>(_aligned_dim) * _elem_size,
                    coords_ptr, static_cast<size_t>(_aligned_dim) * _elem_size);
                found[item.out_idx] = 1;
            }
            ++cursor;
        }
    }
}

bool InPlaceGraphStore::transition_active_to_delete_pending(uint32_t node_id) {
    if (!node_present(node_id)) return false;
    NodeState expected = NodeState::Active;
    if (!_node_states[node_id].compare_exchange_strong(
            expected, NodeState::DeletePending,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    _node_rids[node_id].active.store(0, std::memory_order_release);
    _num_active.fetch_sub(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        _delete_pending_since_maintenance.push_back(node_id);
    }
    _stats.total_deletes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void InPlaceGraphStore::get_entry_points(std::vector<uint32_t>& out) {
    out.clear();
    std::lock_guard<std::mutex> lk(_entry_mtx);
    for (size_t i = 0; i < kEntryPoolSize; ++i) {
        uint32_t node = _entry_pool[i];
        if (node == INVALID_NODE) continue;
        if (is_active(node)) {
            out.push_back(node);
        } else {
            // Lazy repair: find a live replacement from the reservoir.
            uint32_t replacement = INVALID_NODE;
            for (size_t step = 0; step < kReservoirSize; ++step) {
                size_t idx = (_reservoir_cursor + step) % kReservoirSize;
                uint32_t cand = _node_reservoir[idx];
                if (cand != INVALID_NODE && is_active(cand)) {
                    replacement = cand;
                    _reservoir_cursor = (idx + 1) % kReservoirSize;
                    break;
                }
            }
            _entry_pool[i] = replacement;
            if (replacement != INVALID_NODE) out.push_back(replacement);
        }
    }
}

void InPlaceGraphStore::seed_entry_pool_from_reservoir() {
    std::lock_guard<std::mutex> lk(_entry_mtx);
    // Fill slots 1..kEntryPoolSize-1 with random samples from the reservoir.
    // Slot 0 is reserved for the medoid (set via set_entry_point).
    size_t stride = std::max<size_t>(1, kReservoirSize / kEntryPoolSize);
    for (size_t i = 1; i < kEntryPoolSize; ++i) {
        size_t idx = ((i - 1) * stride) % kReservoirSize;
        uint32_t cand = _node_reservoir[idx];
        if (cand != INVALID_NODE && is_active(cand)) {
            _entry_pool[i] = cand;
        }
    }
}

bool InPlaceGraphStore::is_active(uint32_t node_id) const {
    return node_present(node_id) &&
           _node_states[node_id].load(std::memory_order_acquire) ==
               NodeState::Active;
}

bool InPlaceGraphStore::is_delete_pending(uint32_t node_id) const {
    return node_present(node_id) &&
           _node_states[node_id].load(std::memory_order_acquire) ==
               NodeState::DeletePending;
}

bool InPlaceGraphStore::set_node_tag(uint32_t node_id, TagType tag) {
    if (!node_present(node_id) || tag == INVALID_TAG) return false;
    if (is_tag_deleted(tag)) return false;
    _node_tags[node_id] = tag;
    return true;
}

TagType InPlaceGraphStore::node_tag(uint32_t node_id) const {
    return node_present(node_id) ? _node_tags[node_id] : INVALID_TAG;
}

bool InPlaceGraphStore::mark_tag_deleted(TagType tag) {
    if (tag == INVALID_TAG) return false;
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    return _deleted_tags_current.emplace(tag).second;
}

bool InPlaceGraphStore::is_tag_deleted(TagType tag) const {
    if (tag == INVALID_TAG) return false;
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    return _deleted_tags_current.contains(tag) ||
           _deleted_tags_draining.contains(tag);
}

size_t InPlaceGraphStore::deleted_tag_count() const {
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    return _deleted_tags_current.size() + _deleted_tags_draining.size();
}

std::vector<uint32_t> InPlaceGraphStore::drain_deleted_tags_to_delete_pending() {
    auto elapsed_s = [](const std::chrono::steady_clock::time_point& begin) {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin).count();
    };

    size_t drained_tag_count = 0;
    {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        if (!_deleted_tags_current.empty()) {
            _deleted_tags_draining.clear();
            _deleted_tags_current.swap(_deleted_tags_draining);
            drained_tag_count = _deleted_tags_draining.size();
        }
    }

    const auto resolution_begin = std::chrono::steady_clock::now();
    std::cout << "[BufANN] cleanup_deleted_edges(): started tag resolution"
              << " deleted_tags=" << drained_tag_count
              << " max_nodes=" << _max_nodes << std::endl;

    std::vector<uint32_t> resolved_ids;
    resolved_ids.reserve(drained_tag_count);
    if (drained_tag_count != 0) {
        for (size_t nid = 0; nid < _max_nodes; ++nid) {
            const uint32_t node_id = static_cast<uint32_t>(nid);
            if (!node_present(node_id)) continue;
            if (_node_states[node_id].load(std::memory_order_acquire) !=
                NodeState::Active) {
                continue;
            }
            if (!_deleted_tags_draining.contains(_node_tags[node_id])) continue;
            resolved_ids.push_back(node_id);
        }
    }

    std::cout << "[BufANN] cleanup_deleted_edges(): finished tag resolution"
              << " elapsed_s=" << elapsed_s(resolution_begin)
              << " resolved_ids=" << resolved_ids.size() << std::endl;

    const auto tombstone_begin = std::chrono::steady_clock::now();
    std::cout << "[BufANN] cleanup_deleted_edges(): started tombstone marking"
              << " resolved_ids=" << resolved_ids.size() << std::endl;

    std::vector<uint32_t> delete_pending_ids;
    delete_pending_ids.reserve(resolved_ids.size());
    for (uint32_t node_id : resolved_ids) {
        if (transition_active_to_delete_pending(node_id)) {
            delete_pending_ids.push_back(node_id);
        }
    }

    std::cout << "[BufANN] cleanup_deleted_edges(): finished tombstone marking"
              << " elapsed_s=" << elapsed_s(tombstone_begin)
              << " marked_ids=" << delete_pending_ids.size() << std::endl;

    if (drained_tag_count != 0) {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        _deleted_tags_draining.clear();
    }
    return delete_pending_ids;
}

uint32_t InPlaceGraphStore::get_page_id(uint32_t node_id) const {
    RID rid;
    if (!lookup_node_rid(node_id, rid)) return INVALID_PAGE;
    return rid.page_id;
}

uint32_t InPlaceGraphStore::page_pin_count(uint32_t page_id) const {
    return _bp.pin_count_of(page_id);
}

uint32_t InPlaceGraphStore::num_active() const {
    return _num_active.load(std::memory_order_relaxed);
}

// PQ support
void InPlaceGraphStore::load_pq_codes_from_disk_index(const std::string& pq_prefix,
                                                       uint32_t num_chunks) {
    std::string codes_path  = pq_prefix + "_pq_compressed.bin";

    // PQ files are generated by bufann_build after init_store is called, so
    // they may not exist yet during the build phase.  Skip silently; they will
    // be present when the index is loaded for search via bufann_load.
    if (access(codes_path.c_str(), F_OK) != 0) {
        diskann::cout << "InPlace: PQ code file not found at prefix " << pq_prefix
                      << ", skipping PQ code load." << std::endl;
        return;
    }

    if (_n_chunks != 0 && _n_chunks != num_chunks) {
        throw ANNException("PQ chunk mismatch while reloading: requested " +
                               std::to_string(num_chunks) + ", existing " +
                               std::to_string(_n_chunks),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    _n_chunks = num_chunks;
    ensure_pq_storage();
    clear_pending_pq_codes();

    int fd = open(codes_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw ANNException("Unable to open PQ code file: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw ANNException("Unable to stat PQ code file: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (st.st_size < static_cast<off_t>(2 * sizeof(int32_t))) {
        close(fd);
        throw ANNException("Invalid PQ code file header: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    void* mapping = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                         MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        throw ANNException("Unable to mmap PQ code file: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint8_t* mapped_bytes = static_cast<const uint8_t*>(mapping);
    int32_t nr_i32 = 0, nc_i32 = 0;
    memcpy(&nr_i32, mapped_bytes, sizeof(int32_t));
    memcpy(&nc_i32, mapped_bytes + sizeof(int32_t), sizeof(int32_t));
    if (nr_i32 < 0 || nc_i32 < 0) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("Invalid PQ code file header: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t nr = static_cast<size_t>(nr_i32);
    const size_t nc = static_cast<size_t>(nc_i32);
    diskann::cout << "Reading PQ codes " << codes_path << " ..." << std::endl
                  << "Metadata: #pts = " << nr << ", #dims = " << nc
                  << "..." << std::endl;
    if (nc != num_chunks) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code chunk mismatch while loading " + codes_path,
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (nr > 0 &&
        static_cast<size_t>(nr) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(_n_chunks)) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code table too large to stage in memory", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t payload_bytes = nr * static_cast<size_t>(_n_chunks);
    const size_t payload_offset = 2 * sizeof(int32_t);
    if (payload_bytes >
        std::numeric_limits<size_t>::max() - payload_offset) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code table too large to map", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (static_cast<uint64_t>(st.st_size) <
        static_cast<uint64_t>(payload_offset + payload_bytes)) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code file is truncated: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    _pending_pq_code_count = static_cast<size_t>(nr);
    _pending_pq_mmap_base = mapping;
    _pending_pq_mmap_bytes = static_cast<size_t>(st.st_size);
    _pending_pq_codes = mapped_bytes + payload_offset;

    diskann::cout << "InPlace: loaded PQ codes, n=" << nr
                  << ", n_chunks=" << _n_chunks << std::endl;
}

void InPlaceGraphStore::compute_pq_dists_query(const unsigned* ids,
                                                uint64_t n_ids,
                                                const float* pq_dists_in,
                                                float* dists_out) {
    if (_n_chunks == 0) return;
    std::vector<uint8_t> scratch((size_t)n_ids * _n_chunks, 0);
    for (uint64_t i = 0; i < n_ids; ++i) {
        const uint8_t* code = pq_codes_for_node(ids[i]);
        if (code != nullptr) {
            memcpy(scratch.data() + i * _n_chunks, code, _n_chunks);
        }
    }
    pq_dist_lookup(scratch.data(), n_ids, _n_chunks, pq_dists_in, dists_out);
}

template<typename T>
void InPlaceGraphStore::compute_pq_dists_src(uint32_t src,
                                             const unsigned* ids,
                                             float* dists_out,
                                             uint32_t count,
                                             uint8_t* scratch,
                                             FixedChunkPQTable<T>& pq_table) {
    if (_n_chunks == 0) return;
    const uint8_t* src_ptr = pq_codes_for_node(src);
    if (src_ptr == nullptr) return;
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* code = pq_codes_for_node(ids[i]);
        if (code != nullptr) {
            memcpy(scratch + i * _n_chunks, code, _n_chunks);
        } else {
            memset(scratch + i * _n_chunks, 0, _n_chunks);
        }
    }
    pq_table.compute_distances(src_ptr, scratch, dists_out, count);
}

template<typename T>
void InPlaceGraphStore::encode_pq(uint32_t node_id, const T* coords,
                                  FixedChunkPQTable<T>& pq_table) {
    if (_n_chunks == 0) return;
    if (!node_present(node_id)) return;
    ensure_pq_storage();
    uint8_t* codes = _pq_codes.get() + static_cast<size_t>(node_id) * _n_chunks;
    pq_table.deflate_vec(coords, codes);
    _node_rids[node_id].pq_ready.store(1, std::memory_order_release);
}

void InPlaceGraphStore::gather_pq_codes(const unsigned* ids, uint64_t n, uint8_t* out) const {
    if (_n_chunks == 0) return;
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* code = pq_codes_for_node(ids[i]);
        if (code != nullptr) {
            memcpy(out + i * _n_chunks, code, _n_chunks);
        } else {
            memset(out + i * _n_chunks, 0, _n_chunks);
        }
    }
}

const uint8_t* InPlaceGraphStore::pq_codes_for_node(uint32_t node_id) const {
    if (node_present(node_id) && _pq_codes != nullptr &&
        _node_rids[node_id].pq_ready.load(std::memory_order_acquire) != 0) {
        return _pq_codes.get() + static_cast<size_t>(node_id) * _n_chunks;
    }
    return pending_pq_codes(node_id);
}

uint32_t InPlaceGraphStore::dirty_page_count() const {
    return _bp.dirty_page_count();
}

template<typename T, typename TagT>
void InPlaceGraphStore::bulk_load_from_index(diskann::Index<T, TagT>& mem_index,
                                              uint32_t n) {
    const auto* graph = mem_index.get_graph();
    T*          data  = mem_index.get_data();

    for (uint32_t i = 0; i < n; i++) {
        allocate_node(i);
        NodeRID rid;
        PinnedFrame frame = pin_page_for_node(i, WRITE, &rid);
        if (!frame.valid()) continue;
        auto guard = frame.write_guard();
        MutableNodeRef node = guard.write_node(rid);
        if (!node.valid()) continue;
        node.set_coords<T>(data + static_cast<size_t>(_aligned_dim) * i);
        const auto& nbrs = (*graph)[i];
        node.set_neighbors(nbrs.data(), nbrs.size());
        publish_node(i);
    }
    diskann::cout << "InPlace: bulk loaded " << n << " nodes" << std::endl;
}

// explicit instantiations
template void InPlaceGraphStore::bulk_load_from_index<float, uint32_t>(
    diskann::Index<float, uint32_t>&, uint32_t);
template void InPlaceGraphStore::bulk_load_from_index<uint8_t, uint32_t>(
    diskann::Index<uint8_t, uint32_t>&, uint32_t);
template void InPlaceGraphStore::bulk_load_from_index<int8_t, uint32_t>(
    diskann::Index<int8_t, uint32_t>&, uint32_t);

void InPlaceGraphStore::warmup_bfs(uint32_t entry_point, uint32_t num_nodes) {
    if (num_nodes == 0) return;
    tsl::robin_set<uint32_t> visited;
    std::queue<uint32_t> q;
    q.push(entry_point);
    visited.insert(entry_point);
    uint32_t count = 0;
    while (!q.empty() && count < num_nodes) {
        uint32_t cur = q.front();
        q.pop();
        NodeRID rid;
        PinnedFrame frame = pin_page_for_node(cur, READ, &rid);
        if (!frame.valid()) continue;
        auto guard = frame.read_guard();
        ConstNodeRef node = guard.read_node(rid);
        if (!node.valid()) continue;
        const uint32_t* nbrs = node.neighbors();
        for (uint16_t i = 0; i < node.degree(); i++) {
            uint32_t nbr = nbrs[i];
            if (visited.find(nbr) == visited.end()) {
                visited.insert(nbr);
                q.push(nbr);
            }
        }
        count++;
    }
    diskann::cout << "InPlace: warmed up " << count << " nodes" << std::endl;
}

void InPlaceGraphStore::warmup_bfs_pages(uint32_t entry_point, uint32_t num_pages,
                                         uint32_t num_threads) {
    if (num_pages == 0) return;
    NodeRID seed_rid;
    if (!lookup_node_rid(entry_point, seed_rid) || seed_rid.page_id == INVALID_PAGE) {
        return;
    }
    const int threads = static_cast<int>(std::max<uint32_t>(1, num_threads));

    // True node-based BFS from the entry point: the frontier holds graph
    // node IDs, and pages enter the buffer pool as a side effect of pinning
    // each node's host page. visited_nodes dedups the BFS expansion;
    // visited_pages tracks the actual buffer-pool side-effect and serves as
    // the termination budget.
    tsl::robin_set<uint32_t> visited_nodes;
    tsl::robin_set<uint32_t> visited_pages;
    visited_nodes.insert(entry_point);
    std::vector<uint32_t> current_frontier{entry_point};
    std::mt19937 urng{std::random_device{}()};
    uint32_t hop = 0;

    while (!current_frontier.empty() && visited_pages.size() < num_pages) {
        std::shuffle(current_frontier.begin(), current_frontier.end(), urng);

        const uint32_t remaining =
            static_cast<uint32_t>(num_pages - visited_pages.size());
        // When the frontier is larger than the remaining page budget, this is
        // the last hop: parallel-pin only the first `remaining` nodes (since
        // node->page is many-to-one, this pins <= remaining novel pages —
        // typically an undershoot), then sequentially tail-pin the rest of
        // the frontier until the budget is hit. We don't build a next
        // frontier in this case.
        const bool last_hop = (current_frontier.size() > remaining);
        const size_t parallel_count =
            last_hop ? remaining : current_frontier.size();

        // Per-thread accumulators; merged sequentially after the parallel
        // section to avoid locks in the hot loop.
        std::vector<std::vector<uint32_t>> tls_pages(threads);
        std::vector<std::vector<uint32_t>> tls_nbrs(threads);
        const int64_t n = static_cast<int64_t>(parallel_count);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 8)
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t node_id = current_frontier[i];
            NodeRID rid;
            PinnedFrame frame = pin_page_for_node(node_id, READ, &rid);
            if (!frame.valid()) continue;
            const int tid = omp_get_thread_num();
            tls_pages[tid].push_back(rid.page_id);
            if (last_hop) continue;  // skip neighbor scan on last hop
            auto guard = frame.read_guard();
            ConstNodeRef node = guard.read_node(rid);
            if (!node.valid()) continue;
            const uint32_t* nbrs = node.neighbors();
            const uint16_t deg = node.degree();
            auto& local = tls_nbrs[tid];
            for (uint16_t k = 0; k < deg; ++k) {
                local.push_back(nbrs[k]);
            }
        }

        // Merge pinned pages from the parallel section.
        for (auto& local : tls_pages) {
            for (uint32_t pid : local) {
                visited_pages.insert(pid);
            }
        }

        // Last-hop tail pin: parallel top-up to the page budget. Atomic
        // flag for early exit once full; mutex on the shared visited_pages.
        uint32_t tail_pinned = 0;
        if (last_hop && visited_pages.size() < num_pages) {
            std::mutex pages_mtx;
            std::atomic<bool> budget_full{false};
            std::atomic<uint32_t> tail_pinned_atomic{0};
            const int64_t tail_start = static_cast<int64_t>(parallel_count);
            const int64_t tail_end = static_cast<int64_t>(current_frontier.size());
#pragma omp parallel for num_threads(threads) schedule(dynamic, 8)
            for (int64_t i = tail_start; i < tail_end; ++i) {
                if (budget_full.load(std::memory_order_relaxed)) continue;
                NodeRID rid;
                PinnedFrame frame =
                    pin_page_for_node(current_frontier[i], READ, &rid);
                if (!frame.valid()) continue;
                std::lock_guard<std::mutex> lk(pages_mtx);
                if (visited_pages.size() >= num_pages) {
                    budget_full.store(true, std::memory_order_relaxed);
                    continue;
                }
                if (visited_pages.insert(rid.page_id).second) {
                    tail_pinned_atomic.fetch_add(1, std::memory_order_relaxed);
                    if (visited_pages.size() >= num_pages) {
                        budget_full.store(true, std::memory_order_relaxed);
                    }
                }
            }
            tail_pinned = tail_pinned_atomic.load(std::memory_order_relaxed);
        }

        // Build the next frontier by deduping collected neighbors against
        // visited_nodes. Skipped on last hop.
        std::vector<uint32_t> next_frontier;
        if (!last_hop) {
            for (auto& local : tls_nbrs) {
                for (uint32_t nbr : local) {
                    if (visited_nodes.insert(nbr).second) {
                        next_frontier.push_back(nbr);
                    }
                }
            }
        }

        diskann::cout << "InPlace BFS warmup: hop=" << hop
                      << " nodes=" << current_frontier.size()
                      << " parallel=" << parallel_count
                      << " tail=" << tail_pinned
                      << " pages=" << visited_pages.size()
                      << " next_frontier=" << next_frontier.size() << std::endl;
        ++hop;
        current_frontier.swap(next_frontier);
    }
    diskann::cout << "InPlace: BFS warmed up " << visited_pages.size()
                  << " pages over " << hop << " hops" << std::endl;
}

void InPlaceGraphStore::reset_stats() {
    tls_registry().drain_into(_stats);
    _stats.reset();
}

InPlaceIOStats& InPlaceGraphStore::stats() {
    tls_registry().drain_into(_stats);
    return _stats;
}

void InPlaceGraphStore::preload_query_hot_pages(const std::vector<uint32_t>& seed_nodes,
                                                uint32_t page_budget,
                                                uint32_t node_budget,
                                                uint8_t protect_credit) {
    if (seed_nodes.empty() || page_budget == 0 || node_budget == 0) return;

    tsl::robin_set<uint32_t> visited_nodes;
    tsl::robin_set<uint32_t> warmed_pages;
    std::queue<uint32_t> q;
    visited_nodes.reserve(node_budget * 2);
    warmed_pages.reserve(page_budget * 2);

    for (uint32_t seed : seed_nodes) {
        if (!is_active(seed)) continue;
        if (visited_nodes.insert(seed).second) {
            q.push(seed);
        }
    }

    uint32_t visited_count = 0;
    while (!q.empty() && visited_count < node_budget && warmed_pages.size() < page_budget) {
        uint32_t cur = q.front();
        q.pop();
        NodeRID rid;
        PinnedFrame frame = pin_page_for_node(cur, READ, &rid);
        if (!frame.valid()) continue;
        auto guard = frame.read_guard();
        ConstNodeRef node = guard.read_node(rid);
        if (!node.valid()) continue;
        if (warmed_pages.insert(frame.page_id()).second) {
            _bp.protect_page(frame.page_id(), protect_credit, FrameRegion::QUERY);
        }
        ++visited_count;
        const uint32_t* nbrs = node.neighbors();
        for (uint16_t i = 0; i < node.degree() && visited_nodes.size() < node_budget; ++i) {
            uint32_t nbr = nbrs[i];
            if (!is_active(nbr)) continue;
            if (visited_nodes.insert(nbr).second) {
                q.push(nbr);
            }
        }
    }

    diskann::cout << "InPlace: preloaded " << warmed_pages.size()
                  << " hot query pages from " << seed_nodes.size()
                  << " seed nodes" << std::endl;
}

void InPlaceGraphStore::preload_pages(uint32_t max_pages,
                                      uint32_t num_threads) {
    const uint32_t total = total_pages();
    const uint32_t limit = max_pages == 0 ? total : std::min(total, max_pages);
    if (limit == 0) return;

    const int threads = static_cast<int>(std::max<uint32_t>(1, num_threads));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t page_id = 0; page_id < static_cast<int64_t>(limit); ++page_id) {
        auto& frame = _bp.pin(static_cast<uint32_t>(page_id), FrameRegion::QUERY);
        _bp.unpin_frame(&frame, false);
    }
}

void InPlaceGraphStore::start_bg_flush(float high_wm, float low_wm) { _bp.start_bg_flush(high_wm, low_wm); }
void InPlaceGraphStore::stop_bg_flush()  { _bp.stop_bg_flush(); }
void InPlaceGraphStore::flush()          { _bp.flush_all_dirty(); }
uint32_t InPlaceGraphStore::flush_dirty_budget(uint32_t max_pages) {
    return _bp.flush_dirty_budget(max_pages);
}
double InPlaceGraphStore::dirty_ratio() const { return _bp.dirty_ratio(); }
void InPlaceGraphStore::set_bg_flush_budget_pages_per_cycle(uint32_t flush_budget_pages_per_cycle) {
    _bp.set_flush_budget_pages_per_cycle(flush_budget_pages_per_cycle);
}
void InPlaceGraphStore::set_entry_point(uint32_t entry_point) {
    if (!is_active(entry_point)) return;
    std::lock_guard<std::mutex> lk(_entry_mtx);
    _entry_pool[0] = entry_point;
}

void InPlaceGraphStore::save_snapshot(const std::string& meta_path) {
    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t _reserved;     // unused, was entry_point
        uint32_t max_nodes;     // unused
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
        uint64_t page_dir_size;
        uint64_t pages_with_space_size;
        uint64_t reservoir_cursor;
        uint64_t _reserved2;    // unused, was entry_rr_cursor
    } hdr{};

    if (deleted_tag_count() != 0) {
        throw ANNException("save_snapshot: pending deleted tags exist; run maintenance before saving",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    memcpy(hdr.magic, "IPGSNP4", 8);
    hdr.version = 4;
    hdr.dim = _dim;
    hdr.aligned_dim = _aligned_dim;
    hdr.Mmax = _Mmax;
    hdr.elem_size = _elem_size;
    hdr.page_size = _page_size;
    hdr.slot_size = _slot_size;
    hdr.slots_per_page = _slots_per_page;
    hdr.n_chunks = _n_chunks;
    hdr.num_active = _num_active.load(std::memory_order_relaxed);
    hdr.total_pages = _bp.total_pages();
    hdr.rid_size = 0;
    hdr.active_cap = 0;
    // save_snapshot is called at quiesce, so no synchronization is needed.
    {
        for (size_t nid = 0; nid < _max_nodes; ++nid) {
            if (!node_present(static_cast<uint32_t>(nid)) ||
                nid == INVALID_NODE) continue;
            hdr.active_cap = std::max<uint64_t>(
                hdr.active_cap, static_cast<uint64_t>(nid) + 1);
            ++hdr.rid_size;
        }
    }
    hdr.page_dir_size = _page_dir.size();
    hdr.pages_with_space_size = _pages_with_space.size();
    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        hdr.reservoir_cursor = static_cast<uint64_t>(_reservoir_cursor);
    }

    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw ANNException("Failed to open snapshot file: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    // Serialize RID map as a sequence of (node_id, RID) pairs.
    {
        for (size_t nid = 0; nid < _max_nodes; ++nid) {
            if (!node_present(static_cast<uint32_t>(nid)) ||
                nid == INVALID_NODE) continue;
            const uint32_t id = static_cast<uint32_t>(nid);
            out.write(reinterpret_cast<const char*>(&id), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&_node_rids[nid]), sizeof(RID));
        }
    }
    std::vector<uint8_t> active_flags(static_cast<size_t>(hdr.active_cap), 0);
    {
        for (size_t nid = 0; nid < _max_nodes && nid < active_flags.size(); ++nid) {
            if (!node_present(static_cast<uint32_t>(nid)) ||
                nid == INVALID_NODE) continue;
            active_flags[nid] =
                _node_states[nid].load(std::memory_order_relaxed) ==
                    NodeState::Active ? 1 : 0;
        }
    }
    out.write(reinterpret_cast<const char*>(active_flags.data()),
              static_cast<std::streamsize>(active_flags.size()));
    std::vector<TagType> tags(static_cast<size_t>(hdr.active_cap), INVALID_TAG);
    {
        for (size_t nid = 0; nid < _max_nodes && nid < tags.size(); ++nid) {
            if (!node_present(static_cast<uint32_t>(nid)) ||
                nid == INVALID_NODE) continue;
            tags[nid] = _node_tags[nid];
        }
    }
    out.write(reinterpret_cast<const char*>(tags.data()),
              static_cast<std::streamsize>(tags.size() * sizeof(TagType)));
    out.write(reinterpret_cast<const char*>(_page_dir.data()),
              static_cast<std::streamsize>(_page_dir.size() * sizeof(PageDir)));
    for (uint32_t pid : _pages_with_space) {
        out.write(reinterpret_cast<const char*>(&pid), sizeof(pid));
    }
    // Entry pool and reservoir (fixed-size arrays).
    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        out.write(reinterpret_cast<const char*>(_entry_pool.data()),
                  static_cast<std::streamsize>(kEntryPoolSize * sizeof(uint32_t)));
        out.write(reinterpret_cast<const char*>(_node_reservoir.data()),
                  static_cast<std::streamsize>(kReservoirSize * sizeof(uint32_t)));
    }
    // Staged PQ table is redundant once snapshot is written; drop to cut RSS.
    clear_pending_pq_codes();
}

void InPlaceGraphStore::load_snapshot(const std::string& meta_path) {
    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t _reserved;
        uint32_t max_nodes;
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
        uint64_t page_dir_size;
        uint64_t pages_with_space_size;
        uint64_t reservoir_cursor;
        uint64_t _reserved2;
    } hdr{};

    std::ifstream in(meta_path, std::ios::binary);
    if (!in) {
        throw ANNException("Failed to open snapshot file: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    const bool is_v2 = in && memcmp(hdr.magic, "IPGSNP2", 8) == 0 && hdr.version == 2;
    const bool is_v3 = in && memcmp(hdr.magic, "IPGSNP3", 8) == 0 && hdr.version == 3;
    const bool is_v4 = in && memcmp(hdr.magic, "IPGSNP4", 8) == 0 && hdr.version == 4;
    if (!in || (!is_v2 && !is_v3 && !is_v4)) {
        throw ANNException("Invalid snapshot header (expected IPGSNP4 v4, IPGSNP3 v3, or IPGSNP2 v2): " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (hdr.dim != _dim || hdr.page_size != _page_size || hdr.Mmax != _Mmax ||
        hdr.elem_size != _elem_size || hdr.slot_size != _slot_size ||
        hdr.slots_per_page != _slots_per_page) {
        std::ostringstream oss;
        oss << "Snapshot layout mismatch: " << meta_path
            << " [snapshot vs runtime]"
            << " dim=" << hdr.dim << "/" << _dim
            << " page_size=" << hdr.page_size << "/" << _page_size
            << " Mmax=" << hdr.Mmax << "/" << _Mmax
            << " elem_size=" << hdr.elem_size << "/" << _elem_size
            << " slot_size=" << hdr.slot_size << "/" << _slot_size
            << " slots_per_page=" << hdr.slots_per_page << "/" << _slots_per_page;
        throw ANNException(oss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    // Keep runtime PQ mode from current configuration/load path.  We do not
    // auto-enable PQ from snapshot metadata so NoPQ stays zero-overhead.
    // NOTE: _num_active is restored *after* reset_metadata() below, because
    // reset_metadata() zeros it and the materialize_node loop does not
    // increment it back. Setting it here gets silently clobbered.
    _bp.set_total_pages(hdr.total_pages);
    std::vector<std::pair<uint32_t, RID>> rid_entries;
    rid_entries.reserve(static_cast<size_t>(hdr.rid_size));
    for (uint64_t i = 0; i < hdr.rid_size; ++i) {
        uint32_t node_id = 0;
        RID rid{INVALID_PAGE, 0};
        in.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        in.read(reinterpret_cast<char*>(&rid), sizeof(rid));
        if (rid.page_id != INVALID_PAGE) {
            rid_entries.emplace_back(node_id, rid);
        }
    }

    std::vector<uint8_t> active_flags(static_cast<size_t>(hdr.active_cap), 0);
    if (!active_flags.empty()) {
        in.read(reinterpret_cast<char*>(active_flags.data()),
                static_cast<std::streamsize>(active_flags.size()));
    }
    std::vector<TagType> tags(static_cast<size_t>(hdr.active_cap), INVALID_TAG);
    if (is_v4 && !tags.empty()) {
        in.read(reinterpret_cast<char*>(tags.data()),
                static_cast<std::streamsize>(tags.size() * sizeof(TagType)));
    } else {
        for (size_t nid = 0; nid < tags.size(); ++nid) {
            tags[nid] = static_cast<TagType>(nid);
        }
    }
    {
        size_t inactive_count = 0;
        uint32_t first_inactive = INVALID_NODE;
        const size_t af_size = active_flags.size();
        for (const auto& kv : rid_entries) {
            const bool active = kv.first < af_size && active_flags[kv.first] != 0;
            if (active) continue;
            if (first_inactive == INVALID_NODE) {
                first_inactive = kv.first;
            }
            ++inactive_count;
        }
        if (inactive_count != 0) {
            std::ostringstream oss;
            oss << "Snapshot contains " << inactive_count
                << " inactive/delete-pending node entries; first node_id="
                << first_inactive
                << ". Loading snapshots with pending deletes is unsupported; "
                << "run maintenance before saving.";
            throw ANNException(oss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }
    diskann::cout << "InPlace: start metadata array reserve: active_cap = " << hdr.active_cap << std::endl;
    reset_metadata(static_cast<size_t>(hdr.active_cap));
    diskann::cout << "InPlace: done metadata array reserve" << std::endl;
    const size_t n_rid = rid_entries.size();
    const size_t af_size = active_flags.size();
    diskann::cout << "InPlace: start of metadata insertion (" << n_rid
                  << " entries, " << omp_get_max_threads() << " threads)"
                  << std::endl;
    auto t_start = std::chrono::steady_clock::now();
    std::atomic<size_t> progress_counter{0};
    const size_t one_percent = std::max<size_t>(1, n_rid / 100);
    auto t_last_pct = t_start;
#pragma omp parallel for schedule(static, 65536)
    for (size_t idx = 0; idx < n_rid; ++idx) {
        const auto& kv = rid_entries[idx];
        materialize_node(kv.first, kv.second);
        uint8_t active = kv.first < af_size ? active_flags[kv.first] : 0;
        _node_rids[kv.first].active.store(active, std::memory_order_relaxed);
        _node_states[kv.first].store(active != 0 ? NodeState::Active
                                                 : NodeState::DeletePending,
                                     std::memory_order_relaxed);
        const TagType tag = kv.first < tags.size()
                                ? tags[kv.first]
                                : static_cast<TagType>(kv.first);
        _node_tags[kv.first] = tag;
        size_t done = progress_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % one_percent == 0) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - t_last_pct).count();
            t_last_pct = now;
            double mops = (dt > 0) ? (one_percent / dt / 1e6) : 0;
            fprintf(stderr, "\rInPlace: metadata insert %zu%% (%.1f Mop/s)        ",
                    done / one_percent, mops);
        }
    }
    fprintf(stderr, "\n");
    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    diskann::cout << "InPlace: end of metadata insertion; inserted " << n_rid
                  << " elements in " << total_s << "s" << std::endl;
    // Restore num_active now that reset_metadata-induced zeroing is past.
    _num_active.store(hdr.num_active, std::memory_order_relaxed);
    if (_n_chunks > 0) {
        diskann::cout << "InPlace: PQ codes materialized per-node, freeing flat buffer ("
                      << _pending_pq_code_count << " entries, "
                      << (_pending_pq_code_count * _n_chunks / (1024*1024)) << " MB)"
                      << std::endl;
        clear_pending_pq_codes();
    }
    _page_dir.assign(static_cast<size_t>(hdr.page_dir_size), PageDir{0, 0});
    if (!_page_dir.empty()) {
        in.read(reinterpret_cast<char*>(_page_dir.data()),
                static_cast<std::streamsize>(_page_dir.size() * sizeof(PageDir)));
    }
    _pages_with_space.clear();
    for (uint64_t i = 0; i < hdr.pages_with_space_size; ++i) {
        uint32_t pid = 0;
        in.read(reinterpret_cast<char*>(&pid), sizeof(pid));
        _pages_with_space.insert(pid);
    }
    _alloc_page_rr_cursor = 0;

    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        if (is_v2) {
            // v2 stored: single entry_point in hdr._reserved, then a variable-length
            // _entry_candidates array (count in hdr.reservoir_cursor, which was
            // entry_candidates_count in v2).  Reconstruct the pool and reservoir.
            _entry_pool.fill(INVALID_NODE);
            _node_reservoir.fill(INVALID_NODE);
            _reservoir_cursor = 0;
            _entry_pool[0] = hdr._reserved;  // was entry_point
            uint64_t cand_count = hdr.reservoir_cursor;  // was entry_candidates_count
            for (uint64_t i = 0; i < cand_count; ++i) {
                uint32_t cand = INVALID_NODE;
                in.read(reinterpret_cast<char*>(&cand), sizeof(cand));
                if (i < kReservoirSize) _node_reservoir[i] = cand;
            }
            _reservoir_cursor = std::min<size_t>(static_cast<size_t>(cand_count), kReservoirSize);
        } else {
            _reservoir_cursor = static_cast<size_t>(hdr.reservoir_cursor);
            in.read(reinterpret_cast<char*>(_entry_pool.data()),
                    static_cast<std::streamsize>(kEntryPoolSize * sizeof(uint32_t)));
            in.read(reinterpret_cast<char*>(_node_reservoir.data()),
                    static_cast<std::streamsize>(kReservoirSize * sizeof(uint32_t)));
        }
    }
    if (is_v2) seed_entry_pool_from_reservoir();
}

size_t InPlaceGraphStore::memory_usage_bytes() const {
    size_t rss = 0;
    std::ifstream f("/proc/self/statm");
    if (f.is_open()) {
        size_t dummy;
        f >> dummy >> rss;
        rss *= sysconf(_SC_PAGESIZE);
    }
    return rss;
}

size_t InPlaceGraphStore::buffer_pool_bytes() const {
    return (size_t)_bp.num_frames() * _bp.page_size();
}

size_t InPlaceGraphStore::locator_bytes() const {
    return _max_nodes * (sizeof(RID) + sizeof(TagType) +
                         sizeof(std::atomic<NodeState>));
}

size_t InPlaceGraphStore::pq_data_bytes() const {
    if (_n_chunks == 0) return 0;

    const size_t attached =
        _pq_codes == nullptr ? 0 : _max_nodes * static_cast<size_t>(_n_chunks);
    size_t staged = (_pending_pq_codes == nullptr) ? 0 :
                    _pending_pq_code_count * static_cast<size_t>(_n_chunks);
    return attached + staged;
}

template void InPlaceGraphStore::encode_pq<float>(
    uint32_t, const float*, FixedChunkPQTable<float>&);
template void InPlaceGraphStore::encode_pq<uint8_t>(
    uint32_t, const uint8_t*, FixedChunkPQTable<uint8_t>&);
template void InPlaceGraphStore::encode_pq<int8_t>(
    uint32_t, const int8_t*, FixedChunkPQTable<int8_t>&);
template void InPlaceGraphStore::compute_pq_dists_src<float>(
    uint32_t, const unsigned*, float*, uint32_t, uint8_t*,
    FixedChunkPQTable<float>&);
template void InPlaceGraphStore::compute_pq_dists_src<uint8_t>(
    uint32_t, const unsigned*, float*, uint32_t, uint8_t*,
    FixedChunkPQTable<uint8_t>&);
template void InPlaceGraphStore::compute_pq_dists_src<int8_t>(
    uint32_t, const unsigned*, float*, uint32_t, uint8_t*,
    FixedChunkPQTable<int8_t>&);

}
}
