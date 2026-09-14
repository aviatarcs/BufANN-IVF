// Tests for RawVectorHeap: layout, allocate/write/read/free/reuse, concurrent
// bitmap updates, slot-space limit, and the reopen guard.

#include "bufann/ivf_pq_raw_vector_heap.h"
#include "ann_exception.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace diskann::inplace;

namespace {

bool test_layout_matches_design_doc_example() {
    std::cout << "[Test] layout for page_size=4096, elem_size=512 (dim=128 fp32)..." << std::endl;
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, 512);
    bool pass = layout.slots_per_page == 7 && layout.bitmap_bytes == 1 &&
                layout.slots_offset == 9 &&
                (layout.page_size - layout.slots_offset - layout.slots_per_page * layout.elem_size) == 503;
    std::cout << "  slots_per_page=" << layout.slots_per_page
              << " bitmap_bytes=" << layout.bitmap_bytes
              << " slots_offset=" << layout.slots_offset << std::endl;
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool test_allocate_write_read_free_reuse() {
    std::cout << "[Test] allocate/write/read/free/reuse across multiple pages..." << std::endl;
    std::string path = "/tmp/ivf_pq_raw_vector_heap_test_" + std::to_string((uint64_t) getpid()) + ".bin";

    const uint32_t elem_size = 512;
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, elem_size);

    RawVectorHeap heap;
    heap.open(path, layout);
    RawVectorFreeList free_list;

    const uint32_t num_vectors = 50;  // spans multiple pages at 7 slots/page
    std::vector<std::vector<char>> expected(num_vectors, std::vector<char>(elem_size));
    std::vector<uint32_t> flat_slots(num_vectors);

    bool pass = true;
    for (uint32_t i = 0; i < num_vectors; ++i) {
        std::memset(expected[i].data(), static_cast<int>(i + 1), elem_size);
        flat_slots[i] = heap.allocate_slot(free_list);
        heap.write_vector(flat_slots[i], expected[i].data());
    }

    for (uint32_t i = 0; i < num_vectors; ++i) {
        std::vector<char> got(elem_size);
        heap.read_vector(flat_slots[i], got.data());
        if (got != expected[i]) {
            std::cout << "  FAIL: mismatch reading vector " << i << std::endl;
            pass = false;
        }
    }

    // Free every other slot; reallocation must reuse exactly those.
    std::vector<uint32_t> freed;
    for (uint32_t i = 0; i < num_vectors; i += 2) {
        heap.free_slot(flat_slots[i], free_list);
        freed.push_back(flat_slots[i]);
    }

    std::vector<uint32_t> reused;
    for (size_t i = 0; i < freed.size(); ++i) {
        reused.push_back(heap.allocate_slot(free_list));
    }
    std::sort(reused.begin(), reused.end());
    std::vector<uint32_t> freed_sorted = freed;
    std::sort(freed_sorted.begin(), freed_sorted.end());
    if (reused != freed_sorted) {
        std::cout << "  FAIL: reallocation did not reuse freed slots" << std::endl;
        pass = false;
    }

    // A freshly reallocated slot must be writable/readable again.
    std::vector<char> refill(elem_size, static_cast<char>(0x7F));
    heap.write_vector(reused[0], refill.data());
    std::vector<char> got(elem_size);
    heap.read_vector(reused[0], got.data());
    if (got != refill) {
        std::cout << "  FAIL: reused slot did not round-trip after refill" << std::endl;
        pass = false;
    }

    heap.close();
    ::unlink(path.c_str());

    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// All slots of a page share one bitmap byte at this geometry; unsynchronized
// updates would drop bits.
bool test_concurrent_writes_to_one_page_keep_every_occupancy_bit() {
    std::cout << "[Test] concurrent writes into one page keep all occupancy bits..." << std::endl;
    std::string path =
        "/tmp/ivf_pq_raw_vector_heap_test_conc_" + std::to_string((uint64_t) getpid()) + ".bin";

    const uint32_t elem_size = 512;
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, elem_size);

    bool pass = true;
    const int NUM_RUNS = 200;
    for (int run = 0; run < NUM_RUNS && pass; ++run) {
        ::unlink(path.c_str());
        RawVectorHeap heap;
        heap.open(path, layout);
        RawVectorFreeList free_list;

        // One page's worth of slots, written concurrently.
        std::vector<uint32_t> slots;
        for (uint32_t i = 0; i < layout.slots_per_page; ++i) {
            slots.push_back(heap.allocate_slot(free_list));
        }
        std::vector<char> buf(elem_size, 'z');
        std::vector<std::thread> writers;
        for (uint32_t slot : slots) {
            writers.emplace_back([&heap, slot, &buf] { heap.write_vector(slot, buf.data()); });
        }
        for (auto& t : writers) t.join();

        for (uint32_t slot : slots) {
            if (!heap.is_slot_occupied(slot)) {
                std::cout << "  FAIL: run " << run << " lost the occupancy bit for slot "
                          << slot << std::endl;
                pass = false;
            }
        }

        // Concurrent frees must clear every bit too.
        std::vector<std::thread> freers;
        for (uint32_t slot : slots) {
            freers.emplace_back([&heap, slot, &free_list] { heap.free_slot(slot, free_list); });
        }
        for (auto& t : freers) t.join();

        for (uint32_t slot : slots) {
            if (heap.is_slot_occupied(slot)) {
                std::cout << "  FAIL: run " << run << " left slot " << slot
                          << " marked occupied after a concurrent free" << std::endl;
                pass = false;
            }
        }
        if (free_list.free_slots.size() != slots.size()) {
            std::cout << "  FAIL: run " << run << " free list holds "
                      << free_list.free_slots.size() << " of " << slots.size()
                      << " freed slots" << std::endl;
            pass = false;
        }
        heap.close();
    }
    ::unlink(path.c_str());

    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool test_rejects_slots_past_the_rid_slot_space() {
    std::cout << "[Test] allocate_slot refuses to hand out an unaddressable slot..." << std::endl;
    std::string path =
        "/tmp/ivf_pq_raw_vector_heap_test_full_" + std::to_string((uint64_t) getpid()) + ".bin";
    ::unlink(path.c_str());

    // Park the cursor on the last addressable slot with the page count already
    // past it, so neither allocation extends the file.
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, 4096 - 9);
    RawVectorHeap heap;
    heap.open(path, layout);
    RawVectorFreeList free_list;
    heap.restore_slot_cursor(RAW_VECTOR_RID_SLOT_MASK, RAW_VECTOR_RID_SLOT_MASK + 1u);

    bool last_ok = false;
    bool threw = false;
    try {
        uint32_t last = heap.allocate_slot(free_list);  // the last addressable slot
        last_ok = (last == RAW_VECTOR_RID_SLOT_MASK);
        if (!last_ok) {
            std::cout << "  FAIL: expected the last addressable slot, got " << last << std::endl;
        }
        heap.allocate_slot(free_list);  // one past it -- must throw
        std::cout << "  FAIL: handed out a slot that RawVectorRID cannot address" << std::endl;
    } catch (const diskann::ANNException&) {
        threw = true;
    }
    bool pass = last_ok && threw;

    heap.close();
    ::unlink(path.c_str());
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool test_open_refuses_existing_nonempty_file() {
    std::cout << "[Test] open() refuses to reopen an existing non-empty heap..." << std::endl;
    std::string path = "/tmp/ivf_pq_raw_vector_heap_test_reopen_" + std::to_string((uint64_t) getpid()) + ".bin";
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(4096, 512);

    RawVectorHeap heap;
    heap.open(path, layout);
    RawVectorFreeList free_list;
    heap.write_vector(heap.allocate_slot(free_list), std::vector<char>(512, 'x').data());
    heap.close();

    bool pass = false;
    RawVectorHeap heap2;
    try {
        heap2.open(path, layout);
        std::cout << "  FAIL: open() silently truncated an existing non-empty heap" << std::endl;
    } catch (const diskann::ANNException&) {
        pass = true;
    }

    ::unlink(path.c_str());
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

}  // namespace

int main() {
    bool all_pass = true;
    all_pass &= test_layout_matches_design_doc_example();
    all_pass &= test_allocate_write_read_free_reuse();
    all_pass &= test_concurrent_writes_to_one_page_keep_every_occupancy_bit();
    all_pass &= test_rejects_slots_past_the_rid_slot_space();
    all_pass &= test_open_refuses_existing_nonempty_file();
    return all_pass ? 0 : 1;
}
