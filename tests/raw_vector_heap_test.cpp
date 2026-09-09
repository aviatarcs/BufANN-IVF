// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Smoke test for the IVF-PQ raw-vector heap storage primitive:
// allocate/write/read/free/reuse over RawVectorHeap + RawVectorFreeList.

#include "bufann/raw_vector_heap.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
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
    std::string path = "/tmp/raw_vector_heap_test_" + std::to_string((uint64_t) getpid()) + ".bin";

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

    // Free every other slot, then confirm reallocation reuses exactly those
    // flat slots (LIFO) rather than growing the heap further.
    std::vector<uint32_t> freed;
    for (uint32_t i = 0; i < num_vectors; i += 2) {
        heap.free_slot(flat_slots[i]);
        free_list.free_slots.push_back(flat_slots[i]);
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

}  // namespace

int main() {
    bool all_pass = true;
    all_pass &= test_layout_matches_design_doc_example();
    all_pass &= test_allocate_write_read_free_reuse();
    return all_pass ? 0 : 1;
}
