// Tests for RawVectorHeap: layout, allocate/write/read/free/reuse, concurrent
// bitmap updates, slot-space limit, the reopen guard, and the bulk writer.

#include "bufann/ivf_pq_raw_vector_heap.h"
#include "ivf_pq_test_util.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t PAGE = 4096;
const uint32_t ELEM = 512;  // dim=128 fp32: 7 slots per page, 1 bitmap byte

std::vector<char> pattern(uint32_t i, uint32_t elem_size = ELEM) {
    return std::vector<char>(elem_size, static_cast<char>(i % 251 + 1));
}

std::vector<char> read_whole_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool test_layout() {
    TestCase t("layout matches the design-doc example and is maximal for every geometry");
    RawVectorHeapLayout l = compute_raw_vector_heap_layout(PAGE, ELEM);
    t.check(l.slots_per_page == 7 && l.bitmap_bytes == 1 && l.slots_offset == 9,
            "expected 7 slots, 1 bitmap byte, slots at offset 9");

    // Whatever the geometry, the page holds the slots and one more would not fit.
    for (uint32_t page : {64u, 512u, 4096u, 16384u}) {
        for (uint32_t elem = 1; elem + RAW_VECTOR_PAGE_HEADER_BYTES + 1 <= page; elem += 7) {
            RawVectorHeapLayout g = compute_raw_vector_heap_layout(page, elem);
            auto bytes = [&](uint32_t slots) {
                return uint64_t(RAW_VECTOR_PAGE_HEADER_BYTES) + (slots + 7) / 8 + uint64_t(slots) * elem;
            };
            if (!t.check(g.slots_per_page > 0 && bytes(g.slots_per_page) <= page &&
                             bytes(g.slots_per_page + 1) > page &&
                             g.bitmap_bytes == (g.slots_per_page + 7) / 8 &&
                             g.slots_offset == RAW_VECTOR_PAGE_HEADER_BYTES + g.bitmap_bytes,
                         "page " + std::to_string(page) + " elem " + std::to_string(elem) +
                             " gave " + std::to_string(g.slots_per_page) + " slots")) {
                break;
            }
        }
    }
    t.expect_throw("one slot + header + bitmap byte exceeds the page",
                   [] { compute_raw_vector_heap_layout(PAGE, PAGE - 8); });
    t.expect_throw("elem_size == 0", [] { compute_raw_vector_heap_layout(PAGE, 0); });
    return t.done();
}

// Pins the on-disk format: page header zero, bitmap bit i (LSB first) for
// slot i, slot i at slots_offset + i * elem_size, second page at page_size.
bool test_on_disk_format() {
    TestCase t("on-disk page format: header, LSB-first bitmap, slot placement");
    std::string path = temp_path("ivf_heap_format");
    RawVectorHeap heap;
    heap.open(path, compute_raw_vector_heap_layout(PAGE, ELEM));
    RawVectorFreeList free_list;
    for (uint32_t i = 0; i < 8; ++i) heap.allocate_slot(free_list);  // pages 0 and 1
    heap.write_vector(0, pattern(10).data());
    heap.write_vector(2, pattern(12).data());
    heap.write_vector(7, pattern(17).data());  // page 1, index 0
    heap.close();

    std::vector<char> f = read_whole_file(path);
    auto slot_is = [&](size_t offset, uint32_t i) {
        std::vector<char> want = pattern(i);
        return offset + ELEM <= f.size() && std::equal(want.begin(), want.end(), f.begin() + offset);
    };
    t.check(f.size() == 2 * PAGE, "file is not exactly two pages");
    t.check(std::all_of(f.begin(), f.begin() + 8, [](char c) { return c == 0; }), "page header not zero");
    t.check(uint8_t(f[8]) == 0b00000101, "page 0 bitmap is not bits 0 and 2");
    t.check(slot_is(9, 10), "slot 0 not at offset 9");
    t.check(slot_is(9 + 2 * ELEM, 12), "slot 2 not at offset 9 + 2 * elem_size");
    t.check(uint8_t(f[PAGE + 8]) == 0b00000001, "page 1 bitmap is not bit 0");
    t.check(slot_is(PAGE + 9, 17), "slot 7 not at page 1 offset 9");
    ::unlink(path.c_str());
    return t.done();
}

bool test_allocate_write_read_free_reuse() {
    TestCase t("allocate/write/read/free/reuse across multiple pages");
    std::string path = temp_path("ivf_heap_basic");
    RawVectorHeap heap;
    heap.open(path, compute_raw_vector_heap_layout(PAGE, ELEM));
    RawVectorFreeList free_list;

    const uint32_t n = 50;
    std::vector<uint32_t> slots(n);
    for (uint32_t i = 0; i < n; ++i) {
        slots[i] = heap.allocate_slot(free_list);
        heap.write_vector(slots[i], pattern(i).data());
    }
    std::vector<char> got(ELEM);
    for (uint32_t i = 0; i < n; ++i) {
        heap.read_vector(slots[i], got.data());
        t.check(got == pattern(i) && heap.is_slot_occupied(slots[i]),
                "vector " + std::to_string(i) + " did not read back");
    }

    // Free every other slot; reallocation must hand back exactly those.
    std::vector<uint32_t> freed, reused;
    for (uint32_t i = 0; i < n; i += 2) {
        heap.free_slot(slots[i], free_list);
        freed.push_back(slots[i]);
        t.check(!heap.is_slot_occupied(slots[i]), "freed slot still occupied");
    }
    for (size_t i = 0; i < freed.size(); ++i) {
        reused.push_back(heap.allocate_slot(free_list));
    }
    std::sort(freed.begin(), freed.end());
    std::sort(reused.begin(), reused.end());
    t.check(reused == freed, "reallocation did not reuse the freed slots");

    heap.write_vector(reused[0], pattern(999).data());
    heap.read_vector(reused[0], got.data());
    t.check(got == pattern(999), "reused slot did not round-trip after refill");

    heap.close();
    ::unlink(path.c_str());
    return t.done();
}

// All 7 slots of a page share one bitmap byte; unsynchronized updates drop bits.
bool test_concurrent_bitmap_updates() {
    TestCase t("concurrent writes and frees within one page keep every occupancy bit");
    std::string path = temp_path("ivf_heap_conc");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    std::vector<char> buf(ELEM, 'z');

    for (int run = 0; run < 200; ++run) {
        ::unlink(path.c_str());
        RawVectorHeap heap;
        heap.open(path, layout);
        RawVectorFreeList free_list;
        std::vector<uint32_t> slots(layout.slots_per_page);
        for (uint32_t& s : slots) {
            s = heap.allocate_slot(free_list);
        }

        std::vector<std::thread> threads;
        for (uint32_t s : slots) {
            threads.emplace_back([&, s] { heap.write_vector(s, buf.data()); });
        }
        for (auto& th : threads) th.join();
        bool all_set = std::all_of(slots.begin(), slots.end(),
                                   [&](uint32_t s) { return heap.is_slot_occupied(s); });

        threads.clear();
        for (uint32_t s : slots) {
            threads.emplace_back([&, s] { heap.free_slot(s, free_list); });
        }
        for (auto& th : threads) th.join();
        bool all_clear = std::none_of(slots.begin(), slots.end(),
                                      [&](uint32_t s) { return heap.is_slot_occupied(s); });

        heap.close();
        if (!t.check(all_set && all_clear && free_list.free_slots.size() == slots.size(),
                     "run " + std::to_string(run) + " lost a bitmap update or a freed slot")) {
            break;
        }
    }
    ::unlink(path.c_str());
    return t.done();
}

bool test_rejects_slots_past_the_rid_slot_space() {
    TestCase t("allocate_slot hands out the last addressable slot, then refuses");
    std::string path = temp_path("ivf_heap_full");
    RawVectorHeap heap;
    heap.open(path, compute_raw_vector_heap_layout(PAGE, PAGE - 9));  // 1 slot per page
    RawVectorFreeList free_list;

    // Page count already past the cursor, so neither call extends the file.
    heap.restore_slot_cursor(RAW_VECTOR_RID_SLOT_MASK, RAW_VECTOR_RID_SLOT_MASK + 1u);
    t.check(heap.allocate_slot(free_list) == RAW_VECTOR_RID_SLOT_MASK,
            "expected the last addressable slot");
    t.expect_throw("allocating past the 31-bit slot space", [&] { heap.allocate_slot(free_list); });

    heap.close();
    ::unlink(path.c_str());
    return t.done();
}

bool test_open_refuses_existing_nonempty_file() {
    TestCase t("open() refuses to reopen an existing non-empty heap");
    std::string path = temp_path("ivf_heap_reopen");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    RawVectorHeap heap;
    heap.open(path, layout);
    RawVectorFreeList free_list;
    heap.write_vector(heap.allocate_slot(free_list), pattern(1).data());
    heap.close();

    RawVectorHeap again;
    t.expect_throw("reopening a non-empty heap", [&] { again.open(path, layout); });
    ::unlink(path.c_str());
    return t.done();
}

// Bulk-loads `n` vectors and requires the file to be byte-identical to one
// built with allocate_slot/write_vector, with the cursor in the same place.
void check_bulk_load(TestCase& t, const std::string& tag, uint32_t n, uint32_t pages_per_flush) {
    std::string bulk_path = temp_path("ivf_heap_bulk"), slot_path = temp_path("ivf_heap_slot");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    auto fail = [&](const std::string& msg) { t.check(false, "[" + tag + "] " + msg); };

    RawVectorHeap bulk;
    bulk.open(bulk_path, layout);
    {
        RawVectorHeapBulkWriter writer(bulk, pages_per_flush);
        for (uint32_t i = 0; i < n; ++i) {
            if (writer.append(pattern(i).data()) != i) fail("vector " + std::to_string(i) + " not in slot i");
        }
        writer.finish();
    }

    RawVectorHeap per_slot;
    per_slot.open(slot_path, layout);
    RawVectorFreeList unused;
    for (uint32_t i = 0; i < n; ++i) {
        per_slot.write_vector(per_slot.allocate_slot(unused), pattern(i).data());
    }

    if (bulk.next_flat_slot() != per_slot.next_flat_slot() ||
        bulk.allocated_pages() != per_slot.allocated_pages()) {
        fail("cursor differs from per-slot allocation");
    }
    std::vector<char> bulk_bytes = read_whole_file(bulk_path);
    if (bulk_bytes.size() != size_t(bulk.allocated_pages()) * PAGE) fail("file size is not whole pages");
    if (bulk_bytes != read_whole_file(slot_path)) fail("file differs from per-slot writes");

    // Per-slot allocation continues where the load ended and leaves it intact.
    if (n > 0) {
        RawVectorFreeList free_list;
        uint32_t next = bulk.allocate_slot(free_list);
        if (next != n) fail("allocate_slot after load returned " + std::to_string(next));
        bulk.write_vector(next, pattern(999).data());
        std::vector<char> got(ELEM);
        bulk.read_vector(n - 1, got.data());
        if (got != pattern(n - 1)) fail("post-load write clobbered the last bulk slot");
    }

    bulk.close();
    per_slot.close();
    ::unlink(bulk_path.c_str());
    ::unlink(slot_path.c_str());
}

bool test_bulk_writer_matches_per_slot_writes() {
    TestCase t("bulk writer produces the same file as per-slot writes");
    const uint32_t spp = 7, ppf = 3;
    check_bulk_load(t, "empty", 0, ppf);
    check_bulk_load(t, "one slot", 1, ppf);
    check_bulk_load(t, "one full page", spp, ppf);
    check_bulk_load(t, "one full buffer", spp * ppf, ppf);
    check_bulk_load(t, "one buffer + partial page", spp * ppf + 3, ppf);
    check_bulk_load(t, "four buffers exactly", 4 * spp * ppf, ppf);
    check_bulk_load(t, "four buffers + full page", 4 * spp * ppf + spp, ppf);
    check_bulk_load(t, "four buffers + partial page", 4 * spp * ppf + 4, ppf);
    check_bulk_load(t, "single-page flushes", 5 * spp + 2, 1);
    return t.done();
}

bool test_bulk_writer_rejects_misuse() {
    TestCase t("bulk writer rejects zero flush size, append after finish, and a used heap");
    std::string path = temp_path("ivf_heap_misuse");
    RawVectorHeap heap;
    heap.open(path, compute_raw_vector_heap_layout(PAGE, ELEM));

    t.expect_throw("pages_per_flush == 0", [&] { RawVectorHeapBulkWriter w(heap, 0); });

    RawVectorHeapBulkWriter writer(heap, 2);
    writer.append(pattern(0).data());
    writer.finish();
    writer.finish();  // idempotent
    t.expect_throw("append after finish", [&] { writer.append(pattern(1).data()); });
    t.check(heap.next_flat_slot() == 1, "cursor moved by a rejected append");
    t.expect_throw("bulk writer on a non-empty heap", [&] { RawVectorHeapBulkWriter w(heap, 2); });

    heap.close();
    ::unlink(path.c_str());
    return t.done();
}

}  // namespace

int main() {
    bool all_pass = true;
    all_pass &= test_layout();
    all_pass &= test_on_disk_format();
    all_pass &= test_allocate_write_read_free_reuse();
    all_pass &= test_concurrent_bitmap_updates();
    all_pass &= test_rejects_slots_past_the_rid_slot_space();
    all_pass &= test_open_refuses_existing_nonempty_file();
    all_pass &= test_bulk_writer_matches_per_slot_writes();
    all_pass &= test_bulk_writer_rejects_misuse();
    return all_pass ? 0 : 1;
}
