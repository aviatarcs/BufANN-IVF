// Tests for RawVectorHeap: layout, on-disk format, allocate/write/read/free/
// reuse, concurrent bitmap updates, slot-space limit, the fresh-open guard,
// reopening a closed heap (and refusing a bad one or a wrong geometry),
// page-header verification on read, and the bulk writer.

#include "bufann/ivf_pq_raw_vector_heap.h"
#include "ivf_pq_test_util.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
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
    t.check(l.slots_per_page == 7 && l.bitmap_bytes == 1 && l.slots_offset == 17,
            "expected 7 slots, 1 bitmap byte, slots at offset 17");

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
                   [] { compute_raw_vector_heap_layout(PAGE, PAGE - RAW_VECTOR_PAGE_HEADER_BYTES); });
    t.expect_throw("elem_size == 0", [] { compute_raw_vector_heap_layout(PAGE, 0); });
    return t.done();
}

// Little-endian bytes the header of page `id` should carry: "IVFH", the id,
// then the page_size and elem_size the page was written with.
std::vector<char> page_header_bytes(uint32_t id, uint32_t page_size = PAGE, uint32_t elem_size = ELEM) {
    std::vector<char> h = {'I', 'V', 'F', 'H'};
    for (uint32_t v : {id, page_size, elem_size}) {
        for (int i = 0; i < 4; ++i) h.push_back(char((v >> (8 * i)) & 0xFF));
    }
    return h;
}

bool page_header_is(const std::vector<char>& f, uint32_t page_id) {
    std::vector<char> want = page_header_bytes(page_id);
    size_t offset = size_t(page_id) * PAGE;
    return offset + want.size() <= f.size() && std::equal(want.begin(), want.end(), f.begin() + offset);
}

// Pins the on-disk format: page header = magic + page id + page_size +
// elem_size, bitmap bit i (LSB first) for slot i, slot i at
// slots_offset + i * elem_size, second page at page_size.
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
    t.check(page_header_is(f, 0), "page 0 header is not magic + id 0 + geometry");
    t.check(uint8_t(f[16]) == 0b00000101, "page 0 bitmap is not bits 0 and 2");
    t.check(slot_is(17, 10), "slot 0 not at offset 17");
    t.check(slot_is(17 + 2 * ELEM, 12), "slot 2 not at offset 17 + 2 * elem_size");
    t.check(page_header_is(f, 1), "page 1 header is not magic + id 1 + geometry");
    t.check(uint8_t(f[PAGE + 16]) == 0b00000001, "page 1 bitmap is not bit 0");
    t.check(slot_is(PAGE + 17, 17), "slot 7 not at page 1 offset 17");
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
    heap.open(path, compute_raw_vector_heap_layout(PAGE, PAGE - RAW_VECTOR_PAGE_HEADER_BYTES - 1));  // 1 slot per page
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

void patch_bytes(const std::string& path, uint64_t offset, const std::vector<char>& bytes) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(std::streamoff(offset));
    f.write(bytes.data(), std::streamsize(bytes.size()));
}

// Writes vectors 0..n-1 (pattern(i) in slot i) into a fresh heap at `path`
// through per-slot allocation and closes it.
void write_closed_heap(const std::string& path, const RawVectorHeapLayout& layout, uint32_t n) {
    ::unlink(path.c_str());
    RawVectorHeap heap;
    heap.open(path, layout);
    RawVectorFreeList free_list;
    for (uint32_t i = 0; i < n; ++i) {
        heap.write_vector(heap.allocate_slot(free_list), pattern(i).data());
    }
    heap.close();
}

bool test_reopen_resumes_the_heap() {
    TestCase t("open_existing reads every vector back, restores the cursor, and allocation continues");
    std::string path = temp_path("ivf_heap_resume");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    const uint32_t spp = layout.slots_per_page;
    const uint32_t n = 5 * spp + 3;  // six pages, the last one partial
    {
        RawVectorHeap heap;
        heap.open(path, layout);
        RawVectorHeapBulkWriter writer(heap, 2);
        for (uint32_t i = 0; i < n; ++i) writer.append(pattern(i).data());
        writer.finish();
    }
    std::vector<char> before = read_whole_file(path);

    RawVectorHeap heap;
    heap.open_existing(path, layout, n, 6);
    t.check(heap.next_flat_slot() == n && heap.allocated_pages() == 6, "cursor not restored");
    std::vector<char> got(ELEM);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        heap.read_vector(i, got.data());
        bad += got != pattern(i) || !heap.is_slot_occupied(i);
    }
    t.check(bad == 0, std::to_string(bad) + " vectors did not read back after reopen");
    t.check(!heap.is_slot_occupied(n), "slot past the cursor reads as occupied");
    t.check(read_whole_file(path) == before, "reopening changed the file");

    // Allocation fills the partial page first, then grows a page with its own header.
    RawVectorFreeList free_list;
    const uint32_t m = 6 * spp + 1;
    for (uint32_t i = n; i < m; ++i) {
        if (heap.allocate_slot(free_list) != i) t.check(false, "allocation after reopen skipped slot " + std::to_string(i));
        heap.write_vector(i, pattern(i).data());
    }
    t.check(heap.allocated_pages() == 7, "growth after reopen did not add exactly one page");
    heap.close();
    std::vector<char> f = read_whole_file(path);
    t.check(f.size() == 7 * PAGE && page_header_is(f, 6), "page grown after reopen lacks its header");

    heap.open_existing(path, layout, m, 7);
    heap.read_vector(m - 1, got.data());
    t.check(got == pattern(m - 1), "vector written after the first reopen did not survive a second");
    heap.close();
    ::unlink(path.c_str());
    return t.done();
}

bool test_reopen_rejects_bad_file() {
    TestCase t("open_existing refuses a missing file, a wrong size, a bad cursor, and foreign page headers");
    std::string path = temp_path("ivf_heap_resume_bad");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    const uint32_t spp = layout.slots_per_page, pages = 4, n = pages * spp;
    write_closed_heap(path, layout, n);
    std::vector<char> good = read_whole_file(path);
    RawVectorHeap heap;

    t.expect_throw("missing file", [&] { heap.open_existing(path + ".missing", layout, n, pages); });
    t.expect_throw("cursor past the pages", [&] { heap.open_existing(path, layout, n + 1, pages); });
    t.expect_throw("fewer pages than the file", [&] { heap.open_existing(path, layout, n - spp, pages - 1); });
    t.expect_throw("more pages than the file", [&] { heap.open_existing(path, layout, n, pages + 1); });
    RawVectorHeapLayout inconsistent = layout;
    inconsistent.slots_per_page += 1;
    t.expect_throw("inconsistent layout", [&] { heap.open_existing(path, inconsistent, n, pages); });

    // Same byte count as two double-size pages, but the header at the second
    // double-size page is page 2's, not page 1's.
    RawVectorHeapLayout doubled = compute_raw_vector_heap_layout(2 * PAGE, ELEM);
    t.expect_throw("different page_size", [&] { heap.open_existing(path, doubled, 0, 2); });
    // Geometries under which the file's size and its first and last page ids
    // both line up, so only the geometry in the page header tells them apart:
    // the whole file as one page (first == last == page 0), and the same
    // pages holding elements twice the size.
    RawVectorHeapLayout one_page = compute_raw_vector_heap_layout(pages * PAGE, ELEM);
    t.expect_throw("whole file as a single page", [&] { heap.open_existing(path, one_page, 0, 1); });
    RawVectorHeapLayout wider = compute_raw_vector_heap_layout(PAGE, 2 * ELEM);
    t.expect_throw("same pages, different elem_size",
                   [&] { heap.open_existing(path, wider, pages * wider.slots_per_page, pages); });

    t.expect_throw("page 0 without a header", [&] {
        patch_bytes(path, 0, std::vector<char>(RAW_VECTOR_PAGE_HEADER_BYTES, 0));
        heap.open_existing(path, layout, n, pages);
    });
    t.expect_throw("page 0 recording another elem_size", [&] {
        patch_bytes(path, 0, page_header_bytes(0, PAGE, ELEM + 4));
        heap.open_existing(path, layout, n, pages);
    });
    patch_bytes(path, 0, page_header_bytes(0));
    t.expect_throw("last page carrying another page's id", [&] {
        patch_bytes(path, size_t(pages - 1) * PAGE, page_header_bytes(pages));
        heap.open_existing(path, layout, n, pages);
    });
    patch_bytes(path, size_t(pages - 1) * PAGE, page_header_bytes(pages - 1));

    t.expect_throw("truncated by a byte", [&] {
        ::truncate(path.c_str(), off_t(good.size() - 1));
        heap.open_existing(path, layout, n, pages);
    });
    t.expect_throw("extended by a byte", [&] {
        ::truncate(path.c_str(), off_t(good.size() + 1));
        heap.open_existing(path, layout, n, pages);
    });
    ::truncate(path.c_str(), off_t(good.size()));
    patch_bytes(path, good.size() - 1, {good.back()});

    // A rejected open leaves nothing allocated; the restored file opens.
    t.check(heap.allocated_pages() == 0 && heap.next_flat_slot() == 0, "rejected open left a cursor");
    t.check(read_whole_file(path) == good, "corruptions were not undone");
    heap.open_existing(path, layout, n, pages);
    std::vector<char> got(ELEM);
    heap.read_vector(n - 1, got.data());
    t.check(got == pattern(n - 1), "restored file does not read back");
    heap.close();
    ::unlink(path.c_str());
    return t.done();
}

// A heap that was open before must not carry its cursor through close() or
// through a reopen that is rejected, whether before the file is opened
// (missing path) or after (bad page header). Either way allocate_slot must
// throw instead of handing out the old file's next slot, and a rejected
// reopen must not touch the file it looked at.
bool test_rejected_reopen_after_use_leaves_nothing_behind() {
    TestCase t("close() and a rejected open_existing drop the cursor of the heap that was open before");
    std::string path = temp_path("ivf_heap_resume_stale");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    const uint32_t spp = layout.slots_per_page, pages = 3, n = 2 * spp + 3;  // last page partial
    write_closed_heap(path, layout, n);
    std::vector<char> good = read_whole_file(path);
    RawVectorFreeList free_list;

    RawVectorHeap heap;
    heap.open_existing(path, layout, n, pages);
    t.check(heap.next_flat_slot() == n && heap.allocated_pages() == pages, "cursor not restored");
    heap.close();
    t.check(heap.next_flat_slot() == 0 && heap.allocated_pages() == 0, "close() left the cursor");

    t.expect_throw("missing file", [&] { heap.open_existing(path + ".missing", layout, n, pages); });
    t.check(heap.next_flat_slot() == 0 && heap.allocated_pages() == 0, "rejected reopen left the cursor");
    t.expect_throw("allocate_slot after a rejected reopen", [&] { heap.allocate_slot(free_list); });

    // Rejected after the fd was opened: the heap must be closed again, so the
    // allocation attempt neither succeeds nor writes a page into the file.
    heap.open_existing(path, layout, n, pages);
    t.expect_throw("last page carrying another page's id", [&] {
        patch_bytes(path, size_t(pages - 1) * PAGE, page_header_bytes(pages));
        heap.open_existing(path, layout, n, pages);
    });
    t.check(heap.next_flat_slot() == 0 && heap.allocated_pages() == 0, "rejected reopen left the cursor");
    t.expect_throw("allocate_slot after a rejected reopen", [&] { heap.allocate_slot(free_list); });
    std::vector<char> got(ELEM);
    t.expect_throw("read_vector after a rejected reopen", [&] { heap.read_vector(0, got.data()); });
    patch_bytes(path, size_t(pages - 1) * PAGE, page_header_bytes(pages - 1));
    t.check(read_whole_file(path) == good, "a rejected reopen or the allocation after it changed the file");

    heap.open_existing(path, layout, n, pages);
    t.check(heap.allocate_slot(free_list) == n, "allocation after the restored reopen skipped a slot");
    heap.close();
    ::unlink(path.c_str());
    return t.done();
}

// Atomically points `path` at the file `target` (a hard link, so the inode
// the heap opens is `target`'s even while `path` keeps changing).
void relink(const std::string& path, const std::string& target) {
    std::string tmp = path + ".relink";
    ::unlink(tmp.c_str());
    if (::link(target.c_str(), tmp.c_str()) != 0 || ::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("relink failed");
    }
}

// The size open_existing accepts must belong to the descriptor it reads
// through, not to whatever the path named an instant earlier. Two files
// share the path by turns: the heap itself, and a copy one byte longer whose
// slot 1 holds other bytes but whose first and last page headers are intact,
// so nothing after the size check tells the two apart. Every open must
// either be refused or read the true slot 1. Checking the size with
// stat(path) before open() lets the copy through whenever the swap lands in
// between; over this many attempts that window is hit reliably.
bool test_reopen_sizes_the_file_it_opened() {
    TestCase t("open_existing checks the size of the file it opened, not of the path it was given");
    std::string path = temp_path("ivf_heap_swap");
    std::string good = path + ".good", longer = path + ".longer";
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    const uint32_t spp = layout.slots_per_page, pages = 2, n = pages * spp;
    write_closed_heap(good, layout, n);
    std::vector<char> bytes = read_whole_file(good);
    std::vector<char> other = pattern(n + 1);
    std::copy(other.begin(), other.end(), bytes.begin() + layout.slot_offset(1));
    bytes.push_back(0);
    ::unlink(longer.c_str());
    std::ofstream(longer, std::ios::binary).write(bytes.data(), std::streamsize(bytes.size()));
    ::unlink(path.c_str());
    relink(path, good);

    std::atomic<bool> stop{false};
    std::thread swapper([&] {
        while (!stop.load()) {
            relink(path, longer);
            relink(path, good);
        }
    });

    RawVectorHeap heap;
    std::vector<char> got(ELEM);
    uint32_t accepted = 0, refused = 0, wrong = 0;
    for (uint32_t attempt = 0; attempt < 20000; ++attempt) {
        try {
            heap.open_existing(path, layout, n, pages);
        } catch (const diskann::ANNException&) {
            ++refused;
            continue;
        }
        ++accepted;
        heap.read_vector(1, got.data());
        wrong += got != pattern(1);
        heap.close();
    }
    stop.store(true);
    swapper.join();

    t.check(accepted > 0 && refused > 0, "the swap never raced an open (" + std::to_string(accepted) +
                                              " accepted, " + std::to_string(refused) + " refused)");
    t.check(wrong == 0, std::to_string(wrong) + " opens accepted the longer file and read its slot 1");
    ::unlink(path.c_str());
    ::unlink(good.c_str());
    ::unlink(longer.c_str());
    return t.done();
}

bool test_read_rejects_misplaced_page() {
    TestCase t("a page whose header names another page or geometry is rejected on read, and on bulk write");
    std::string path = temp_path("ivf_heap_misplaced");
    RawVectorHeapLayout layout = compute_raw_vector_heap_layout(PAGE, ELEM);
    const uint32_t spp = layout.slots_per_page, pages = 5, n = pages * spp;
    write_closed_heap(path, layout, n);
    patch_bytes(path, 1 * PAGE, page_header_bytes(2));                  // page 1 now claims to be page 2
    patch_bytes(path, 2 * PAGE, page_header_bytes(2, 2 * PAGE, ELEM));  // page 2 claims a wider page
    patch_bytes(path, 3 * PAGE, page_header_bytes(3, PAGE, ELEM / 2));  // page 3 claims smaller elements

    RawVectorHeap heap;
    heap.open_existing(path, layout, n, pages);  // first and last pages are intact
    std::vector<char> got(ELEM);
    uint32_t wrong = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t page = layout.page_of(i);
        bool in_bad_page = page >= 1 && page <= 3;
        try {
            heap.read_vector(i, got.data());
            wrong += in_bad_page || got != pattern(i);
        } catch (const diskann::ANNException&) {
            wrong += !in_bad_page;
        }
    }
    t.check(wrong == 0, std::to_string(wrong) + " slots were accepted from a bad page or rejected elsewhere");

    std::vector<char> before = read_whole_file(path);
    std::vector<char> page(PAGE, 0);
    auto stamp = [&](std::vector<char> header) { std::copy(header.begin(), header.end(), page.begin()); };
    stamp(page_header_bytes(3));
    t.expect_throw("write_pages with a header for another page", [&] { heap.write_pages(2, page.data(), 1); });
    stamp(page_header_bytes(2, PAGE, 2 * ELEM));
    t.expect_throw("write_pages with a header for another elem_size", [&] { heap.write_pages(2, page.data(), 1); });
    t.check(read_whole_file(path) == before, "rejected write_pages touched the file");
    heap.close();
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
    for (uint32_t p = 0; p < bulk.allocated_pages(); ++p) {
        if (!page_header_is(bulk_bytes, p)) fail("page " + std::to_string(p) + " lacks its header");
    }

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

bool test_rejects_bad_layout_and_out_of_range_slots() {
    TestCase t("heap rejects an inconsistent layout, slots past the allocated pages, a bad cursor");
    std::string path = temp_path("ivf_heap_guard");
    RawVectorHeap heap;
    t.expect_throw("default-constructed layout", [&] { heap.open(path, RawVectorHeapLayout{}); });
    RawVectorHeapLayout too_many = compute_raw_vector_heap_layout(PAGE, ELEM);
    too_many.slots_per_page += 1;  // slots would run past the page
    t.expect_throw("slots overrunning the page", [&] { heap.open(path, too_many); });

    heap.open(path, compute_raw_vector_heap_layout(PAGE, ELEM));
    RawVectorFreeList free_list;
    uint32_t last = 0;
    for (int i = 0; i < 3; ++i) last = heap.allocate_slot(free_list);  // page 0 only
    std::vector<char> buf(ELEM);
    uint32_t past = 7;  // first slot of page 1, which does not exist
    t.expect_throw("write past allocated pages", [&] { heap.write_vector(past, buf.data()); });
    t.expect_throw("read past allocated pages", [&] { heap.read_vector(past, buf.data()); });
    t.expect_throw("occupancy past allocated pages", [&] { heap.is_slot_occupied(past); });
    t.expect_throw("free past allocated pages", [&] { heap.free_slot(past, free_list); });
    heap.write_vector(last, buf.data());  // slots within page 0 still work
    t.check(heap.is_slot_occupied(last) && heap.allocated_pages() == 1, "in-range slot rejected");

    t.expect_throw("cursor past the pages", [&] { heap.restore_slot_cursor(8, 1); });
    heap.restore_slot_cursor(7, 1);  // exactly full is fine
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
    all_pass &= test_reopen_resumes_the_heap();
    all_pass &= test_reopen_rejects_bad_file();
    all_pass &= test_rejected_reopen_after_use_leaves_nothing_behind();
    all_pass &= test_reopen_sizes_the_file_it_opened();
    all_pass &= test_read_rejects_misplaced_page();
    all_pass &= test_bulk_writer_matches_per_slot_writes();
    all_pass &= test_bulk_writer_rejects_misuse();
    all_pass &= test_rejects_bad_layout_and_out_of_range_slots();
    return all_pass ? 0 : 1;
}
