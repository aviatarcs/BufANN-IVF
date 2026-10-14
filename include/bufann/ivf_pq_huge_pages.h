#pragma once

#include <cstddef>

namespace diskann {
namespace inplace {

// Asks the kernel to back the 2 MB-aligned pages inside [p, p + bytes) with
// transparent huge pages (MADV_HUGEPAGE, then MADV_COLLAPSE to do it now
// rather than whenever khugepaged gets there). For the large arrays a query
// jumps around in (PQ codes, centroids): with 4 KB pages nearly every list
// or centroid row costs a TLB miss. Advice only: the contents are unchanged
// and a kernel without THP or MADV_COLLAPSE ignores it. Returns the bytes
// advised, 0 when the range holds no whole huge page.
size_t advise_huge_pages(const void* p, size_t bytes);

}  // namespace inplace
}  // namespace diskann
