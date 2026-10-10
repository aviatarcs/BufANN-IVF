#include "bufann/ivf_pq_huge_pages.h"

#include <sys/mman.h>

#include <cstdint>

namespace diskann {
namespace inplace {

namespace {
const uintptr_t HUGE_PAGE = uintptr_t(2) << 20;
// Linux 6.1; older headers lack the name.
const int ADVICE_COLLAPSE = 25;
}  // namespace

size_t advise_huge_pages(const void* p, size_t bytes) {
    const uintptr_t begin = (uintptr_t(p) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
    const uintptr_t end = (uintptr_t(p) + bytes) & ~(HUGE_PAGE - 1);
    if (end <= begin) return 0;
    madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
    madvise(reinterpret_cast<void*>(begin), end - begin, ADVICE_COLLAPSE);
    return end - begin;
}

}  // namespace inplace
}  // namespace diskann
