#include "bufann/ivf_pq_fastscan.h"

#include <algorithm>
#include <atomic>
#include <cmath>

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
#include <immintrin.h>
#define IVF_FASTSCAN_HAVE_VBMI 1
#endif

#include "bufann/ivf_pq_require.h"

namespace diskann {
namespace inplace {

IVFPQBlockedCodes make_ivf_blocked_layout(const std::vector<uint32_t>& offsets, uint32_t chunks) {
    IVF_PQ_REQUIRE(chunks > 0 && !offsets.empty(), "blocked layout needs chunks > 0 and posting offsets");
    const size_t nlist = offsets.size() - 1;
    IVFPQBlockedCodes b;
    b.chunks = chunks;
    b.block_start.resize(nlist + 1);
    uint32_t next = 0;
    for (size_t p = 0; p < nlist; ++p) {
        b.block_start[p] = next;
        next += (offsets[p + 1] - offsets[p] + IVF_FASTSCAN_BLOCK - 1) / IVF_FASTSCAN_BLOCK;
    }
    b.block_start[nlist] = next;
    b.blocks.assign(size_t(next) * chunks * IVF_FASTSCAN_BLOCK, 0);
    return b;
}

void place_ivf_blocked_rows(IVFPQBlockedCodes& b, const std::vector<uint32_t>& offsets, uint32_t first_row,
                            const uint8_t* rows, uint32_t nrows) {
    IVF_PQ_REQUIRE(uint64_t(first_row) + nrows <= offsets.back(), "rows past the posting lists");
    const size_t block_bytes = size_t(b.chunks) * IVF_FASTSCAN_BLOCK;
    // The list holding first_row; rows are in posting order, so later ones
    // walk forward through the lists.
    size_t p = size_t(std::upper_bound(offsets.begin(), offsets.end(), first_row) - offsets.begin()) - 1;
    for (uint32_t r = 0; r < nrows; ++r) {
        const uint32_t row = first_row + r;
        while (row >= offsets[p + 1]) ++p;
        const uint32_t local = row - offsets[p];
        uint8_t* block = b.blocks.data() + (size_t(b.block_start[p]) + local / IVF_FASTSCAN_BLOCK) * block_bytes;
        const uint8_t* code = rows + size_t(r) * b.chunks;
        for (uint32_t c = 0; c < b.chunks; ++c) block[size_t(c) * IVF_FASTSCAN_BLOCK + local % IVF_FASTSCAN_BLOCK] = code[c];
    }
}

IVFPQBlockedCodes build_ivf_blocked_codes(const PostingLists& lists, uint32_t chunks) {
    IVF_PQ_REQUIRE(chunks > 0 && !lists.offsets.empty() && lists.codes.size() == lists.ids.size() * chunks,
                   "posting-list codes do not cover the lists");
    IVFPQBlockedCodes b = make_ivf_blocked_layout(lists.offsets, chunks);
    place_ivf_blocked_rows(b, lists.offsets, 0, lists.codes.data(), uint32_t(lists.ids.size()));
    return b;
}

bool ivf_blocked_codes_cover(const IVFPQBlockedCodes& b, const PostingLists& lists, uint32_t chunks) {
    return b.chunks == chunks && chunks > 0 && !lists.offsets.empty() && b.block_start.size() == lists.offsets.size() &&
           b.blocks.size() == size_t(b.block_start.back()) * chunks * IVF_FASTSCAN_BLOCK;
}

void quantize_pq_table(const float* table, uint32_t chunks, float dmax, IVFPQQuantizedTable& out) {
    out.chunks = chunks;
    out.entries.resize(size_t(chunks) * 256);
    std::vector<float> mins(chunks);
    float dmin = 0.0f;
    for (uint32_t c = 0; c < chunks; ++c) {
        mins[c] = *std::min_element(table + size_t(c) * 256, table + size_t(c + 1) * 256);
        dmin += mins[c];
    }
    out.dmin = dmin;
    // A degenerate range (every code equally close) still needs a positive bin.
    out.delta = dmax > dmin ? (dmax - dmin) / 255.0f : 1.0f;
    const float inv = 1.0f / out.delta;
    for (uint32_t c = 0; c < chunks; ++c) {
        for (uint32_t j = 0; j < 256; ++j) {
            const float q = std::nearbyint((table[size_t(c) * 256 + j] - mins[c]) * inv);
            out.entries[size_t(c) * 256 + j] = uint8_t(std::min(q, 255.0f));
        }
    }
}

void scan_block_scalar(const uint8_t* block, const IVFPQQuantizedTable& qt, uint16_t* out) {
    for (uint32_t lane = 0; lane < IVF_FASTSCAN_BLOCK; ++lane) {
        uint32_t s = 0;
        for (uint32_t c = 0; c < qt.chunks; ++c) {
            s += qt.entries[size_t(c) * 256 + block[size_t(c) * IVF_FASTSCAN_BLOCK + lane]];
        }
        out[lane] = uint16_t(std::min<uint32_t>(s, 0xFFFF));
    }
}

#ifdef IVF_FASTSCAN_HAVE_VBMI
namespace {
#define IVF_FASTSCAN_TARGET __attribute__((target("avx512f,avx512bw,avx512vbmi")))

// One sub-quantizer's 256 entries in four registers; vpermi2b indexes 128 of
// them by a code's low 7 bits, and the top bit picks the half.
IVF_FASTSCAN_TARGET inline __m512i lookup(const __m512i t[4], __m512i codes) {
    return _mm512_mask_blend_epi8(_mm512_movepi8_mask(codes), _mm512_permutex2var_epi8(t[0], codes, t[1]),
                                  _mm512_permutex2var_epi8(t[2], codes, t[3]));
}

IVF_FASTSCAN_TARGET inline void accumulate(__m512i& lo, __m512i& hi, __m512i v) {
    lo = _mm512_add_epi16(lo, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v)));
    hi = _mm512_add_epi16(hi, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(v, 1)));
}

// Sums fit 16 bits: at most 255 per sub-quantizer, and chunks <= 257. Blocks
// go in pairs so each table load serves two (the loads, not the lookups,
// bound a single block), and the pair after next is prefetched: a list's
// codes come from DRAM.
IVF_FASTSCAN_TARGET void scan_blocks_vbmi(const uint8_t* first, uint32_t nblocks, const IVFPQQuantizedTable& qt,
                                          uint16_t* out) {
    const size_t block_bytes = size_t(qt.chunks) * IVF_FASTSCAN_BLOCK;
    const uint8_t* entries = qt.entries.data();
    uint32_t b = 0;
    for (; b + 2 <= nblocks; b += 2) {
        const uint8_t* b0 = first + size_t(b) * block_bytes;
        const uint8_t* b1 = b0 + block_bytes;
        if (b + 4 <= nblocks) {
            for (size_t off = 0; off < 2 * block_bytes; off += 64) _mm_prefetch((const char*)b0 + 2 * block_bytes + off, _MM_HINT_T0);
        }
        __m512i lo0 = _mm512_setzero_si512(), hi0 = lo0, lo1 = lo0, hi1 = lo0;
        for (uint32_t c = 0; c < qt.chunks; ++c) {
            const uint8_t* t = entries + size_t(c) * 256;
            const __m512i tab[4] = {_mm512_loadu_si512(t), _mm512_loadu_si512(t + 64), _mm512_loadu_si512(t + 128),
                                    _mm512_loadu_si512(t + 192)};
            accumulate(lo0, hi0, lookup(tab, _mm512_loadu_si512(b0 + size_t(c) * IVF_FASTSCAN_BLOCK)));
            accumulate(lo1, hi1, lookup(tab, _mm512_loadu_si512(b1 + size_t(c) * IVF_FASTSCAN_BLOCK)));
        }
        uint16_t* o = out + size_t(b) * IVF_FASTSCAN_BLOCK;
        _mm512_storeu_si512(o, lo0);
        _mm512_storeu_si512(o + 32, hi0);
        _mm512_storeu_si512(o + 64, lo1);
        _mm512_storeu_si512(o + 96, hi1);
    }
    if (b < nblocks) {
        const uint8_t* b0 = first + size_t(b) * block_bytes;
        __m512i lo = _mm512_setzero_si512(), hi = lo;
        for (uint32_t c = 0; c < qt.chunks; ++c) {
            const uint8_t* t = entries + size_t(c) * 256;
            const __m512i tab[4] = {_mm512_loadu_si512(t), _mm512_loadu_si512(t + 64), _mm512_loadu_si512(t + 128),
                                    _mm512_loadu_si512(t + 192)};
            accumulate(lo, hi, lookup(tab, _mm512_loadu_si512(b0 + size_t(c) * IVF_FASTSCAN_BLOCK)));
        }
        uint16_t* o = out + size_t(b) * IVF_FASTSCAN_BLOCK;
        _mm512_storeu_si512(o, lo);
        _mm512_storeu_si512(o + 32, hi);
    }
}

IVF_FASTSCAN_TARGET uint64_t lanes_below_avx512(const uint16_t* sums, uint16_t bound) {
    const __m512i b = _mm512_set1_epi16(short(bound));
    return uint64_t(_mm512_cmplt_epu16_mask(_mm512_loadu_si512(sums), b)) |
           uint64_t(_mm512_cmplt_epu16_mask(_mm512_loadu_si512(sums + 32), b)) << 32;
}
}  // namespace
#endif

namespace {
std::atomic<bool> g_fastscan_enabled{true};
}  // namespace

void set_fastscan_enabled(bool enabled) { g_fastscan_enabled.store(enabled); }
bool fastscan_enabled() { return g_fastscan_enabled.load(); }

bool fastscan_simd_available() {
#ifdef IVF_FASTSCAN_HAVE_VBMI
    static const bool ok = __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw");
    return ok;
#else
    return false;
#endif
}

void scan_blocks(const uint8_t* first, uint32_t nblocks, const IVFPQQuantizedTable& qt, uint16_t* out) {
#ifdef IVF_FASTSCAN_HAVE_VBMI
    if (fastscan_simd_available()) {
        scan_blocks_vbmi(first, nblocks, qt, out);
        return;
    }
#endif
    const size_t block_bytes = size_t(qt.chunks) * IVF_FASTSCAN_BLOCK;
    for (uint32_t b = 0; b < nblocks; ++b) {
        scan_block_scalar(first + b * block_bytes, qt, out + size_t(b) * IVF_FASTSCAN_BLOCK);
    }
}

uint64_t lanes_below(const uint16_t* sums, uint32_t lanes, uint32_t bound) {
    const uint64_t valid = lanes >= 64 ? ~uint64_t(0) : (uint64_t(1) << lanes) - 1;
    if (bound > 0xFFFF) return valid;
#ifdef IVF_FASTSCAN_HAVE_VBMI
    if (fastscan_simd_available()) return lanes_below_avx512(sums, uint16_t(bound)) & valid;
#endif
    uint64_t mask = 0;
    for (uint32_t l = 0; l < lanes && l < 64; ++l) mask |= uint64_t(sums[l] < bound) << l;
    return mask;
}

float adc_block_lane(const uint8_t* block, uint32_t chunks, uint32_t lane, const float* table) {
    float d = 0.0f;
    for (uint32_t c = 0; c < chunks; ++c) d += table[size_t(c) * 256 + block[size_t(c) * IVF_FASTSCAN_BLOCK + lane]];
    return d;
}

}  // namespace inplace
}  // namespace diskann
