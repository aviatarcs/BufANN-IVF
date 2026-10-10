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

IVFPQBlockedCodes build_ivf_blocked_codes(const PostingLists& lists, uint32_t chunks) {
    IVF_PQ_REQUIRE(chunks > 0 && !lists.offsets.empty() && lists.codes.size() == lists.ids.size() * chunks,
                   "posting-list codes do not cover the lists");
    const size_t nlist = lists.offsets.size() - 1;
    IVFPQBlockedCodes b;
    b.chunks = chunks;
    b.block_start.resize(nlist + 1);
    uint32_t next = 0;
    for (size_t p = 0; p < nlist; ++p) {
        b.block_start[p] = next;
        next += (lists.offsets[p + 1] - lists.offsets[p] + IVF_FASTSCAN_BLOCK - 1) / IVF_FASTSCAN_BLOCK;
    }
    b.block_start[nlist] = next;
    const size_t block_bytes = size_t(chunks) * IVF_FASTSCAN_BLOCK;
    b.blocks.assign(size_t(next) * block_bytes, 0);
#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t p = 0; p < int64_t(nlist); ++p) {
        const uint32_t begin = lists.offsets[size_t(p)], end = lists.offsets[size_t(p) + 1];
        uint8_t* first = b.blocks.data() + size_t(b.block_start[size_t(p)]) * block_bytes;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t local = i - begin;
            uint8_t* block = first + size_t(local / IVF_FASTSCAN_BLOCK) * block_bytes;
            const uint8_t* code = lists.codes.data() + size_t(i) * chunks;
            for (uint32_t c = 0; c < chunks; ++c) block[size_t(c) * IVF_FASTSCAN_BLOCK + local % IVF_FASTSCAN_BLOCK] = code[c];
        }
    }
    return b;
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
// Sums fit 16 bits: at most 255 per sub-quantizer, and chunks <= 257.
__attribute__((target("avx512f,avx512bw,avx512vbmi"))) void scan_block_vbmi(const uint8_t* block,
                                                                           const IVFPQQuantizedTable& qt,
                                                                           uint16_t* out) {
    __m512i acc_lo = _mm512_setzero_si512(), acc_hi = _mm512_setzero_si512();
    const uint8_t* entries = qt.entries.data();
    for (uint32_t c = 0; c < qt.chunks; ++c) {
        const uint8_t* t = entries + size_t(c) * 256;
        const __m512i t0 = _mm512_loadu_si512(t), t1 = _mm512_loadu_si512(t + 64);
        const __m512i t2 = _mm512_loadu_si512(t + 128), t3 = _mm512_loadu_si512(t + 192);
        const __m512i codes = _mm512_loadu_si512(block + size_t(c) * IVF_FASTSCAN_BLOCK);
        // vpermi2b indexes 128 entries by the low 7 bits; the top bit picks the half.
        const __m512i low = _mm512_permutex2var_epi8(t0, codes, t1);
        const __m512i high = _mm512_permutex2var_epi8(t2, codes, t3);
        const __m512i v = _mm512_mask_blend_epi8(_mm512_movepi8_mask(codes), low, high);
        acc_lo = _mm512_add_epi16(acc_lo, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v)));
        acc_hi = _mm512_add_epi16(acc_hi, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(v, 1)));
    }
    _mm512_storeu_si512(out, acc_lo);
    _mm512_storeu_si512(out + 32, acc_hi);
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

void scan_block(const uint8_t* block, const IVFPQQuantizedTable& qt, uint16_t* out) {
#ifdef IVF_FASTSCAN_HAVE_VBMI
    if (fastscan_simd_available()) {
        scan_block_vbmi(block, qt, out);
        return;
    }
#endif
    scan_block_scalar(block, qt, out);
}

float adc_block_lane(const uint8_t* block, uint32_t chunks, uint32_t lane, const float* table) {
    float d = 0.0f;
    for (uint32_t c = 0; c < chunks; ++c) d += table[size_t(c) * 256 + block[size_t(c) * IVF_FASTSCAN_BLOCK + lane]];
    return d;
}

}  // namespace inplace
}  // namespace diskann
