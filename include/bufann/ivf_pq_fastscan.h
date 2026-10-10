// SIMD scan of 8-bit PQ codes (Quicker ADC split tables: Andre, Kermarrec
// and Le Scouarnec, TPAMI 2019, arXiv:1812.09162). The PQ codebooks and
// codes are unchanged; each posting list's codes are transposed into blocks
// of IVF_FASTSCAN_BLOCK vectors so that one 512-bit register holds the same
// sub-code of 64 vectors, and a query's float lookup tables are quantized to
// 8-bit integers. One sub-quantizer's lookup for a block is then two AVX-512
// VBMI vpermi2b shuffles (the low and high 128 entries of the 256-entry
// table, indexed by the code's low 7 bits) and a blend on its top bit, with
// 16-bit accumulation.

#pragma once

#include <cstdint>
#include <vector>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

// Built from lists.codes (posting order). Throws unless those cover the lists.
IVFPQBlockedCodes build_ivf_blocked_codes(const PostingLists& lists, uint32_t chunks);

// A query's lookup tables quantized for the scan: entry [c][j] is
// min(255, round((table[c][j] - min_j table[c][j]) / delta)), so a code's
// summed entries s estimate its PQ distance as dmin + s * delta. With delta =
// (dmax - dmin) / 255, any code whose distance reaches dmax sums to about
// 255 or more, so saturating single entries at 255 loses nothing below dmax.
struct IVFPQQuantizedTable {
    uint32_t chunks = 0;
    float dmin = 0.0f;
    float delta = 1.0f;
    std::vector<uint8_t> entries;  // [chunks x 256]
};

// `table` is [chunks x 256] floats; `dmax` an upper bound on the distances
// that matter (the rerank_m-th smallest seen so far).
void quantize_pq_table(const float* table, uint32_t chunks, float dmax, IVFPQQuantizedTable& out);

// Sums of quantized entries for the 64 lanes of one block, into out[64].
// The SIMD kernel when the CPU has AVX-512 VBMI, else the scalar one.
void scan_block(const uint8_t* block, const IVFPQQuantizedTable& qt, uint16_t* out);
// The same integer arithmetic, one lane at a time; the kernel's reference.
void scan_block_scalar(const uint8_t* block, const IVFPQQuantizedTable& qt, uint16_t* out);
// True when scan_block runs the AVX-512 VBMI kernel.
bool fastscan_simd_available();

// Process-wide switch for the search's SIMD scan (on by default); off makes
// every search use the float scan, for tests and benchmarks that compare.
void set_fastscan_enabled(bool enabled);
bool fastscan_enabled();

// Float PQ distance of one lane of a block (the scalar ADC on the blocked
// layout), for `table` [chunks x 256].
float adc_block_lane(const uint8_t* block, uint32_t chunks, uint32_t lane, const float* table);

}  // namespace inplace
}  // namespace diskann
