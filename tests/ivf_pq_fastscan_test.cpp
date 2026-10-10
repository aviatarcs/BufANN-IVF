// Tests for the SIMD scan's pieces against references written here: the
// blocked layout against the posting-order codes it is built from, the
// quantized lookup table against its definition, and the block kernel
// (AVX-512 VBMI when the CPU has it) against a plain sum of table bytes,
// exactly, on random codes and tables including the extremes.

#include "bufann/ivf_pq_fastscan.h"
#include "ivf_pq_test_util.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace diskann::inplace;

namespace {

const uint32_t CHUNKS = 32;

// Lists of sizes 0, 1, 63, 64, 65, 200 with runs of empty lists between
// them, and random ones, with random codes.
PostingLists random_lists(std::mt19937& gen) {
    std::vector<uint32_t> sizes{0, 1, 0, 0, 63, 64, 0, 65, 200, 0};
    std::uniform_int_distribution<uint32_t> size(0, 300);
    for (int i = 0; i < 20; ++i) sizes.push_back(size(gen));
    PostingLists lists;
    lists.offsets.push_back(0);
    for (uint32_t s : sizes) lists.offsets.push_back(lists.offsets.back() + s);
    const uint32_t n = lists.offsets.back();
    lists.ids.resize(n);
    std::iota(lists.ids.begin(), lists.ids.end(), 0u);
    std::shuffle(lists.ids.begin(), lists.ids.end(), gen);
    lists.codes.resize(size_t(n) * CHUNKS);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& c : lists.codes) c = uint8_t(byte(gen));
    return lists;
}

bool test_blocked_layout() {
    TestCase t("blocked codes hold each list's codes transposed, padded with zeros");
    std::mt19937 gen(1);
    const PostingLists lists = random_lists(gen);
    const IVFPQBlockedCodes b = build_ivf_blocked_codes(lists, CHUNKS);
    const size_t nlist = lists.offsets.size() - 1, block_bytes = size_t(CHUNKS) * IVF_FASTSCAN_BLOCK;
    size_t wrong = 0, blocks = 0;
    for (size_t p = 0; p < nlist; ++p) {
        const uint32_t n = lists.offsets[p + 1] - lists.offsets[p];
        const uint32_t want_blocks = (n + IVF_FASTSCAN_BLOCK - 1) / IVF_FASTSCAN_BLOCK;
        wrong += b.block_start[p + 1] - b.block_start[p] != want_blocks;
        blocks += want_blocks;
        for (uint32_t local = 0; local < want_blocks * IVF_FASTSCAN_BLOCK; ++local) {
            const uint8_t* block = b.blocks.data() + (b.block_start[p] + local / IVF_FASTSCAN_BLOCK) * block_bytes;
            for (uint32_t c = 0; c < CHUNKS; ++c) {
                const uint8_t got = block[size_t(c) * IVF_FASTSCAN_BLOCK + local % IVF_FASTSCAN_BLOCK];
                const uint8_t want = local < n ? lists.codes[size_t(lists.offsets[p] + local) * CHUNKS + c] : 0;
                wrong += got != want;
            }
        }
    }
    t.check(b.chunks == CHUNKS && b.block_start.size() == nlist + 1 && b.blocks.size() == blocks * block_bytes,
            "blocked codes have the wrong shape");
    t.check(wrong == 0, std::to_string(wrong) + " blocked bytes differ from the posting-order codes");
    PostingLists short_codes = lists;
    short_codes.codes.pop_back();
    t.expect_throw("codes not covering the lists", [&] { build_ivf_blocked_codes(short_codes, CHUNKS); });
    t.expect_throw("zero chunks", [&] { build_ivf_blocked_codes(lists, 0); });
    return t.done();
}

// The loader streams the codes in pieces; any split into consecutive pieces,
// including ones that end mid-list and empty lists between them, must give
// the layout built in one go.
bool test_blocked_rows_in_pieces() {
    TestCase t("blocked rows placed piece by piece equal the layout built in one go");
    std::mt19937 gen(4);
    const PostingLists lists = random_lists(gen);
    const IVFPQBlockedCodes whole = build_ivf_blocked_codes(lists, CHUNKS);
    const uint32_t n = lists.offsets.back();
    for (uint32_t piece : {1u, 7u, 64u, 65u, 1000u}) {
        IVFPQBlockedCodes b = make_ivf_blocked_layout(lists.offsets, CHUNKS);
        for (uint32_t r = 0; r < n; r += piece) {
            const uint32_t nr = std::min(piece, n - r);
            place_ivf_blocked_rows(b, lists.offsets, r, lists.codes.data() + size_t(r) * CHUNKS, nr);
        }
        t.check(b.block_start == whole.block_start && b.blocks == whole.blocks,
                "pieces of " + std::to_string(piece) + " rows give a different layout");
    }
    IVFPQBlockedCodes b = make_ivf_blocked_layout(lists.offsets, CHUNKS);
    t.expect_throw("rows past the lists", [&] { place_ivf_blocked_rows(b, lists.offsets, n, lists.codes.data(), 1); });
    return t.done();
}

bool test_quantized_table() {
    TestCase t("quantized table entries are min(255, round((t - min) / delta)), delta = (dmax - dmin) / 255");
    std::mt19937 gen(2);
    std::uniform_real_distribution<float> val(0.0f, 1000.0f);
    std::vector<float> table(size_t(CHUNKS) * 256);
    for (float& v : table) v = val(gen);
    double dmin = 0;
    for (uint32_t c = 0; c < CHUNKS; ++c) dmin += *std::min_element(table.begin() + c * 256, table.begin() + (c + 1) * 256);
    const float dmax = float(dmin) + 500.0f;  // delta ~2: entries above ~510 saturate
    IVFPQQuantizedTable qt;
    quantize_pq_table(table.data(), CHUNKS, dmax, qt);
    t.check(qt.chunks == CHUNKS && std::fabs(qt.dmin - dmin) <= 1e-3 * dmin &&
                std::fabs(qt.delta - (dmax - qt.dmin) / 255.0f) <= 1e-4f * qt.delta,
            "dmin or delta is not what the definition gives");
    size_t wrong = 0, saturated = 0;
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        const float mn = *std::min_element(table.begin() + c * 256, table.begin() + (c + 1) * 256);
        for (uint32_t j = 0; j < 256; ++j) {
            const double raw = (table[c * 256 + j] - mn) / qt.delta;
            const int want = int(std::min(std::round(raw), 255.0));
            // Within a hair of a half bin, float and double may round apart.
            const int slack = std::fabs(raw - std::floor(raw) - 0.5) < 1e-3 ? 1 : 0;
            wrong += std::abs(int(qt.entries[c * 256 + j]) - want) > slack;
            saturated += want == 255;
        }
    }
    t.check(wrong == 0, std::to_string(wrong) + " entries differ from the definition");
    t.check(saturated > 0, "no entry saturated; the test does not reach 255");
    IVFPQQuantizedTable flat;
    quantize_pq_table(table.data(), CHUNKS, float(dmin) - 1.0f, flat);
    t.check(flat.delta == 1.0f, "dmax below dmin must fall back to a unit bin");
    return t.done();
}

bool test_kernel_matches_plain_sum() {
    TestCase t(std::string("block kernel (") + (fastscan_simd_available() ? "AVX-512 VBMI" : "scalar") +
               ") equals a plain sum of table bytes");
    std::mt19937 gen(3);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> block(size_t(CHUNKS) * IVF_FASTSCAN_BLOCK);
    std::vector<uint16_t> got(IVF_FASTSCAN_BLOCK), ref(IVF_FASTSCAN_BLOCK);
    IVFPQQuantizedTable qt;
    qt.chunks = CHUNKS;
    qt.entries.resize(size_t(CHUNKS) * 256);
    size_t wrong = 0;
    for (int trial = 0; trial < 200; ++trial) {
        // Trial 0: every code 255 and every entry 255 (the largest sum, the
        // top half of every table); trial 1: every code 127 (the low half's
        // last entry); the rest random.
        for (size_t i = 0; i < block.size(); ++i) block[i] = trial == 0 ? 255 : trial == 1 ? 127 : uint8_t(byte(gen));
        for (size_t i = 0; i < qt.entries.size(); ++i) qt.entries[i] = trial == 0 ? 255 : uint8_t(byte(gen));
        scan_block(block.data(), qt, got.data());
        scan_block_scalar(block.data(), qt, ref.data());
        for (uint32_t lane = 0; lane < IVF_FASTSCAN_BLOCK; ++lane) {
            uint32_t want = 0;
            for (uint32_t c = 0; c < CHUNKS; ++c) want += qt.entries[c * 256 + block[c * IVF_FASTSCAN_BLOCK + lane]];
            wrong += got[lane] != want || ref[lane] != want;
        }
    }
    t.check(wrong == 0, std::to_string(wrong) + " lane sums differ from the plain sum");
    return t.done();
}

}  // namespace

int main() {
    bool all_pass = true;
    try {
        all_pass &= test_blocked_layout();
        all_pass &= test_blocked_rows_in_pieces();
        all_pass &= test_quantized_table();
        all_pass &= test_kernel_matches_plain_sum();
    } catch (const diskann::ANNException& e) {
        std::cout << "  FAIL: " << e.message() << std::endl;
        all_pass = false;
    }
    return all_pass ? 0 : 1;
}
