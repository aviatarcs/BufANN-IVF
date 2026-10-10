// Exact L2 top-K of every query over a uint8 .bin base; writes a DiskANN
// truthset (npts, K, ids, dists). ||x||^2 + ||q||^2 - 2 x.q with GEMM per block.
#include <mkl.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "exact_gt base.u8.bin query.u8.bin K out.bin\n"); return 2; }
    const int K = std::atoi(argv[3]);
    FILE* fb = std::fopen(argv[1], "rb"); FILE* fq = std::fopen(argv[2], "rb");
    int32_t n, d, nq, dq; std::fread(&n, 4, 1, fb); std::fread(&d, 4, 1, fb); std::fread(&nq, 4, 1, fq); std::fread(&dq, 4, 1, fq);
    if (d != dq) return 3;
    std::vector<uint8_t> q8(size_t(nq) * d); std::fread(q8.data(), 1, q8.size(), fq);
    std::vector<float> q(q8.begin(), q8.end()), qn(nq, 0);
    for (int i = 0; i < nq; ++i) for (int j = 0; j < d; ++j) qn[i] += q[size_t(i) * d + j] * q[size_t(i) * d + j];
    using P = std::pair<float, uint32_t>;
    std::vector<std::priority_queue<P>> top(nq);
    const size_t B = 131072;
    std::vector<uint8_t> b8(B * d); std::vector<float> b(B * d), bn(B), dist(size_t(nq) * B);
    for (size_t s = 0; s < size_t(n); s += B) {
        const size_t m = std::min(B, size_t(n) - s);
        std::fread(b8.data(), 1, m * d, fb);
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) { float t = 0; for (int j = 0; j < d; ++j) { float v = b8[i * d + j]; b[i * d + j] = v; t += v * v; } bn[i] = t; }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, m, d, -2.0f, q.data(), d, b.data(), d, 0.0f, dist.data(), m);
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < nq; ++i) {
            auto& h = top[i]; const float* row = &dist[size_t(i) * m];
            for (size_t j = 0; j < m; ++j) {
                const float v = row[j] + bn[j] + qn[i];
                if (int(h.size()) < K) h.emplace(v, uint32_t(s + j));
                else if (v < h.top().first) { h.pop(); h.emplace(v, uint32_t(s + j)); }
            }
        }
        if ((s / B) % 64 == 0) { std::fprintf(stderr, "%zu / %d\n", s + m, n); }
    }
    std::vector<uint32_t> ids(size_t(nq) * K); std::vector<float> ds(size_t(nq) * K);
    for (int i = 0; i < nq; ++i) for (int r = K - 1; r >= 0; --r) { ids[size_t(i) * K + r] = top[i].top().second; ds[size_t(i) * K + r] = top[i].top().first; top[i].pop(); }
    FILE* fo = std::fopen(argv[4], "wb"); std::fwrite(&nq, 4, 1, fo); std::fwrite(&K, 4, 1, fo);
    std::fwrite(ids.data(), 4, ids.size(), fo); std::fwrite(ds.data(), 4, ds.size(), fo); std::fclose(fo);
    return 0;
}
