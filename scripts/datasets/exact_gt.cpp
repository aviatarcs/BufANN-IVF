// Exact L2 top-K of every query over the first N rows of a .bin base
// (uint8, int8 or float32 rows; the header's count when N is omitted);
// writes a DiskANN truthset (npts, K, ids, dists). ||x||^2 + ||q||^2 - 2 x.q
// with one GEMM per block of 131072 base rows.
//
//   g++ -O3 -march=native -fopenmp exact_gt.cpp -lmkl_rt -o exact_gt
//   ./exact_gt uint8|int8|float base.bin query.bin K out.bin [N]
#include <mkl.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

template<typename T>
std::vector<float> read_rows(FILE* f, size_t rows, int d) {
    std::vector<T> raw(rows * d);
    if (std::fread(raw.data(), sizeof(T), raw.size(), f) != raw.size()) {
        std::fprintf(stderr, "short read\n");
        std::exit(4);
    }
    return std::vector<float>(raw.begin(), raw.end());
}

template<typename T>
int run(int argc, char** argv) {
    const int K = std::atoi(argv[4]);
    FILE* fb = std::fopen(argv[2], "rb");
    FILE* fq = std::fopen(argv[3], "rb");
    if (fb == nullptr || fq == nullptr) {
        std::fprintf(stderr, "cannot open base or query file\n");
        return 2;
    }
    int32_t n, d, nq, dq;
    if (std::fread(&n, 4, 1, fb) != 1 || std::fread(&d, 4, 1, fb) != 1 || std::fread(&nq, 4, 1, fq) != 1 ||
        std::fread(&dq, 4, 1, fq) != 1 || d != dq) {
        std::fprintf(stderr, "bad headers or dimension mismatch\n");
        return 3;
    }
    if (argc > 6) n = std::min<int32_t>(n, std::atoi(argv[6]));
    std::vector<float> q = read_rows<T>(fq, size_t(nq), d), qn(nq, 0);
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < d; ++j) qn[i] += q[size_t(i) * d + j] * q[size_t(i) * d + j];
    }
    using P = std::pair<float, uint32_t>;
    std::vector<std::priority_queue<P>> top(nq);
    const size_t B = 131072;
    std::vector<float> bn(B), dist(size_t(nq) * B);
    for (size_t s = 0; s < size_t(n); s += B) {
        const size_t m = std::min(B, size_t(n) - s);
        std::vector<float> b = read_rows<T>(fb, m, d);
#pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            float t = 0;
            for (int j = 0; j < d; ++j) t += b[i * d + j] * b[i * d + j];
            bn[i] = t;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, m, d, -2.0f, q.data(), d, b.data(), d, 0.0f,
                    dist.data(), m);
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < nq; ++i) {
            auto& h = top[i];
            const float* row = &dist[size_t(i) * m];
            for (size_t j = 0; j < m; ++j) {
                const float v = row[j] + bn[j] + qn[i];
                if (int(h.size()) < K) h.emplace(v, uint32_t(s + j));
                else if (v < h.top().first) { h.pop(); h.emplace(v, uint32_t(s + j)); }
            }
        }
        if ((s / B) % 64 == 0) std::fprintf(stderr, "%zu / %d\n", s + m, n);
    }
    std::vector<uint32_t> ids(size_t(nq) * K);
    std::vector<float> ds(size_t(nq) * K);
    for (int i = 0; i < nq; ++i) {
        for (int r = K - 1; r >= 0; --r) {
            ids[size_t(i) * K + r] = top[i].top().second;
            ds[size_t(i) * K + r] = top[i].top().first;
            top[i].pop();
        }
    }
    FILE* fo = std::fopen(argv[5], "wb");
    std::fwrite(&nq, 4, 1, fo);
    std::fwrite(&K, 4, 1, fo);
    std::fwrite(ids.data(), 4, ids.size(), fo);
    std::fwrite(ds.data(), 4, ds.size(), fo);
    std::fclose(fo);
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr, "exact_gt uint8|int8|float base.bin query.bin K out.bin [N]\n");
        return 2;
    }
    const std::string type = argv[1];
    if (type == "uint8") return run<uint8_t>(argc, argv);
    if (type == "int8") return run<int8_t>(argc, argv);
    if (type == "float") return run<float>(argc, argv);
    std::fprintf(stderr, "type must be uint8, int8 or float\n");
    return 2;
}
