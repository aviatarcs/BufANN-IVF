# SIFT10M / SIFT100M as benchmarked here

Both are prefixes of SIFT1B (`base.1B.u8bin` from the big-ann-benchmarks
bucket), shuffled with `numpy.random.default_rng(0).permutation(N)`.
Queries are BigANN's `query.public.10K.u8bin` (stored as `query.10K.bin`).

- **SIFT100M**: `make_sift100m.py` downloads the first 100M rows (~230 s
  at ~62 MB/s), shuffles them, and writes `base.90M.bin` (the query-workload
  base) and `tail.10M.bin` (the update tail). BufANN's scripts want the
  whole `base.100M.bin`: a `<ii>` header (100000000, 128), then the
  90M rows, then the 10M rows (`benchmark_results/sift100m_2026-10-02_prep.sh`, step 2).
- **SIFT10M**: the same procedure with N = 10M gives `base.10M.bin`; the
  query-workload base `base.9M.bin` is its first 9M rows.
- **Ground truth**: `exact_gt.cpp` is an exact L2 top-K over a uint8, int8
  or float32 base, optionally its first N rows (GEMM per 131072-row block,
  MKL + OpenMP), writing a DiskANN truthset (`npts, K, ids, dists`). The 90M
  top-100 used by both systems at 100M:

      g++ -O3 -march=native -fopenmp exact_gt.cpp -lmkl_rt -o exact_gt
      ./exact_gt uint8 base.90M.bin query.10K.bin 100 sift100m_90000000_gt100.bin
      ./exact_gt float base.100M.fbin query.10K.fbin 100 deep100m_90000000_gt100.bin 90000000

  Checked against the uint8-only version (identical output on 1M SIFT
  rows), float against uint8 on the same values, the row limit against a
  truncated file, and int8 against a NumPy brute force (SPACEV).
- **DEEP100M, SPACEV100M, Turing100M**: the first 100M rows of each 1B base
  from big-ann-benchmarks, unshuffled (BufANN's convention); the first 90M
  are the query base, and the first 10K of each provided query set are the
  queries.

  BufANN's own `gen_deep_gt.sh` (DiskANN's compute_groundtruth over the
  full 100M) ran out of memory on a 125 GB node; the query comparison only
  needs the 90M base's ground truth.
