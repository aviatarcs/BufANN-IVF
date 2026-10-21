# Benchmark results

One JSON object per run: bufann_driver's result file, or the IVF-PQ query
bench's result line, plus `dataset`, `N` and either `search_L`/`driver`
(BufANN) or `index_version` (IVF-PQ). Field names follow bufann_driver.
The `*_run.sh` / `*_prep.sh` files are the scripts that produced them, with
the node's paths as they were.

- `sift10m_2026-09-30.jsonl`: 9M base, 10K queries, 32 threads. BufANN at
  L 20-200 with a 262144-frame (1 GB) pool; IVF-PQ nlist 65536, graph beam
  2 x nprobe, pools of 262144 and 160000 frames (equal total RSS).
- `sift100m_2026-10-03.jsonl`: 90M base, same setup. BufANN and IVF-PQ
  (nlist 131072 and 262144, index file v5) at pools of 262144 and 2097152
  frames (1 GB and 8 GB).
- `100m_2026-10-15.jsonl`: SIFT100M, DEEP100M, SPACEV100M and Turing100M
  (90M base, 10K queries, scripts/datasets/README.md), IVF-PQ from
  agent/rerank-batches (nlist 131072, 32-byte PQ, graph centroid search,
  fixed nprobe and per-query cutoff, re-rank 100-400, re-rank schedules;
  `run` names the process) and BufANN (vectordb-bench a9f031d bufann_driver,
  canonical DiskANN R 64 / L 100 from vectordb-baselines' shard helpers,
  32-chunk PQ, search_L 20-400, one process per L). 1 GB pool, 32 threads,
  1000-row warmup per dataset. A new CloudLab node: everything read from a
  volume striped over its four NVMe drives only (its /tmpdata also spans the
  SATA system disk), so absolute numbers are not comparable to the October
  files. The `100m_2026-10-15_*` scripts produced it (prep.sh ran on
  /tmpdata and its outputs were copied to /nvme); `_compare.py` prints the
  comparison table.
- `rabitq_ssd_100m_2026-10-20.jsonl`: RaBitQ-SSD (arXiv 2610.02652,
  github.com/VectorDB-NTU/RaBitQ-SSD 7ab80aa) on the same four 90M bases,
  same queries, ground truth, node and 32 threads, its paper defaults
  (C = N/1024, IRQ coarse quantizer), nprobe 8-1536, one CSV row per
  point (its own schema; recall is a fraction and tie-aware). Three
  configurations in `config`: 9-bit codes on SSD (paper default), raw
  vectors on SSD, and raw vectors on SSD with a 2-bit code in RAM, which
  needs `2026-10-20_rabitq_ssd_rawmemex.patch` (store=rawmemex: the B-bit
  estimate and its bound f_error*g_error/2^(B-1) prune before any page is
  read; 54 B/vector in RAM against 31 B for the 1-bit code). SPACEV was
  converted to float32, which RaBitQ-SSD reads (it does not read int8).
  RaBitQ-SSD reads through O_DIRECT with no buffer pool.
- `laion512_10m_2026-10-20.jsonl`: LAION-400M CLIP ViT-B/32 image
  embeddings (512-d, unit length): a 10M base from img_emb_0..10.npy,
  10K queries from img_emb_405.npy, exact top-100 ground truth (made with
  the fixed exact_gt; the first two were corrupt). IVF-PQ (nlist 16384)
  and BufANN with 32- and 64-byte PQ, RaBitQ-SSD with 256 and 512 bits
  in RAM (exbits and raw SSD records): PQ bytes = in-memory RaBitQ bits
  / 8, the paper's two memory budgets. `2026-10-20_laion512.sh` made it.

Both systems read through BufANN's buffer pool (O_DIRECT) from the same
NVMe-backed volume, after the same 1000-row warmup sampled from the base,
one process per setting. bufann_driver was built from a copy of
BufANN-CS395T with the pin_batch fix (785d976) applied.
