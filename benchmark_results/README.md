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

- `highdim_2026-11-20.jsonl`: IVF-PQ, BufANN and RaBitQ-SSD (7ab80aa) on
  four high-dimensional sets at RaBitQ-SSD's two memory budgets (PQ bytes =
  in-RAM RaBitQ bits / 8 = D/16 and D/8 bytes): GIST1M (960-d; queries =
  the first 10K rows of gist_learn), DBpedia-OpenAI-1M (1536-d, 990K base,
  10K held out), LAION-512 (as above, 10M) and Cohere wiki-22-12 768-d
  (VectorDBBench cohere_large_10m, 9.99M base, 10K held out). Exact top-100
  ground truth, 10K queries, 1000-row warmup from the base, 32 threads,
  recall@10, one process per setting. `2026-11-20_highdim_*` made it
  (`highdim.sh <dataset> prep|search_ivf|search_bufann|search_rabitq`;
  `fetch.sh`/`prep_data.py` make the data, `collect.py` the jsonl and
  tables). Rows carry `set`:
  - `nvme` (the result): searches read from /nvme, an LVM stripe over the
    node's four NVMe drives only (~1.8M 4 KB random reads/s in fio), with
    agent/highdim f23e6df: the buffer-pool pin-count fix (d113eca) and
    lowdim's re-rank work. IVF-PQ reads its re-rank pages uncached with
    O_DIRECT (`io_mode` direct, no pool, like RaBitQ-SSD) at
    `pipeline_depth` 0 and 2; BufANN keeps its 1 GB buffer pool. Every
    timed run held bench.lock while the other sessions' builds were
    paused (1-minute load at lock acquisition 7.6-12.0, median 10.8; the
    log header records it). One RaBitQ-SSD config per dataset was rerun
    after the IVF-PQ runs (`aba_repeat`): it moved 0-2%.
  - `tmpdata` (superseded): the first pass on /tmpdata, a stripe that also
    spans the SATA system disk and caps 4 KB random reads at ~0.46M/s for
    every system, at load 60-80 from concurrent builds, with binaries from
    before the pin-count fix and IVF-PQ through a 1 GB buffer pool
    (`io_mode` pool). Recall and pages/query match the nvme set; q/s does
    not. gist960 there has only IVF-PQ, and dbpedia1536 misses one
    adaptive IVF-PQ point (an abort from the pin-count bug).
  Pages are each system's own: IVF-PQ heap pages are 8 KB on DBpedia
  (6 KB records) and 4 KB elsewhere, BufANN's 8 KB on GIST and DBpedia,
  RaBitQ-SSD reads 4 KB pages; reads/s = q/s x pages/query. DiskANN's
  layout and fast_convert were extended for nodes larger than a sector
  (c41e3ab, dbee09e). RaBitQ-SSD's two-level k-means hit faiss 1.16's
  0-centroid split on Cohere (a 14-point coarse cell);
  `2026-11-20_highdim_kmeans_min1.py` gives every cell at least one.

  Best q/s at recall@10 >= target on /nvme (pages read per query); IVF-PQ
  the better of depth 0 and 2, RaBitQ-SSD the better of exbits and raw:

  | dataset | budget | system | R@10 90 | 95 | 98 | 99 |
  |---|---|---|---|---|---|---|
  | gist960 | 64 B / 512 bit | IVF-PQ | 7059 (200) | 4737 (400) | 2768 (800) | 1444 (1600) |
  | gist960 | 64 B / 512 bit | BufANN | 3380 (138) | 2293 (207) | 1162 (413) | - |
  | gist960 | 64 B / 512 bit | RaBitQ-SSD | 7065 (223) | 5923 (259) | 3943 (332) | 3116 (354) |
  | gist960 | 120 B / 960 bit | IVF-PQ | 8698 (100) | 6365 (200) | 4882 (200) | 3406 (400) |
  | gist960 | 120 B / 960 bit | BufANN | 6182 (67) | 4279 (100) | 2238 (199) | 1689 (266) |
  | gist960 | 120 B / 960 bit | RaBitQ-SSD | 9598 (130) | 6109 (151) | 4562 (159) | 3668 (163) |
  | dbpedia1536 | 96 B / 768 bit | IVF-PQ | 11974 (100) | 9658 (100) | 5462 (200) | 3007 (400) |
  | dbpedia1536 | 96 B / 768 bit | BufANN | 13458 (34) | 7613 (62) | 3251 (148) | 1699 (292) |
  | dbpedia1536 | 96 B / 768 bit | RaBitQ-SSD | 19779 (65) | 12012 (149) | 5807 (170) | 3425 (176) |
  | dbpedia1536 | 192 B / 1536 bit | IVF-PQ | 11384 (100) | 7127 (100) | 4929 (100) | 3649 (100) |
  | dbpedia1536 | 192 B / 1536 bit | BufANN | 12628 (34) | 9341 (49) | 4152 (113) | 2200 (221) |
  | dbpedia1536 | 192 B / 1536 bit | RaBitQ-SSD | 19684 (97) | 9833 (58) | 5794 (119) | 3231 (121) |
  | laion512 | 32 B / 256 bit | IVF-PQ | 5833 (400) | 2915 (800) | - | - |
  | laion512 | 32 B / 256 bit | BufANN | 2360 (288) | 1208 (570) | - | - |
  | laion512 | 32 B / 256 bit | RaBitQ-SSD | 12944 (120) | 9271 (164) | 5469 (241) | 2630 (335) |
  | laion512 | 64 B / 512 bit | IVF-PQ | 15347 (100) | 10528 (100) | 4704 (400) | 2471 (800) |
  | laion512 | 64 B / 512 bit | BufANN | 9536 (65) | 7606 (84) | 3375 (196) | 2318 (291) |
  | laion512 | 64 B / 512 bit | RaBitQ-SSD | 21212 (63) | 15114 (78) | 7070 (98) | 4167 (105) |
  | cohere768 | 48 B / 384 bit | IVF-PQ | 7385 (200) | 3085 (400) | 1386 (1600) | - |
  | cohere768 | 48 B / 384 bit | BufANN | 4378 (150) | 2302 (288) | - | - |
  | cohere768 | 48 B / 384 bit | RaBitQ-SSD | 9844 (145) | 4966 (232) | 3073 (282) | 1805 (308) |
  | cohere768 | 96 B / 768 bit | IVF-PQ | 7832 (200) | 5392 (200) | 2344 (200) | 1834 (400) |
  | cohere768 | 96 B / 768 bit | BufANN | 12955 (46) | 7247 (81) | 3243 (186) | 1670 (364) |
  | cohere768 | 96 B / 768 bit | RaBitQ-SSD | 10354 (83) | 4590 (96) | 2559 (101) | 1955 (110) |
