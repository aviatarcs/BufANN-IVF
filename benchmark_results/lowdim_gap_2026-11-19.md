# IVF-PQ vs RaBitQ-SSD at low dimension (96-128 d)

Question: why does IVF-PQ (agent/rerank-batches) not match RaBitQ-SSD
(arXiv 2610.02652, 7ab80aa) on SIFT, DEEP, SPACEV and Turing?

## 1. There is no low-dimension gap in IVF-PQ's disfavour

The recorded 100M runs (`100m_2026-10-15.jsonl`,
`rabitq_ssd_100m_2026-10-20.jsonl`; same node, queries, ground truth, 32
threads) show IVF-PQ ahead of every RaBitQ-SSD configuration on all four
low-dimensional sets. Best q/s at recall >= 90 / 95 / 98 / 99
(`2026-11-19_lowdim_gap_table.py`):

| 90M base | IVF-PQ | RaBitQ 9-bit (paper default) | RaBitQ raw | RaBitQ 2-bit in RAM + raw (rawmemex) |
|---|---|---|---|---|
| SIFT 128d   | 14603 / 11791 / 8405 / 6242 | 5825 / 4234 / 2924 / 1912 | 6024 / 4356 / 2994 / 2355 | 12373 / 8075 / 5064 / 3838 |
| DEEP 96d    | 14520 / 11870 / 6574 / 4110 | 5839 / 4348 / 2763 / 1867 | 5080 / 3908 / 2914 / 2100 | 12654 / 9023 / 5737 / 3763 |
| SPACEV 100d | 14825 / 6098 / 1803 / -     | 6805 / 2818 / 1011 / -    | 6374 / 2772 / 1008 / -    | 10886 / 4074 / 1282 / -    |
| Turing 100d | 10320 / 3487 / - / -        | 3070 / 1168 / - / -       | 2920 / 1144 / - / -       | 4972 / 1762 / - / -        |

IVF-PQ leads the paper-default RaBitQ-SSD by 2.1-3.4x and the 2-bit
rawmemex variant by 1.2-2.1x (equal at SPACEV >= 90 within 1.4x). The
crossover is at higher dimension: on LAION 512d (`laion512_10m_2026-10-20.jsonl`)
RaBitQ-SSD leads IVF-PQ 1.3-1.6x at its 512-bit budget. The paper itself
evaluates only 512-1536d sets, so it makes no low-dimension claim to match.
What follows explains why the ranking flips with dimension, measured at
1M-10M on this node, and what in IVF-PQ is still worth taking from RaBitQ-SSD.

## 2. Why RaBitQ-SSD loses at low dimension (ranked by measured effect)

### (1) The 1-bit bound cannot prune at D ~ 100: 3-8x more pages per query

RaBitQ-SSD's paper budget is one bit per dimension in RAM (16 B + factors
at 128d, against IVF-PQ's 32 B PQ code, i.e. 2-2.7 bits/dim). A candidate
is read from SSD unless its lower bound est - f_error * g_error exceeds the
current k-th distance, and the bound's width is
2 * 1.9 * ||o-c|| ||q-c|| * sqrt(1 - <o',o>^2) / <o',o> / sqrt(D)
(`rabitq_impl.hpp`, kConstEpsilon = 1.9): ~0.25 of ||o-c|| ||q-c|| at 128d,
0.13 at 512d, 0.09 at 960d. Measured share of scanned codes that pass the
bound at nprobe 16 (RaBitQ-SSD's own scan counters, 32 threads):

| | D | passed / scanned | pages / query at recall ~91-93 |
|---|---|---|---|
| SIFT1M  | 128 | 4313 / 17087 = 25% | 121 (exbits), 153 (raw) |
| DEEP10M | 96  | 3914 / 17265 = 23% | 174 (exbits), 221 (raw) |
| GIST1M  | 960 | 1921 / 20247 = 9.5% | 100 (exbits), 102 (raw) |

At 100M the default reads 202-445 pages per query at recall 90 and
370-1007 at 95, against IVF-PQ's 37-55 and 37-97 misses (a fixed re-rank of
100 PQ candidates, part of it served by the pool). RaBitQ-SSD is then
IOPS-bound: 1.37-1.68M reads/s, the four-NVMe ceiling of that node, at
every one of those points, so its q/s is ceiling / pages
(5825 x 275 = 1.60M on SIFT). IVF-PQ used 0.46-1.15M reads/s at the same
recalls. Giving RaBitQ a second bit in RAM (rawmemex B2, 54 B/vector, more
than our 32 B) cuts its pages 2.9-4.2x (SIFT 275 -> 96) and doubles its
q/s, which is the whole of its recovery; it then stops being IOPS-bound
(0.2-1.3M) and is still behind IVF-PQ.

This is the low-dimension effect: at fixed bits per dimension the code
shrinks with D, and RaBitQ's error shrinks only as 1/sqrt(D), so the
bound-gated read set grows as D falls. PQ at 32 B keeps ~2+ bits/dim at
96-128d and ranks well enough that a fixed top-100 re-rank reaches 95-99
(SIFT1M here: re-rank 50 -> 92.3 / 95.5 / 98.1 at nprobe 16 / 24 / 48,
re-rank 100 -> 99.0 at nprobe 48). At 960d the same 32 B (or 120 B) PQ
code is coarse: GIST1M with 120 B needs a re-rank of 200 (172 pages) for
95, while RaBitQ-SSD 9-bit needs 113 pages and rawmemex 67.

### (2) The 9-bit SSD record does not help at low D

Record size matters only if several useful records share a page. In
RaBitQ-SSD's posting-order layout a page read yields only 1.3-2.5 records
that survive to the re-rank (boosts per page: SIFT1M exbits 2.5 at
nprobe 16, GIST1M exbits 1.3, SIFT1M raw 1.5). At 100M the 9-bit and raw
stores read the same pages on SIFT (275 vs 272 at recall 90; uint8 raw is
128 B, the 9-bit record 152 B) and 17% fewer on DEEP (274 vs 330; float
raw is 384 B). Its 9-bit records are therefore not where its throughput
comes from, and for uint8 SIFT the 9-bit record is larger than the raw one.

For IVF-PQ (`2026-11-19_lowdim_page_locality_sim.py`: FAISS IVF-PQ with our
nlist and 32 B codes, each query's top-R shortlist mapped to pages), pages
touched by a re-rank of R = 100 at recall ~92-99:

| | raw, id order (today) | raw, posting order | 9-bit, id order | 9-bit, posting order |
|---|---|---|---|---|
| SIFT1M (7 raw / 26 9-bit per page)  | 99.6 | 85.6-88.2 | 98.6 | 70.0-76.3 |
| DEEP10M (10 raw / 34 9-bit per page), recall 90.5-97.9 | 100.0 | 92.2-93.1 | 100.0 | 77.8-80.5 |

So a posting-order 9-bit store should save 20-30% of re-rank pages at
R = 100 (DEEP 20-22%, SIFT 24-30%) and 27-34% at R = 200, not the 3.4-3.7x
the record size suggests: the shortlist of one query spreads over many
lists and over many pages within each list. All of the saving comes from
the posting-order layout with a dense record, none from 9 bits in id order.

**Measured (agent/ssd-9bit 25ddf59, `logs/9bit/runs/deep10m_{raw,ex9}.log`,
DEEP10M, same nlist 16384 / 32 B PQ):** the 9-bit store touched
78.9-82.2 pages per query at R = 100 (nprobe 32-256, recall 90.7-98.5)
against 100.0 for raw, i.e. 18-21% fewer, and 138.7-150.2 at R = 200
(25-31% fewer); the simulation's 77.8-80.5 and 136-145 are within 1-3
pages. Recall: ranking by the 9-bit estimate alone loses 0.13-0.31 points
against raw at the same nprobe and R (95.09 vs 95.32 at nprobe 64,
R = 100); re-ranking its 20 nearest exactly (20 more raw pages) restores
raw's recall to the second decimal. The page saving is modest, but the
store is 3.4x smaller, so the same 1 GB of pool (768 MB ex + 256 MB raw
vs 1 GB raw) misses 2.2x less often (31.6-32.6 vs 71.4 misses per query
at R = 100). Best q/s at recall >= 90 / 95 / 98 / 99 on that run: raw
8929 / 7861 / 3169 / 2032; 9-bit 13627 / 9066 / 5728 / 2059; 9-bit + 20
exact 8806 / 7661 / 4657 / 3377, i.e. 1.5 / 1.15 / 1.8 / 1.7x, and it
comes from the pool hit rate, not the page count. (Both runs at load
average 60-85; q/s points are noisy, pages and misses are not.)

### (3) I/O overlap: the one structural advantage RaBitQ-SSD has

IVF-PQ fetches a query's re-rank pages with one blocking pin_batch, so the
thread idles for the whole read: at 100M the fetch is 47-53% of query
latency at recall 90 on SIFT/DEEP (31-38% on Turing/SPACEV, 15-16% at 95
where the scan dominates). RaBitQ-SSD keeps 8 reads in flight per thread and
scans the next list meanwhile (cpu_util 2900-3100% of 32 cores in every
run). This is the lever for IVF-PQ at low dimension, where the scan and
centroid work per query is small (scan 27% / centroid 17-22% of latency at
recall 90).

Measured on DEEP10M (32 threads, 256 MB pool), the difference shows up as
reads per second, not only as idle threads. IVF-PQ's fetch is 88-97% of its
latency and it sustained 0.23-0.34M reads/s across the sweep, while
RaBitQ-SSD got 0.40-0.45M out of the same volume. That is the one place
where RaBitQ-SSD beats IVF-PQ on a low-dimensional set here: its 2-bit
rawmemex reads more pages (59 at recall 91, 72 at 96) than IVF-PQ (46
misses at R = 50) and is still faster (7572 vs 6776 q/s at >= 90, 6238 vs
5867 at >= 95). A back-to-back control (`2026-11-19_lowdim_control_deep10m.sh`,
iostat alongside; the first IVF-PQ leg overlapped an unlocked run from
another session and was repeated) gave the same split:

| DEEP10M control, 21:54 | recall | q/s | pages (misses) / query | reads/s | NVMe r_await |
|---|---|---|---|---|---|
| IVF-PQ np48 R50 (two runs)  | 91.98 | 6005 / 6452 | 46.4 | 0.28 / 0.30M | 0.24-0.29 ms |
| IVF-PQ np64 R100 (two runs) | 95.28 | 2633 / 2799 | 92.7 | 0.24 / 0.26M | |
| RaBitQ rawmemex B2 np24     | 91.05 | 7438 | 60 | 0.44M | 0.08 ms |
| RaBitQ rawmemex B2 np48     | 95.95 | 5850 | 76 | 0.45M | |
| RaBitQ 9-bit np24 / np48    | 90.9 / 95.6 | 1936 / 1399 | 174 / 230 | 0.32-0.34M | (sda r_await 1.1-1.3 ms) |

All reads are 4 KB and spread evenly over the five stripe members. In the
IVF-PQ phases each NVMe drive served ~50K reads/s at 0.24-0.30 ms; in
RaBitQ's rawmemex phase ~85K/s at 0.08 ms, with the SATA member at
0.44-0.57 ms in both. A query's pin_batch completes only when its slowest
read does, and about one read in five of every batch lands on the SATA
disk, so each batch waits on the SATA latency; RaBitQ's continuous 8-deep
stream per thread keeps the fast drives busy meanwhile. (Its 9-bit store,
which reads up to 300 pages per query, saturated the SATA disk instead.)

More threads do not fix it: at 64 query threads IVF-PQ got 3035-3252 q/s at
R = 100 (-4 to +27% over 32 threads, 0.27-0.30M reads/s), at 128 threads
2511-2908 (no gain), the fetch growing to 20-49 ms per query. With the heap
in the page cache (pool 0, the 3.9 GB heap fits in RAM) IVF-PQ reached
13715 q/s at recall 95.3, which bounds what overlap could buy at 10M:
~4x over the 32-thread 0.29M reads/s figure on this volume. (SIFT1M IVF-PQ
sustained 0.43-0.47M reads/s; why its batches did not hit the same ceiling
was not investigated.)

### (4) Smaller differences

- Coarse quantizer: IRQ over C = N/1024 vs our graph over N/687; our
  centroid search is 15-36% of latency at 100M. Not a cause of a gap in
  RaBitQ's favour; IRQ's cost is in its own runs' CPU time.
- Buffer pool vs O_DIRECT without a pool: the 1 GB pool serves part of the
  re-rank (47 misses of 100 re-ranked on SIFT100M at recall 90) and costs
  CPU per pin; RaBitQ-SSD caches nothing.
- Per-query probe cutoff: both have one (our tau cutoff; RaBitQ's drain
  alpha 0.4 prunes the tail of pending pages).

## 3. Recommendations

1. Overlap the re-rank fetch with work: keep a fixed number of reads in
   flight per thread and complete them out of order (scan or serve another
   query meanwhile), or start the fetch per list as the shortlist forms.
   More query threads than cores is not a substitute (-4 to +27% at 64
   threads, nothing at 128 on DEEP10M). Expected gain at low D: up to the
   fetch share at 100M (1.3-1.9x at recall 90); on this node's volume,
   where the batch also waits on the SATA member, up to the 0.29 -> 0.45M
   reads/s RaBitQ-SSD reaches (1.5x).
2. Lay out the re-rank records in posting order with a dense record
   (agent/ssd-9bit's store does this): 18-21% fewer pages at R = 100 on
   DEEP10M, as simulated, and 2.2x fewer misses from the same 1 GB of pool;
   keep 20 exact re-ranks to hold raw's recall. For uint8 data (SIFT),
   posting-order raw records (128 B) are smaller than 9-bit ones (152 B);
   use them.
3. Keep the fixed-R PQ re-rank at low D; RaBitQ's bound-gated read is the
   right tool only where its bound is tight (>= ~512 bits per vector).
4. For comparisons on this node: /tmpdata is striped over the four NVMe
   drives and the SATA disk, which caps 4 KB random reads at ~0.45M/s for
   both systems (section 4); rerun I/O-bound comparisons on an NVMe-only
   volume, as the October runs did.

## 4. Measurements on this node (2026-11-19)

All runs 32 threads, top-10, 10K queries (GIST 1K), RaBitQ-SSD with its
defaults (IRQ, drain alpha 0.4, QD 8/thread, O_DIRECT, C = N/1024),
IVF-PQ with nlist ~N/1000 (1M) or N/610 (10M), beam 1.25 x nprobe, fixed
nprobe, buffer pool as noted, 1000 base-sampled warmup rows; each run under
`flock /tmpdata/work/bench.lock`. Results: `lowdim_2026-11-19.jsonl`
(`2026-11-19_lowdim_collect.py`). Caveats:

- I/O ceiling: no run here exceeded 0.47M reads/s whatever its pages per
  query: RaBitQ-SSD mostly 0.40-0.45M at 36-564 pages (0.28-0.31M at nprobe 4) (0.32-0.34M for its 9-bit
  store in the control), IVF-PQ 0.43-0.47M on SIFT1M but 0.23-0.34M on
  DEEP10M (section 2, item 3). The volume's SATA stripe member caps them, not the
  NVMe drives (lvs: /tmpdata = 5 stripes, /dev/sda4 + 4 NVMe). q/s on this
  node is therefore at most ~0.45M / pages, and pages per query (hardware
  independent) is the quantity compared above.
- Other agents ran builds and downloads on the same volume throughout
  (load average 50-85); individual q/s points dip by up to 3x. Pages,
  pass rates and recall are unaffected.
- The IVF-PQ pool held 23% (SIFT1M, 32768 frames; 8192 frames threw
  "every frame of the shard pinned" at re-rank 100) or 3.3% (GIST1M) /
  6.5% (DEEP10M) of the heap.

SIFT1M (128d), best q/s at recall >= 90 / 95 / 98 / 99 (pages per query;
for IVF-PQ, pool misses):
IVF-PQ 13238 (35) / 13047 (35) / 10833 (35) / 6679 (70);
RaBitQ 9-bit 3100 (121) / 2208 (187) / 2208 (187) / 2205 (204);
raw 2929 (153) / 2454 (175) / 1803 (243) / 1803 (243);
rawmemex B2 9748 (45) / 8892 (49) / 7401 (59) / 6724 (64).

GIST1M (960d, IVF-PQ 120 B): IVF-PQ 4952 (86) at >= 90, max 98.4 at
re-rank 200 (172 pages); RaBitQ 9-bit 3951 (100) / 3191 (122) / 2550 (132)
/ 2089 (137); rawmemex B2 7526 (64) / 5617 (67) / 3599 (75) / 2511 (75).

DEEP10M (96d, IVF-PQ nlist 16384, 256 MB pool unless noted):
IVF-PQ 6776 (46) / 5867 (46) / 2886 (93) / 1873 (93);
IVF-PQ 1 GB pool 4034 (71) / 4034 (71) / 3166 (72) / 1632 (72);
RaBitQ 9-bit 2451 (174) / 1919 (229) / 1228 (317) / 1131 (388);
raw 1986 (221) / 1496 (280) / 1223 (344) / 957 (425);
rawmemex B2 7572 (59) / 6238 (72) / 4804 (90) / 3147 (100).
RaBitQ pass rate at nprobe 16: 9-bit 23%, raw 17%, rawmemex B2 9.0%
(1547 / 17266). (The 1 GB-pool run's best points at >= 90 sit at higher
nprobe because its R = 100-only grid had no R = 50.)

## 5. Not verified

- Nothing at 100M was rerun (the 100M data is not on this node); the 100M
  conclusions rest on the October files, whose per-phase timings and
  page counts are internally consistent with the 1M-10M measurements.
- Absolute q/s on this node are not comparable to October's (SATA-capped
  volume, shared load).
- The 9-bit store was measured only on DEEP10M (agent/ssd-9bit 25ddf59, run
  by that session, not rerun here). Its SIFT1M and GIST1M ex9 runs aborted
  (diskann::ANNException, exit 134) after nprobe 4-6, which that session is
  debugging; the points that exist agree with the simulation's direction
  (SIFT1M nprobe 4, R = 100: 62.0 pages vs raw 99.6, recall 68.9 in both;
  GIST1M, 3-4 records per page: 96.1-96.7 vs 100). The SIFT prediction
  (24-30% fewer at R = 100, recall >= 90) is therefore unchecked.
- Some 9-bit DEEP10M points report up to 0.64M ex-store misses/s, above
  every other run on this volume; whether a miss shared by concurrent
  queries is counted once per query was not checked. Its page counts and
  recall do not depend on this.
- The SATA-latency explanation of IVF-PQ's lower reads/s on DEEP10M rests on
  per-device iostat in one control; it was not tested by moving the heap to
  an NVMe-only volume, and SIFT1M's higher reads/s is unexplained.
