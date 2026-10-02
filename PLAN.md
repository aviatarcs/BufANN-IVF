# IVF-PQ plan

The worker takes the first unchecked item whose dependencies are checked,
unless `REVIEW.md` has open findings, which come first. Each item is one
PR-sized change with tests. Check an item off in the same commit that
completes it, citing the commit subject.

## Queue

- [x] **Heap reopen.** `RawVectorHeap::open_existing(path, layout, next_flat_slot, allocated_pages)`
      driven by `IVFPQIndexFileHeader`; refuse a file whose size disagrees
      with `allocated_pages * page_size`. Put a page id in the 8-byte page
      header (bump `IVF_PQ_INDEX_VERSION` and the bulk writer) and verify it
      on `read_vector`, so a misplaced page is detected. Tests: build, close,
      reopen, read every vector back; corrupt one page id and see it rejected.
      (Done in "Add raw-vector heap reopen and page-id headers"; the header
      now also records the slot cursor, index file version 2.)
- [x] **Query path, single query.** `ivf_pq_search(const IVFPQIndex&, const RawVectorHeap&, const float* q, k, nprobe, rerank_m) -> ids+dists`
      in `include/bufann/ivf_pq_search.h`, implementing exactly the reference
      search in `tests/ivf_pq_recall_test.cpp` (which then calls it instead of
      its own loop). Centroid distances via one GEMM against the padded
      centroids; PQ table in a per-query scratch; re-rank from the heap.
      Recall must match the reference to within ties.
      (Done in "Add the single-query IVF-PQ search path"; the reference loop
      now lives in `tests/ivf_pq_search_test.cpp` as the oracle.)
- [x] **Query path, batched.** Batch the centroid GEMM over many queries and
      parallelize across queries with OpenMP; measure QPS on SIFT1M at nprobe
      16 and 64 and record it in the commit message. (Add the batched IVF-PQ
      search path)
- [x] **Wire `BufANNConfig::index_type`.** `IndexType::IvfPq` selects the
      IVF-PQ build (steps 1-6 from `ivf_nlist`, `pq_chunks`) and search
      (`ivf_nprobe`) through the existing BufANN entry points; the graph path
      is untouched. Test: build + search through the public API on the blob
      fixture. (Wire IndexType::IvfPq into the BufANN API)
- [x] **Free-slot grace period.** Freed slots enter a deferred list and are
      only reusable after every search that could hold the slot has finished
      (epoch counter). Required before concurrent search + delete. Test with
      threads: a reader holding a slot across a free never observes the
      replacement's bytes. (Defer freed heap slots until in-flight readers
      have left)
- [x] **Mutation: insert.** Allocate a slot (free list or grow), write raw
      bytes, assign to nearest centroid, encode PQ code into `DynamicPQCodes`,
      append to `PostingListDelta::pending_inserts`, set RID active. Search
      merges pending inserts. Tests: inserted vectors are found at nprobe
      covering their partition. (Add IVF-PQ insert; the delta is consulted by
      search and published through bufann_insert)
- [x] **Mutation: delete.** Tombstone in `PostingListDelta`, clear RID active
      bit with `store_rid` (seq_cst, which the grace period relies on) before
      freeing the slot through it. Search drops tombstoned ids. (Add IVF-PQ
      delete; tombstoned base ids are dropped before selection and freed
      slots go through the grace period)
- [x] **Measure QPS-recall against nlist.** LindormVector (SIGMOD Companion
      '26) finds the best nlist at 100M near 200K (~20 sqrt(N)); we use
      ~1.4 sqrt(N) at 9M (4096) and ~1.7 sqrt(N) at 90M (16384), and at
      9M / nprobe 64 a query scans ~140K codes while its centroid GEMM reads
      2 MB, so the scan is the cost a finer partition cuts. Build SIFT10M 9M
      at nlist 4096, 16384, 32768 and 65536 with the default training sample
      (64 points per centroid; the earlier sweep's ad-hoc build trained 32K
      and 64K on 16 and 5), run `ivf_pq_query_bench` over nprobe for each,
      and record QPS at recall 95/98/99, build time and RSS. Add the sweep
      as a script so it can be rerun. The items below that depend on this
      one are only worth doing if a larger nlist wins.
      (Done in "Add the nlist sweep script and record the SIFT10M 9M
      sweep". A larger nlist wins. Best q/s, single-query / batched
      centroid search, 32 threads:

      | nlist | build  | recall >= 95   | >= 98          | >= 99          |
      |-------|--------|----------------|----------------|----------------|
      | 4096  | 85 s   | 3909 / 3648    | 2106 / 1929    | 2106 / 1929    |
      | 16384 | 310 s  | 5797 / 5990    | 3400 / 3415    | 3400 / 3415    |
      | 32768 | 976 s  | 7256 / 8424    | 4828 / 5351    | 2934 / 3074    |
      | 65536 | 3562 s | 4260 / 6442    | 3176 / 4378    | 3176 / 4378    |

      32768 gives 2.3x (batched 2.8x) the q/s of 4096 at recall 98. At
      65536 the single-query path has half the batched one's q/s; the two
      differ only in the centroid search (a GEMV per query, 32 MB read,
      against one GEMM per batch). Build peak RSS 4.5 / 3.4 /
      4.2 / 6.2 GB; bench RSS 1.2-1.6 GB at every nlist. Logs:
      `/var/tmp/bufann-ivf-evpeng/nlist/`.)
- [x] **Centroid search that scales with nlist.** The per-query centroid
      GEMV reads nlist x aligned_dim floats: 32 MB at 64K, ~125 GB/s at
      4K q/s. Search a proximity graph over the centroids (the in-repo
      in-memory Vamana; the paper uses HNSW, refines its bottom layer and
      reinserts zero in-degree nodes) for a multiple of nprobe candidates,
      re-rank them exactly, keep nprobe. Oracle: the exact GEMM probe set;
      report probe-set recall and end-to-end QPS-recall. The paper's PCA
      compression of centroids during traversal pays at 768-1024 dims, not
      obviously at 128; measure before adding it. Depends on the nlist
      measurement, which found the GEMV halves single-query q/s at 65536
      (5180 vs 8866 batched at recall 90), so the target is 65536 and up
      at the batched path's q/s or better.
      (Done in "Search the IVF centroids through a proximity graph". A
      Vamana graph of its own (R 32, build L 100, alpha 1.2) rather than
      the in-repo Index, which builds only from a file, seeds its build
      from random_device, sets the global OpenMP thread count and
      allocates hash sets per query. Built after load in 0.47 s at 32768
      and 1.24 s at 65536; the beam's distances are exact, so it is its
      own re-rank. SIFT10M 9M, 32 threads, beam 2 x nprobe, single-query
      q/s against the exact path's single / batched, recall unchanged to
      0.02 points, probe-set recall >= 0.998:

      | nlist | nprobe | recall | exact single / batched | graph  |
      |-------|--------|--------|------------------------|--------|
      | 32768 | 128    | 96.26  | 7163 / 7782            | 10341  |
      | 32768 | 256    | 98.55  | 4788 / 5271            | 5978   |
      | 65536 | 64     | 88.91  | 5508 / 9746            | 23285  |
      | 65536 | 256    | 97.57  | 4196 / 6462            | 8496   |
      | 65536 | 512    | 99.11  | 3135 / 4374            | 4823   |

      Best at recall >= 95 / 98 / 99 moves from 8424 / 5351 / 3074
      (32768, batched) to 10341 / 5978 (32768, graph) / 4823 (65536,
      graph). A 4 x beam costs 8-17% q/s for no recall. PCA not tried:
      the graph visits a few hundred 128-d centroids, not the cost.
      Logs: `/var/tmp/bufann-ivf-evpeng/nlist/bench_graph_n*.log`.)
- [ ] **Graph centroid search in the backend.** The backend and the batched
      path still use the exact GEMM/GEMV. Build the graph when the backend
      loads an index (or persist it in the index file if load time
      matters at 90M+), rebuild it with the centroids, and expose
      centroid_L (as a multiple of nprobe, default 2) next to nprobe.
      Test through the public API against the exact path at a full beam.
- [ ] **nlist sweep with graph search.** With the per-query centroid cost
      gone, the sweep's optimum may move up: rerun
      `GRAPH_BEAMS=2 scripts/nlist_sweep.sh` at 9M over nlist 32768,
      65536 and 131072 (~20 sqrt(N) is 60K) with nprobes between the
      powers of two, and at 90M; build time (59 min at 65536) is then
      the cost, so this and the build-time item inform each other.
      First points, beam 2 x nprobe, best q/s at recall >= 95 / 98 / 99
      (nprobe): 9M at 65536 10773 (192) / 6209 (384) / 4871 (512), ahead
      of 32768's 10472 / 6001 / 4231 everywhere; 90M at 16384 1919 (48) /
      763 (128) / 518 (192); 90M at 131072 (built in 5 h 23 min, peak RSS
      12.5 GB) 3042 (192) / 1637 (384) / 869 (768). At 90M the scan is the
      cost: recall 99 scans ~530K codes per query (768 lists of ~690)
      against ~70K at 9M. Logs: `/var/tmp/bufann-ivf-evpeng/final/`.
- [ ] **Build time at large nlist.** Assignment of 9M to 32K centroids took
      378 s, ~200 GFLOP/s against a ~1.2 TFLOP/s peak; find where it goes
      before changing the algorithm. Then, if still needed, assign through
      the centroid graph (k-means and the final assignment), seeding each
      point from its previous centroid; the paper reports 5-10x with the
      same MSE. Oracle: brute-force assignment within the
      compute_closest_centers tolerance, and k-means MSE against the exact
      run. Depends on the nlist measurement: whole builds at 9M took 85,
      310, 976 and 3562 s at nlist 4096 to 65536, ~3.4x per doubling of
      nlist (the training sample grows with nlist, and so does each pass).
      Where it goes (9M, nlist 32768, 996 s, phase timers in the build
      log): k-means++ seeding 376 s, 15 Lloyd's iterations 440 s,
      assignment and heap write 138 s, PQ training and encoding 38 s.
      k-means++ does two serial passes over the sample per centroid
      picked, so it grows with nlist x sample (~16x from 32768 to 131072);
      Lloyd's and the assignment run their GEMMs at ~550-600 GFLOP/s and
      grow with nlist x points. Scaled to 90M at 131072 each is ~1.5-2 h,
      matching the 5 h 23 min measured. Order: final assignment through
      the graph (largest at 90M, and exact-checkable), then Lloyd's, then
      k-means++ (parallel sums, or seeding from a subsample).
      Final assignment through the graph: done in "Assign vectors through
      the centroid graph from nlist 32768 on" (138 -> 35 s at 9M/32768,
      recall within 0.02 points; ~8x at 90M/131072 by extrapolation).
      Lloyd's through the graph: done in "Run Lloyd's iterations through
      the centroid graph from nlist 32768 on" (2.1M x 32768 from the same
      init: 429 s -> 60 s, mean squared distance +0.006%; the 9M build
      16:36 -> 8:31, recall within 0.1 points). k-means++: done in "Drop
      k-means++'s two serial passes per pick" (374 -> 302 s) and "Seed
      k-means++ from 4 points per centroid" (302 -> ~20 s; the 9M build
      7:26 before it).
- [ ] **nlist cost model.** For 90M+, where one build takes 41 min: sample
      m vectors, cluster them at each candidate nlist, take ~100 sample
      queries with exact neighbours within the sample, estimate nprobe as
      the centroids no farther than the farthest one holding a true
      neighbour, scan cost from the codes in those lists scaled by N/m and a
      measured per-code time, plus measured centroid-search time; pick the
      minimum (paper section 4.1). Validate against the 9M measurements.
      Depends on the nlist measurement.
- [ ] **4-bit fast-scan PQ.** The scan does one scalar table lookup per
      chunk per code. 64 x 4-bit sub-quantizers are the same 32 B/vector as
      32 x 8-bit for SIFT, and their tables fit SIMD registers (PQFastScan,
      blocks of 32 codes). Prototype the kernel on the SIFT10M codes against
      the scalar scan as oracle, then compare QPS-recall end to end. Small
      lists (large nlist) leave partial blocks; measure at nlist 32768
      (~275 codes per list at 9M), the sweep's best.
- [ ] **Raw-vector heap through BufANN's buffer pool.** The heap is read
      with plain pread, so the query bench serves re-rank reads from an
      unbounded OS page cache (the 9M heap, 1.2 GB, is fully cached after
      warmup) while BufANN's runs pay 75 misses per query against a 1 GB
      pool (262144 frames; I/O is 70% of its latency). Serve the heap's
      pages through BufANN's buffer pool so both are measured at the same
      pool size and miss accounting, then compare at 9M and 90M. This is
      what makes the IVF-PQ vs BufANN numbers like for like; it subsumes
      the cold-heap item's bench mode below.
      Serving is done ("Let the raw-vector heap serve its pages through
      BufANN's buffer pool", "Re-rank from one batched heap read ..."):
      `ivf_pq_query_bench --buffer_pool_frames` (default 262144, BufANN's
      1 GB; 0 = page cache) reports hits, misses/query and I/O time under
      bufann_driver's field names, and a query's re-rank pages are fetched
      with one pin_batch. Still open: the 9M and 90M comparison on a quiet
      node. Two things to settle first. (1) The budget: 1 GB is 87% of
      the 9M heap (300000 pages) but ~28% of BufANN's 9M graph, and
      IVF-PQ's in-memory part (codes twice, lists) is ~720 MB at 9M, so
      compare at equal total RSS as well as equal pool size. (2) Warmup:
      a 4-thread smoke run with a 1000-query warmup showed 22.2 / 13.5
      misses/query at nprobe 16 / 64 (recall unchanged at 88.248 /
      97.987), mostly first touches, so both systems should use the same
      warmup (or preload) before misses are compared.
      9M, done on a quiet node: both through the pool (O_DIRECT) on the
      same volume (/tmpdata; /var/tmp is the SATA system disk, which
      capped IVF-PQ at ~95K reads/s in a first run), the same 1000-row
      base-sampled warmup file, one process per setting, 32 threads,
      10K queries. IVF-PQ nlist 65536, graph beam 2 x nprobe, rerank 100.
      Best q/s at recall >= 90 / 95 / 98 / 99 / 99.5:
      BufANN (1 GB pool, RSS ~1590 MB) 11513 / 6827 / 5338 / 3088 / 2360;
      IVF-PQ, 1 GB pool (RSS 1.9-2.1 GB) 12932 / 9280 / 5781 / 4629 / 3319;
      IVF-PQ, 160000 frames (RSS 1.47-1.68 GB) 8263 / 7854 / 5407 / 4389 /
      3209 (rerun on the pin_batch fix, 785d976; before it the 160000-frame
      runs were ~15% faster at nprobe <= 128, the rest within 3%). IVF-PQ misses a constant 23.4 (47.5) pages per query, BufANN
      23 to 143 growing with L; IVF-PQ's I/O share falls from ~50-80% at
      low nprobe to 4-6% at nprobe 768, where the PQ scan is the cost.
      Logs: `/var/tmp/bufann-ivf-evpeng/cmp10m/` (bufann_L*, nvme_ivf_*).
      One run in ~40 aborted before the pin_batch fix; none of 20 after.
      BufANN's numbers come from BufANN-CS395T's bufann_driver, whose
      pin_batch has the same race (recall matched its earlier runs, so no
      sign it bit at 1 GB). Still open: 90M.
- [ ] **Re-rank reads under a cold heap.** The bench reads the heap through
      a warm page cache, while BufANN's numbers pay ~75 buffer-pool misses
      per query; the re-rank's up to rerank_m (100) random reads are our
      counterpart of the paper's batch get. Add a cold-heap mode to the
      bench (O_DIRECT or a bounded cache), issue a query's re-rank reads
      concurrently (the aligned async reader DiskANN already has) instead of
      one pread at a time, and sweep rerank_m for recall.
- [ ] **Posting-list rebuild.** With a large nlist, lists are short enough
      to rebuild one at a time (copy, fold its delta, publish the list's
      pointer, reclaim after a grace period) rather than folding the whole
      index; decide the granularity after the nlist measurement. Fold `PostingListDelta` and `DynamicPQCodes`
      into fresh `PostingLists`/`PQMetadata` off to the side, publish with
      one atomic pointer store in `IVFPQSearchConfig`, reclaim the old
      structures after a grace period. Searches must run throughout; test
      with a search thread hammering during a rebuild. Then rewrite the
      index file (with the heap's current page count and cursor) so a
      prefix with inserts is loadable again (deletes already survive a
      load through the heap's occupancy bitmap, `ivf_pq_recover_deletes`);
      the backend's inserted-tag maps must survive the fold or be persisted
      with it.
- [ ] **One in-memory copy of the PQ codes.** Search reads the posting-order
      `PostingLists::codes`; `PQMetadata::codes` (by id) is kept only for the
      index-file writer and the tests' oracles, doubling the codes' memory
      (32 B/vector at 32 chunks: +2.7 GB at 90M). Persist the codes in
      posting order (bump the index version), drop the by-id copy after load,
      and have the rebuild produce posting-order codes directly. Depends on
      the rebuild above.

## Not planned

- Non-uniform PQ chunks (`dim % chunks != 0`): upstream's file supports it;
  add `chunk_offsets` to `PQMetadata` only if a dataset needs it.
- Two-level coarse quantizer: only if `nlist` grows to ~1e6.
