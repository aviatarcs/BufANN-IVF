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
- [ ] **Measure QPS-recall against nlist.** LindormVector (SIGMOD Companion
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
- [ ] **Centroid search that scales with nlist.** The per-query centroid
      GEMV reads nlist x aligned_dim floats: 32 MB at 64K, ~125 GB/s at
      4K q/s. Search a proximity graph over the centroids (the in-repo
      in-memory Vamana; the paper uses HNSW, refines its bottom layer and
      reinserts zero in-degree nodes) for a multiple of nprobe candidates,
      re-rank them exactly, keep nprobe. Oracle: the exact GEMM probe set;
      report probe-set recall and end-to-end QPS-recall. The paper's PCA
      compression of centroids during traversal pays at 768-1024 dims, not
      obviously at 128; measure before adding it. Depends on the nlist
      measurement.
- [ ] **Build time at large nlist.** Assignment of 9M to 32K centroids took
      378 s, ~200 GFLOP/s against a ~1.2 TFLOP/s peak; find where it goes
      before changing the algorithm. Then, if still needed, assign through
      the centroid graph (k-means and the final assignment), seeding each
      point from its previous centroid; the paper reports 5-10x with the
      same MSE. Oracle: brute-force assignment within the
      compute_closest_centers tolerance, and k-means MSE against the exact
      run. Depends on the nlist measurement.
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
      lists (large nlist) leave partial blocks; measure with the nlist
      chosen above if that has landed.
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
