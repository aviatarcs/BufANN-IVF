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
- [ ] **Posting-list rebuild.** Fold `PostingListDelta` and `DynamicPQCodes`
      into fresh `PostingLists`/`PQMetadata` off to the side, publish with
      one atomic pointer store in `IVFPQSearchConfig`, reclaim the old
      structures after a grace period. Searches must run throughout; test
      with a search thread hammering during a rebuild. Then rewrite the
      index file (with the heap's current page count and cursor) so a
      prefix with inserts is loadable again and one with deletes stops
      resurrecting them on load (the file's RID table still marks deleted
      base vectors active; only the heap's occupancy bit says otherwise);
      the backend's inserted-tag maps must survive the fold or be persisted
      with it.

## Not planned

- Non-uniform PQ chunks (`dim % chunks != 0`): upstream's file supports it;
  add `chunk_offsets` to `PQMetadata` only if a dataset needs it.
- Two-level coarse quantizer: only if `nlist` grows to ~1e6.
