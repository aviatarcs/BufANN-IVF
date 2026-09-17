# IVF-PQ plan

The worker takes the first unchecked item whose dependencies are checked,
unless `REVIEW.md` has open findings, which come first. Each item is one
PR-sized change with tests. Check an item off in the same commit that
completes it, with the commit hash.

## Queue

- [x] **Heap reopen.** `RawVectorHeap::open_existing(path, layout, next_flat_slot, allocated_pages)`
      driven by `IVFPQIndexFileHeader`; refuse a file whose size disagrees
      with `allocated_pages * page_size`. Put a page id in the 8-byte page
      header (bump `IVF_PQ_INDEX_VERSION` and the bulk writer) and verify it
      on `read_vector`, so a misplaced page is detected. Tests: build, close,
      reopen, read every vector back; corrupt one page id and see it rejected.
      (Done in "Add raw-vector heap reopen and page-id headers"; the header
      now also records the slot cursor, index file version 2.)
- [ ] **Query path, single query.** `ivf_pq_search(const IVFPQIndex&, const RawVectorHeap&, const float* q, k, nprobe, rerank_m) -> ids+dists`
      in `include/bufann/ivf_pq_search.h`, implementing exactly the reference
      search in `tests/ivf_pq_recall_test.cpp` (which then calls it instead of
      its own loop). Centroid distances via one GEMM against the padded
      centroids; PQ table in a per-query scratch; re-rank from the heap.
      Recall must match the reference to within ties.
- [ ] **Query path, batched.** Batch the centroid GEMM over many queries and
      parallelize across queries with OpenMP; measure QPS on SIFT1M at nprobe
      16 and 64 and record it in the commit message.
- [ ] **Wire `BufANNConfig::index_type`.** `IndexType::IvfPq` selects the
      IVF-PQ build (steps 1-6 from `ivf_nlist`, `pq_chunks`) and search
      (`ivf_nprobe`) through the existing BufANN entry points; the graph path
      is untouched. Test: build + search through the public API on the blob
      fixture.
- [ ] **Free-slot grace period.** Freed slots enter a deferred list and are
      only reusable after every search that could hold the slot has finished
      (epoch counter). Required before concurrent search + delete. Test with
      threads: a reader holding a slot across a free never observes the
      replacement's bytes.
- [ ] **Mutation: insert.** Allocate a slot (free list or grow), write raw
      bytes, assign to nearest centroid, encode PQ code into `DynamicPQCodes`,
      append to `PostingListDelta::pending_inserts`, set RID active. Search
      merges pending inserts. Tests: inserted vectors are found at nprobe
      covering their partition.
- [ ] **Mutation: delete.** Tombstone in `PostingListDelta`, clear RID active
      bit, free the slot through the grace period. Search drops tombstoned ids.
- [ ] **Posting-list rebuild.** Fold `PostingListDelta` and `DynamicPQCodes`
      into fresh `PostingLists`/`PQMetadata` off to the side, publish with
      one atomic pointer store in `IVFPQSearchConfig`, reclaim the old
      structures after a grace period. Searches must run throughout; test
      with a search thread hammering during a rebuild.

## Not planned

- Non-uniform PQ chunks (`dim % chunks != 0`): upstream's file supports it;
  add `chunk_offsets` to `PQMetadata` only if a dataset needs it.
- Two-level coarse quantizer: only if `nlist` grows to ~1e6.
