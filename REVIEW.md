# Review findings

The reviewer appends a section per reviewed commit range. Each finding has a
severity (crash / correctness / concurrency / test-gap / style), a
`file:line`, a concrete failure scenario, and a status. The worker sets
status to `fixed <hash>` or `wontfix <reason>`; the reviewer may reopen.
Findings with status `open` block new plan items.

(no findings yet)

## Worker note after 18d647f (addressed to the human)

- **style** `scripts/check_stop.sh:17` — the hook treats a non-empty
  `build/Testing/Temporary/LastTestsFailed.log` as "the last ctest run
  failed", but the CTest on this host does not remove that file on a green
  run: after the one red run during this cycle, three consecutive 5/5 passes
  (`scripts/dev_env.sh test` and `ctest --test-dir build`) left it in place,
  so the hook blocked a cycle whose suite is green and whose work is
  committed and pushed. The worker's sandbox cannot delete or truncate the
  file (the build tree resolves to `/var/tmp`). Suggested fix: compare it
  against `LastTest.log` by mtime, or parse the pass/fail summary out of
  `LastTest.log` instead. Not fixed here because `scripts/` and `.claude/`
  had uncommitted edits from another session at the time. Status: fixed (check_stop.sh now parses LastTest.log, which every ctest run rewrites)

## Review of 47f5324..f0b986b (2026-09-16)

### Findings
- **[correctness] src/bufann/ivf_pq_raw_vector_heap.cpp:106 — open_existing pins page ids only, not the geometry; the "different page_size" / "foreign file" claims in the commit message hold only when the last-page offset lands on a wrong id.** Failure scenario: a heap written with `elem_size` 512 is reopened with the layout of an index recording `elem_size` 1024 and the same `page_size`/page count -> both header checks pass (ids 0 and N-1 are where they were), `read_vector` returns bytes straddling two old slots with no throw. Likewise a 2-page heap of `page_size` P opened as 1 page of 2P: first == last page, id 0 matches, every read from the second half returns page 1's header+bitmap as vector data. The heap file is described only by the index file that names it, so this needs a mismatched index/heap pair; with the page header carrying only `page_id`, nothing in the file itself can catch it. Status: fixed (page header now records page_size and elem_size; verified on open_existing, read_vector, and write_pages — see "Record the heap geometry in every page header")
- **[correctness] src/bufann/ivf_pq_raw_vector_heap.cpp:119 — close() does not reset the cursor, so a rejected open_existing on a heap that was open before leaves stale `next_flat_slot`/`allocated_pages` on a closed fd, and the test that "no cursor is left behind" only runs on a never-opened heap.** Failure scenario: heap opened, 3 pages / cursor n, `close()`, then `open_existing(missing_path, ...)` throws at the stat check -> `allocated_pages()` still reports 3; `allocate_slot` returns n without touching the file (page_of(n) < 3, so no grow, no throw), and only the subsequent `write_vector` fails on `pwrite(-1)`. A caller that writes the index header from `next_flat_slot()`/`allocated_pages()` after a failed reopen records the previous heap's cursor. The same test would also pass if the `catch { close(); throw; }` were deleted (an fd leak is unobservable to it). Status: fixed (close() now zeroes the cursor; see "Drop the heap cursor on close()" — the new test opens a heap, rejects a reopen both before and after the fd is opened, and requires allocate_slot/read_vector to throw and the file to stay byte-identical)
- **[test-gap] src/bufann/ivf_pq_index_file.cpp:155 — the new header bound `raw_vector_next_slot <= 2^31` has no corruption test.** Failure scenario: remove the clause; `index.heap_next_slot = uint32_t(h.raw_vector_next_slot)` truncates a patched value of `2^32 + heap_next_slot` to the true cursor, `require_consistent` passes, and the file loads. The existing "slot cursor past the heap's pages" corruption patches `slots + 1`, which is caught by `require_consistent`, not by this guard (the sibling `raw_vectors_bytes / page_size <= UINT32_MAX` clause is defended by the later heap-file size check; this one is not). Status: fixed (see "Test the header bound on raw_vector_next_slot" — the build test patches the cursor to `2^32 + heap_next_slot`, which truncates back to the true cursor; verified the case fails with the clause removed and passes with it)
- **[style] src/bufann/ivf_pq_raw_vector_heap.cpp:93 — size is taken from `stat(path)` before `open`, not `fstat(_fd)` after.** Failure scenario: the file is replaced between the two calls; the size accepted belongs to a file the heap never opened. Outcome is an eventual `pread` short-read throw, not silent wrong data, since page headers are still verified on read. Status: open

### Verified
- `scripts/dev_env.sh test`: the library compile fails on the worker's uncommitted `src/bufann/ivf_pq_search.cpp` (`error: expected primary-expression before 'long'`), which is outside this range; ctest then ran the previously linked binaries: 4/4 passed (`ivf_pq_raw_vector_heap` 0.13 s, `ivf_pq_build` 2.36 s, `ivf_pq_scale_smoke` 20.39 s, `ivf_pq_recall_synthetic` 13.26 s). Those binaries contain this commit's tests: the heap test prints the three new `open_existing`/misplaced-page cases and the build test prints "combined index file round-trips, reopens the heap, ...", all PASS. I could not build the commit in an isolated tree (worktree/archive/g++ were not permitted in this session).
- `build/tests/ivf_pq_scale_test 1000000 128 1024 32`: "invariants over every vector" PASS, "brute-force spot checks on 1000 random vectors" PASS, peak RSS 2.81 GB. Note the scale test at this commit still searches through the build heap; only the index-file test and the recall test exercise `open_existing`.
- Cursor/page arithmetic: `next_flat_slot <= allocated_pages * slots_per_page` is checked in u64 in both `open_existing` and `require_consistent`; `heap_next_slot` truncation to u32 is guarded by the 2^31 bound; allocation after reopen at exactly a page boundary (`cursor == pages * spp`) grows and stamps a new page, at a partial page continues without growing (checked by `test_reopen_resumes_the_heap` with n = 5·spp + 3 and m = 6·spp + 1).
- `allocated_pages == 0` with an empty file: size check requires 0 bytes, no header read, `RawVectorHeapBulkWriter` still accepts the heap. `allocated_pages > 0` with `page_size >= 9` guarantees both header reads lie inside the file.
- Bulk writer stamps the header on the first slot of every page (`index == 0`) after `flush()` zero-fills the buffer, and `write_pages` re-verifies every buffer page against `first_page_id + p` before the single pwrite; the bulk-vs-per-slot byte-for-byte comparison plus the per-page `page_header_is` check would catch a wrong id, a missing stamp, or a stamp on the wrong buffer page.
- `read_vector` reads `[page start, slot end)` in one pread, which is ≤ `page_size` by the layout invariant; header verified before the memcpy. The misplaced-page test compares every accepted slot to the independent `pattern(i)` and requires exactly the page-1 slots to throw, so dropping the header check or checking the wrong page id fails it.
- Concurrency: the page (with header) is written before `_allocated_pages` is published, so a concurrent `read_vector` that passes `require_allocated` sees a stamped page; `open_existing` stores the cursor only after the fd and headers are validated.
- Mutations the tests do catch: `<=` in the RID-below-cursor check ("RID at the heap's slot cursor"), `<=`/`>=` in the file-size equality ("truncated/extended by a byte"), checking page 1 instead of the last page ("last page carrying another page's id" and "different page_size"), assigning the cursor before the header checks (the post-rejection cursor check).
- Commit message claims checked against the diff: header 168 -> 176 bytes and version 2, `raw_vector_next_slot` written by `make_header` and read by the loader, `open()` guard message updated, PLAN.md item ticked. The recall test now closes the build heap and searches through one reopened from the loaded index, as stated.
