# BufANN-IVF

IVF-PQ index type on BufANN (a DiskANN derivative). Design: `IVF on BufANN
Design Document.pdf`. IVF-PQ code lives in `include/bufann/ivf_pq*.h`,
`src/bufann/ivf_pq*.cpp`, `tests/ivf_pq_*`.

## Build and test

- `scripts/dev_env.sh test` — configures `build/` with the host's MKL shim,
  builds the IVF-PQ targets, runs `ctest`. Use this; do not hand-roll cmake.
- `scripts/dev_env.sh scale` — the 1M x 128 scale run (~70 s).
- `build/tests/ivf_pq_recall_test <base.bin> <query.bin> <gt> <nlist> <chunks>`
  for recall on a real dataset (SIFT1M: nlist 1024, chunks 32; expect
  recall@10 >= 0.99 at nprobe 64).
- `bufann_driver` does not link here (stubbed tcmalloc lacks MallocExtension);
  that is environmental, not a regression.

## Conventions

- Never commit red: `ctest` must pass before every commit.
- One logical change per commit. Commit message: imperative subject, a body
  that says why and what a reader could not infer from the diff. Match the
  style in `git log`.
- Preconditions use `IVF_PQ_REQUIRE(cond, msg)`; every loader checks the
  file exists first (`diskann::load_bin` reads garbage from a missing file).
- Prefer code whose correctness is visible from reading it over comments;
  comments state what the code cannot (why, invariants, non-obvious costs).
- Tests use `tests/ivf_pq_test_util.h` (`TestCase::check/expect_throw/done`).
  Every new guard gets a test that would fail without it. A test should
  compare against an independent oracle (brute force, byte-for-byte files),
  not the code under test's own reads.
- Nearest-centre checks must tolerate the float expansion of
  `||x||^2 + ||c||^2 - 2x.c` in `compute_closest_centers`: a few ulps of the
  norms involved (for upstream PQ, of `||x - mean||^2`). A mis-indexed result
  is off by whole pivot spacings; do not loosen tolerances beyond that.
- Temp files go under `/tmp/ivf_*` and are removed by the test that made them.
- `main` is protected: agents commit to `agent/work` and never push to `main`.

## Autonomous agents

`PLAN.md` is the task queue, `REVIEW.md` the reviewer's findings;
`.claude/prompts/` holds the worker and reviewer prompts; `scripts/worker.sh`
and `scripts/reviewer.sh` run them in loops.
