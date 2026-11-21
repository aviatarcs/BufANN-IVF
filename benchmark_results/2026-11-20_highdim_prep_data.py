#!/usr/bin/env python3
"""Writes base/query .fbin files (float32, <ii n d> header) for the high-dim
comparison.  prep_data.py <dataset> <out_dir> <raw_dir>

  gist960     base: GIST1M's 1M base; queries: the first 10K rows of
              gist_learn (disjoint from the base, same distribution; the
              official set has only 1000 queries).
  laion512    as benchmark_results/2026-10-20_laion512.sh: base = the first
              10M rows of img_emb_0..10.npy, queries = img_emb_405[:10000].
  cohere768   Cohere wikipedia-22-12 768-d (VectorDBBench cohere_large_10m,
              train-00..09): 10K rows held out at random (seed 0) as queries,
              the other 9,990,000 rows (in file order) are the base.
  dbpedia1536 OpenAI ada-002 on DBpedia (KShivendu/dbpedia-entities-openai-1M,
              float64 -> float32): 10K held out at random (seed 0), 990,000 base.
"""
import glob, os, struct, sys
import numpy as np

ds, out, raw = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)


def write_fbin(path, x):
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<ii", *x.shape))
        x.astype(np.float32).tofile(f)
    os.replace(path + ".tmp", path)


class Writer:
    def __init__(self, path, n, d):
        self.path, self.n, self.left = path, n, n
        self.f = open(path + ".tmp", "wb")
        self.f.write(struct.pack("<ii", n, d))

    def add(self, x):
        x = x[: self.left]
        np.ascontiguousarray(x, dtype=np.float32).tofile(self.f)
        self.left -= x.shape[0]

    def close(self):
        assert self.left == 0, (self.path, self.left)
        self.f.close()
        os.replace(self.path + ".tmp", self.path)


def parquet_holdout(files, col, d, nq, total):
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    qidx = np.sort(rng.choice(total, nq, replace=False))
    is_q = np.zeros(total, bool)
    is_q[qidx] = True
    base = Writer(f"{out}/base.fbin", total - nq, d)
    queries = np.empty((nq, d), np.float32)
    row = qi = 0
    for fn in files:
        pf = pq.ParquetFile(fn)
        for b in pf.iter_batches(batch_size=200_000, columns=[col]):
            arr = b.column(0)
            x = arr.values.to_numpy(zero_copy_only=False).reshape(-1, d)
            assert x.shape[0] == len(arr)
            m = is_q[row : row + x.shape[0]]
            queries[qi : qi + m.sum()] = x[m]
            qi += m.sum()
            base.add(x[~m])
            row += x.shape[0]
        print(fn, row, flush=True)
    assert row == total and qi == nq, (row, qi)
    base.close()
    write_fbin(f"{out}/query.10K.fbin", queries)


if ds == "gist960":
    os.symlink("/local/data/gist/gist_base.fbin", f"{out}/base.fbin") if not os.path.exists(f"{out}/base.fbin") else None
    lr = np.memmap(f"{raw}/gist_learn.fvecs", dtype=np.float32, mode="r").reshape(-1, 961)
    assert lr[:, 0].view(np.int32).max() == 960
    write_fbin(f"{out}/query.10K.fbin", np.array(lr[:10000, 1:]))
elif ds == "laion512":
    n, d = 10_000_000, 512
    base = Writer(f"{out}/base.fbin", n, d)
    for i in range(11):
        x = np.load(f"{raw}/img_emb_{i}.npy", mmap_mode="r")
        take = min(base.left, x.shape[0])
        for s in range(0, take, 500_000):
            base.add(x[s : min(take, s + 500_000)])
        if base.left == 0:
            break
    base.close()
    write_fbin(f"{out}/query.10K.fbin", np.load(f"{raw}/img_emb_405.npy", mmap_mode="r")[:10000])
elif ds == "cohere768":
    parquet_holdout(sorted(glob.glob(f"{raw}/train-*-of-10.parquet")), "emb", 768, 10000, 10_000_000)
elif ds == "dbpedia1536":
    parquet_holdout(sorted(glob.glob(f"{raw}/train-*.parquet")), "openai", 1536, 10000, 1_000_000)
else:
    sys.exit(f"unknown dataset {ds}")

b = np.memmap(f"{out}/base.fbin", dtype=np.float32, mode="r", offset=8)
n, d = np.fromfile(f"{out}/base.fbin", dtype=np.int32, count=2)
nr = np.linalg.norm(b.reshape(n, d)[:100000], axis=1)
print(ds, "base", n, d, "norm min/mean/max", nr.min(), nr.mean(), nr.max(), flush=True)
