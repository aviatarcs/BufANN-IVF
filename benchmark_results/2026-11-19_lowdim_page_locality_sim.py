"""How many 4 KB pages does IVF-PQ's re-rank shortlist touch under each heap layout?

Builds a FAISS IVF-PQ (same nlist / 32-byte PQ as ours), takes each query's
top-R PQ shortlist at a given nprobe, and counts the distinct pages those R
vectors occupy when the SSD records are stored
  id     : in base-file order (our raw-vector heap today; ~1 page per candidate),
  post   : in posting-list order (RaBitQ-SSD's layout, agent/ssd-9bit's ex-code store),
for raw records (dim*elem bytes) and 9-bit ex-code records (8 + 9*dim/8 bytes).
Slots cost record + 4 B (owner id), pages hold (4096 - 16) // slot.
Also reports recall@10 of exact re-rank of the shortlist (layout-independent).

usage: page_locality_sim.py base.fbin query.fbin gt nlist nprobes R[,R] [nq] [elem_bytes]
"""
import sys, numpy as np, faiss

def read_bin(path, dtype=np.float32):
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(int(n), int(d)))

base_p, q_p, gt_p, nlist = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
nprobes = [int(x) for x in sys.argv[5].split(',')]
Rs = [int(x) for x in sys.argv[6].split(',')]
nq = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
raw_elem = int(sys.argv[8]) if len(sys.argv) > 8 else 4
xb = np.ascontiguousarray(read_bin(base_p)); n, d = xb.shape
xq = np.ascontiguousarray(read_bin(q_p)[:nq])
g = np.fromfile(gt_p, dtype=np.int32); gk = int(g[1]); gt = g[2:2 + int(g[0]) * gk].reshape(-1, gk)[:nq, :10]
faiss.omp_set_num_threads(16)
m = 32 if d % 32 == 0 else (25 if d % 25 == 0 else 32)
q = faiss.IndexFlatL2(d)
ix = faiss.IndexIVFPQ(q, d, nlist, m, 8)
rng = np.random.default_rng(0)
ix.train(xb[rng.choice(n, min(n, 50 * nlist, 500000), replace=False)])
ix.add(xb)
# posting position of every id
pos = np.empty(n, dtype=np.int64); off = 0
invl = ix.invlists
for l in range(nlist):
    sz = invl.list_size(l)
    ids = faiss.rev_swig_ptr(invl.get_ids(l), sz).copy()
    pos[ids] = off + np.arange(sz); off += sz
def per_page(rec): return (4096 - 16) // (rec + 4)
recs = {'raw': d * raw_elem, 'ex9': 8 + (d + 7) // 8 + d}
print(f'n={n} d={d} nlist={nlist} pq_m={m} nq={nq}; records/page: raw {per_page(recs["raw"])}, ex9 {per_page(recs["ex9"])}')
for npb in nprobes:
    ix.nprobe = npb
    for R in Rs:
        _, I = ix.search(xq, R)
        # exact re-rank of the shortlist
        hit = 0
        for i in range(nq):
            c = I[i][I[i] >= 0]
            dd = ((xb[c] - xq[i]) ** 2).sum(1)
            top = c[np.argsort(dd)[:10]]
            hit += len(np.intersect1d(top, gt[i]))
        out = [f'nprobe {npb:4d} R {R:4d} recall@10 {100*hit/(10*nq):6.2f}']
        for name, rec in recs.items():
            pp = per_page(rec)
            for lay, key in (('id', lambda c: c), ('post', lambda c: pos[c])):
                pages = np.mean([len(np.unique(key(I[i][I[i] >= 0]) // pp)) for i in range(nq)])
                out.append(f'{name}/{lay} {pages:6.1f}')
        print('  '.join(out), flush=True)
