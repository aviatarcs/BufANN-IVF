# check_gt.py base.fbin query.fbin gt.bin [nq]: brute-force top-10 of nq random queries vs the truthset.
import sys, numpy as np
b, q, g = sys.argv[1:4]; nq = int(sys.argv[4]) if len(sys.argv) > 4 else 20
n, d = np.fromfile(b, np.int32, 2); B = np.memmap(b, np.float32, 'r', 8, (n, d))
Q = np.fromfile(q, np.float32, offset=8).reshape(-1, d)
gn, gk = np.fromfile(g, np.int32, 2); G = np.fromfile(g, np.uint32, gn * gk, offset=8).reshape(gn, gk)
rng = np.random.default_rng(1); qs = rng.choice(len(Q), nq, replace=False)
dist = np.zeros((nq, n), np.float32)
for s in range(0, n, 1_000_000):
    X = np.asarray(B[s:s + 1_000_000]); dist[:, s:s + len(X)] = ((X ** 2).sum(1)[None] - 2 * Q[qs] @ X.T)
ok = ties = 0
for i, qi in enumerate(qs):
    order = np.argsort(dist[i]); top = order[:10]; ok += len(set(top) & set(G[qi, :10]))
    # A miss is a tie when the truthset's member is as close as the brute-force 10th.
    kth = dist[i][order[9]]
    ties += sum(1 for g in set(G[qi, :10]) - set(top) if dist[i][g] <= kth + 1e-4 * max(1.0, abs(kth)))
print(f"gt check: {ok}/{nq*10} of brute-force top-10 in truthset top-10, {ties} of the misses tied with the 10th (n={n}, d={d}, k={gk})")
