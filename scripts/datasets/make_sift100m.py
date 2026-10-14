# SIFT100M the way /tmpdata/sift10m was made: the first 100M rows of SIFT1B
# (base.1B.u8bin), shuffled by numpy.random.default_rng(0).permutation.
# Writes the 90M query-workload base and the 10M update tail separately
# (cat-able into base.100M.bin: fix the header to 100M) to save disk.
import numpy as np, struct, subprocess, sys, time
N, D, NB = 100_000_000, 128, 90_000_000
url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
arr = np.empty((N, D), dtype=np.uint8); buf = memoryview(arr.reshape(-1))
p = subprocess.Popen(["curl", "-sf", "-r", f"8-{8 + N * D - 1}", url], stdout=subprocess.PIPE, bufsize=1 << 24)
got, t0 = 0, time.time()
while got < N * D:
    k = p.stdout.readinto(buf[got:got + (1 << 26)])
    if not k: sys.exit(f"short download: {got} of {N * D}")
    got += k
    if got % (1 << 30) < k: print(f"{got / 1e9:.1f} GB {got / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
p.wait()
perm = np.random.default_rng(0).permutation(N)
np.save("/tmpdata/sift100m/shuffle_perm_seed0.npy", perm)
out = arr[perm]; del arr
with open("/tmpdata/sift100m/base.90M.bin", "wb") as f:
    f.write(struct.pack("<ii", NB, D)); out[:NB].tofile(f)
with open("/tmpdata/sift100m/tail.10M.bin", "wb") as f:
    f.write(struct.pack("<ii", N - NB, D)); out[NB:].tofile(f)
print("done", time.time() - t0, "s")
