#!/bin/bash
# RaBitQ-SSD (arXiv 2610.02652, github.com/VectorDB-NTU/RaBitQ-SSD @ 7ab80aa) on
# the four 90M bases, with the paper's defaults (C = ceil(N/1024), full 1-bit
# code in DRAM, B = 9 on SSD, IRQ coarse quantizer), our exact ground truth,
# 32 threads, top-10, fixed nprobes. One step at a time, all on /nvme.
set -x
R=/nvme/RaBitQ-SSD
export DATA_ROOT=/nvme/rabitq/data INDEX_ROOT=/nvme/rabitq/indexes
export PYTHON_ENV=/nvme/bench100m/py PYTHONPATH=/tmpdata/pylib_faiss:/tmpdata/pylib
export BUDGET_GB_ENV=96
L=/nvme/bench100m/rabitq_logs
mkdir -p $L $DATA_ROOT $INDEX_ROOT
cd $R
NPROBES=8,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536

prep() {  # name ext(type of train/test) src_base src_query
  local name=$1 ext=$2 d=$DATA_ROOT/$1
  mkdir -p $d
  [[ -e $d/train.$ext ]] || ln -s $3 $d/train.$ext
  [[ -e $d/test.$ext ]] || ln -s $4 $d/test.$ext
  for k in 10 100; do [[ -e $d/gt$k ]] || ln -s /nvme/100m/$name/${name}_90000000_gt100.bin $d/gt$k; done
}
one() {  # name ext
  local name=$1 ext=$2
  ls $INDEX_ROOT/$name/clustering/centroids_*.fvecs > /dev/null 2>&1 ||
    /usr/bin/time -v env DATA_ENV=$DATA_ROOT/$name/train.$ext OUTDIR_ENV=$INDEX_ROOT/$name/clustering \
      clustering/run_kmeans.sh $name > $L/${name}_cluster.log 2>&1
  echo "$name cluster exit=$? $(date +%T)"
  /usr/bin/time -v scripts/build_index/build_index.sh $name > $L/${name}_build.log 2>&1
  echo "$name build exit=$? $(date +%T)"
  TOPKS_ENV=10 TS_ENV=32 NPROBES_ENV=$NPROBES scripts/main_exp/run_main_exp.sh $name > $L/${name}_search.log 2>&1
  echo "$name search exit=$? $(date +%T)"
}

# SPACEV is int8, which RaBitQ-SSD does not read: convert to float32 (exact for L2).
S=/nvme/100m/spacev100m
for f in base.90M query.10K; do
  [[ -f $S/$f.fbin ]] || $PYTHON_ENV -c "
import numpy as np, struct
src, dst = '$S/$f.i8bin', '$S/$f.fbin.tmp'
n, d = np.fromfile(src, dtype=np.int32, count=2)
x = np.memmap(src, dtype=np.int8, mode='r', offset=8, shape=(n, d))
with open(dst, 'wb') as out:
    out.write(struct.pack('<ii', n, d))
    for i in range(0, n, 4_000_000): x[i:i + 4_000_000].astype(np.float32).tofile(out)
" && mv $S/$f.fbin.tmp $S/$f.fbin
done
echo "spacev convert exit=$?"

prep sift100m u8bin /nvme/100m/sift100m/base.90M.bin /nvme/100m/sift100m/query.10K.u8bin
prep deep100m fbin /nvme/100m/deep100m/base.90M.fbin /nvme/100m/deep100m/query.10K.fbin
prep spacev100m fbin $S/base.90M.fbin $S/query.10K.fbin
prep turing100m fbin /nvme/100m/turing100m/base.90M.fbin /nvme/100m/turing100m/query.10K.fbin

for ds in "sift100m u8bin" "deep100m fbin" "spacev100m fbin" "turing100m fbin"; do one $ds; done
echo ALL_DONE
