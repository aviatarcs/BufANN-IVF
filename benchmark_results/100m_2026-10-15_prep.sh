#!/bin/bash
# DEEP100M / SPACEV100M / Turing100M (first 100M rows of each 1B base,
# unshuffled; queries are the first 10K of each official query set, the same
# files vectordb-baselines ships under benchmark/datasets/1b/):
# 90M query bases, exact top-100 ground truth over them, IVF-PQ indexes.
# Each step logs to /tmpdata/bench100m/<step>.log.
set -x
L=/tmpdata/bench100m
D=/tmpdata/100m
IVF=/tmpdata/BufANN-IVF/build/tests
GT=$L/exact_gt
N=90000000

# name type elem_bytes dim ext
SETS="deep100m float 4 96 fbin
spacev100m int8 1 100 i8bin
turing100m float 4 100 fbin"

# 1. base.90M: the first 90M rows under a 90M header.
while read -r name type eb dim ext; do
  out=$D/$name/base.90M.$ext
  [[ -f $out ]] && continue
  python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<ii', $N, $dim))" > $out.tmp &&
    tail -c +9 $D/$name/base.100M.$ext | head -c $((N * dim * eb)) >> $out.tmp &&
    [[ $(stat -c %s $out.tmp) == $((8 + N * dim * eb)) ]] && mv $out.tmp $out
  echo "$name base.90M exit=$?"
done <<< "$SETS"

# 2. Exact top-100 over the 90M base.
[[ -x $GT ]] || g++ -O3 -march=native -fopenmp /tmpdata/BufANN-IVF/scripts/datasets/exact_gt.cpp \
  -I/tmpdata/build-root/build-env/include -L/lib/x86_64-linux-gnu -lmkl_rt -o $GT
while read -r name type eb dim ext; do
  out=$D/$name/${name}_90000000_gt100.bin
  [[ -f $out ]] && continue
  /usr/bin/time -v $GT $type $D/$name/base.90M.$ext $D/$name/query.10K.$ext 100 $out > $L/gt_$name.log 2>&1
  echo "$name gt exit=$?"
done <<< "$SETS"

# 3. IVF-PQ indexes, nlist 131072 (SIFT100M's best), 32 PQ chunks (32 B per
#    vector, as BufANN's PQ), uneven chunks for the 100-d sets.
while read -r name type eb dim ext; do
  p=/tmpdata/ivf_bench/$name/n131072
  [[ -f ${p}_ivf_pq_index.bin ]] && continue
  mkdir -p $(dirname $p)
  /usr/bin/time -v $IVF/ivf_pq_build_index --data_type $type --data_file $D/$name/base.90M.$ext \
    --index_prefix $p --dim $dim --ivf_nlist 131072 --ivf_pq_chunks 32 > $L/ivf_build_$name.log 2>&1
  echo "$name ivf build exit=$?"
done <<< "$SETS"
echo ALL_DONE
