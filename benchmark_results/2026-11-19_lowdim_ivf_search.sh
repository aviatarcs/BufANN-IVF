#!/bin/bash
# IVF-PQ query bench, timed under the shared bench lock. 32 threads, top-10,
# extra args come first: get_arg takes the first match. Fixed nprobe, centroid beam 1.25 x nprobe, 1000 base-sampled warmup rows.
#   ivf_search.sh <name> <prefix> <query.fbin> <gt> <nprobes> <rerank_ms> <pool_frames> [extra args]
B=/tmpdata/work/wt-lowdim/build/tests/ivf_pq_query_bench
name=$1 p=$2 q=$3 gt=$4 np=$5 rm=$6 pf=$7; shift 7
flock /tmpdata/work/bench.lock $B "$@" --data_type float --index_prefix $p --query_file $q --gt_file $gt \
  --nprobes $np --graph_beams 1.25 --probe_taus 0 --rerank_m $rm --skip_exact 1 --scan simd \
  --warmup_query_file /tmpdata/work/data/warm/${name%%_*}_n1000.fbin --buffer_pool_frames $pf --threads 32 \
  > /tmpdata/work/logs/lowdim/ivf_${name}.log 2>&1
echo "$name exit=$? lines=$(grep -c '^{' /tmpdata/work/logs/lowdim/ivf_${name}.log)"
