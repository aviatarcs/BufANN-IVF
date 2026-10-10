#!/bin/bash
# SIFT100M (90M base) rerun with the October changes: SIMD (Quicker ADC)
# scan, heap+prefetch centroid beam at 1.25 x nprobe, one copy of the codes.
# Same method as 2026-10-03: BufANN's buffer pool on /tmpdata for both,
# same 1000-row base-sampled warmup, exact 90M top-100 GT, one process per
# setting, 32 threads, 10K queries. BufANN rerun in the same session.
set -x
L=/tmpdata/bench100m; R=$L/results_oct10
QF=/tmpdata/sift100m/query.10K.bin
GT=/tmpdata/sift100m/sift100m_90000000_gt100.bin
WU=$L/warmup_90M_n1000_seed0.bin
DRV=/var/tmp/bufann-ivf-evpeng/cs395t-fixed/build/tests/bufann_driver
W=/tmpdata/BufANN-CS395T/eval_tempfiles/sift100m/BufANN/query/index/sift100m
IVF=/tmpdata/BufANN-IVF/build/tests/ivf_pq_query_bench
for frames in 262144 2097152; do
  for Lval in 20 30 40 60 80 100 150 200; do
    $DRV --data_type uint8 --index_prefix $W --dim 128 --workload search_only --query_file $QF --gt_file $GT \
      --recall_at 10 --search_L $Lval --beamwidth 4 --query_threads 32 --maintenance_threads 32 \
      --buffer_pool_frames $frames --pq_chunks 32 --R 64 --L 100 --result_file $R/bufann_f${frames}_L$Lval.json \
      --delete_micro_batch 1 --max_dataset_size 100000000 --warmup_query_file $WU --warmup_threads 32 \
      > $R/bufann_f${frames}_L$Lval.log 2>&1
    echo "bufann frames $frames L $Lval exit=$?"
  done
done
for n in 262144 131072; do
  for frames in 262144 2097152; do
    for p in 48 64 80 96 112 128 160 192 224 256 320 384 448 512 640 768 1024; do
      $IVF --data_type uint8 --index_prefix /tmpdata/ivf_bench/sift100m_v5/n$n --query_file $QF --gt_file $GT \
        --nprobes $p --graph_beams 1.25 --skip_exact 1 --scan simd --warmup_query_file $WU --buffer_pool_frames $frames \
        > $R/ivf_n${n}_f${frames}_p$p.log 2>&1
      echo "ivf n $n frames $frames nprobe $p exit=$?"
    done
  done
done
echo ALL_DONE
