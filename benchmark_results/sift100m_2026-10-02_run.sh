#!/bin/bash
# Finish BufANN's 100M canonical (PQ, then conversion) and run the SIFT100M
# (90M base) IVF-PQ vs BufANN comparison. Both read through BufANN's buffer
# pool from /tmpdata (NVMe-only segment), same 1000-row base-sampled warmup,
# exact 90M top-100 GT, one process per setting, 32 threads, 10K queries.
set -x
L=/tmpdata/bench100m
CS=/tmpdata/BufANN-CS395T
export DATASET=sift100m SIFT100M_ROOT=/tmpdata/sift100m SIFT100M_INDEX_DIR=/tmpdata/indexes/sift100m
cd $CS
bash benchmark/scripts/build_pq.sh sift100m > $L/build_pq.log 2>&1;                     echo "build pq exit=$?"
bash benchmark/BufANN/sift/query_prepare.sh > $L/bufann_query_prepare2.log 2>&1;         echo "bufann prepare exit=$? (GT step expected to fail)"
W=$CS/eval_tempfiles/sift100m/BufANN/query/index/sift100m
[[ -f $W.heap ]] || { echo "NO BUFANN HEAP; stopping"; echo ALL_DONE; exit 1; }

QF=/tmpdata/sift100m/query.10K.bin
GT=/tmpdata/sift100m/sift100m_90000000_gt100.bin
WU=$L/warmup_90M_n1000_seed0.bin
DRV=/var/tmp/bufann-ivf-evpeng/cs395t-fixed/build/tests/bufann_driver
IVF=/tmpdata/BufANN-IVF/build/tests/ivf_pq_query_bench
mkdir -p $L/results
for frames in 262144 2097152; do
  for Lval in 20 30 40 60 80 100 150 200; do
    $DRV --data_type uint8 --index_prefix $W --dim 128 --workload search_only --query_file $QF --gt_file $GT \
      --recall_at 10 --search_L $Lval --beamwidth 4 --query_threads 32 --maintenance_threads 32 \
      --buffer_pool_frames $frames --pq_chunks 32 --R 64 --L 100 --result_file $L/results/bufann_f${frames}_L$Lval.json \
      --delete_micro_batch 1 --max_dataset_size 100000000 --warmup_query_file $WU --warmup_threads 32 \
      > $L/results/bufann_f${frames}_L$Lval.log 2>&1
    echo "bufann frames $frames L $Lval exit=$?"
  done
done
for n in 131072 262144; do
  for frames in 262144 2097152; do
    for p in 48 64 96 128 192 256 384 512 768 1024; do
      $IVF --data_type uint8 --index_prefix /tmpdata/ivf_bench/sift100m_v5/n$n --query_file $QF --gt_file $GT \
        --nprobes $p --graph_beams 2 --skip_exact 1 --warmup_query_file $WU --buffer_pool_frames $frames \
        > $L/results/ivf_n${n}_f${frames}_p$p.log 2>&1
      echo "ivf n $n frames $frames nprobe $p exit=$?"
    done
  done
done
echo ALL_DONE
