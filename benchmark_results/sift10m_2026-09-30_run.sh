#!/bin/bash
# Everything the advisor email states, rerun in one pass on a quiet node.
# 1-2: SIFT10M (9M) query comparison: BufANN (driver built from a copy of
#      BufANN-CS395T with the pin_batch fix, 785d976) and IVF-PQ (agent/bp-bench),
#      both through BufANN's buffer pool on /tmpdata (NVMe LVM), same 1000-row
#      base-sampled warmup, one process per setting, 32 threads, 10K queries.
# 3:   9M nlist 32768 build with every build-speed change.
# 4:   90M nlist 131072 build on a quiet node.
set -x
O=/var/tmp/bufann-ivf-evpeng/final2
Q=/tmpdata/BufANN-CS395T/eval_tempfiles/sift10m/BufANN/query
W=$(readlink -f $Q/warmup_queries.bin)
GT=$Q/groundtruth/sift10m_9000000_gt10.bin
QF=/tmpdata/sift10m/query.10K.bin
DRV=/var/tmp/bufann-ivf-evpeng/cs395t-fixed/build/tests/bufann_driver
T=/tmp/claude-20014/-tmpdata-BufANN-IVF/da143b61-2de8-484d-beb4-616ea0944e2d/scratchpad/bpbench-wt/build/tests
sync; sleep 30
for L in 20 30 40 60 80 100 150 200; do
  $DRV --data_type uint8 --index_prefix $Q/index/sift10m --dim 128 \
    --workload search_only --query_file $QF --gt_file $GT --recall_at 10 --search_L $L --beamwidth 4 \
    --query_threads 32 --maintenance_threads 32 --buffer_pool_frames 262144 --pq_chunks 32 --R 64 --L 100 \
    --result_file $O/bufann_L$L.json --delete_micro_batch 1 --max_dataset_size 10000000 \
    --warmup_query_file $W --warmup_threads 32 > $O/bufann_L$L.log 2>&1
  echo "bufann L$L exit=$?"
done
for frames in 262144 160000; do
  for p in 32 48 64 96 128 192 256 384 512 768; do
    $T/ivf_pq_query_bench --data_type uint8 --index_prefix /tmpdata/ivf_bench/cmp10m/n65536 --query_file $QF --gt_file $GT \
      --nprobes $p --graph_beams 2 --skip_exact 1 --warmup_query_file $W --buffer_pool_frames $frames \
      > $O/ivf_f${frames}_p$p.log 2>&1
    echo "ivf frames $frames nprobe $p exit=$?"
  done
done
rm -f /tmpdata/ivf_bench/cmp10m/n65536_ivf_raw_vectors.bin
/usr/bin/time -v $T/ivf_pq_build_index --data_type uint8 --data_file /tmpdata/ivf_bench/sift10m/base.9M.bin \
  --index_prefix $O/b9M_n32768 --dim 128 --ivf_nlist 32768 --ivf_pq_chunks 32 > $O/build_9M_n32768.log 2>&1
echo "build 9M exit=$?"
rm -f $O/b9M_n32768_*
rm -f /var/tmp/bufann-ivf-evpeng/final/fast_90M_n131072_*
/usr/bin/time -v $T/ivf_pq_build_index --data_type uint8 --data_file /tmpdata/sift100m/base.90M.bin \
  --index_prefix /var/tmp/bufann-ivf-evpeng/final/fast_90M_n131072 --dim 128 --ivf_nlist 131072 --ivf_pq_chunks 32 \
  > $O/build_90M_n131072.log 2>&1
echo "build 90M exit=$?"
echo ALL_DONE
