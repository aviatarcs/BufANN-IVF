#!/bin/bash
# SIFT100M A/B of the faiss-borrow changes (Oct-10 binary vs agent/faiss-borrow).
# Pauses the DEEP IVF build that was sharing the node (SIGSTOP, resumed on exit).
# Old and new alternate per setting, one rep each, 1 GB pool, 32 threads.
PID=$(pgrep -f "^/tmpdata/BufANN-IVF/build/tests/ivf_pq_build_index" | head -1)
trap '[[ -n $PID ]] && kill -CONT $PID; echo resumed $PID' EXIT
[[ -n $PID ]] && kill -STOP $PID && echo "paused $PID"
OLD=/tmpdata/BufANN-IVF/build/tests/ivf_pq_query_bench
NEW=/tmpdata/BufANN-IVF-faiss/build/tests/ivf_pq_query_bench
D=$(dirname $0)/clean; mkdir -p $D
for n in 131072 262144; do
  for p in 64 128 256 512 1024; do
    for v in old new; do
      [[ $v == old ]] && B=$OLD || B=$NEW
      $B --data_type uint8 --index_prefix /tmpdata/ivf_bench/sift100m_v5/n$n --query_file /tmpdata/sift100m/query.10K.bin \
        --gt_file /tmpdata/sift100m/sift100m_90000000_gt100.bin --nprobes $p --graph_beams 1.25 --skip_exact 1 --scan simd \
        --warmup_query_file /tmpdata/bench100m/warmup_90M_n1000_seed0.bin --buffer_pool_frames 262144 > $D/${v}_n${n}_p${p}.log 2>&1
      echo "$v n$n p$p: $(grep -o 'in [0-9.]* s; RSS [0-9]* MB' $D/${v}_n${n}_p${p}.log) $(grep -o '"query_qps":[0-9.]*\|"query_lat_p99_us":[0-9.]*\|"recall":[0-9.]*\|"query_io_avg_us_per_query":[0-9.]*' $D/${v}_n${n}_p${p}.log | tr '\n' ' ') load $(cut -d' ' -f1 /proc/loadavg)"
    done
  done
done
