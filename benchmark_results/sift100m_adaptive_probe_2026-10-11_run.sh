#!/bin/bash
# SIFT100M: per-query probe cutoff (agent/adaptive-probe) vs fixed nprobe,
# same binary, one process per setting, 1 GB pool, 32 threads, 10K queries.
# Fixed: nprobe P, beam 1.25 P. Adaptive: cap 1.25 P, beam = cap (the same
# beam width), tau swept.
B=/tmpdata/BufANN-IVF-faiss/build/tests/ivf_pq_query_bench
R=/tmpdata/bench100m/results_adaptive
run() {  # tag nlist nprobe beams tau
  $B --data_type uint8 --index_prefix /tmpdata/ivf_bench/sift100m_v5/n$2 --query_file /tmpdata/sift100m/query.10K.bin \
    --gt_file /tmpdata/sift100m/sift100m_90000000_gt100.bin --nprobes $3 --graph_beams $4 --probe_taus $5 --skip_exact 1 \
    --scan simd --warmup_query_file /tmpdata/bench100m/warmup_90M_n1000_seed0.bin --buffer_pool_frames 262144 > $R/$1.log 2>&1
  echo "$1 exit=$? $(grep -o '"probes_avg":[0-9.]*\|"codes_avg":[0-9]*\|"query_qps":[0-9.]*\|"query_lat_p99_us":[0-9.]*\|"recall":[0-9.]*' $R/$1.log | tr '\n' ' ')"
}
for n in 131072 262144; do
  for P in 128 256 512 1024; do
    run fixed_n${n}_p$P $n $P 1.25 0
    cap=$((P * 5 / 4))
    for tau in 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.7; do run adapt_n${n}_c${cap}_t$tau $n $cap 1 $tau; done
  done
done
echo ALL_DONE
