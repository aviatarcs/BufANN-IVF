#!/bin/bash
# DEEP100M / SPACEV100M / Turing100M IVF-PQ query runs (agent/rerank-batches):
# nlist 131072, 1 GB pool, 32 threads, 10K queries, 1000 base-sampled warmup
# rows. One process per nprobe (or cap); within it, each tau / rerank_m /
# schedule gets its own warmup pass. Fixed: nprobe P, beam 1.25 P. Adaptive:
# cap C, beam C, tau swept (as on SIFT100M, Oct 11). Schedules: 0:0 one
# batch, 10:0 batches of 10, 10:1 and 10:2 batches of 10 with early stop.
# SPACEV also re-ranks 200 and 400 ("more exact vectors").
B=/tmpdata/BufANN-IVF/build/tests/ivf_pq_query_bench
D=/nvme/100m
R=/nvme/bench100m/results_ivf
mkdir -p $R
run() {  # tag name type ext nprobe beam taus rerank_ms schedules
  local tag=$1 name=$2 type=$3 ext=$4
  $B --data_type $type --index_prefix /nvme/ivf_bench/$name/n131072 --query_file $D/$name/query.10K.$ext \
    --gt_file $D/$name/${name}_90000000_gt100.bin --nprobes $5 --graph_beams $6 --probe_taus $7 --rerank_m $8 \
    --rerank_schedules $9 --skip_exact 1 --scan simd --warmup_query_file /nvme/bench100m/warmup_${name}_90M_n1000_seed0.bin \
    --buffer_pool_frames 262144 > $R/$tag.log 2>&1
  echo "$tag exit=$? lines=$(grep -c '^{' $R/$tag.log)"
}
while read -r name type ext rms; do
  for P in 32 64 128 256 512 1024; do
    run ${name}_fixed_p$P $name $type $ext $P 1.25 0 $rms 0:0,10:0,10:1,10:2
  done
  for C in 160 320 640 1280; do
    run ${name}_adapt_c$C $name $type $ext $C 1 1.3,1.4,1.5,1.6,1.7 $rms 0:0,10:2
  done
done <<'EOS'
deep100m float fbin 100
spacev100m int8 i8bin 100,200,400
turing100m float fbin 100
EOS
echo ALL_DONE
