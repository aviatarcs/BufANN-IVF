#!/bin/bash
# LAION-400M CLIP ViT-B/32 image embeddings (512-d, fp16 -> float32): a 10M base
# from img_emb_0..9.npy (first 10,000,000 rows) and 10K queries from
# img_emb_405.npy (disjoint from the base). Exact top-100 ground truth.
# Three systems at the paper's two memory budgets, PQ bytes = in-memory
# RaBitQ bits / 8: D_m = 256 (PQ 32 B) and D_m = 512 (PQ 64 B).
#   IVF-PQ: nlist 16384 (~610 per list), graph beam, fixed + adaptive probing,
#           re-rank 100/200, 1 GB pool, 32 threads.
#   BufANN: one DiskANN graph (R 64, L 100), PQ 32 and 64, search_L 10-400.
#   RaBitQ-SSD: paper defaults, memdim 256 / 512, store exbits and raw.
set -x
H=/nvme/hd/laion512
L=/nvme/bench100m/laion512_logs
PY=/nvme/bench100m/py
N=10000000 D=512
mkdir -p $L
until grep -q 'LAION512_DONE\|failed' $H/fetch.log; do sleep 30; done
grep -q LAION512_DONE $H/fetch.log || { echo "download failed"; exit 1; }

# 1. Data: float32 .fbin base and queries. Shards differ in size (img_emb_8
#    holds 642,675 rows), so 0..10 are read until the base has N rows.
[[ -f $H/img_emb_10.npy ]] || curl -fsSL --retry 5 -o $H/img_emb_10.npy \
  https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_10.npy
[[ $(stat -c %s $H/base.10M.fbin 2>/dev/null) == $((8 + N * D * 4)) ]] || $PY - <<EOF && mv $H/base.10M.fbin.tmp $H/base.10M.fbin
import numpy as np, struct
n, d = $N, $D
with open("$H/base.10M.fbin.tmp", "wb") as out:
    out.write(struct.pack("<ii", n, d)); left = n
    for i in range(11):
        x = np.load(f"$H/img_emb_{i}.npy", mmap_mode="r")
        take = min(left, x.shape[0])
        for s in range(0, take, 500_000): x[s:min(take, s + 500_000)].astype(np.float32).tofile(out)
        left -= take
        if left == 0: break
assert left == 0
q = np.load("$H/img_emb_405.npy", mmap_mode="r")[:10000].astype(np.float32)
with open("$H/query.10K.fbin", "wb") as out:
    out.write(struct.pack("<ii", q.shape[0], q.shape[1])); q.tofile(out)
norms = np.linalg.norm(np.load("$H/img_emb_0.npy", mmap_mode="r")[:100000].astype(np.float32), axis=1)
print("base norm min/mean/max", norms.min(), norms.mean(), norms.max())
EOF
[[ $(stat -c %s $H/base.10M.fbin) == $((8 + N * D * 4)) && -f $H/query.10K.fbin ]] || { echo "data step failed"; exit 1; }
echo "data ok $(date +%T)"

# 2. Exact top-100 ground truth, backed up with the others.
GT=$H/laion512_10000000_gt100.bin
[[ -f $GT ]] || /nvme/bench100m/exact_gt float $H/base.10M.fbin $H/query.10K.fbin 100 $GT > $L/gt.log 2>&1
[[ -s $GT ]] || { echo "gt failed"; exit 1; }; echo "gt ok $(date +%T)"
mkdir -p /proj/ut-data-PG0/evpeng/gt100m/laion512 && cp $GT $H/query.10K.fbin /proj/ut-data-PG0/evpeng/gt100m/laion512/
$PY /tmpdata/BufANN-IVF/benchmark/sample_base_warmup.py $H/base.10M.fbin float 1000 $H/warmup_n1000_seed0.bin --seed 0 > $L/warmup.log 2>&1

# 3. Builds, one at a time.
IB=/tmpdata/BufANN-IVF/build/tests
for C in 32 64; do
  P=/nvme/ivf_bench/laion512/n16384_pq$C; mkdir -p $(dirname $P)
  [[ -f ${P}_ivf_pq_index.bin ]] || /usr/bin/time -v $IB/ivf_pq_build_index --data_type float --data_file $H/base.10M.fbin \
    --index_prefix $P --dim $D --ivf_nlist 16384 --ivf_pq_chunks $C > $L/ivf_build_pq$C.log 2>&1
  [[ -f ${P}_ivf_pq_index.bin ]] || { echo "ivf build pq$C failed"; exit 1; }; echo "ivf build pq$C ok $(date +%T)"
done

DB=/tmpdata/vectordb-baselines/DiskANN/build/tests
VB=/tmpdata/vectordb-bench/build/tests
CAN=/nvme/indexes/laion512/laion512_10M_R64_L100_noPQ; mkdir -p $(dirname $CAN)
if [[ ! -f ${CAN}_disk.index ]]; then
  {
    TCMALLOC_RELEASE_RATE=10 $DB/partition_only float $H/base.10M.fbin $CAN 64 90 $N &&
    NS=$(ls ${CAN}_mem.index_tempFiles_subshard-[0-9]*_ids_uint32.bin | wc -l) &&
    for ((p = 0; p < NS; p++)); do TCMALLOC_RELEASE_RATE=10 $DB/build_shard float ${CAN}_mem.index_tempFiles_subshard-$p.bin \
      ${CAN}_mem.index_tempFiles_subshard-${p}_mem.index 64 100 32 l2 || exit 1; done &&
    $DB/merge_shards_only ${CAN}_mem.index_tempFiles_subshard- $NS 64 ${CAN}_mem.index ${CAN}_disk.index_medoids.bin &&
    $DB/finalize_index float ${CAN}_mem.index $H/base.10M.fbin ${CAN}_disk.index $N
  } > $L/diskann_build.log 2>&1
fi
[[ -f ${CAN}_disk.index ]] || { echo "diskann build failed"; exit 1; }; echo "diskann build ok $(date +%T)"
GT10=$H/laion512_10000000_gt10.bin
[[ -f $GT10 ]] || $DB/utils/incremental_groundtruth $GT 10 $GT10 --base_trunc $N > $L/gt10.log 2>&1
for C in 32 64; do
  V=/nvme/indexes/laion512/pq$C; mkdir -p $V
  # The canonical family under a per-PQ prefix: links to the graph, its own PQ files.
  for f in ${CAN}_disk.index*; do ln -sfn $f $V/laion512${f#$CAN}; done
  [[ -f $V/laion512_pq_compressed.bin ]] || $DB/build_pq_standalone float $H/base.10M.fbin $V/laion512 $C > $L/bufann_pq$C.log 2>&1
  [[ -f $V/laion512_disk.index.tags ]] || $PY -c "
import struct, numpy as np
with open('$V/laion512_disk.index.tags', 'wb') as f: f.write(struct.pack('<ii', $N, 1)); np.arange($N, dtype=np.uint32).tofile(f)"
  W=/nvme/bufann/laion512_pq$C; mkdir -p $W
  [[ -f $W/laion512.heap ]] || $VB/fast_convert_diskann_to_inplaceann float $V/laion512 $W/laion512 $D 64 $C 262144 > $L/bufann_convert_pq$C.log 2>&1
  [[ -f $W/laion512.heap ]] || { echo "bufann prep pq$C failed"; exit 1; }; echo "bufann prep pq$C ok $(date +%T)"
done

R=/nvme/RaBitQ-SSD
export DATA_ROOT=/nvme/rabitq/data INDEX_ROOT=/nvme/rabitq/indexes PYTHON_ENV=$PY PYTHONPATH=/tmpdata/pylib_faiss:/tmpdata/pylib BUDGET_GB_ENV=96
RD=$DATA_ROOT/laion512; mkdir -p $RD
ln -sfn $H/base.10M.fbin $RD/train.fbin; ln -sfn $H/query.10K.fbin $RD/test.fbin
ln -sfn $GT $RD/gt10; ln -sfn $GT $RD/gt100
cd $R
ls $INDEX_ROOT/laion512/clustering/centroids_*.fvecs > /dev/null 2>&1 ||
  DATA_ENV=$RD/train.fbin OUTDIR_ENV=$INDEX_ROOT/laion512/clustering clustering/run_kmeans.sh laion512 > $L/rabitq_cluster.log 2>&1
ls $INDEX_ROOT/laion512/clustering/centroids_*.fvecs > /dev/null 2>&1 || { echo "rabitq cluster failed"; exit 1; }; echo "rabitq cluster ok $(date +%T)"
for M in 256 512; do for S in exbits raw; do
  MEMDIM_ENV=$M STORE_ENV=$S scripts/build_index/build_index.sh laion512 > $L/rabitq_build_m${M}_$S.log 2>&1
  rc=$?; (( rc == 0 )) || { echo "rabitq build memdim $M $S failed ($rc)"; exit 1; }; echo "rabitq build memdim $M $S ok $(date +%T)"
done; done

# 4. Searches, each on a quiet node.
until awk '{exit !($1 < 2.0)}' /proc/loadavg; do sleep 20; done
QB=$IB/ivf_pq_query_bench
for C in 32 64; do
  P=/nvme/ivf_bench/laion512/n16384_pq$C
  common="--data_type float --index_prefix $P --query_file $H/query.10K.fbin --gt_file $GT --skip_exact 1 --scan simd \
    --warmup_query_file $H/warmup_n1000_seed0.bin --buffer_pool_frames 262144 --rerank_m 100,200 --rerank_schedules 0:0"
  for Pn in 16 32 64 128 256 512 1024; do $QB $common --nprobes $Pn --graph_beams 1.25 --probe_taus 0 > $L/ivf_pq${C}_fixed_p$Pn.log 2>&1; done
  for Cap in 80 160 320 640 1280; do $QB $common --nprobes $Cap --graph_beams 1 --probe_taus 1.3,1.4,1.5,1.6,1.7 > $L/ivf_pq${C}_adapt_c$Cap.log 2>&1; done
  echo "ivf search pq$C exit=$? $(date +%T)"
done
for C in 32 64; do
  W=/nvme/bufann/laion512_pq$C
  for Lv in 10 20 30 40 60 80 100 150 200 300 400; do
    $VB/bufann_driver --data_type float --index_prefix $W/laion512 --dim $D --workload search_only \
      --query_file $H/query.10K.fbin --gt_file $GT10 --recall_at 10 --search_L $Lv --beamwidth 4 \
      --query_threads 32 --maintenance_threads 32 --buffer_pool_frames 262144 --pq_chunks $C --R 64 --L 100 \
      --result_file $L/bufann_pq${C}_L$Lv.json --delete_micro_batch 1 --max_dataset_size $N \
      --compaction_enabled 0 --compaction_auto_trigger 1 --compaction_fragmentation_ratio 0.50 --compaction_threads 1 \
      --warmup_query_file $H/warmup_n1000_seed0.bin --warmup_threads 32 > $L/bufann_pq${C}_L$Lv.log 2>&1
  done
  echo "bufann search pq$C exit=$? $(date +%T)"
done
for M in 256 512; do for S in exbits raw; do
  MEMDIM_ENV=$M STORE_ENV=$S OUT_TAG_SUFFIX=_$S TOPKS_ENV=10 TS_ENV=32 NPROBES_ENV=4,8,16,24,32,48,64,96,128,192,256,384,512 \
    scripts/main_exp/run_main_exp.sh laion512 > $L/rabitq_search_m${M}_$S.log 2>&1
  echo "rabitq search memdim $M $S exit=$? $(date +%T)"
done; done
echo ALL_DONE
