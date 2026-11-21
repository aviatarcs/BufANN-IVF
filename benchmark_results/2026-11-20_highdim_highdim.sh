#!/bin/bash
# High-dim comparison (November 2026): IVF-PQ (agent/highdim), BufANN
# (bufann_driver from the same tree) and RaBitQ-SSD (7ab80aa) on four >= 512-d
# sets at the LAION-512 run's two memory budgets (PQ bytes = in-RAM RaBitQ
# bits / 8 = D/16 and D/8 bytes, i.e. 0.5 and 1 bit per dimension), 32
# threads, 1000-row warmup sampled from the base, 10K queries, exact top-100
# ground truth (exact_gt), recall@10. Template: 2026-10-20_laion512.sh.
#
#   highdim.sh <dataset> prep     ground truth, warmup, all index builds
#   highdim.sh <dataset> search   the timed runs, each under bench.lock
#
# Every step runs through run_step: in the background, registered with the
# hang watchdog (/tmpdata/work/highdim_watchdog.sh), retried once if the
# watchdog kills it, then skipped.
set -u
ds=$1 phase=$2
W=/tmpdata/work
WT=$W/wt-highdim
IB=${IB_DIR:-$W/highdim-build/build-wt-highdim-fix/tests}        # ivf_pq_*, bufann_driver, fast_convert
DB=$W/highdim-build/diskann/tests                  # DiskANN shard helpers, build_pq_standalone
GTBIN=$W/highdim-build/exact_gt
R=$W/highdim-src/RaBitQ-SSD
PY=$W/highdim-py/bin/python
LOCK=$W/bench.lock
HD=${HD_ROOT:-$W/data/highdim}   # HD_ROOT=/nvme/highdim for the /nvme reruns
H=$HD/$ds
RUN=${RUN_TAG:-}                  # RUN_TAG=_nvme: separate logs and RaBitQ result dirs
L=$W/logs/highdim$RUN/$ds
JOBS=$W/logs/highdim/jobs
STATUS=$W/logs/highdim/status.log
mkdir -p $L $JOBS

case $ds in  # N D nlist pq_small pq_large memdim_small memdim_large page fixed_nprobes adaptive_caps rabitq_nprobes
  gist960)     N=1000000  D=960  NL=2048  CS=64 CL=120 MS=512 ML=960  PG=8192
               FIX="4 8 16 32 64 128 256 512" CAPS="10 20 40 80 160 320" RNP=2,4,8,12,16,24,32,48,64,96,128,192,256,384 ;;
  dbpedia1536) N=990000   D=1536 NL=2048  CS=96 CL=192 MS=768 ML=1536 PG=8192
               FIX="4 8 16 32 64 128 256 512" CAPS="10 20 40 80 160 320" RNP=2,4,8,12,16,24,32,48,64,96,128,192,256,384 ;;
  laion512)    N=10000000 D=512  NL=16384 CS=32 CL=64  MS=256 ML=512  PG=4096
               FIX="16 32 64 128 256 512 1024 2048" CAPS="80 160 320 640 1280 2560" RNP=4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024 ;;
  cohere768)   N=9990000  D=768  NL=16384 CS=48 CL=96  MS=384 ML=768  PG=4096
               FIX="16 32 64 128 256 512 1024 2048" CAPS="80 160 320 640 1280 2560" RNP=4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024 ;;
  *) echo "unknown dataset $ds"; exit 2 ;;
esac
# IVF-PQ's raw-vector heap fits one 1536-d float vector only in an 8 KB page;
# BufANN's heap slot (vector + 64 neighbours + header) needs 8 KB from 960-d.
# Both pools are 1 GB whatever the page size.
IVF_PG=$(( D * 4 + 24 > 4096 ? 8192 : 4096 ))
IVF_FRAMES=$(( (1 << 30) / IVF_PG ))
BUF_FRAMES=$(( (1 << 30) / PG ))
BASE=$H/base.fbin Q=$H/query.10K.fbin GT=$H/gt100.bin GT10=$H/gt10.bin WARM=$H/warmup_n1000_seed0.bin
CAN=$H/diskann/${ds}_R64_L100_noPQ
export DATA_ROOT=$HD/rabitq/data INDEX_ROOT=$HD/rabitq/indexes PYTHON_ENV=$PY BUDGET_GB_ENV=48
export RABITQ_BUILD_SCRATCH_DIR=$W/data/highdim/rabitq/scratch

say() { echo "$(date '+%F %T') [$ds$RUN] $*" | tee -a $STATUS; }
run_step() {  # name log cmd... : background + watchdog registration, one retry
  local step=$1 name=${ds}${RUN}_$1 lg=$2 rc pid; shift 2
  for attempt in 1 2; do
    rm -f $JOBS/$name.killed
    [[ -f $lg ]] && mv $lg $lg.prev
    echo "# $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg): $*" > $lg
    "$@" >> $lg 2>&1 &
    pid=$!
    echo "$pid $lg" > $JOBS/$name.job
    wait $pid; rc=$?
    rm -f $JOBS/$name.job
    # ivf_pq_query_bench intermittently aborts (exit 134, uncaught
    # ANNException from the buffer pool; being fixed in the 9bit session):
    # rerun the same command once. A run that completes is unaffected.
    if [[ ${RETRY_ABORT:-1} == 1 && $rc == 134 && $attempt == 1 && ! -e $JOBS/$name.killed ]]; then say "$step exit=134 (abort), retrying once"; continue; fi
    [[ -e $JOBS/$name.killed ]] || { say "$step exit=$rc"; return $rc; }
    say "$step KILLED by watchdog (attempt $attempt)"
  done
  say "$step SKIPPED after two hangs"; return 99
}
timed() {  # every timed run holds bench.lock; logs the load once it has it
  # Builds under /tmpdata/work/build_gate.sh pause while the lock is held;
  # give the 1-minute load average up to 5 min to fall below 12 first.
  flock $LOCK bash -c 'for i in $(seq 30); do awk "{exit !(\$1 < 12)}" /proc/loadavg && break; sleep 10; done
    echo "# $(date "+%F %T") lock acquired, load $(cut -d" " -f1-3 /proc/loadavg)"; exec "$@"' timed "$@"
}
build_threads() { echo 24; }     # builds leave a few cores to the other agents

prep() {
  [[ -s $BASE && -s $Q ]] || { say "no data"; exit 1; }
  [[ -s $GT ]] || run_step gt $L/gt.log env OMP_NUM_THREADS=$(build_threads) nice -n 5 $GTBIN float $BASE $Q 100 $GT || exit 1
  [[ -s $L/gt_check.log ]] || run_step gtcheck $L/gt_check.log $PY -I $W/highdim-scripts/check_gt.py $BASE $Q $GT 20
  [[ -s $WARM ]] || run_step warmup $L/warmup.log $PY -I $WT/benchmark/sample_base_warmup.py $BASE float 1000 $WARM --seed 0 || exit 1

  # IVF-PQ, both budgets.
  export OMP_NUM_THREADS=$(build_threads)
  for C in $CS $CL; do
    P=$H/ivf/n${NL}_pq$C; mkdir -p $H/ivf
    [[ -f ${P}_ivf_pq_index.bin ]] || run_step ivf_build_pq$C $L/ivf_build_pq$C.log nice -n 5 /usr/bin/time -v \
      $IB/ivf_pq_build_index --data_type float --data_file $BASE --index_prefix $P --dim $D \
      --ivf_nlist $NL --ivf_pq_chunks $C --page_size $IVF_PG
  done

  # BufANN: one DiskANN graph (R 64, L 100), PQ at both budgets.
  mkdir -p $H/diskann
  if [[ ! -f ${CAN}_disk.index ]]; then
    run_step diskann_build $L/diskann_build.log nice -n 5 bash -c "
      set -e
      TCMALLOC_RELEASE_RATE=10 $DB/partition_only float $BASE $CAN 64 40 $N
      NS=\$(ls ${CAN}_mem.index_tempFiles_subshard-[0-9]*_ids_uint32.bin | wc -l)
      for ((p = 0; p < NS; p++)); do TCMALLOC_RELEASE_RATE=10 $DB/build_shard float ${CAN}_mem.index_tempFiles_subshard-\$p.bin \
        ${CAN}_mem.index_tempFiles_subshard-\${p}_mem.index 64 100 $(build_threads) l2; done
      $DB/merge_shards_only ${CAN}_mem.index_tempFiles_subshard- \$NS 64 ${CAN}_mem.index ${CAN}_disk.index_medoids.bin
      $DB/finalize_index float ${CAN}_mem.index $BASE ${CAN}_disk.index $N" || exit 1
    rm -f ${CAN}_mem.index_tempFiles_subshard-*   # shard temp files: tens of GB at 10M
  fi
  [[ -f $GT10 ]] || run_step gt10 $L/gt10.log $DB/utils/incremental_groundtruth $GT 10 $GT10 --base_trunc $N || exit 1
  for C in $CS $CL; do
    V=$H/diskann/pq$C; mkdir -p $V
    for f in ${CAN}_disk.index*; do ln -sfn $f $V/$ds${f#$CAN}; done
    [[ -f $V/${ds}_pq_compressed.bin ]] || run_step bufann_pq$C $L/bufann_pq$C.log nice -n 5 $DB/build_pq_standalone float $BASE $V/$ds $C
    [[ -f $V/${ds}_disk.index.tags ]] || $PY -c "
import struct, numpy as np
with open('$V/${ds}_disk.index.tags', 'wb') as f: f.write(struct.pack('<ii', $N, 1)); np.arange($N, dtype=np.uint32).tofile(f)"
    B=$H/bufann/pq$C; mkdir -p $B
    [[ -f $B/$ds.heap ]] || run_step bufann_convert_pq$C $L/bufann_convert_pq$C.log \
      $IB/fast_convert_diskann_to_inplaceann float $V/$ds $B/$ds $D 64 $C $BUF_FRAMES $PG
  done
  rm -f $H/diskann/pq*/${ds}_pq_compressed.bin.tmp 2>/dev/null

  # RaBitQ-SSD: one clustering, then memdim x store.
  RD=$DATA_ROOT/$ds; mkdir -p $RD $RABITQ_BUILD_SCRATCH_DIR
  ln -sfn $BASE $RD/train.fbin; ln -sfn $Q $RD/test.fbin; ln -sfn $GT $RD/gt10; ln -sfn $GT $RD/gt100
  cd $R
  ls $INDEX_ROOT/$ds/clustering/centroids_*.fvecs > /dev/null 2>&1 ||
    run_step rabitq_cluster $L/rabitq_cluster.log env DATA_ENV=$RD/train.fbin OUTDIR_ENV=$INDEX_ROOT/$ds/clustering \
      nice -n 5 clustering/run_kmeans.sh $ds || exit 1
  for M in $MS $ML; do for S in exbits raw; do
    run_step rabitq_build_m${M}_$S $L/rabitq_build_m${M}_$S.log env MEMDIM_ENV=$M STORE_ENV=$S nice -n 5 scripts/build_index/build_index.sh $ds
  done; done
  say "prep done"
}

search_ivf() {
  # IVF_IO=direct (the /nvme set): uncached O_DIRECT re-rank reads, pipelined
  # 0 and 2 queries deep. IVF_IO=pool: the /tmpdata set's 1 GB buffer pool.
  if [[ ${IVF_IO:-pool} == direct ]]; then IVF_IO_ARGS="--io_mode direct --pipeline_depths ${IVF_PIPE:-0,2}"
  else IVF_IO_ARGS="--io_mode pool --buffer_pool_frames $IVF_FRAMES"; fi
  QB=$IB/ivf_pq_query_bench
  for C in $CS $CL; do
    P=$H/ivf/n${NL}_pq$C
    [[ -f ${P}_ivf_pq_index.bin ]] || { say "no ivf pq$C index"; continue; }
    common="--data_type float --index_prefix $P --query_file $Q --gt_file $GT --skip_exact 1 --scan simd
      --warmup_query_file $WARM $IVF_IO_ARGS --threads 32 --rerank_m 100,200,400,800,1600 --rerank_schedules 0:0"
    for Pn in $FIX; do run_step ivf_pq${C}_fixed_p$Pn $L/ivf_pq${C}_fixed_p$Pn.log timed $QB $common --nprobes $Pn --graph_beams 1.25 --probe_taus 0; done
    for Cap in $CAPS; do run_step ivf_pq${C}_adapt_c$Cap $L/ivf_pq${C}_adapt_c$Cap.log timed $QB $common --nprobes $Cap --graph_beams 1 --probe_taus 1.3,1.4,1.5,1.6,1.7; done
  done
  say "search_ivf done"
}
search_bufann() {
  for C in $CS $CL; do
    B=$H/bufann/pq$C
    [[ -f $B/$ds.heap ]] || { say "no bufann pq$C heap"; continue; }
    for Lv in 10 20 30 40 60 80 100 150 200 300 400 600; do
      run_step bufann_pq${C}_L$Lv $L/bufann_pq${C}_L$Lv.log timed $IB/bufann_driver --data_type float --index_prefix $B/$ds --dim $D \
        --workload search_only --query_file $Q --gt_file $GT10 --recall_at 10 --search_L $Lv --beamwidth 4 \
        --query_threads 32 --maintenance_threads 32 --buffer_pool_frames $BUF_FRAMES --page_size $PG --pq_chunks $C --R 64 --L 100 \
        --result_file $L/bufann_pq${C}_L$Lv.json --delete_micro_batch 1 --max_dataset_size $N \
        --compaction_enabled 0 --compaction_auto_trigger 1 --compaction_fragmentation_ratio 0.50 --compaction_threads 1 \
        --warmup_query_file $WARM --warmup_threads 32
    done
  done
  say "search_bufann done"
}
search_rabitq() {
  cd $R
  for M in $MS $ML; do for S in exbits raw; do
    run_step rabitq_search_m${M}_$S $L/rabitq_search_m${M}_$S.log timed env MEMDIM_ENV=$M STORE_ENV=$S OUT_TAG_SUFFIX=${RUN}_$S \
      TOPKS_ENV=10 TS_ENV=32 NPROBES_ENV=$RNP scripts/main_exp/run_main_exp.sh $ds
  done; done
  say "search_rabitq done"
}
search_rabitq_check() {  # A-B-A: one RaBitQ-SSD config again after the IVF runs, for drift
  cd $R
  run_step rabitq_search_m${MS}_exbits_after $L/rabitq_search_m${MS}_exbits_after.log timed env MEMDIM_ENV=$MS STORE_ENV=exbits \
    OUT_TAG_SUFFIX=${RUN}_after_exbits TOPKS_ENV=10 TS_ENV=32 NPROBES_ENV=$RNP scripts/main_exp/run_main_exp.sh $ds
}
search() { search_ivf; search_bufann; search_rabitq; say "search done"; }

dbpedia_rerun() {  # the one IVF point that aborted in the main pass
  C=192; P=$H/ivf/n${NL}_pq$C; QB=$IB/ivf_pq_query_bench
  common="--data_type float --index_prefix $P --query_file $Q --gt_file $GT --skip_exact 1 --scan simd
    --warmup_query_file $WARM --buffer_pool_frames $IVF_FRAMES --threads 32 --rerank_m 100,200,400,800,1600 --rerank_schedules 0:0"
  run_step ivf_pq${C}_adapt_c80 $L/ivf_pq${C}_adapt_c80.log timed $QB $common --nprobes 80 --graph_beams 1 --probe_taus 1.3,1.4,1.5,1.6,1.7
}

$phase
