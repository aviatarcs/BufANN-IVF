#!/bin/bash
# BufANN (vectordb-bench a9f031d) on one 100M dataset, as the harness's query
# workload does it (vectordb-baselines benchmark/BufANN/{common,sift/query*}.sh)
# but with the steps called directly so everything stays on /nvme:
#   1. 32-chunk DiskANN PQ for the canonical (build_pq_standalone on the 90M base)
#   2. fast_convert_diskann_to_inplaceann -> BufANN heap/meta/medoids family
#   3. top-10 GT from our exact 90M top-100 (incremental_groundtruth --base_trunc)
#   4. bufann_driver search_only, one process per search_L, flags as
#      run_bufann_driver passes them; warmup = the IVF-PQ runs' 1000-row file.
#   bufann_one.sh <name> <type> <ext> <dim>
set -x
name=$1 type=$2 ext=$3 dim=$4
DB=/tmpdata/vectordb-baselines/DiskANN/build/tests
VB=/tmpdata/vectordb-bench/build/tests
C=/nvme/indexes/$name/${name}_90M_R64_L100_noPQ
W=/nvme/bufann/$name
R=/nvme/bench100m/results_bufann
mkdir -p $W/index $R /nvme/bufann/gt
[[ -f ${C}_disk.index ]] || { echo "no canonical for $name"; exit 1; }
[[ -f ${C}_pq_compressed.bin ]] || $DB/build_pq_standalone $type /nvme/100m/$name/base.90M.$ext $C 32 > $R/${name}_pq.log 2>&1 || exit 1
[[ -f ${C}_disk.index.tags ]] || PYTHONPATH=/tmpdata/pylib python3 -c "
import struct, numpy as np
with open('${C}_disk.index.tags','wb') as f:
    f.write(struct.pack('<ii', 90000000, 1)); np.arange(90000000, dtype=np.uint32).tofile(f)"
[[ -f $W/index/$name.heap ]] || /usr/bin/time -v $VB/fast_convert_diskann_to_inplaceann $type $C $W/index/$name $dim 64 32 262144 \
  > $R/${name}_convert.log 2>&1 || exit 1
GT=/nvme/bufann/gt/${name}_90000000_gt10.bin
[[ -f $GT ]] || $DB/utils/incremental_groundtruth /nvme/100m/$name/${name}_90000000_gt100.bin 10 $GT --base_trunc 90000000 || exit 1
for L in 20 30 40 60 80 100 150 200 300 400; do
  $VB/bufann_driver --data_type $type --index_prefix $W/index/$name --dim $dim --workload search_only \
    --query_file /nvme/100m/$name/query.10K.$ext --gt_file $GT --recall_at 10 --search_L $L --beamwidth 4 \
    --query_threads 32 --maintenance_threads 32 --buffer_pool_frames 262144 --pq_chunks 32 --R 64 --L 100 \
    --result_file $R/${name}_L$L.json --delete_micro_batch 1 --max_dataset_size 100000000 \
    --compaction_enabled 0 --compaction_auto_trigger 1 --compaction_fragmentation_ratio 0.50 --compaction_threads 1 \
    --warmup_query_file /nvme/bench100m/warmup_${name}_90M_n1000_seed0.bin --warmup_threads 32 > $R/${name}_L$L.log 2>&1
  echo "$name L $L exit=$?"
done
