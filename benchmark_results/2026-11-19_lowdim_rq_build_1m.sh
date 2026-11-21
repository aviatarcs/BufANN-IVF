#!/bin/bash
# RaBitQ-SSD (7ab80aa + rawmemex patch) indexes for SIFT1M (128-d) and GIST1M
# (960-d): paper defaults (C = N/1024, IRQ), three SSD stores. Untimed.
set -x
cd /tmpdata/work/rabitq-ssd
export DATA_ROOT=/tmpdata/work/data/rq/data INDEX_ROOT=/tmpdata/work/data/rq/idx
export PYTHON_ENV=/tmpdata/work/lowdim/py/bin/python BUDGET_GB_ENV=32
L=/tmpdata/work/logs/lowdim
for ds in ${DS:-sift1m gist1m}; do
  ls $INDEX_ROOT/$ds/clustering/centroids_*.fvecs >/dev/null 2>&1 ||
    DATA_ENV=$DATA_ROOT/$ds/train.fbin OUTDIR_ENV=$INDEX_ROOT/$ds/clustering clustering/run_kmeans.sh $ds > $L/rq_${ds}_cluster.log 2>&1
  echo "$ds cluster $?"
  scripts/build_index/build_index.sh $ds > $L/rq_${ds}_build_B9.log 2>&1; echo "$ds B9 $?"
  STORE_ENV=raw scripts/build_index/build_index.sh $ds > $L/rq_${ds}_build_raw.log 2>&1; echo "$ds raw $?"
  STORE_ENV=rawmemex B_ENV=2 scripts/build_index/build_index.sh $ds > $L/rq_${ds}_build_memexB2.log 2>&1; echo "$ds memexB2 $?"
done
echo ALL_DONE
