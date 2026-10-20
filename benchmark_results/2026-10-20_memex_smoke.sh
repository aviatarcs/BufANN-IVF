#!/bin/bash
# Validation of STORE_ENV=rawmemex on a 1M DEEP slice: recall must match raw
# at every nprobe (the B-bit bound prunes nothing it should keep), with fewer
# pages read.
set -x
export DATA_ROOT=/nvme/rabitq_smoke/data INDEX_ROOT=/nvme/rabitq_smoke/indexes
export PYTHON_ENV=/nvme/bench100m/py PYTHONPATH=/tmpdata/pylib_faiss:/tmpdata/pylib BUDGET_GB_ENV=32
L=/nvme/bench100m/memex_smoke_logs; mkdir -p $L
cd /nvme/RaBitQ-SSD
DATA_ENV=$DATA_ROOT/deep1m/train.fbin OUTDIR_ENV=$INDEX_ROOT/deep1m/clustering clustering/run_kmeans.sh deep1m > $L/cluster.log 2>&1
echo "cluster exit=$?"
STORE_ENV=raw scripts/build_index/build_index.sh deep1m > $L/build_raw.log 2>&1; echo "build raw exit=$?"
for B in 2 3 9; do STORE_ENV=rawmemex B_ENV=$B scripts/build_index/build_index.sh deep1m > $L/build_memex_B$B.log 2>&1; echo "build memex B$B exit=$?"; done
NP=4,8,16,32,64,128,256
STORE_ENV=raw OUT_TAG_SUFFIX=_raw TOPKS_ENV=10 TS_ENV=16 NPROBES_ENV=$NP scripts/main_exp/run_main_exp.sh deep1m > $L/search_raw.log 2>&1; echo "search raw exit=$?"
for B in 2 3 9; do STORE_ENV=rawmemex B_ENV=$B OUT_TAG_SUFFIX=_memexB$B TOPKS_ENV=10 TS_ENV=16 NPROBES_ENV=$NP scripts/main_exp/run_main_exp.sh deep1m > $L/search_memex_B$B.log 2>&1; echo "search memex B$B exit=$?"; done
echo ALL_DONE
