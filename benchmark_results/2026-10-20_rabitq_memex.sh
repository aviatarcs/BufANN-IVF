#!/bin/bash
# RaBitQ-SSD with STORE_ENV=rawmemex B_ENV=2 on the four 90M bases (branch
# memex-2bit of the clone): 2-bit RaBitQ codes in RAM (1-bit code + 1 ex bit
# per dimension, ~52 B/vector at padded dim 128), raw vectors on SSD, the
# 2-bit bound f_error*g_error/2 pruning before any page is read, exact re-rank.
# Same clustering, paper defaults otherwise, 32 threads, top-10, fixed nprobes.
set -x
cd /nvme/RaBitQ-SSD
export DATA_ROOT=/nvme/rabitq/data INDEX_ROOT=/nvme/rabitq/indexes
export PYTHON_ENV=/nvme/bench100m/py PYTHONPATH=/tmpdata/pylib_faiss:/tmpdata/pylib
export BUDGET_GB_ENV=96 STORE_ENV=rawmemex B_ENV=2 OUT_TAG_SUFFIX=_memexB2
L=/nvme/bench100m/rabitq_logs
NPROBES=8,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536
DS="sift100m deep100m spacev100m turing100m"
for ds in $DS; do
  /usr/bin/time -v scripts/build_index/build_index.sh $ds > $L/${ds}_memexB2_build.log 2>&1
  echo "$ds memex build exit=$? $(date +%T)"
done
until awk '{exit !($1 < 2.0)}' /proc/loadavg; do sleep 20; done
for ds in $DS; do
  TOPKS_ENV=10 TS_ENV=32 NPROBES_ENV=$NPROBES scripts/main_exp/run_main_exp.sh $ds > $L/${ds}_memexB2_search.log 2>&1
  echo "$ds memex search exit=$? $(date +%T)"
done
echo ALL_DONE
