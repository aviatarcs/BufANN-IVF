#!/bin/bash
# RaBitQ-SSD with STORE_ENV=raw on the four 90M bases: the SSD records hold the
# vectors themselves (uint8 for SIFT, float32 for DEEP/SPACEV/Turing) and the
# re-rank is exact, as in IVF-PQ and BufANN. Everything else as rabitq_run.sh
# (paper defaults, full 1-bit code in DRAM, the same clustering, 32 threads,
# top-10, fixed nprobes). Searches wait until the high-dim downloads that share
# the NVMe volume have finished.
set -x
R=/nvme/RaBitQ-SSD
export DATA_ROOT=/nvme/rabitq/data INDEX_ROOT=/nvme/rabitq/indexes
export PYTHON_ENV=/nvme/bench100m/py PYTHONPATH=/tmpdata/pylib_faiss:/tmpdata/pylib
export BUDGET_GB_ENV=96 STORE_ENV=raw OUT_TAG_SUFFIX=_raw
L=/nvme/bench100m/rabitq_logs
cd $R
NPROBES=8,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536
DS="sift100m deep100m spacev100m turing100m"
for ds in $DS; do
  /usr/bin/time -v scripts/build_index/build_index.sh $ds > $L/${ds}_raw_build.log 2>&1
  echo "$ds raw build exit=$? $(date +%T)"
done
until grep -q WIKI_DONE /nvme/hd/wiki10m/fetch.log && grep -q YFCC_DONE /nvme/hd/yfcc10m/fetch.log; do
  pgrep -f 'curl -fsSL' > /dev/null || break   # a failed download must not block the searches forever
  sleep 60
done
echo "downloads settled $(date +%T)"
until awk '{exit !($1 < 2.0)}' /proc/loadavg; do sleep 20; done
for ds in $DS; do
  TOPKS_ENV=10 TS_ENV=32 NPROBES_ENV=$NPROBES scripts/main_exp/run_main_exp.sh $ds > $L/${ds}_raw_search.log 2>&1
  echo "$ds raw search exit=$? $(date +%T)"
done
echo ALL_DONE
