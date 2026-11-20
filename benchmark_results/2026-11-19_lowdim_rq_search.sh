#!/bin/bash
# RaBitQ-SSD searches, timed under the shared bench lock. 32 threads, top-10,
# paper defaults (IRQ, drain alpha 0.4, QD 8/thread at T>16), O_DIRECT.
#   rq_search.sh <dataset> <nprobes>   (STORE_ENV/B_ENV select the index)
cd /tmpdata/work/rabitq-ssd
export DATA_ROOT=/tmpdata/work/data/rq/data INDEX_ROOT=/tmpdata/work/data/rq/idx
export PYTHON_ENV=/tmpdata/work/lowdim/py/bin/python
ds=$1
tag=${STORE_ENV:-exbits}${B_ENV:-}
TOPKS_ENV=10 TS_ENV=${TS_ENV:-32} NPROBES_ENV=$2 OUT_TAG_SUFFIX=_${tag}${OUT_SUFFIX:-} \
  flock /tmpdata/work/bench.lock scripts/main_exp/run_main_exp.sh $ds > /tmpdata/work/logs/lowdim/rq_${ds}_${tag}${OUT_SUFFIX:-}_search.log 2>&1
echo "$ds $tag search exit=$?"
