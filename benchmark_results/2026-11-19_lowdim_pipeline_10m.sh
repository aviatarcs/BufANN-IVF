#!/bin/bash
# DEEP10M (96-d): after the 1M runs, build both systems' indexes (untimed),
# then time the searches. IVF-PQ nlist 16384 (~610/list, as 131072 at 90M);
# RaBitQ-SSD C = N/1024 = 9766.
L=/tmpdata/work/logs/lowdim S=/tmpdata/work/lowdim D=/tmpdata/work/data/rq/data
until grep -q PIPE_DONE $L/pipeline_1m.log; do sleep 15; done
until grep -q "wrote.*gt100" $L/gt_deep10m.log; do sleep 15; done
mkdir -p /tmpdata/work/data/ivf/deep10m
/usr/bin/time -v /tmpdata/work/wt-lowdim/build/tests/ivf_pq_build_index --data_type float --data_file /tmpdata/work/data/deep10m/base.10M.fbin \
  --index_prefix /tmpdata/work/data/ivf/deep10m/n16384 --dim 96 --ivf_nlist 16384 --ivf_pq_chunks 32 > $L/ivf_build_deep10m.log 2>&1
echo "ivf build exit=$?"
DS=deep10m $S/rq_build_1m.sh > $L/rq_build_deep10m.log 2>&1; echo "rq build: $(grep -c ' 0$' $L/rq_build_deep10m.log) ok"
NP=4,8,12,16,24,32,48,64,96,128,192,256,384,512
$S/rq_search.sh deep10m $NP
STORE_ENV=raw $S/rq_search.sh deep10m $NP
STORE_ENV=rawmemex B_ENV=2 $S/rq_search.sh deep10m $NP
# pool: 1 GB holds 26% of the 3.8 GB heap; 65536 frames (256 MB, 6.5% of the heap) ~ the 100M runs' 3% share.
$S/ivf_search.sh deep10m_p65536 /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 16,32,48,64,96,128,192,256,384,512 50,100,200 65536
$S/ivf_search.sh deep10m_p262144 /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 16,32,64,128,256,512 100 262144
echo PIPE10_DONE $(date +%T)
