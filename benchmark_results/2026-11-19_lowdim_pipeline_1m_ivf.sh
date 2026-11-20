#!/bin/bash
# SIFT1M IVF-PQ rerun: 8192 frames threw (pool shard fully pinned at rerank 100+).
S=/tmpdata/work/lowdim D=/tmpdata/work/data/rq/data
$S/ivf_search.sh sift1m_p32768 /tmpdata/work/data/ivf/sift1m/n1024 $D/sift1m/test.fbin $D/sift1m/gt100 2,4,8,12,16,24,32,48,64,96,128 50,100,200 32768
echo PIPE1I_DONE
