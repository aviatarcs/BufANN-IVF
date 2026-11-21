#!/bin/bash
# Ablation: does IVF-PQ gain from more threads than cores (its re-rank fetch
# blocks the thread; RaBitQ-SSD overlaps reads with scanning)? And RaBitQ at T=64.
L=/tmpdata/work/logs/lowdim S=/tmpdata/work/lowdim D=/tmpdata/work/data/rq/data
until grep -q PIPE10_DONE $L/pipeline_10m.log; do sleep 15; done
for T in 64 128; do
  $S/ivf_search.sh deep10m_p65536_T$T /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 32,64,128,256 100 65536 --threads $T
done
$S/ivf_search.sh deep10m_p65536_pagecache /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 32,64,128,256 100 0
TS_ENV=64 OUT_SUFFIX=_T64 STORE_ENV=rawmemex B_ENV=2 $S/rq_search.sh deep10m 16,32,64,128,256
echo EXTRA_DONE $(date +%T)
