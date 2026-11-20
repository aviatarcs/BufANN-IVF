#!/bin/bash
# Wait for the 1M builds, then run every timed 1M search, one at a time.
L=/tmpdata/work/logs/lowdim S=/tmpdata/work/lowdim
until grep -q ALL_DONE $L/rq_build_1m.log && grep -q ALL_DONE $L/ivf_build_1m.log; do sleep 10; done
echo "builds done $(date +%T)"
NP=1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256
for ds in sift1m gist1m; do
  $S/rq_search.sh $ds $NP
  STORE_ENV=raw $S/rq_search.sh $ds $NP
  STORE_ENV=rawmemex B_ENV=2 $S/rq_search.sh $ds $NP
done
D=/tmpdata/work/data/rq/data
$S/ivf_search.sh sift1m_p8192 /tmpdata/work/data/ivf/sift1m/n1024 $D/sift1m/test.fbin $D/sift1m/gt100 2,4,8,12,16,24,32,48,64,96,128 50,100,200 8192
$S/ivf_search.sh gist1m_p32768 /tmpdata/work/data/ivf/gist1m/n1024 $D/gist1m/test.fbin $D/gist1m/gt100 2,4,8,12,16,24,32,48,64,96,128 50,100,200 32768
echo PIPE_DONE $(date +%T)
