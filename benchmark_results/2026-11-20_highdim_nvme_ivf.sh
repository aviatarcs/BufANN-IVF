#!/bin/bash
# /nvme IVF-PQ in direct io mode (uncached O_DIRECT re-rank, pipeline 0 and 2),
# after the /nvme BufANN set; then signal WAVE1_DONE to the ablation sessions.
set -u
W=/tmpdata/work; S=$W/logs/highdim/status.log
until grep -q '\[nvme\] bufann done' $S; do sleep 60; done
cd $W/highdim-scripts
for ds in gist960 dbpedia1536 laion512 cohere768; do
  RUN_TAG=_nvme HD_ROOT=/nvme/highdim RETRY_ABORT=0 IVF_IO=direct ./highdim.sh $ds search_ivf
  RUN_TAG=_nvme HD_ROOT=/nvme/highdim RETRY_ABORT=0 ./highdim.sh $ds search_rabitq_check
done
echo "$(date '+%F %T') [nvme] ivf done; all /nvme timed reruns complete" >> $S
touch $W/WAVE1_DONE
