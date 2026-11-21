#!/bin/bash
# /nvme wave 1 after staging: RaBitQ-SSD, then BufANN. Holds the first timed
# step until load < 12 or LADDERS_GATED exists (both ladder sessions build
# through build_gate.sh).
set -u
W=/tmpdata/work; S=$W/logs/highdim/status.log
cd $W/highdim-scripts
until [[ -e LADDERS_GATED ]] || awk '{exit !($1 < 12)}' /proc/loadavg; do sleep 30; done
echo "$(date '+%F %T') [nvme] wave 1 starting, load $(cut -d' ' -f1-3 /proc/loadavg)" >> $S
for ds in gist960 dbpedia1536 laion512 cohere768; do
  RUN_TAG=_nvme HD_ROOT=/nvme/highdim RETRY_ABORT=0 ./highdim.sh $ds search_rabitq
done
echo "$(date '+%F %T') [nvme] rabitq done" >> $S
until [[ -e FIX_READY ]]; do sleep 60; done
for ds in gist960 dbpedia1536 laion512 cohere768; do
  RUN_TAG=_nvme HD_ROOT=/nvme/highdim RETRY_ABORT=0 ./highdim.sh $ds search_bufann
done
echo "$(date '+%F %T') [nvme] bufann done; IVF next" >> $S
