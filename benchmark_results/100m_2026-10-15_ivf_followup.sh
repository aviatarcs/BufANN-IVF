#!/bin/bash
# Follow-ups to ivf_runs.sh: where each dataset's recall ceiling comes from.
# Deeper probes (nprobe 2048 / cap 2560) and re-ranks of 200-400, one batch.
source <(sed -n '/^B=/,/^}/p' /nvme/bench100m/ivf_runs.sh)
run turing100m_fixed_p2048 turing100m float fbin 2048 1.25 0 100,200,400 0:0
run turing100m_adapt_c2560 turing100m float fbin 2560 1 1.4,1.5,1.6,1.8 100,200 0:0
run spacev100m_fixed_p2048 spacev100m int8 i8bin 2048 1.25 0 100,200 0:0
run spacev100m_adapt_c2560 spacev100m int8 i8bin 2560 1 1.5,1.7,2.0 100,200 0:0
run deep100m_fixed_p1024r deep100m float fbin 1024 1.25 0 200,400 0:0
run deep100m_adapt_c640r deep100m float fbin 640 1 1.4,1.5,1.7 200,400 0:0
echo ALL_DONE
