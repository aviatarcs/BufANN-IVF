#!/bin/bash
# Control: does IVF-PQ sustain fewer reads/s than RaBitQ-SSD on the same volume
# (DEEP10M, 32 threads)? A-B-A order, run back to back; iostat alongside to
# see which device saturates.
L=/tmpdata/work/logs/lowdim S=/tmpdata/work/lowdim D=/tmpdata/work/data/rq/data
(command -v iostat >/dev/null && iostat -dxm -t 2 > $L/control_iostat.log 2>&1) & IO=$!
$S/ivf_search.sh deep10m_ctlA /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 48,64 50,100 65536
OUT_SUFFIX=_ctl STORE_ENV=rawmemex B_ENV=2 $S/rq_search.sh deep10m 24,48
OUT_SUFFIX=_ctl $S/rq_search.sh deep10m 24,48
$S/ivf_search.sh deep10m_ctlB /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 48,64 50,100 65536
kill $IO
echo CTL_DONE $(date +%T)
