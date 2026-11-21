#!/bin/bash
# ctlA overlapped an unlocked highdim run at 21:53:42 (audit log); repeat it as ctlC.
L=/tmpdata/work/logs/lowdim S=/tmpdata/work/lowdim D=/tmpdata/work/data/rq/data
until grep -q CTL_DONE $L/control_deep10m.log; do sleep 10; done
$S/ivf_search.sh deep10m_ctlC /tmpdata/work/data/ivf/deep10m/n16384 $D/deep10m/test.fbin $D/deep10m/gt100 48,64 50,100 65536
echo CTLC_DONE $(date +%T)
