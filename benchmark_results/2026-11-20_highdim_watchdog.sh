#!/bin/bash
# Hang watchdog for the high-dim comparison. Every INTERVAL s it checks each
# registered job (a file $JOBS/<name>.job holding "<pid> <log>", written by
# run_step in highdim-scripts/lib.sh) over its whole process tree for
#   (a) log/output not advancing for > STALE_S,
#   (b) tree alive with ~0 CPU and no /proc/<pid>/io progress since last check,
#   (c) every thread of every non-shell process waiting in a futex.
# A job waiting for bench.lock (a flock with no child yet) is not a hang.
# When (a) holds together with (b) or (c) for SUSPECT_CHECKS consecutive
# checks, it captures `gdb thread apply all bt` (eu-stack as fallback) of every
# non-shell process in the tree into $LOGDIR/hang_<job>_<time>.txt, then kills
# the tree and touches $JOBS/<name>.killed so run_step retries once / skips.
# Status lines go to $LOGDIR/watchdog.log.
LOGDIR=/tmpdata/work/logs/highdim
JOBS=$LOGDIR/jobs
INTERVAL=${INTERVAL:-120}
STALE_S=${STALE_S:-600}
SUSPECT_CHECKS=${SUSPECT_CHECKS:-3}
mkdir -p $JOBS
declare -A last_cpu last_io suspect

tree() {  # pid -> pid and all descendants
  local p=$1; echo $p
  for c in $(cat /proc/$p/task/*/children 2>/dev/null); do tree $c; done
}
log() { echo "$(date '+%F %T') $*" >> $LOGDIR/watchdog.log; }

while true; do
  for jf in $JOBS/*.job; do
    [[ -e $jf ]] || continue
    name=$(basename $jf .job); read -r pid lg < $jf
    if ! kill -0 $pid 2>/dev/null; then continue; fi
    pids=$(tree $pid)
    cpu=0 io=0 nthreads=0 nfutex=0 waiting_lock=0 workers=""
    now=$(date +%s)
    for p in $pids; do
      [[ -r /proc/$p/stat ]] || continue
      read -r -a st < <(sed 's/.*) //' /proc/$p/stat 2>/dev/null)
      (( ${#st[@]} > 12 )) || continue   # the pid exited between the listing and the read
      cpu=$((cpu + st[11] + st[12]))
      # A zombie (exited, not yet reaped by a stopped parent) passes kill -0 but
      # has no readable io file: gawk then prints nothing, so default to 0.
      pio=$(awk '/^(rchar|wchar|read_bytes|write_bytes):/{s+=$2} END{print s+0}' /proc/$p/io 2>/dev/null)
      io=$((io + ${pio:-0}))
      comm=$(cat /proc/$p/comm 2>/dev/null)
      case $comm in
        bash|sh|sleep|time|tee|nice|timeout|env) ;;
        flock) [[ -z $(cat /proc/$p/task/*/children 2>/dev/null) ]] && waiting_lock=1 ;;
        *) workers+=" $p"
           for t in /proc/$p/task/*; do
             nthreads=$((nthreads + 1))
             [[ $(cat $t/wchan 2>/dev/null) == futex* ]] && nfutex=$((nfutex + 1))
           done ;;
      esac
    done
    age=$(( now - $(stat -c %Y "$lg" 2>/dev/null || echo $now) ))
    dcpu=$(( cpu - ${last_cpu[$name]:-0} )); dio=$(( io - ${last_io[$name]:-0} ))
    first=${last_cpu[$name]:+0}; first=${first:-1}
    last_cpu[$name]=$cpu; last_io[$name]=$io
    idle=0; ((first == 0 && dcpu <= 2 && dio == 0)) && idle=1      # (b): <= 2 ticks in INTERVAL s
    allfutex=0; ((nthreads > 0 && nfutex == nthreads)) && allfutex=1 # (c)
    stale=0; ((age > STALE_S)) && stale=1                           # (a)
    state="ok"
    if ((waiting_lock)); then state="waiting-for-bench.lock"; suspect[$name]=0
    elif ((stale && (idle || allfutex))); then
      suspect[$name]=$(( ${suspect[$name]:-0} + 1 )); state="SUSPECT(${suspect[$name]}/$SUSPECT_CHECKS)"
    else suspect[$name]=0; ((stale)) && state="silent-but-busy"
    fi
    log "$name pid=$pid log_age=${age}s dcpu=${dcpu}ticks dio=${dio}B threads=$nthreads futex=$nfutex $state"
    if (( ${suspect[$name]:-0} >= SUSPECT_CHECKS )); then
      out=$LOGDIR/hang_${name}_$(date +%Y%m%d_%H%M%S).txt
      { echo "job $name pid $pid log $lg age ${age}s threads $nthreads futex $nfutex"
        ps -o pid,ppid,stat,pcpu,etime,wchan:32,args -p "$(echo $pids | tr ' ' ,)"
        for w in $workers; do
          echo "=== gdb $w $(tr '\0' ' ' < /proc/$w/cmdline)"
          timeout 300 sudo -n gdb -p $w -batch -ex 'set pagination off' -ex 'thread apply all bt' 2>&1 ||
            timeout 120 sudo -n eu-stack -p $w 2>&1
        done
        echo "=== log tail"; tail -50 "$lg"
      } > $out 2>&1
      log "$name HANG captured to $out; killing tree $pids"
      touch $JOBS/$name.killed
      kill -TERM $pids 2>/dev/null; sleep 10; kill -KILL $pids 2>/dev/null
      suspect[$name]=0
    fi
  done
  sleep $INTERVAL
done
