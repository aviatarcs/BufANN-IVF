#!/usr/bin/env bash
# Prepares a fresh CloudLab node from the bufann-ivf-bench profile for one
# isolated benchmark run (docs/cloudlab_bench.md): checks the hardware type,
# checks out REF, stages the datasets on local disk and verifies their
# checksums, builds and runs ctest, then waits for the node to go idle and
# writes $BENCH_ROOT/results/node.json recording what the run ran on.
# Exits non-zero, before any benchmark starts, if any of that fails.
#
#   scripts/cloudlab_node_prep.sh <dataset_dir> <ref>
#
# <dataset_dir> holds the files listed in its SHA256SUMS (the layout is in
# docs/cloudlab_bench.md). Benchmarks read it as $BENCH_ROOT/data: a copy if
# it is network-backed (a remote dataset), so benchmarks never read over the
# network, otherwise a link. Safe to rerun.
#
# Environment: BENCH_ROOT (default /tmpdata, the profile's local scratch),
# EXPECT_NODETYPE (default sm110p), IDLE_LOAD (default 1.0),
# IDLE_WAIT_S (default 600).
set -euo pipefail

[[ $# -eq 2 ]] || { sed -n '2,18p' "$0" >&2; exit 2; }
DATASET=$(realpath "$1") REF=$2
ROOT=${BENCH_ROOT:-/tmpdata}
EXPECT=${EXPECT_NODETYPE:-sm110p}
IDLE_LOAD=${IDLE_LOAD:-1.0}
IDLE_WAIT_S=${IDLE_WAIT_S:-600}
REPO_URL=https://github.com/aviatarcs/BufANN-IVF.git
SRC=$ROOT/BufANN-IVF
RESULTS=$ROOT/results
die() { echo "PREP FAILED: $*" >&2; exit 1; }
step() { echo "== $*" >&2; }

step "hardware"
nodetype=$(cat /var/emulab/boot/nodetype 2>/dev/null || echo unknown)
[[ $nodetype == "$EXPECT" ]] || die "node type is $nodetype, expected $EXPECT; results would not be comparable"
[[ -f /lib/x86_64-linux-gnu/libmkl_rt.so ]] || die "no /lib/x86_64-linux-gnu/libmkl_rt.so; this node is not on the bench image"
[[ -d $ROOT && -w $ROOT ]] || die "$ROOT is not a writable directory"
case $(findmnt -n -o FSTYPE -T "$ROOT") in
    nfs*|cifs|fuse*) die "$ROOT is on a network filesystem" ;;
esac

step "source at $REF"
[[ -d $SRC/.git ]] || git clone -q "$REPO_URL" "$SRC"
git -C "$SRC" fetch -q origin
# origin/ first: a branch name must mean the branch as just fetched, not the
# clone's local copy of it from an earlier run.
git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "origin/$REF" 2>/dev/null ||
    git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$REF" ||
    die "cannot check out $REF"
[[ -z $(git -C "$SRC" status --porcelain --untracked-files=no) ]] || die "$SRC has local changes"
commit=$(git -C "$SRC" rev-parse HEAD)

step "datasets"
[[ -f $DATASET/SHA256SUMS ]] || die "no SHA256SUMS in $DATASET"
# $ROOT/data is where benchmarks read: a copy of a network-backed dataset
# (remote datasets are iSCSI), otherwise a link to the local one, which at
# 1B scale is too large to copy for nothing.
DATA=$ROOT/data
case $(findmnt -n -o FSTYPE -T "$DATASET") in nfs*|cifs|fuse*) network=1 ;; *) network= ;; esac
lsblk -s -n -o TRAN "$(findmnt -n -o SOURCE -T "$DATASET")" 2>/dev/null | grep -q iscsi && network=1
if [[ $network ]]; then
    step "copying $DATASET to $DATA (network-backed)"
    [[ -L $DATA ]] && rm "$DATA"
    mkdir -p "$DATA"
    rsync -a "$DATASET/" "$DATA/"
else
    [[ -e $DATA && ! -L $DATA ]] && die "$DATA exists and is not a link to the local dataset"
    ln -sfn "$DATASET" "$DATA"
fi
(cd "$DATA" && sha256sum --quiet -c SHA256SUMS) || die "checksum mismatch in $DATA"

step "build and ctest"
# A build root on the scratch disk, not the image's /var/tmp one, so a stale
# build tree captured in the image cannot leak into the run.
export BUFANN_BUILD_ROOT=$ROOT/build-root MKL_LIB_DIR=/lib/x86_64-linux-gnu
"$SRC/scripts/dev_env.sh" test || die "build or ctest failed"

step "waiting for the node to go idle (1-min load < $IDLE_LOAD)"
# Load, not process names: the command line that launched this script (and
# will launch the benchmark after it) names the benchmark too. The build
# leaves the 1-minute load average high for a few minutes.
deadline=$((SECONDS + IDLE_WAIT_S))
while load=$(cut -d' ' -f1 /proc/loadavg); ! awk -v l="$load" -v m="$IDLE_LOAD" 'BEGIN { exit !(l < m) }'; do
    ((SECONDS < deadline)) || die "node not idle after ${IDLE_WAIT_S}s: load $load
$(ps -eo pcpu,pid,user,args --sort=-pcpu | head -6)"
    sleep 15
done

step "manifest"
mkdir -p "$RESULTS"
python3 - "$RESULTS/node.json" <<EOF
import json, os, platform, subprocess, sys, time
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def cat(p):
    try: return open(p).read().strip()
    except OSError: return None
json.dump({
    "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": platform.node(),
    "experiment": cat("/var/emulab/boot/nickname"),
    "nodetype": "$nodetype",
    "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
    "nproc": os.cpu_count(),
    "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    "kernel": platform.release(),
    "governor": cat("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    "no_turbo": cat("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    "thp": cat("/sys/kernel/mm/transparent_hugepage/enabled"),
    "ref": "$REF",
    "commit": "$commit",
    "data_dir": "$DATA",
    "data_sha256sums": cat("$DATA/SHA256SUMS"),
    "load_at_ready": "$load",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "READY commit=$commit data=$DATA build=$SRC/build/tests results=$RESULTS"
