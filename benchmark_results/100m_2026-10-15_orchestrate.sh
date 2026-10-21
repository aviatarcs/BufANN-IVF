#!/bin/bash
# Replaces diskann_builds.sh's loop (2026-10-14 ~20:50) so each dataset's
# BufANN query runs follow its canonical build on a quiet node: DEEP's last
# shards, merge and finalize, then BufANN on DEEP; then build + BufANN for
# SPACEV and Turing. Steps and arguments are diskann_builds.sh's.
set -x
BIN=/tmpdata/vectordb-baselines/DiskANN/build/tests
L=/nvme/bench100m
N=90000000
build() {  # name type ext first_shard (partition only when first_shard == 0 and nothing is there)
  local name=$1 type=$2 ext=$3 first=$4
  local P=/nvme/indexes/$name/${name}_90M_R64_L100_noPQ B=/nvme/100m/$name/base.100M.$ext
  [[ -f ${P}_disk.index ]] && return 0
  mkdir -p $(dirname $P)
  {
    date
    if ! ls ${P}_mem.index_tempFiles_subshard-0_ids_uint32.bin > /dev/null 2>&1; then
      TCMALLOC_RELEASE_RATE=10 $BIN/partition_only $type $B $P 64 90 $N || return 1
    fi
    local NS=$(ls ${P}_mem.index_tempFiles_subshard-[0-9]*_ids_uint32.bin | wc -l)
    for ((p = first; p < NS; p++)); do
      TCMALLOC_RELEASE_RATE=10 $BIN/build_shard $type ${P}_mem.index_tempFiles_subshard-$p.bin \
        ${P}_mem.index_tempFiles_subshard-${p}_mem.index 64 100 32 l2 || return 1
    done
    $BIN/merge_shards_only ${P}_mem.index_tempFiles_subshard- $NS 64 ${P}_mem.index ${P}_disk.index_medoids.bin || return 1
    $BIN/finalize_index $type ${P}_mem.index $B ${P}_disk.index $N || return 1
    date
  } >> $L/diskann_build_$name.log 2>&1
}
# DEEP: shard 1 (of 0-2) is still building under the old loop's pid.
while [[ -e /proc/37993 ]]; do sleep 30; done
[[ -f /nvme/indexes/deep100m/deep100m_90M_R64_L100_noPQ_mem.index_tempFiles_subshard-1_mem.index ]] || { echo "deep shard 1 missing"; exit 1; }
build deep100m float fbin 2; echo "deep100m diskann build exit=$? $(date +%T)"
$L/bufann_one.sh deep100m float fbin 96 > $L/bufann_deep100m.log 2>&1; echo "deep100m bufann exit=$? $(date +%T)"
build spacev100m int8 i8bin 0; echo "spacev100m diskann build exit=$? $(date +%T)"
$L/bufann_one.sh spacev100m int8 i8bin 100 > $L/bufann_spacev100m.log 2>&1; echo "spacev100m bufann exit=$? $(date +%T)"
build turing100m float fbin 0; echo "turing100m diskann build exit=$? $(date +%T)"
$L/bufann_one.sh turing100m float fbin 100 > $L/bufann_turing100m.log 2>&1; echo "turing100m bufann exit=$? $(date +%T)"
echo ALL_DONE
