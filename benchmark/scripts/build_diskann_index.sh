#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/utils.sh"

if (( $# != 1 )); then
    echo "Usage: bash benchmark/scripts/build_diskann_index.sh <dataset>" >&2
    echo "This script builds the DiskANN index (\"Canonical Graph\")." >&2
    echo "Input:  <dataset> -- see benchmark/datasets/{dataset}.sh to configure paths, including:" >&2
    echo "        base vector (.bin) should contain \"NPTS_FULL\" vectors" >&2
    echo "Output: DiskANN graph index (_disk.index)"
    exit 2
fi

DATASET="$1"
source "$REPO_DIR/benchmark/datasets/_load.sh"

BIN="$REPO_DIR/baselines/DiskANN/build/tests"
PREFIX="$CANONICAL_INDEX_0P9B"
BUILD_L="${BUILD_L:-100}"
BUILD_RAM_GB="${BUILD_RAM_GB:-200}"
BUILD_THREADS="${BUILD_THREADS:-$(nproc)}"

for helper in partition_only build_shard merge_shards_only finalize_index; do
    [[ -x "$BIN/$helper" ]] || error "missing DiskANN helper: $BIN/$helper; run bash benchmark/scripts/build.sh first"
done

ensure_dataset
[[ "$(bin_count "$DATA_BIN")" == "$NPTS_FULL" ]] || error "base-vector count does not match NPTS_FULL=$NPTS_FULL: $DATA_BIN"

if [[ -f "${PREFIX}_disk.index" ]]; then
    note "Canonical graph already exists: ${PREFIX}_disk.index"
    exit 0
fi

mkdir -p "$(dirname "$PREFIX")"
note "Partitioning the first $NPTS_BASE vectors of $DATA_BIN"
"$BIN/partition_only" "$DATA_TYPE" "$DATA_BIN" "$PREFIX" "$CANONICAL_R" "$BUILD_RAM_GB" "$NPTS_BASE"

shopt -s nullglob
shard_id_files=("${PREFIX}_mem.index_tempFiles_subshard-"[0-9]*_ids_uint32.bin)
shopt -u nullglob
NSHARDS="${#shard_id_files[@]}"
(( NSHARDS > 0 )) || error "partition_only produced no shards for $PREFIX"

for ((p = 0; p < NSHARDS; p++)); do
    note "Building shard $((p + 1))/$NSHARDS"
    "$BIN/build_shard" "$DATA_TYPE" "${PREFIX}_mem.index_tempFiles_subshard-${p}.bin" "${PREFIX}_mem.index_tempFiles_subshard-${p}_mem.index" "$CANONICAL_R" "$BUILD_L" "$BUILD_THREADS" "$METRIC"
done

note "Merging $NSHARDS shards"
"$BIN/merge_shards_only" "${PREFIX}_mem.index_tempFiles_subshard-" "$NSHARDS" "$CANONICAL_R" "${PREFIX}_mem.index" "${PREFIX}_disk.index_medoids.bin"

note "Finalizing the canonical graph"
"$BIN/finalize_index" "$DATA_TYPE" "${PREFIX}_mem.index" "$DATA_BIN" "${PREFIX}_disk.index" "$NPTS_BASE"

[[ -f "${PREFIX}_disk.index" ]] || error "finalize_index did not create ${PREFIX}_disk.index"
note "Canonical index complete: $PREFIX"
