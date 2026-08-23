#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/utils.sh"

if (( $# != 1 )); then
    echo "Usage: bash benchmark/scripts/build_pq.sh <dataset>" >&2
    echo "Builds PQ pivots on NPTS_FULL vectors and PQ codes for the first NPTS_BASE vectors." >&2
    echo "Set PQ_CHUNKS to override the default of 32 bytes per vector." >&2
    exit 2
fi

DATASET="$1"
source "$REPO_DIR/benchmark/datasets/_load.sh"

BIN="$REPO_DIR/baselines/DiskANN/build/tests"
PREFIX="$DISKANN_INDEX_0P9B"
PQ_CHUNKS="${PQ_CHUNKS:-32}"
PIVOTS="${PREFIX}_pq_pivots.bin"
CODES="${PREFIX}_pq_compressed.bin"

ensure_dataset
[[ -x "$BIN/build_pq_pivot_standalone" && \
   -x "$BIN/build_pq_codes_standalone" ]] || \
    error "PQ helpers are not built; run bash benchmark/scripts/build.sh first"
[[ ! -e "$PIVOTS" && ! -e "$CODES" ]] || \
    error "PQ output already exists under $PREFIX; remove it before rebuilding"

mkdir -p "$(dirname "$PREFIX")"
note "Training $PQ_CHUNKS-chunk PQ pivots on all $NPTS_FULL vectors"
"$BIN/build_pq_pivot_standalone" \
    "$DATA_TYPE" "$DATA_BIN" "$PREFIX" "$PQ_CHUNKS"

note "Encoding the first $NPTS_BASE vectors into PQ codes"
"$BIN/build_pq_codes_standalone" \
    "$DATA_TYPE" "$DATA_BIN" "$PREFIX" "$PQ_CHUNKS" "$NPTS_BASE"

note "PQ build complete: $PREFIX"
