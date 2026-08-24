#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="SPFresh"
WORKLOAD="insert"
# eval_tempfiles contract: consume the tree insert_prepare.sh produced, in
# place; spfresh mutates WORK_DIR/index. Re-running requires re-running
# insert_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
# INSERT_POINTS, if set, overrides the per-percent count without shifting
# BASE_POINTS (mirrors insert_prepare.sh).
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"

# SPFresh insert tuning parameters.
SPFRESH_INSERT_THREADS="${SPFRESH_INSERT_THREADS:-$(nproc)}"
SPFRESH_APPEND_THREADS="${SPFRESH_APPEND_THREADS:-$SPFRESH_INSERT_THREADS}"
SPFRESH_REASSIGN_THREADS="${SPFRESH_REASSIGN_THREADS:-$SPFRESH_INSERT_THREADS}"
SPFRESH_MUTATION_DAYS="1"
SPFRESH_MUTATION_DELETE_QPS="-1"
SPFRESH_MUTATION_SAMPLING="1"
SPFRESH_MUTATION_MERGE_THRESHOLD="10"
SPFRESH_MUTATION_LATENCY_LIMIT="100.0"
SPFRESH_MUTATION_REASSIGN_K="64"
SPFRESH_DISABLE_REASSIGN="false"
SPFRESH_IN_PLACE="true"
SPFRESH_LOAD_ALL_VECTORS="true"

# Separate final query tuning parameters. Query runs through ssdserving over the mutated output index.
SPFRESH_QUERY_THREADS="${SPFRESH_QUERY_THREADS:-$(nproc)}"
SPFRESH_QUERY_INTERNAL_RESULT_NUM="256"
SPFRESH_QUERY_SEARCH_TIMES="1"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${INSERT_POINTS:-}" ]]; then
    [[ "$INSERT_POINTS" =~ ^[0-9]+$ && "$INSERT_POINTS" -gt 0 ]] \
        || error "INSERT_POINTS must be a positive integer (got: $INSERT_POINTS)"
    UPDATE_POINTS="$INSERT_POINTS"
fi
[[ -d "$WORK_DIR/index" ]] || error "missing insert index; run insert_prepare.sh first"
FINAL_POINTS=$((BASE_POINTS + UPDATE_POINTS))
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$GT_FILE" ]] || error "missing insert groundtruth; run insert_prepare.sh first"
spfresh_apply_insert "$WORK_DIR" "$GT_FILE" "$WORK_DIR/data/full.bin"
