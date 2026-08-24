#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="SPFresh"
WORKLOAD="delete"
# eval_tempfiles contract: consume the tree delete_prepare.sh produced, in
# place; spfresh mutates WORK_DIR/index. Re-running requires re-running
# delete_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
# DELETE_POINTS, if set, overrides the per-percent count without shifting
# BASE_POINTS (mirrors delete_prepare.sh).
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"

# SPFresh delete tuning parameters.
SPFRESH_DELETE_THREADS="${SPFRESH_DELETE_THREADS:-$(nproc)}"
SPFRESH_REASSIGN_THREADS="${SPFRESH_REASSIGN_THREADS:-$SPFRESH_DELETE_THREADS}"
SPFRESH_MUTATION_DAYS="1"
SPFRESH_MUTATION_DELETE_QPS="-1"
SPFRESH_MUTATION_SAMPLING="1"
SPFRESH_MUTATION_MERGE_THRESHOLD="10"
SPFRESH_MUTATION_LATENCY_LIMIT="100.0"
SPFRESH_MUTATION_REASSIGN_K="64"
SPFRESH_DISABLE_REASSIGN="false"
SPFRESH_IN_PLACE="true"
SPFRESH_END_VECTOR_NUM=""
SPFRESH_LOAD_ALL_VECTORS="true"

# Separate final query tuning parameters. Query runs through ssdserving over the mutated output index.
SPFRESH_QUERY_THREADS="${SPFRESH_QUERY_THREADS:-$(nproc)}"
SPFRESH_QUERY_INTERNAL_RESULT_NUM="256"
SPFRESH_QUERY_SEARCH_TIMES="1"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${DELETE_POINTS:-}" ]]; then
    [[ "$DELETE_POINTS" =~ ^[0-9]+$ && "$DELETE_POINTS" -gt 0 ]] \
        || error "DELETE_POINTS must be a positive integer (got: $DELETE_POINTS)"
    (( DELETE_POINTS < BASE_POINTS )) \
        || error "DELETE_POINTS ($DELETE_POINTS) must be < BASE_POINTS ($BASE_POINTS)"
    UPDATE_POINTS="$DELETE_POINTS"
fi
FINAL_POINTS=$((BASE_POINTS - UPDATE_POINTS))
SPFRESH_END_VECTOR_NUM="$BASE_POINTS"
[[ -d "$WORK_DIR/index" ]] || error "missing delete index; run delete_prepare.sh first"
[[ -f "$WORK_DIR/data/final.bin" ]] || error "missing delete final.bin; run delete_prepare.sh first"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$GT_FILE" ]] || error "missing delete groundtruth; run delete_prepare.sh first"
spfresh_apply_delete "$WORK_DIR" "$GT_FILE" "$WORK_DIR/data/final.bin"
