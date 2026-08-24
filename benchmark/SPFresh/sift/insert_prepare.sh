#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="SPFresh"
WORKLOAD="insert"
# eval_tempfiles contract: prepare materializes the full consumable here;
# insert.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
# INSERT_POINTS, if set, overrides the per-percent count *without* shifting
# BASE_POINTS off the canonical (compute_split derives BASE from
# UPDATE_PERCENT, then we re-pin UPDATE_POINTS below).
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
source "$REPO_DIR/benchmark/$BASELINE/common.sh"
ensure_dataset
compute_split
if [[ -n "${INSERT_POINTS:-}" ]]; then
    [[ "$INSERT_POINTS" =~ ^[0-9]+$ && "$INSERT_POINTS" -gt 0 ]] \
        || error "INSERT_POINTS must be a positive integer (got: $INSERT_POINTS)"
    (( INSERT_POINTS < TOTAL_POINTS - BASE_POINTS + 1 )) \
        || error "INSERT_POINTS ($INSERT_POINTS) exceeds available tail rows ($((TOTAL_POINTS - BASE_POINTS)))"
    UPDATE_POINTS="$INSERT_POINTS"
fi
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
BASE_BIN="$WORK_DIR/data/base.bin"
FINAL_POINTS=$((BASE_POINTS + UPDATE_POINTS))
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
cached_write_bin_prefix "$WORK_DIR/data/full.bin" "$BASE_BIN" "$BASE_POINTS"

spfresh_provision_index_into "$WORK_DIR/index" "$BASE_BIN"

# Exact GT for the post-insert prefix.
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$FINAL_POINTS"

note "Prepared $BASELINE insert work tree at $WORK_DIR"
