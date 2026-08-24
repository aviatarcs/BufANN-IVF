#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="SPFresh"
WORKLOAD="update"
# eval_tempfiles contract: prepare materializes the full consumable here;
# update.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
UPDATE_BATCH="${UPDATE_BATCH:-}"
if [[ -n "$UPDATE_BATCH" ]]; then
    INSERT_CAP="${INSERT_CAP:-$UPDATE_BATCH}"
    DELETE_CAP="${DELETE_CAP:-$UPDATE_BATCH}"
fi
source "$REPO_DIR/benchmark/$BASELINE/common.sh"
ensure_dataset
compute_update_layout
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"

BASE_BIN="$WORK_DIR/data/base.bin"
INSERT_IDS="$WORK_DIR/ids/insert_ids.bin"
DELETE_IDS="$WORK_DIR/ids/delete_ids.bin"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_update_i${INSERT_POINTS}_d${DELETE_POINTS}_gt${RECALL_AT}.bin"

cached_write_bin_prefix "$WORK_DIR/data/full.bin" "$BASE_BIN" "$BASE_POINTS"
write_counted_ids_range "$INSERT_IDS" "$BASE_POINTS" "$INSERT_POINTS"
write_counted_ids_range "$DELETE_IDS" "$((BASE_POINTS - DELETE_POINTS))" "$DELETE_POINTS"

spfresh_provision_index_into "$WORK_DIR/index" "$BASE_BIN"

resolve_incremental_gt "$GT_FILE" "$RECALL_AT" \
    --inserts "$INSERT_IDS" --deletes "$DELETE_IDS" \
    --base_trunc "$((BASE_POINTS + INSERT_POINTS))"
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" \
    "$UPDATE_ROUNDS" "$BASE_POINTS" "$INSERT_CAP" \
    "$((BASE_POINTS - DELETE_POINTS))" "$DELETE_CAP"

note "Prepared $BASELINE update work tree at $WORK_DIR"
