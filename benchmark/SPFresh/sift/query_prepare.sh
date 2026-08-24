#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="SPFresh"
WORKLOAD="query"
# eval_tempfiles contract: prepare materializes the full consumable here;
# query.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"
ensure_dataset
BASE_POINTS="$(canonical_base_points)"
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
BASE_BIN="$WORK_DIR/data/base.bin"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
cached_write_bin_prefix "$WORK_DIR/data/full.bin" "$BASE_BIN" "$BASE_POINTS"
spfresh_provision_index_into "$WORK_DIR/index" "$BASE_BIN"
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$BASE_POINTS"
note "Prepared $BASELINE query work tree at $WORK_DIR"
