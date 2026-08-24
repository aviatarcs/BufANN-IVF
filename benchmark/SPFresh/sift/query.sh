#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="SPFresh"
WORKLOAD="query"
# eval_tempfiles contract: consume the tree query_prepare.sh produced, in
# place. No re-materialization. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# SPFresh query tuning parameters.
SPFRESH_QUERY_THREADS="${SPFRESH_QUERY_THREADS:-$(nproc)}"
SPFRESH_QUERY_INTERNAL_RESULT_NUM="256"
SPFRESH_QUERY_SEARCH_TIMES="1"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
BASE_POINTS="$(canonical_base_points)"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$WORK_DIR/index/indexloader.ini" ]] || error "missing query index; run query_prepare.sh first"
[[ -f "$GT_FILE" ]] || error "missing query groundtruth; run query_prepare.sh first"
spfresh_query_index "$WORK_DIR/index" "$WORK_DIR/data/base.bin" "$GT_FILE" "$WORK_DIR"
