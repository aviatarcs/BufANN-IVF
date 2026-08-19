#!/usr/bin/env bash
# Load and validate benchmark/datasets/<DATASET>.sh.
# Source after setting REPO_DIR and DATASET, before system-specific common.sh.

: "${REPO_DIR:?REPO_DIR must be set before sourcing benchmark/datasets/_load.sh}"
: "${DATASET:?DATASET must be set before sourcing benchmark/datasets/_load.sh}"

__bench_datasets_dir="${BENCH_DATASETS_DIR:-$REPO_DIR/benchmark/datasets}"
__dataset_file="$__bench_datasets_dir/$DATASET.sh"
__requested_dataset="$DATASET"

if [[ ! -f "$__dataset_file" ]]; then
    __avail="$(cd "$__bench_datasets_dir" 2>/dev/null && \
        find . -maxdepth 1 -type f -name '*.sh' ! -name '_*' -printf '%f\n' 2>/dev/null | \
        sed 's/\.sh$//' | sort | tr '\n' ' ')"
    echo "ERROR: unknown dataset '$DATASET' (no such file: $__dataset_file)" >&2
    echo "       available datasets: ${__avail:-<none>}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$__dataset_file"

[[ "$DATASET" == "$__requested_dataset" ]] || {
    echo "ERROR: dataset profile mismatch: requested '$__requested_dataset', loaded '$DATASET'" >&2
    exit 1
}

__required_dataset_vars=(
    DATASET DATA_TYPE DIM METRIC
    NPTS_FULL NPTS_BASE UPDATE_POINTS
    DATA_BIN QUERY_BIN GT_BIN_KCACHED
    CANONICAL_INDEX_0P9B CANONICAL_R
)
for __var in "${__required_dataset_vars[@]}"; do
    [[ -n "${!__var:-}" ]] || {
        echo "ERROR: dataset '$DATASET' does not define $__var" >&2
        exit 1
    }
done

for __var in DIM NPTS_FULL NPTS_BASE UPDATE_POINTS CANONICAL_R; do
    [[ "${!__var}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: dataset '$DATASET' has invalid $__var='${!__var}'" >&2
        exit 1
    }
done

(( NPTS_BASE < NPTS_FULL )) || {
    echo "ERROR: dataset '$DATASET' requires NPTS_BASE < NPTS_FULL" >&2
    exit 1
}
(( NPTS_BASE + UPDATE_POINTS == NPTS_FULL )) || {
    echo "ERROR: dataset '$DATASET' requires NPTS_BASE + UPDATE_POINTS == NPTS_FULL" >&2
    exit 1
}

RECALL_AT="${RECALL_AT:-10}"
R="${R:-$CANONICAL_R}"

unset __required_dataset_vars __requested_dataset __dataset_file
unset __bench_datasets_dir __dataset_root __index_root __var
