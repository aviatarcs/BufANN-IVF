#!/usr/bin/env bash
# Placeholder profile: 100M-vector subset of SIFT-1B.
# Replace the paths below before running any benchmark.

# Identity and benchmark split.
DATASET="sift100m"
DATA_TYPE="uint8"
DIM=128
METRIC="l2"
NPTS_FULL=100000000
NPTS_BASE=90000000
UPDATE_POINTS=10000000
CANONICAL_R="${CANONICAL_R:-64}"

# Placeholder artifact roots.
__dataset_root="${SIFT100M_ROOT:-/path/to/datasets/sift100m}"
__index_root="${SIFT100M_INDEX_DIR:-/path/to/indexes/sift100m}"

# Required artifacts.
DATA_BIN="${DATA_BIN:-$__dataset_root/base.100M.u8bin}"
QUERY_BIN="${QUERY_BIN:-$__dataset_root/query.10K.u8bin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__dataset_root/sift100m_100000000_gt100.bin}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__index_root/sift100m_90M_R64_L100_B32_M8_noPQ}"
