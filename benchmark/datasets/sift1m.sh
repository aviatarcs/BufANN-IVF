#!/usr/bin/env bash
# Placeholder profile: 1M-vector subset of SIFT-1B.
# Replace the paths below before running any benchmark.

# Identity and benchmark split.
DATASET="sift1m"
DATA_TYPE="float"
DIM=128
METRIC="l2"
NPTS_FULL=1000000
NPTS_BASE=900000
UPDATE_POINTS=100000
CANONICAL_R="${CANONICAL_R:-64}"

# Dataset source/output directory and canonical-index output directory.
__dataset_root="${SIFT1M_ROOT:-/path/to/datasets/sift1m}"
__index_root="${SIFT1M_INDEX_DIR:-/path/to/indexes/sift1m}"

# Input artifacts.
DATA_FVECS="${DATA_FVECS:-$__dataset_root/sift_base.fvecs}"
QUERY_FVECS="${QUERY_FVECS:-$__dataset_root/sift_query.fvecs}"

# Generated artifact paths.
DATA_BIN="${DATA_BIN:-$__dataset_root/sift_base.bin}"
QUERY_BIN="${QUERY_BIN:-$__dataset_root/sift_query.bin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__dataset_root/sift1m_1000000_gt100.bin}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__index_root/sift1m_0.9M_R64_L100_B32_M8_noPQ}"
