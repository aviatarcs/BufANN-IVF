#!/usr/bin/env bash
# Placeholder profile: 1M-vector subset of SIFT-1B.

# Identity and benchmark split.
DATASET="sift1m"
DATA_TYPE="float"
DIM=128
METRIC="l2"
NPTS_FULL=1000000
NPTS_BASE=900000
UPDATE_POINTS=100000
DISKANN_R="${DISKANN_R:-64}"

# TODO: Configure the dataset and DiskANN-index directories.
__dataset_root="${SIFT1M_ROOT:-/path/to/datasets/sift1m}"
__index_root="${SIFT1M_INDEX_DIR:-/path/to/indexes/sift1m}"

# TODO: Configure the source vector files.
DATA_FVECS="${DATA_FVECS:-$__dataset_root/sift_base.fvecs}"
QUERY_FVECS="${QUERY_FVECS:-$__dataset_root/sift_query.fvecs}"

# Generated artifact paths; no configuration needed.
DATA_BIN="${DATA_BIN:-$__dataset_root/sift_base.bin}"
QUERY_BIN="${QUERY_BIN:-$__dataset_root/sift_query.bin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__dataset_root/sift1m_1000000_gt100.bin}"
DISKANN_INDEX_0P9B="${DISKANN_INDEX_0P9B:-$__index_root/sift1m_0.9M_R64_L100_B32_M8_noPQ}"
