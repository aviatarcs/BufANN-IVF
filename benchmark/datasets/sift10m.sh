#!/usr/bin/env bash
# Placeholder profile: 10M-vector subset of SIFT-1B.
# Students should create the dataset by sampling from SIFT-1B and configuring this script correctly.

# Identity and benchmark split.
DATASET="sift10m"
DATA_TYPE="uint8"
DIM=128
METRIC="l2"
NPTS_FULL=10000000
NPTS_BASE=9000000
UPDATE_POINTS=1000000
DISKANN_R="${DISKANN_R:-64}"

# TODO: Configure the dataset and DiskANN-index directories.
__dataset_root="${SIFT10M_ROOT:-/path/to/datasets/sift10m}"
__index_root="${SIFT10M_INDEX_DIR:-/path/to/indexes/sift10m}"

# TODO: Configure the .bin input vector files.
DATA_BIN="${DATA_BIN:-$__dataset_root/base.10M.bin}"
QUERY_BIN="${QUERY_BIN:-$__dataset_root/query.10K.bin}"

# Alternatively, uncomment these to convert bvecs inputs to the paths above.
# DATA_BVECS="${DATA_BVECS:-$__dataset_root/base.10M.bvecs}"
# QUERY_BVECS="${QUERY_BVECS:-$__dataset_root/query.10K.bvecs}"

# Generated artifact paths; no configuration needed.
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__dataset_root/sift10m_10000000_gt100.bin}"
DISKANN_INDEX_0P9B="${DISKANN_INDEX_0P9B:-$__index_root/sift10m_9M_R64_L100_B32_M8_noPQ}"
