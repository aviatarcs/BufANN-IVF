#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/utils.sh"

note "Building BufANN"
configure_cmake_repo "$REPO_DIR" bufann_driver

note "Building DiskANN canonical-index helpers"
configure_cmake_repo "$REPO_DIR/baselines/DiskANN" \
    build_pq_standalone build_pq_pivot_standalone build_pq_codes_standalone \
    partition_only build_shard merge_shards_only finalize_index

note "Build complete"
