#!/usr/bin/env python3
"""Utility helpers for SPFresh SIFT benchmark scripts."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def tie_aware_recall(truth_path: Path, result_path: Path, recall_at: int) -> float:
    """Calculate Recall@K with the same tie-aware GT expansion used elsewhere.

    When the K-th ground-truth distance ties with later entries, all tied IDs
    are treated as valid hits, matching DiskANN/Greator/PipeANN/InplaceANN.
    """
    if not result_path.exists():
        raise SystemExit(f"ERROR: search result file not found: {result_path}")

    with truth_path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise SystemExit(f"ERROR: invalid truth file header: {truth_path}")
        truth_rows, truth_k_file = struct.unpack("<ii", header)
        truth_k = min(recall_at, truth_k_file)
        truth_ids = []
        truth_dists = []
        for _ in range(truth_rows):
            row = f.read(4 * truth_k_file)
            if len(row) != 4 * truth_k_file:
                raise SystemExit(f"ERROR: truncated truth IDs in {truth_path}")
            truth_ids.append(struct.unpack(f"<{truth_k_file}i", row))
        for _ in range(truth_rows):
            row = f.read(4 * truth_k_file)
            if len(row) != 4 * truth_k_file:
                raise SystemExit(f"ERROR: truncated truth distances in {truth_path}")
            truth_dists.append(struct.unpack(f"<{truth_k_file}f", row))

    with result_path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise SystemExit(f"ERROR: invalid search result header: {result_path}")
        result_rows, result_k_file = struct.unpack("<ii", header)
        result_k = min(recall_at, result_k_file)
        if result_rows != truth_rows:
            raise SystemExit(
                f"ERROR: result query count ({result_rows}) != truth query count ({truth_rows})"
            )

        hits = 0
        for i in range(result_rows):
            row_hits = set()
            for j in range(result_k_file):
                item = f.read(8)
                if len(item) != 8:
                    raise SystemExit(f"ERROR: truncated search results in {result_path}")
                vid, _dist = struct.unpack("<if", item)
                if j < result_k and vid >= 0:
                    row_hits.add(vid)

            tie_breaker = truth_k
            if truth_k > 0:
                kth_dist = truth_dists[i][truth_k - 1]
                tie_breaker = truth_k - 1
                while tie_breaker < truth_k_file and truth_dists[i][tie_breaker] == kth_dist:
                    tie_breaker += 1

            valid_truth_ids = set(truth_ids[i][:tie_breaker])
            hits += len(valid_truth_ids.intersection(row_hits))

    denom = result_rows * truth_k
    recall = hits / denom if denom else 0.0
    print(f"TieAwareRecall{truth_k}@{result_k}: {recall:.6f}")
    print(f"Recall{truth_k}@{result_k}: {recall:.6f}")
    return recall


def main() -> None:
    parser = argparse.ArgumentParser(description="SPFresh SIFT script utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recall_parser = subparsers.add_parser(
        "tie-aware-recall",
        help="Calculate tie-aware Recall@K from SPFresh result and truth files.",
    )
    recall_parser.add_argument("--truth", required=True, type=Path)
    recall_parser.add_argument("--result", required=True, type=Path)
    recall_parser.add_argument("--recall-at", required=True, type=int)

    args = parser.parse_args()
    if args.command == "tie-aware-recall":
        tie_aware_recall(args.truth, args.result, args.recall_at)


if __name__ == "__main__":
    main()
