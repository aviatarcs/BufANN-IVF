#!/usr/bin/env python3
"""Convert dimension-prefixed uint8 .bvecs to a DiskANN-style uint8 .bin."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.input.open("rb") as src:
        raw_dim = src.read(4)
    if len(raw_dim) != 4:
        raise SystemExit(f"ERROR: invalid bvecs file: {args.input}")

    (dim,) = struct.unpack("<i", raw_dim)
    record_size = 4 + dim
    file_size = args.input.stat().st_size
    if dim <= 0 or file_size % record_size:
        raise SystemExit(f"ERROR: invalid bvecs layout: {args.input}")
    npts = file_size // record_size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    try:
        with args.input.open("rb") as src, tmp.open("wb") as dst:
            dst.write(struct.pack("<ii", npts, dim))
            while raw := src.read(record_size * 65536):
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(-1, record_size)
                dims = rows[:, :4].copy().view("<i4").reshape(-1)
                if np.any(dims != dim):
                    raise SystemExit(f"ERROR: inconsistent dimensions in {args.input}")
                dst.write(rows[:, 4:].tobytes())
        os.replace(tmp, args.output)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"Converted {args.input} -> {args.output} ({npts} x {dim}, uint8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
