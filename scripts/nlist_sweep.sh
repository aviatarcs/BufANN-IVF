#!/usr/bin/env bash
# QPS-recall against nlist on one dataset: builds an IVF-PQ index per nlist
# (skipped if OUT/n<nlist>_ivf_pq_index.bin exists), runs ivf_pq_query_bench
# over NPROBES for each, and prints, per nlist, the highest measured QPS at
# or above each recall target. Bench lines go to OUT/bench_n<nlist>.log and
# build logs to OUT/build_n<nlist>.log. Builds use the default training
# sample (IVF_TRAIN_POINTS_PER_CENTROID points per centroid).
#
#   scripts/nlist_sweep.sh <data_type> <base.bin> <query.bin> <gt.bin> <out_dir> \
#                          <nlist,nlist,...> [nprobe,nprobe,...] [pq_chunks]
#
# e.g. SIFT10M 9M:
#   scripts/nlist_sweep.sh uint8 /tmpdata/ivf_bench/sift10m/base.9M.bin \
#       .../query/data/query.bin .../groundtruth/sift10m_9000000_gt10.bin \
#       /var/tmp/bufann-ivf-$USER/nlist 4096,16384,32768,65536
#
# Each 9M index takes ~1.9 GB of disk. Run scripts/dev_env.sh first.
# GRAPH_BEAMS=2,4 also runs each nprobe through the centroid graph with
# beams of 2 and 4 x nprobe (ivf_pq_query_bench --graph_beams), adding a
# "graph" row per nlist: the best q/s over those beams.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 6 ]] || { sed -n '2,16p' "$0" >&2; exit 2; }
TYPE=$1 BASE=$2 QUERY=$3 GT=$4 OUT=$5 NLISTS=$6
NPROBES=${7:-8,16,32,64,128,256,512}
CHUNKS=${8:-32}
DIM=$(python3 -c "import struct,sys; print(struct.unpack('<ii', open(sys.argv[1],'rb').read(8))[1])" "$BASE")
TESTS="$REPO/build/tests"
[[ -x "$TESTS/ivf_pq_build_index" && -x "$TESTS/ivf_pq_query_bench" ]] || {
    echo "ERROR: build the IVF-PQ targets first (scripts/dev_env.sh)" >&2
    exit 1
}
mkdir -p "$OUT"

for n in ${NLISTS//,/ }; do
    prefix="$OUT/n$n"
    if [[ ! -f "${prefix}_ivf_pq_index.bin" ]]; then
        echo "building nlist $n" >&2
        /usr/bin/time -v "$TESTS/ivf_pq_build_index" --data_type "$TYPE" --data_file "$BASE" \
            --index_prefix "$prefix" --dim "$DIM" --ivf_nlist "$n" --ivf_pq_chunks "$CHUNKS" \
            > "$OUT/build_n$n.log" 2>&1 || { echo "ERROR: build failed, see $OUT/build_n$n.log" >&2; exit 1; }
    fi
    echo "benchmarking nlist $n" >&2
    "$TESTS/ivf_pq_query_bench" --data_type "$TYPE" --index_prefix "$prefix" --query_file "$QUERY" \
        --gt_file "$GT" --nprobes "$NPROBES" ${GRAPH_BEAMS:+--graph_beams "$GRAPH_BEAMS"} > "$OUT/bench_n$n.log" 2>&1 ||
        { echo "ERROR: bench failed, see $OUT/bench_n$n.log" >&2; exit 1; }
done

python3 - "$OUT" ${NLISTS//,/ } <<'EOF'
import json, re, sys
out, nlists = sys.argv[1], sys.argv[2:]
targets = [90.0, 95.0, 98.0, 99.0]
# "single" is one query per thread (a centroid GEMV each), "batched" the
# batched centroid GEMM; they differ once the centroids outgrow the cache.
# "graph" is one query per thread through the centroid graph.
print(f"{'nlist':>6} {'build_s':>8} {'path':>8}  " + "  ".join(f"{f'qps@{t:g} (nprobe)':>16}" for t in targets))
for n in nlists:
    rows = [json.loads(l) for l in open(f"{out}/bench_n{n}.log") if l.startswith("{")]
    exact = [r for r in rows if r.get("centroid_search", "exact") == "exact"]
    graph = [r for r in rows if r.get("centroid_search") == "graph"]
    build = open(f"{out}/build_n{n}.log").read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", build)
    secs = sum(float(p) * 60 ** i for i, p in enumerate(reversed(m.group(1).split(":")))) if m else float("nan")
    for path, qps, rs in (("single", "query_qps", exact), ("batched", "batched_qps", exact),
                          ("graph", "query_qps", graph)):
        if not rs:
            continue
        cells = []
        for t in targets:
            ok = [r for r in rs if r["recall"] >= t]
            best = max(ok, key=lambda r: r[qps]) if ok else None
            cells.append(f"{best[qps]:9.0f} ({best['nprobe']:4d})" if best else f"{'-':>16}")
        print(f"{n:>6} {secs:8.0f} {path:>8}  " + "  ".join(cells))
EOF
