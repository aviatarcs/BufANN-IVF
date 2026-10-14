#!/bin/bash
# Download the first N rows of a big-ann-benchmarks .*bin file (an
# <npts, dim> int32 header, then rows) with parallel HTTP range requests,
# and write them under a header saying N.
#
#   fetch_prefix.sh <url> <rows> <elem_bytes> <out> [jobs]
#
# The 100M datasets (DEEP, SPACEV, Turing) are the first 100M rows of their
# 1B bases, unshuffled (scripts/datasets/README.md). Rerunning resumes:
# finished pieces are kept.
set -euo pipefail
[[ $# -ge 4 ]] || { sed -n '2,10p' "$0" >&2; exit 2; }
URL=$1 ROWS=$2 ELEM=$3 OUT=$4 JOBS=${5:-8}

read -r total dim < <(curl -fsS -r 0-7 "$URL" | od -An -tu4)
((ROWS <= total)) || { echo "$URL has $total rows, asked for $ROWS" >&2; exit 1; }
row_bytes=$((dim * ELEM))
bytes=$((ROWS * row_bytes))
piece=$((1 << 30))
npieces=$(((bytes + piece - 1) / piece))
parts=$OUT.parts
mkdir -p "$parts"

fetch() {  # piece index -> $parts/<i>, complete or absent
    local i=$1 start len f
    start=$((i * piece)); len=$((bytes - start < piece ? bytes - start : piece))
    f=$parts/$i
    [[ -f $f && $(stat -c %s "$f") == "$len" ]] && return 0
    for _ in 1 2 3 4 5; do
        curl -fsS --retry 5 -r "$((8 + start))-$((8 + start + len - 1))" -o "$f.tmp" "$URL" &&
            [[ $(stat -c %s "$f.tmp") == "$len" ]] && { mv "$f.tmp" "$f"; return 0; }
        sleep 5
    done
    echo "piece $i failed" >&2; return 1
}
export -f fetch; export URL bytes piece parts
seq 0 $((npieces - 1)) | xargs -P "$JOBS" -I{} bash -c 'fetch {}'

python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<ii', $ROWS, $dim))" > "$OUT.tmp"
for i in $(seq 0 $((npieces - 1))); do cat "$parts/$i" >> "$OUT.tmp"; done
[[ $(stat -c %s "$OUT.tmp") == $((8 + bytes)) ]] || { echo "size mismatch" >&2; exit 1; }
mv "$OUT.tmp" "$OUT"
rm -r "$parts"
echo "$OUT: $ROWS x $dim"
