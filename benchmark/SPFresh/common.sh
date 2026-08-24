#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/utils.sh"

BASELINE_REPO="$REPO_DIR/baselines/SPFresh"

case "$DATA_TYPE" in
    float) SPFRESH_DATA_TYPE="Float" ;;
    uint8) SPFRESH_DATA_TYPE="UInt8" ;;
    int8)  SPFRESH_DATA_TYPE="Int8" ;;
    *) error "SPFresh does not support DATA_TYPE=$DATA_TYPE" ;;
esac
SPFRESH_DIST_CALC_METHOD="L2"
SPFRESH_INDEX_ALGO_TYPE="BKT"

SPFRESH_BUILD_THREADS="${SPFRESH_BUILD_THREADS:-$(nproc)}"
SPFRESH_SELECT_HEAD_TREE_NUMBER="1"
SPFRESH_SELECT_HEAD_BKT_KMEANS_K="32"
SPFRESH_SELECT_HEAD_BKT_LEAF_SIZE="8"
SPFRESH_SELECT_HEAD_SAMPLES="1000"
SPFRESH_SELECT_HEAD_THRESHOLD="12"
SPFRESH_SELECT_HEAD_SPLIT_FACTOR="6"
SPFRESH_SELECT_HEAD_SPLIT_THRESHOLD="25"
SPFRESH_SELECT_HEAD_RATIO="0.12"
SPFRESH_BUILD_HEAD_NEIGHBORHOOD_SIZE="32"
SPFRESH_BUILD_HEAD_TPT_NUMBER="32"
SPFRESH_BUILD_HEAD_TPT_LEAF_SIZE="2000"
SPFRESH_BUILD_HEAD_MAX_CHECK="16324"
SPFRESH_BUILD_HEAD_MAX_CHECK_FOR_REFINE_GRAPH="16324"
SPFRESH_BUILD_HEAD_REFINE_ITERATIONS="3"
SPFRESH_BUILD_SSD_INTERNAL_RESULT_NUM="128"
SPFRESH_BUILD_SSD_REPLICA_COUNT="8"
SPFRESH_BUILD_SSD_POSTING_PAGE_LIMIT="3"
SPFRESH_BUILD_SSD_MAX_CHECK="16324"

choose_spfresh_compiler() {
    if [[ -n "${CXX:-}" ]]; then
        return 0
    fi
    if [[ -x /lusr/bin/g++-11 && -x /lusr/bin/gcc-11 ]]; then
        export CXX=/lusr/bin/g++-11 CC=/lusr/bin/gcc-11
        return 0
    fi
    choose_compiler
}

configure_spfresh_repo() {
    local repo selected_cc selected_cxx rocksdb_dir
    repo="$1"
    choose_spfresh_compiler
    setup_build_env
    selected_cc="${CC:-$(command -v gcc)}"
    selected_cxx="${CXX:-$(command -v g++)}"
    rocksdb_dir="${SPFRESH_ROCKSDB_DIR:-${RocksDB_DIR:-$BENCH_DEPS_PREFIX/lib/cmake/rocksdb}}"
    cmake -S "$repo" -B "$repo/build" \
        -DCMAKE_BUILD_TYPE:STRING="$BENCHMARK_CMAKE_BUILD_TYPE" \
        -DCMAKE_C_COMPILER="$selected_cc" \
        -DCMAKE_CXX_COMPILER="$selected_cxx" \
        -DCMAKE_PREFIX_PATH="$BENCH_DEPS_PREFIX" \
        -DRocksDB_DIR="$rocksdb_dir"
    cmake --build "$repo/build" --config "$BENCHMARK_CMAKE_BUILD_TYPE" --target ssdserving spfresh usefultool -j"$(nproc)"
}

find_spfresh_binary() {
    local repo name candidate
    repo="$1"
    name="$2"
    for candidate in "$repo/build/$BENCHMARK_CMAKE_BUILD_TYPE/$name" "$repo/build/$name" "$repo/$BENCHMARK_CMAKE_BUILD_TYPE/$name"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    error "missing SPFresh binary: $name"
}

# DeletedIDs.bin uses a 12-byte header (deleted, rows, cols), not the 8-byte
# (npts, dim) layout that bin_header/bin_count expect.
spfresh_deletedids_row_count() {
    local path="$1"
    [[ -f "$path" ]] || {
        echo 0
        return 0
    }
    python3 - "$path" <<'PYEOF'
import struct, sys

path = sys.argv[1]
with open(path, "rb") as f:
    header = f.read(12)
if len(header) != 12:
    raise SystemExit("0")
_deleted, rows, cols = struct.unpack("<iii", header)
if cols != 1 or rows < 0:
    raise SystemExit("0")
print(rows)
PYEOF
}

write_spfresh_indexloader() {
    local build_ini head_loader out_loader
    build_ini="$1"
    head_loader="$2"
    out_loader="$3"
    python3 - "$build_ini" "$head_loader" "$out_loader" <<'PYEOF'
import sys
from pathlib import Path

build_ini = Path(sys.argv[1])
head_loader = Path(sys.argv[2])
out_loader = Path(sys.argv[3])

def extract_sections(path):
    sections = {}
    current = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped
            sections[current] = [line]
        elif current is not None:
            sections[current].append(line)
    return sections

build_sections = extract_sections(build_ini)
head_sections = extract_sections(head_loader)

value_type = "Float"
for line in build_sections.get("[Base]", []):
    if line.startswith("ValueType="):
        value_type = line.split("=", 1)[1]
        break

parts = [
    "[Index]",
    "IndexAlgoType=SPANN",
    f"ValueType={value_type}",
]

for name in ("[Base]", "[SelectHead]"):
    block = build_sections.get(name)
    if block:
        parts.append("")
        parts.extend(block)

build_head = build_sections.get("[BuildHead]", ["[BuildHead]"])
merged_build_head = []
seen_keys = set()
for line in build_head:
    merged_build_head.append(line)
    if "=" in line:
        seen_keys.add(line.split("=", 1)[0])

for line in head_sections.get("[Index]", []):
    if line.startswith("["):
        continue
    if "=" not in line:
        merged_build_head.append(line)
        continue
    key, value = line.split("=", 1)
    if key in {"IndexAlgoType", "ValueType"}:
        continue
    if key in seen_keys:
        for i, existing in enumerate(merged_build_head):
            if existing.startswith(f"{key}="):
                merged_build_head[i] = f"{key}={value}"
                break
    else:
        merged_build_head.append(f"{key}={value}")
        seen_keys.add(key)

parts.append("")
parts.extend(merged_build_head)

build_ssd = build_sections.get("[BuildSSDIndex]")
if build_ssd:
    parts.append("")
    parts.extend(build_ssd)

out_loader.write_text("\n".join(parts).rstrip() + "\n")
PYEOF
}

spfresh_build_index_into() {
    local index_dir data_bin npts ini_file tmp_dir deleted_ids ssd_info kv_path ssd_log ssdserving existing_npts
    index_dir="$1"
    data_bin="$2"
    npts="$(bin_count "$data_bin")"
    ini_file="$index_dir/build.ini"
    tmp_dir="$index_dir/tmp"
    deleted_ids="$index_dir/DeletedIDs.bin"
    ssd_info="$index_dir/SsdInfoFile"
    kv_path="$index_dir/KVDatabase"
    ssd_log="$index_dir/ssd_build.log"
    mkdir -p "$index_dir" "$tmp_dir"
    if [[ -f "$index_dir/indexloader.ini" ]]; then
        existing_npts=0
        if [[ -f "$deleted_ids" ]]; then
            existing_npts="$(spfresh_deletedids_row_count "$deleted_ids" || echo 0)"
        fi
        if [[ "$existing_npts" != "$npts" ]]; then
            note "Existing SPFresh index npts ($existing_npts) mismatches data npts ($npts); rebuilding $index_dir"
            rm -rf "$index_dir/head_index" "$kv_path" "$tmp_dir"
            rm -f "$deleted_ids" "$ssd_info" "$index_dir/indexloader.ini" "$index_dir/build.log" "$ssd_log"
        fi
    fi
    if [[ -f "$index_dir/indexloader.ini" ]]; then
        if ! grep -q '^IndexAlgoType=SPANN$' "$index_dir/indexloader.ini"; then
            if [[ -f "$ini_file" && -f "$index_dir/head_index/indexloader.ini" ]]; then
                write_spfresh_indexloader "$ini_file" "$index_dir/head_index/indexloader.ini" "$index_dir/indexloader.ini"
            fi
        fi
        note "Reusing SPFresh index $index_dir"
        return 0
    fi
    configure_spfresh_repo "$BASELINE_REPO"
    rm -f "$deleted_ids" "$ssd_info" "$index_dir/indexloader.ini" "$index_dir/build.log" "$ssd_log"
    cat > "$ini_file" <<EOF_INI
[Base]
ValueType=$SPFRESH_DATA_TYPE
DistCalcMethod=$SPFRESH_DIST_CALC_METHOD
IndexAlgoType=$SPFRESH_INDEX_ALGO_TYPE
Dim=$DIM
VectorPath=$data_bin
VectorType=DEFAULT
VectorSize=$npts
QueryPath=
QueryType=DEFAULT
WarmupPath=
WarmupType=DEFAULT
TruthPath=
TruthType=DEFAULT
GenerateTruth=false
IndexDirectory=$index_dir
HeadIndexFolder=head_index
DeletedIDs=$deleted_ids

[SelectHead]
isExecute=true
TreeNumber=$SPFRESH_SELECT_HEAD_TREE_NUMBER
BKTKmeansK=$SPFRESH_SELECT_HEAD_BKT_KMEANS_K
BKTLeafSize=$SPFRESH_SELECT_HEAD_BKT_LEAF_SIZE
SamplesNumber=$SPFRESH_SELECT_HEAD_SAMPLES
NumberOfThreads=$SPFRESH_BUILD_THREADS
SelectThreshold=$SPFRESH_SELECT_HEAD_THRESHOLD
SplitFactor=$SPFRESH_SELECT_HEAD_SPLIT_FACTOR
SplitThreshold=$SPFRESH_SELECT_HEAD_SPLIT_THRESHOLD
Ratio=$SPFRESH_SELECT_HEAD_RATIO

[BuildHead]
isExecute=true
NeighborhoodSize=$SPFRESH_BUILD_HEAD_NEIGHBORHOOD_SIZE
TPTNumber=$SPFRESH_BUILD_HEAD_TPT_NUMBER
TPTLeafSize=$SPFRESH_BUILD_HEAD_TPT_LEAF_SIZE
MaxCheck=$SPFRESH_BUILD_HEAD_MAX_CHECK
MaxCheckForRefineGraph=$SPFRESH_BUILD_HEAD_MAX_CHECK_FOR_REFINE_GRAPH
RefineIterations=$SPFRESH_BUILD_HEAD_REFINE_ITERATIONS
NumberOfThreads=$SPFRESH_BUILD_THREADS

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=$SPFRESH_BUILD_SSD_INTERNAL_RESULT_NUM
ReplicaCount=$SPFRESH_BUILD_SSD_REPLICA_COUNT
PostingPageLimit=$SPFRESH_BUILD_SSD_POSTING_PAGE_LIMIT
NumberOfThreads=$SPFRESH_BUILD_THREADS
MaxCheck=$SPFRESH_BUILD_SSD_MAX_CHECK
TmpDir=$tmp_dir
UseKV=true
SsdInfoFile=$ssd_info
SSDIndex=SPTAGFullList.bin
LogFile=$ssd_log
ExcludeHead=true
KVPath=$kv_path

[SearchSSDIndex]
isExecute=false
EOF_INI
    ssdserving="$(find_spfresh_binary "$BASELINE_REPO" ssdserving)"
    "$ssdserving" "$ini_file" 2>&1 | tee "$index_dir/build.log"
    write_spfresh_indexloader "$ini_file" "$index_dir/head_index/indexloader.ini" "$index_dir/indexloader.ini"
    [[ -f "$index_dir/indexloader.ini" ]] || error "SPFresh did not produce $index_dir/indexloader.ini"
}

# Drop-in for spfresh_build_index_into under the eval_tempfiles contract.
# SPFresh is native: it builds its SPTAG/SPANN index from scratch (slow, no
# canonical to fastcopy from), so the built index family is cached in
# eval_cached keyed by dataset + npts and restored on the next prepare.
# Called only from *_prepare.sh. Keeps spfresh_build_index_into unchanged.
spfresh_provision_index_into() {
    local work_index_dir build_bin npts cache
    work_index_dir="$1"
    build_bin="$2"
    npts="${3:-$(bin_count "$build_bin")}"
    cache="$EVAL_CACHED/$DATASET/SPFresh/index_npts${npts}"
    if eval_cache_try_restore "$cache" "$work_index_dir"; then
        note "Reusing cached SPFresh built index (npts=$npts) -> $work_index_dir"
    else
        spfresh_build_index_into "$work_index_dir" "$build_bin"
        eval_cache_store "$work_index_dir" "$cache"
    fi
}

spfresh_query_index() {
    local index_dir data_bin gt_file output_dir npts ini_file result_file ssdserving
    index_dir="$1"
    data_bin="$2"
    gt_file="$3"
    output_dir="$4"
    npts="$(bin_count "$data_bin")"
    ini_file="$output_dir/query.ini"
    result_file="$output_dir/search_result.bin"
    mkdir -p "$output_dir"
    configure_spfresh_repo "$BASELINE_REPO"
    cat > "$ini_file" <<EOF_INI
[Base]
ValueType=$SPFRESH_DATA_TYPE
DistCalcMethod=$SPFRESH_DIST_CALC_METHOD
IndexAlgoType=$SPFRESH_INDEX_ALGO_TYPE
Dim=$DIM
VectorPath=$data_bin
VectorType=DEFAULT
VectorSize=$npts
QueryPath=$QUERY_BIN
QueryType=DEFAULT
TruthPath=$gt_file
TruthType=DEFAULT
GenerateTruth=false
IndexDirectory=$index_dir
HeadIndexFolder=head_index
DeletedIDs=$index_dir/DeletedIDs.bin

[SelectHead]
isExecute=false

[BuildHead]
isExecute=false

[BuildSSDIndex]
isExecute=false
UseKV=true
SsdInfoFile=$index_dir/SsdInfoFile
ExcludeHead=true
KVPath=$index_dir/KVDatabase

[SearchSSDIndex]
isExecute=true
ResultNum=$RECALL_AT
SearchInternalResultNum=$SPFRESH_QUERY_INTERNAL_RESULT_NUM
SearchThreadNum=$SPFRESH_QUERY_THREADS
SearchTimes=$SPFRESH_QUERY_SEARCH_TIMES
SearchResult=$result_file
Update=false
UseKV=true
SsdInfoFile=$index_dir/SsdInfoFile
ExcludeHead=true
MaxCheck=${SPFRESH_QUERY_MAX_CHECK:-16324}
KVPath=$index_dir/KVDatabase
EOF_INI
    ssdserving="$(find_spfresh_binary "$BASELINE_REPO" ssdserving)"
    "$ssdserving" "$ini_file" 2>&1 | tee "$output_dir/run.log"
    python3 "$BENCHMARK_DIR/SPFresh/utils.py" tie-aware-recall \
        --truth "$gt_file" \
        --result "$result_file" \
        --recall-at "$RECALL_AT" | tee -a "$output_dir/run.log"
}

write_spfresh_insert_trace() {
    local trace_file
    trace_file="$1"
    python3 - "$trace_file" "$BASE_POINTS" "$UPDATE_POINTS" <<'PYEOF'
import array
import struct
import sys
from pathlib import Path
path = Path(sys.argv[1])
base = int(sys.argv[2])
count = int(sys.argv[3])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('wb') as f:
    f.write(struct.pack('<i', count))
    if count:
        f.seek(count * 4 - 1, 1)
        f.write(b'\0')
    array.array('i', range(base, base + count)).tofile(f)
PYEOF
}

write_spfresh_delete_trace() {
    local trace_file
    trace_file="$1"
    python3 - "$trace_file" "$BASE_POINTS" "$UPDATE_POINTS" <<'PYEOF'
import array
import struct
import sys
from pathlib import Path
path = Path(sys.argv[1])
base = int(sys.argv[2])
count = int(sys.argv[3])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('wb') as f:
    f.write(struct.pack('<i', count))
    array.array('i', range(base - count, base)).tofile(f)
    if count:
        f.seek(count * 4 - 1, 1)
        f.write(b'\0')
PYEOF
}

write_spfresh_update_trace_rounds() {
    local trace_prefix base_points insert_points delete_points insert_cap delete_cap
    trace_prefix="$1"
    base_points="$2"
    insert_points="$3"
    delete_points="$4"
    insert_cap="$5"
    delete_cap="$6"
    python3 - "$trace_prefix" "$base_points" "$insert_points" "$delete_points" "$insert_cap" "$delete_cap" <<'PYEOF'
import array
import struct
import sys
from pathlib import Path

trace_prefix = sys.argv[1]
base_points = int(sys.argv[2])
insert_points = int(sys.argv[3])
delete_points = int(sys.argv[4])
insert_cap = int(sys.argv[5])
delete_cap = int(sys.argv[6])

insert_rounds = insert_points // insert_cap
delete_rounds = delete_points // delete_cap
if insert_rounds != delete_rounds:
    raise SystemExit("SPFresh mixed update requires equal insert/delete round count")
if insert_cap != delete_cap:
    raise SystemExit(
        "SPFresh mixed update trace format requires INSERT_CAP == DELETE_CAP; "
        f"got INSERT_CAP={insert_cap}, DELETE_CAP={delete_cap}"
    )

delete_start = base_points - delete_points
insert_start = base_points
for day in range(insert_rounds):
    path = Path(f"{trace_prefix}{day}")
    path.parent.mkdir(parents=True, exist_ok=True)
    delete_ids = array.array('i', range(delete_start + day * delete_cap,
                                        delete_start + (day + 1) * delete_cap))
    insert_ids = array.array('i', range(insert_start + day * insert_cap,
                                        insert_start + (day + 1) * insert_cap))
    with path.open("wb") as f:
        f.write(struct.pack('<i', insert_cap))
        delete_ids.tofile(f)
        insert_ids.tofile(f)
PYEOF
}

patch_spfresh_ini() {
    local ini_file
    ini_file="$1"
    shift
    python3 - "$ini_file" "$@" <<'PYEOF'
from pathlib import Path
import sys
path = Path(sys.argv[1])
updates = dict(arg.split('=', 1) for arg in sys.argv[2:])
lines = path.read_text(errors='replace').splitlines()
seen = set()
out = []
for line in lines:
    stripped = line.strip()
    if '=' in stripped and not stripped.startswith(('[', '#', ';')):
        key = stripped.split('=', 1)[0]
        if key in updates:
            out.append(f'{key}={updates[key]}')
            seen.add(key)
            continue
    out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f'{key}={value}')
path.write_text('\n'.join(out) + '\n')
PYEOF
}

patch_spfresh_mutation_ini() {
    local output_dir gt_file final_bin trace_prefix truth_prefix loader npts
    output_dir="$1"
    gt_file="$2"
    final_bin="$3"
    shift 3
    trace_prefix="$output_dir/index/update_trace_"
    truth_prefix="$output_dir/index/truth_"
    loader="$output_dir/index/indexloader.ini"
    npts="$(bin_count "$final_bin")"
    cp -f "$gt_file" "${truth_prefix}0"
    mkdir -p "$output_dir/index/tmp"
    patch_spfresh_ini "$loader" \
        "IndexDirectory=$output_dir/index" \
        "VectorPath=$final_bin" \
        "VectorSize=$npts" \
        "QueryPath=$QUERY_BIN" \
        "TruthPath=$gt_file" \
        "TmpDir=$output_dir/index/tmp" \
        "DeletedIDs=$output_dir/index/DeletedIDs.bin" \
        "BuildSsdIndex=false" \
        "ExcludeHead=true" \
        "UseKV=true" \
        "SsdInfoFile=$output_dir/index/SsdInfoFile" \
        "ResultNum=$RECALL_AT" \
        "SearchInternalResultNum=${SPFRESH_MUTATION_SEARCH_INTERNAL_RESULT_NUM:-256}" \
        "SearchPostingPageLimit=${SPFRESH_MUTATION_SEARCH_POSTING_PAGE_LIMIT:-4}" \
        "SearchThreadNum=${SPFRESH_MUTATION_SEARCH_THREADS:-1}" \
        "SearchTimes=${SPFRESH_MUTATION_SEARCH_TIMES:-1}" \
        "Update=true" \
        "SteadyState=true" \
        "CalTruth=true" \
        "LoadAllVectors=$SPFRESH_LOAD_ALL_VECTORS" \
        "OnlySearchFinalBatch=${SPFRESH_MUTATION_ONLY_SEARCH_FINAL_BATCH:-true}" \
        "SearchDuringUpdate=${SPFRESH_MUTATION_SEARCH_DURING_UPDATE:-false}" \
        "TruthFilePrefix=$truth_prefix" \
        "FullVectorPath=$final_bin" \
        "UpdateFilePrefix=$trace_prefix" \
        "Days=$SPFRESH_MUTATION_DAYS" \
        "DeleteQPS=$SPFRESH_MUTATION_DELETE_QPS" \
        "Sampling=$SPFRESH_MUTATION_SAMPLING" \
        "MergeThreshold=$SPFRESH_MUTATION_MERGE_THRESHOLD" \
        "LatencyLimit=$SPFRESH_MUTATION_LATENCY_LIMIT" \
        "ReassignK=$SPFRESH_MUTATION_REASSIGN_K" \
        "$@" \
        "InPlace=$SPFRESH_IN_PLACE" \
        "DisableReassign=$SPFRESH_DISABLE_REASSIGN" \
        "EndVectorNum=${SPFRESH_END_VECTOR_NUM:-$npts}" \
        "KVPath=$output_dir/index/KVDatabase"
}

spfresh_apply_insert() {
    local output_dir gt_file final_bin spfresh_bin
    output_dir="$1"
    gt_file="$2"
    final_bin="$3"
    configure_spfresh_repo "$BASELINE_REPO"
    write_spfresh_insert_trace "$output_dir/index/update_trace_0"
    patch_spfresh_mutation_ini "$output_dir" "$gt_file" "$final_bin" \
        "DeleteThreadNum=0" \
        "InsertThreadNum=$SPFRESH_INSERT_THREADS" \
        "AppendThreadNum=$SPFRESH_APPEND_THREADS" \
        "ReassignThreadNum=$SPFRESH_REASSIGN_THREADS"
    spfresh_bin="$(find_spfresh_binary "$BASELINE_REPO" spfresh)"
    "$spfresh_bin" "$output_dir/index" 2>&1 | tee "$output_dir/run.log"
}

spfresh_apply_delete() {
    local output_dir gt_file final_bin spfresh_bin
    output_dir="$1"
    gt_file="$2"
    final_bin="$3"
    configure_spfresh_repo "$BASELINE_REPO"
    write_spfresh_delete_trace "$output_dir/index/update_trace_0"
    patch_spfresh_mutation_ini "$output_dir" "$gt_file" "$final_bin" \
        "DeleteThreadNum=$SPFRESH_DELETE_THREADS" \
        "InsertThreadNum=0" \
        "AppendThreadNum=0" \
        "ReassignThreadNum=$SPFRESH_REASSIGN_THREADS"
    spfresh_bin="$(find_spfresh_binary "$BASELINE_REPO" spfresh)"
    "$spfresh_bin" "$output_dir/index" 2>&1 | tee "$output_dir/run.log"
}

spfresh_apply_update() {
    local output_dir gt_file full_bin round_count spfresh_bin truth_prefix trace_prefix reassign_threads
    output_dir="$1"
    gt_file="$2"
    full_bin="$3"
    round_count="$4"
    truth_prefix="$output_dir/index/truth_"
    trace_prefix="$output_dir/index/update_trace_"
    reassign_threads="$SPFRESH_REASSIGN_THREADS"
    if [[ "$SPFRESH_DISABLE_REASSIGN" == "true" ]]; then
        reassign_threads="0"
    fi

    configure_spfresh_repo "$BASELINE_REPO"
    write_spfresh_update_trace_rounds "$trace_prefix" "$BASE_POINTS" "$INSERT_POINTS" "$DELETE_POINTS" "$INSERT_CAP" "$DELETE_CAP"
    for ((day = 0; day < round_count; ++day)); do
        cp -f "$output_dir/groundtruth/rounds/round_${day}_gt${RECALL_AT}.bin" "${truth_prefix}${day}"
    done

    patch_spfresh_mutation_ini "$output_dir" "$gt_file" "$full_bin" \
        "DeleteThreadNum=$SPFRESH_DELETE_THREADS" \
        "InsertThreadNum=$SPFRESH_INSERT_THREADS" \
        "AppendThreadNum=$SPFRESH_APPEND_THREADS" \
        "ReassignThreadNum=$reassign_threads" \
        "Days=$round_count"

    spfresh_bin="$(find_spfresh_binary "$BASELINE_REPO" spfresh)"
    "$spfresh_bin" "$output_dir/index" 2>&1 | tee "$output_dir/run.log"
}
