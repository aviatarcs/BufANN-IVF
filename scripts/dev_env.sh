#!/usr/bin/env bash
# Reproducible build environment for this host, which has no system MKL,
# iomp5 or tcmalloc. The home directory is NFS with a small quota, so the
# shim and the build tree live on local disk under BUFANN_BUILD_ROOT
# (default /var/tmp/bufann-ivf-$USER): one shared build-env, and one build
# directory per checkout (keyed by the checkout's directory name, so git
# worktrees do not share object files). Both are symlinked into the checkout
# as .build-env and build. Idempotent; safe to run at the start of every
# agent cycle.
#
#   scripts/dev_env.sh            configure + build the IVF-PQ test targets
#   scripts/dev_env.sh test       ... then run ctest
#   scripts/dev_env.sh scale      ... then the 1M-vector scale run
#
# Override MKL_LIB_DIR if the oneMKL shared libraries live elsewhere.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${BUFANN_BUILD_ROOT:-/var/tmp/bufann-ivf-$USER}"
CHECKOUT="$(basename "$REPO")"
mkdir -p "$ROOT/build-env" "$ROOT/build-$CHECKOUT"
for link in .build-env:build-env build:build-$CHECKOUT; do
    name="${link%%:*}" target="$ROOT/${link##*:}"
    if [[ -e "$REPO/$name" && ! -L "$REPO/$name" ]]; then
        echo "ERROR: $REPO/$name exists and is not a symlink; remove it (it belongs on local disk)" >&2
        exit 1
    fi
    ln -sfn "$target" "$REPO/$name"
done
ENV="$ROOT/build-env"
BUILD="$ROOT/build-$CHECKOUT"
MKL_VERSION="2025.2.0"
MKL_LIB_DIR="${MKL_LIB_DIR:-/lusr/opt/julia-depot/1.12/artifacts/27edf95310a71d47422663c3aea849f56efb1360/lib}"

[[ -f "$MKL_LIB_DIR/libmkl_rt.so" ]] || {
    echo "ERROR: no libmkl_rt.so in $MKL_LIB_DIR; set MKL_LIB_DIR to a oneMKL $MKL_VERSION lib dir" >&2
    exit 1
}

mkdir -p "$ENV/shimlib" "$ENV/dl"
if [[ ! -f "$ENV/include/mkl.h" ]]; then
    echo "Fetching MKL headers $MKL_VERSION"
    pip3 download --no-deps -q "mkl-include==$MKL_VERSION" -d "$ENV/dl"
    unzip -q -o "$ENV"/dl/mkl_include-*.whl -d "$ENV/dl/mklinc"
    mkdir -p "$ENV/include"
    cp -r "$ENV"/dl/mklinc/mkl_include-*.data/data/include/. "$ENV/include/"
fi
if [[ ! -s "$ENV/shimlib/libiomp5.so" ]]; then
    echo "Fetching libiomp5 $MKL_VERSION"
    pip3 download --no-deps -q "intel-openmp==$MKL_VERSION" -d "$ENV/dl"
    unzip -q -o "$ENV"/dl/intel_openmp-*.whl -d "$ENV/dl/iomp"
    cp "$(find "$ENV/dl/iomp" -name libiomp5.so | head -1)" "$ENV/shimlib/"  # copy, not symlink
fi
if [[ ! -s "$ENV/shimlib/libtcmalloc.so" ]]; then
    # Tests only need the symbol to link; glibc malloc is used at runtime.
    # Not suitable for performance benchmarking.
    echo "" | g++ -shared -x c++ - -o "$ENV/shimlib/libtcmalloc.so"
fi

LD="-L$MKL_LIB_DIR -L$ENV/shimlib -Wl,-rpath,$MKL_LIB_DIR -Wl,-rpath,$ENV/shimlib"
cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-isystem $ENV/include" \
    -DCMAKE_EXE_LINKER_FLAGS="$LD" -DCMAKE_SHARED_LINKER_FLAGS="$LD" > "$BUILD.cmake.log" 2>&1 || {
    cat "$BUILD.cmake.log" >&2
    exit 1
}
rm -f "$BUILD.cmake.log"
make -C "$BUILD" -j"$(nproc)" \
    ivf_pq_raw_vector_heap_test ivf_pq_build_test ivf_pq_scale_test ivf_pq_recall_test \
    ivf_pq_search_test ivf_pq_api_test ivf_pq_build_index 2>&1 | \
    grep -E "error|warning: unused|Built target" || true
[[ -x "$BUILD/tests/ivf_pq_build_test" ]] || { echo "ERROR: build failed" >&2; exit 1; }

case "${1:-}" in
    test)  (cd "$BUILD" && ctest --output-on-failure) ;;
    scale) (cd "$BUILD" && ./tests/ivf_pq_scale_test 1000000 128 1024 32) ;;
    "")    ;;
    *)     echo "unknown mode: $1" >&2; exit 2 ;;
esac
