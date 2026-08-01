#!/bin/bash
# Launch a repo-local Parakeet CUDA build with the ONNX provider libraries on
# LD_LIBRARY_PATH. Output behavior comes from config/CLI args; type mode avoids
# touching the clipboard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY="${VOXTYPE_BINARY:-$PROJECT_ROOT/target/release/voxtype}"
RELEASE_DIR="$(dirname "$BINARY")"
PROVIDER_LIB="$RELEASE_DIR/libonnxruntime_providers_shared.so"

find_cuda12_home() {
    local candidate

    for candidate in "${VOXTYPE_CUDA_HOME:-}" "${CUDA_HOME:-}" /usr/local/cuda-12*; do
        [[ -n "$candidate" && -d "$candidate" ]] || continue
        if [[ -e "$candidate/lib64/libcudart.so.12" ]] || \
           [[ -e "$candidate/targets/x86_64-linux/lib/libcudart.so.12" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

cuda_lib_dir() {
    local cuda_home="$1"

    if [[ -d "$cuda_home/lib64" ]]; then
        printf '%s\n' "$cuda_home/lib64"
    elif [[ -d "$cuda_home/targets/x86_64-linux/lib" ]]; then
        printf '%s\n' "$cuda_home/targets/x86_64-linux/lib"
    else
        return 1
    fi
}

if [[ ! -x "$BINARY" ]]; then
    echo "Error: voxtype binary not found: $BINARY" >&2
    echo "Run ./scripts/setup-local-parakeet-cuda.sh first." >&2
    exit 1
fi

if [[ ! -e "$PROVIDER_LIB" ]]; then
    echo "Error: ONNX provider library not found: $PROVIDER_LIB" >&2
    echo "Rebuild with ./scripts/setup-local-parakeet-cuda.sh." >&2
    exit 1
fi

RESOLVED_PROVIDER="$(readlink -f "$PROVIDER_LIB")"
PROVIDER_DIR="$(dirname "$RESOLVED_PROVIDER")"

LD_PATH="$RELEASE_DIR:$RELEASE_DIR/deps:$PROVIDER_DIR"
if CUDA12_HOME="$(find_cuda12_home)"; then
    CUDA12_LIB_DIR="$(cuda_lib_dir "$CUDA12_HOME")"
    export CUDA_HOME="$CUDA12_HOME"
    export CUDA_PATH="$CUDA12_HOME"
    export PATH="$CUDA12_HOME/bin:$PATH"
    LD_PATH="$CUDA12_LIB_DIR:$LD_PATH"
else
    echo "Warning: CUDA 12 runtime not found; ONNX Runtime CUDA may fall back to CPU." >&2
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LD_PATH="$LD_PATH:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$LD_PATH"

CMD=("$BINARY")
CMD+=("$@")

exec "${CMD[@]}"
