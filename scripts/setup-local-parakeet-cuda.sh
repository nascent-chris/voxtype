#!/bin/bash
# Build a local Parakeet CUDA source checkout, download a model, and configure
# the user's Voxtype config for repo-local runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="$PROJECT_ROOT/target/release"
BINARY="$RELEASE_DIR/voxtype"
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/voxtype/config.toml"
MODEL="${VOXTYPE_MODEL:-parakeet-tdt-0.6b-v3-int8}"
FEATURES="${VOXTYPE_FEATURES:-parakeet-cuda}"
PASTE_KEYS=""
RESTORE_CLIPBOARD=false
SKIP_BUILD=false

usage() {
    cat <<'EOF'
Usage: ./scripts/setup-local-parakeet-cuda.sh [options]

Options:
  --model NAME           Parakeet model to configure
                         Default: parakeet-tdt-0.6b-v3-int8
  --features FEATURES    Cargo feature set to build
                         Default: parakeet-cuda
  --paste-keys KEYS      Persist paste_keys in ~/.config/voxtype/config.toml
  --restore-clipboard    Persist restore_clipboard = true in config
  --skip-build           Skip cargo build and reuse an existing release binary
  -h, --help             Show this help

Environment overrides:
  VOXTYPE_MODEL
  VOXTYPE_FEATURES
  VOXTYPE_CUDA_HOME      Path to CUDA 12 install (default: auto-detect)
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

warn() {
    echo "Warning: $*" >&2
}

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

use_cuda12_runtime() {
    if CUDA12_HOME="$(find_cuda12_home)"; then
        CUDA12_LIB_DIR="$(cuda_lib_dir "$CUDA12_HOME")"
        export CUDA_HOME="$CUDA12_HOME"
        export CUDA_PATH="$CUDA12_HOME"
        export PATH="$CUDA12_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA12_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "==> Using CUDA 12 runtime: $CUDA12_HOME"
        return 0
    fi

    warn "CUDA 12 runtime was not found. Parakeet CUDA may fall back to CPU."
    return 1
}

upsert_toml_key() {
    local file="$1"
    local section="$2"
    local key="$3"
    local value="$4"
    local tmp

    tmp="$(mktemp)"
    awk \
        -v target_section="[$section]" \
        -v key="$key" \
        -v value="$value" '
        BEGIN {
            in_target = 0
            section_found = 0
            key_written = 0
        }

        /^\[/ {
            if (in_target && !key_written) {
                print key " = " value
            }

            in_target = ($0 == target_section)
            if (in_target) {
                section_found = 1
                key_written = 0
            }

            print
            next
        }

        {
            if (in_target && $0 ~ "^[[:space:]]*" key "[[:space:]]*=") {
                if (!key_written) {
                    print key " = " value
                    key_written = 1
                }
                next
            }

            print
        }

        END {
            if (in_target && !key_written) {
                print key " = " value
            }

            if (!section_found) {
                if (NR > 0) {
                    print ""
                }
                print target_section
                print key " = " value
            }
        }
        ' "$file" > "$tmp"

    mv "$tmp" "$file"
}

ensure_output_dependencies() {
    if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        if ! command -v wtype >/dev/null 2>&1 && \
           ! command -v eitype >/dev/null 2>&1 && \
           ! command -v dotool >/dev/null 2>&1 && \
           ! command -v ydotool >/dev/null 2>&1; then
            echo "Error: Wayland type mode needs wtype, eitype, dotool, or ydotool." >&2
            exit 1
        fi
    elif [[ -n "${DISPLAY:-}" ]]; then
        if ! command -v dotool >/dev/null 2>&1 && ! command -v ydotool >/dev/null 2>&1; then
            echo "Error: X11 type mode needs dotool or ydotool." >&2
            exit 1
        fi
    else
        warn "Neither DISPLAY nor WAYLAND_DISPLAY is set. Skipping output dependency checks."
    fi
}

ensure_cuda_runtime_visible() {
    use_cuda12_runtime || true
}

ensure_config_exists() {
    mkdir -p "$(dirname "$CONFIG_FILE")"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        cp "$PROJECT_ROOT/config/default.toml" "$CONFIG_FILE"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --paste-keys)
            PASTE_KEYS="$2"
            shift 2
            ;;
        --restore-clipboard)
            RESTORE_CLIPBOARD=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_cmd cargo
ensure_output_dependencies
ensure_cuda_runtime_visible
ensure_config_exists

cd "$PROJECT_ROOT"

if [[ "$SKIP_BUILD" != true ]]; then
    echo "==> Building voxtype with features: $FEATURES"
    cargo build --release --features "$FEATURES"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Error: expected binary not found: $BINARY" >&2
    exit 1
fi

if [[ ! -e "$RELEASE_DIR/libonnxruntime_providers_shared.so" ]]; then
    echo "Error: ONNX Runtime provider library not found in $RELEASE_DIR" >&2
    echo "Rebuild the project with the correct ONNX feature set." >&2
    exit 1
fi

echo "==> Downloading/configuring Parakeet model: $MODEL"
"$BINARY" setup --download --model "$MODEL"

if [[ "$MODEL" == parakeet-tdt-* ]]; then
    upsert_toml_key "$CONFIG_FILE" "parakeet" "model_type" '"tdt"'
fi

if [[ -n "$PASTE_KEYS" ]]; then
    upsert_toml_key "$CONFIG_FILE" "output" "paste_keys" "\"$PASTE_KEYS\""
fi

if [[ "$RESTORE_CLIPBOARD" == true ]]; then
    upsert_toml_key "$CONFIG_FILE" "output" "restore_clipboard" "true"
fi

echo
echo "==> Setup complete"
echo
echo "Run:"
echo "  ./scripts/run-local-parakeet-cuda.sh"
echo
echo "If direct ydotool typing is too fast for an app:"
echo "  ./scripts/run-local-parakeet-cuda.sh --type-delay 12"
echo
echo "Config file:"
echo "  $CONFIG_FILE"
