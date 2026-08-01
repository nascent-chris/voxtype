#!/bin/bash
# Run voxtype with Parakeet engine on CPU, using keyboard typing (not clipboard).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY="${VOXTYPE_BINARY:-$PROJECT_ROOT/target/parakeet-cpu/release/voxtype}"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: voxtype binary not found: $BINARY" >&2
    echo "Build with: CARGO_TARGET_DIR=target/parakeet-cpu cargo build --release --features parakeet" >&2
    exit 1
fi

exec "$BINARY" --engine parakeet "$@"
