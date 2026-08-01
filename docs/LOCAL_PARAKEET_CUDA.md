# Local Parakeet CUDA Setup

This guide covers the local source-build setup for:

- NVIDIA GPU acceleration via Parakeet + ONNX Runtime
- Direct type-mode output that does not touch the clipboard
- Running the binary directly from `./target/release/voxtype`

Use this guide if you are working from a checkout of this repo and want a repeatable setup you can hand to someone else.

## What This Solves

A local `parakeet-cuda` source build needs two things beyond the normal Voxtype setup:

1. A Parakeet model downloaded and configured.
2. `LD_LIBRARY_PATH` set so Linux can find the ONNX provider shared libraries sitting next to the release build artifacts.

Without that library path, you can see errors like:

```text
Failed to load library libonnxruntime_providers_shared.so
```

The helper scripts added for this path are:

- `./scripts/setup-local-parakeet-cuda.sh`
- `./scripts/run-local-parakeet-cuda.sh`

## Requirements

### Base Requirements

- Rust + Cargo
- ALSA development headers
- `clang`
- `cmake`
- `pkg-config`
- NVIDIA GPU drivers
- CUDA 12 runtime visible to the dynamic linker

You can check the CUDA runtime with:

```bash
ldconfig -p | grep libcudart.so.12
```

If multiple CUDA versions are installed, the helper scripts prefer a CUDA 12
install because the bundled ONNX Runtime CUDA provider expects CUDA 12.x. Set
`VOXTYPE_CUDA_HOME` to override auto-detection:

```bash
VOXTYPE_CUDA_HOME=/usr/local/cuda-12.8 ./scripts/setup-local-parakeet-cuda.sh
VOXTYPE_CUDA_HOME=/usr/local/cuda-12.8 ./scripts/run-local-parakeet-cuda.sh
```

### Output Requirements

Voxtype chooses different tools depending on your display server.

**Wayland**

- `wtype`, `eitype`, `dotool`, or `ydotool` for direct type mode
- `wl-copy` only if you explicitly use paste or clipboard mode

**X11**

- `dotool` or `ydotool` for direct type mode
- `xclip` plus `xdotool` or `ydotool` only if you explicitly use paste or clipboard mode

If you rely on built-in hotkey capture instead of compositor keybindings, your user also needs to be in the `input` group:

```bash
sudo usermod -aG input $USER
```

Then log out and back in.

## Fast Path

From the repo root:

```bash
./scripts/setup-local-parakeet-cuda.sh
./scripts/run-local-parakeet-cuda.sh
```

The run script does not force paste mode. It launches the binary with your config and any CLI args you pass through. If direct `ydotool` typing is too fast for a target app:

```bash
./scripts/run-local-parakeet-cuda.sh --type-delay 12
```

If the CUDA provider is troublesome on a newer GPU, use CPU Parakeet instead:

```bash
CARGO_TARGET_DIR=target/parakeet-cpu cargo build --release --features parakeet
./scripts/run-parakeet-cpu.sh
```

## Manual Setup

### 1. Build The CUDA-Enabled Parakeet Binary

```bash
cargo build --release --features parakeet-cuda
```

This produces `./target/release/voxtype` plus the ONNX Runtime provider libraries used by the CUDA execution provider.

### 2. Create A Config File If You Do Not Have One Yet

```bash
mkdir -p ~/.config/voxtype
cp config/default.toml ~/.config/voxtype/config.toml
```

### 3. Download And Configure A Parakeet Model

Recommended smaller model:

```bash
./target/release/voxtype setup --download --model parakeet-tdt-0.6b-v3-int8
```

Larger model:

```bash
./target/release/voxtype setup --download --model parakeet-tdt-0.6b-v3
```

If `~/.config/voxtype/config.toml` already exists, that command updates it to:

- `engine = "parakeet"`
- `[parakeet].model = "<selected model>"`

For TDT models, set `model_type = "tdt"` to avoid the auto-detect warning:

```toml
engine = "parakeet"

[parakeet]
model = "parakeet-tdt-0.6b-v3-int8"
model_type = "tdt"
```

### 4. Output Settings

Direct type mode avoids consuming the clipboard:

```toml
[output]
mode = "type"
type_delay_ms = 0  # backend default pacing; try 12-20 if text is garbled
```

Paste mode copies transcription text to the clipboard and sends a paste shortcut. Use it only when that clipboard tradeoff is acceptable.

For normal desktop apps:

```toml
[output]
mode = "paste"
paste_keys = "ctrl+v"
```

For terminals or apps that expect `Ctrl+Shift+V`:

```toml
[output]
mode = "paste"
paste_keys = "ctrl+shift+v"
```

### 5. Run The Daemon

For a direct source-build launch from the repo root, export the release artifact directories on `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH="$(pwd)/target/release:$(pwd)/target/release/deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./target/release/voxtype
```

If you need the exact provider cache directory added too, use the helper script:

```bash
./scripts/run-local-parakeet-cuda.sh
```

The helper run script handles `LD_LIBRARY_PATH` automatically.
It also prepends the detected CUDA 12 library directory when available, so a
system-wide `/usr/local/cuda` symlink pointing at CUDA 13 will not shadow the
CUDA 12 runtime.

## What The Helper Scripts Do

### `setup-local-parakeet-cuda.sh`

- Verifies the repo-local build prerequisites that matter for this path
- Prefers CUDA 12 via `VOXTYPE_CUDA_HOME`, `CUDA_HOME`, or `/usr/local/cuda-12*`
- Builds `./target/release/voxtype` with `--features parakeet-cuda`
- Creates `~/.config/voxtype/config.toml` from `config/default.toml` if it does not exist yet
- Downloads the selected Parakeet model if needed
- Updates config to use Parakeet
- Writes `model_type = "tdt"` for TDT Parakeet models
- Optionally writes `paste_keys` and `restore_clipboard`, but does not force paste mode

### `run-local-parakeet-cuda.sh`

- Prepends `target/release` and `target/release/deps` to `LD_LIBRARY_PATH`
- Also adds the resolved ONNX provider cache directory when available
- Prepends the detected CUDA 12 library directory when available
- Launches `./target/release/voxtype` with the CLI args you provide
- Does not force paste mode or read `VOXTYPE_PASTE_KEYS`

## Script Usage

### Setup Script

```bash
./scripts/setup-local-parakeet-cuda.sh
./scripts/setup-local-parakeet-cuda.sh --model parakeet-tdt-0.6b-v3
./scripts/setup-local-parakeet-cuda.sh --paste-keys ctrl+shift+v --restore-clipboard
```

### Run Script

```bash
./scripts/run-local-parakeet-cuda.sh
./scripts/run-local-parakeet-cuda.sh --type-delay 12
./scripts/run-local-parakeet-cuda.sh --paste --paste-keys ctrl+shift+v --restore-clipboard
```

### CPU Parakeet Fallback

```bash
CARGO_TARGET_DIR=target/parakeet-cpu cargo build --release --features parakeet
./scripts/run-parakeet-cpu.sh
```

## Troubleshooting

### `libonnxruntime_providers_shared.so` Not Found

Run the helper wrapper:

```bash
./scripts/run-local-parakeet-cuda.sh
```

Or export the release artifact directory yourself:

```bash
export LD_LIBRARY_PATH="$(pwd)/target/release:$(pwd)/target/release/deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### CUDA Runtime Detected But CUDA Execution Provider Still Fails

Check:

- `ldconfig -p | grep libcudart.so.12`
- `ldconfig -p | grep libcublas.so.12`
- `ldconfig -p | grep libcudnn.so`

The current ONNX CUDA path expects a CUDA 12 runtime.

### Direct Typing Is Garbled

On X11, Voxtype may fall back to `ydotool`. If an application cannot keep up with direct typing, keep type mode and add a small delay:

```toml
[output]
mode = "type"
type_delay_ms = 12
```

Or pass it for one run:

```bash
./scripts/run-local-parakeet-cuda.sh --type-delay 12
```

### Paste Mode Falls Back To Clipboard-Only

Check the detected output chain:

```bash
./target/release/voxtype --paste config
```

On X11, install `xclip` plus `xdotool` or `ydotool`.

On Wayland, install `wl-copy` plus `wtype` or `ydotool`.

### Built-In Hotkey Does Not Trigger

If you are not using compositor keybindings, make sure your user is in the `input` group and has logged out and back in.

## Recommended Shareable Commands

For people working from a local clone of this repo:

```bash
./scripts/setup-local-parakeet-cuda.sh
./scripts/run-local-parakeet-cuda.sh
```

For applications that need slower direct typing:

```bash
./scripts/run-local-parakeet-cuda.sh --type-delay 12
```
