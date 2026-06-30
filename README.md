# Comprehensive-Head-Classification
Comprehensive Head Classification. Presence/absence of sunglasses, presence/absence of a mask, eyes open/closed, mouth open/closed, simple/complex background, face image quality assessment (FIQA).

It is capable of rapidly performing six types of classification and inference in a single inference pass.

<img width="1884" height="1075" alt="image" src="https://github.com/user-attachments/assets/41dfd45d-2f34-4c10-b0a2-f20d72a921f3" />

## Build merged ONNX models

Install dependencies and run the builder with uv:

```bash
uv sync
source .venv/bin/activate
uv run build-chc-onnx --variant s
```

The uv environment uses Python 3.13 from `.python-version`.
Core dependencies are pinned in `pyproject.toml`.

Useful options:

```bash
uv run build-chc-onnx --variant s --verify
uv run build-chc-onnx --variant s --disable-fiqa
uv run build-chc-onnx --variant s --disable-onnxsim
uv run build-chc-onnx --variant s --output /path/to/output.onnx
```

## Benchmark ONNX models

Use `sit4onnx` through uv after building the merged ONNX files:

```bash
uv run sit4onnx -if chc_s.onnx -tlc 100 -oep cpu
uv run sit4onnx -if chc_s_wo_fiqa.onnx -tlc 100 -oep cpu
```

`-tlc` is the inference loop count. `sit4onnx` reports total elapsed time and
average elapsed time per inference.

To benchmark with GPU providers from `onnxruntime-gpu`, change `-oep`:

```bash
uv run sit4onnx -if chc_s.onnx -tlc 1000 -oep cuda
uv run sit4onnx -if chc_s.onnx -tlc 1000 -oep tensorrt
```

Profiling output can be enabled with `-pro`:

```bash
uv run sit4onnx -if chc_s.onnx -tlc 100 -oep cpu -pro
```

For reproducible benchmarks with fixed input tensors, save each input as a
`.npy` file and pass them in graph input order:

```bash
uv run sit4onnx \
-if chc_s.onnx \
-tlc 100 \
-oep cpu \
-ifp head_image_48x48.npy \
-ifp eye_images_24x40.npy \
-ifp mouth_image_30x48.npy \
-ifp head_image_352x352.npy
```

For `chc_s_wo_fiqa.onnx`, omit `head_image_352x352.npy`.

## Browser benchmark app

The Electron benchmark app lives in `benchmark-app/` and runs ONNX Runtime Web
or LiteRT.js inside the Chromium renderer. JavaScript dependencies are pinned
exactly in `package.json` and locked by `pnpm-lock.yaml`.

```bash
cd benchmark-app
pnpm install --frozen-lockfile
pnpm dev
```

During dev, root-level `chc_*.onnx`, `chc_*_float32.tflite`, ONNX Runtime Web
assets, and LiteRT.js Wasm assets are served directly by the Vite asset plugin.
During `vite build`, the same plugin copies models into
`benchmark-app/dist/models/`, ONNX Runtime Web assets into
`benchmark-app/dist/ort/`, and LiteRT.js assets into
`benchmark-app/dist/litert/wasm/`. These copied assets are generated files and
are not tracked by git.

<img width="900" alt="image" src="https://github.com/user-attachments/assets/c271b3b0-56a3-4fb8-be85-d7de9ed845bb" />

During dev, root-level `chc_*.onnx` files and ONNX Runtime Web assets are served
directly by the Vite asset plugin. During `vite build`, the same plugin copies
models into `benchmark-app/dist/models/` and ONNX Runtime Web assets into
`benchmark-app/dist/ort/`. These copied assets are generated files and are not
tracked by git.

Build and smoke-test the app:

```bash
cd benchmark-app
pnpm build
pnpm benchmark:wasm
pnpm benchmark:litert:wasm
pnpm benchmark:webgpu
pnpm benchmark:litert:webgpu
```

The WASM smoke scripts should run anywhere Electron can start. WebGPU smoke
scripts require a Chromium WebGPU-capable environment and may report that the
backend is unsupported when no GPU adapter is available.
