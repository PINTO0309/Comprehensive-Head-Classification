# Comprehensive-Head-Classification
Comprehensive head classification. Presence/absence of hats, sunglasses, and masks; eyes open/closed; mouth open/closed; background simplicity/complexity; and Face Image Quality Assessment (FIQA).

It is capable of rapidly performing seven types of classification and inference in a single inference pass.

Merged model inputs:

- `head_image_48x48`: `[1, 3, 48, 48]`, head crop used for background, mask, sunglasses, and hat classification
- `eye_images_24x40`: `[2, 3, 24, 40]`, two eye crops used for eye-open classification
- `mouth_image_30x48`: `[1, 3, 30, 48]`, mouth crop used for mouth-open classification
- `head_image_352x352`: `[1, 3, 352, 352]`, head crop used for FIQA in FIQA-enabled models

Input normalization:

- `head_image_48x48`, `eye_images_24x40`, and `mouth_image_30x48`: RGB `float32`, normalized to `0.0..1.0` by dividing pixel values by `255`
- `head_image_352x352`: RGB `float32`, normalized to `0.0..1.0` and then ImageNet-normalized with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

Merged model outputs:

- `prob_bg_plain`: `[1]`, probability that the background is plain/complicated
- `prob_masked`: `[1]`, probability that the person is wearing a mask
- `prob_sunglass`: `[1]`, probability that the person is wearing sunglasses
- `prob_hat`: `[1]`, probability that the person is wearing a hat
- `prob_eye_open`: `[2]`, probability that each eye is open
- `prob_mouth_open`: `[1]`, probability that the mouth is open
- `quality_score`: `[1, 1]`, face image quality score for FIQA-enabled models

<img width="2101" height="1073" alt="image" src="https://github.com/user-attachments/assets/456599bf-d18a-4d8a-b1be-3d5f88e52514" />

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

Download `chc_*.onnx` from [models](https://github.com/PINTO0309/Comprehensive-Head-Classification/releases/tag/models) and place them in the root folder.

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

Download `chc_*.onnx` and `chc_*.tflite`, `yolomit_*.onnx`, `yolomit_*.tflite` from [models](https://github.com/PINTO0309/Comprehensive-Head-Classification/releases/tag/models) and place them in the root folder.

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

<img width="1177" height="1230" alt="image" src="https://github.com/user-attachments/assets/a7ef3c22-3bd0-4abf-9963-cd4e9ca3b57d" />

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

## Cited - Ultra-lightweight classification model series

1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License
8. [UHD: Ultra-lightweight human detection](https://github.com/PINTO0309/UHD) - MIT License
9. [MWC: Mask wearing classifier.](https://github.com/PINTO0309/MWC) - MIT License
10. [SGC: Classification of wearing vs. not wearing sunglasses. 48x48.](https://github.com/PINTO0309/SGC) - MIT License
11. [HHC: Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.](https://github.com/PINTO0309/HHC) - MIT License
