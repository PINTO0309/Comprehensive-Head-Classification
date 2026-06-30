import './styles.css';

type Backend = 'wasm' | 'webgpu';
type Runtime = 'onnxruntime-web' | 'litert';
type InputMode = 'synthetic_tensor' | 'demo_image';
type TensorLayout = 'nchw' | 'nhwc';

type ModelEntry = {
  name: string;
  path: string;
  bytes: number;
  withFiqa: boolean;
  format: 'onnx' | 'tflite';
  runtime: Runtime;
};

type DetectorEntry = Omit<ModelEntry, 'withFiqa'>;

type DemoImageEntry = {
  name: string;
  path: string;
  bytes: number;
};

type Manifest = {
  generatedAt: string;
  models: ModelEntry[];
  detectors?: DetectorEntry[];
  demoImages?: DemoImageEntry[];
};

type OrtModule = typeof import('onnxruntime-web');
type LiteRtModule = typeof import('@litertjs/core');
type OnnxSession = Awaited<ReturnType<OrtModule['InferenceSession']['create']>>;
type LiteRtCompiledModel = Awaited<ReturnType<LiteRtModule['loadAndCompile']>>;

type BenchmarkResult = {
  mode: InputMode;
  modelName: string;
  detectorName?: string;
  imageCount?: number;
  runtime: Runtime;
  backend: Backend;
  warmup: number;
  runs: number;
  loadMs: number;
  avgMs: number;
  medianMs: number;
  p95Ms: number;
  minMs: number;
  maxMs: number;
  avgYoloMs?: number;
  avgChcMs?: number;
  samplesMs: number[];
  outputs: Array<{ name: string; dims: readonly number[]; type: string }>;
};

type BenchmarkRun = {
  result: BenchmarkResult;
  headCrops?: HeadCropPreview[];
};

type DemoPipelineRun = {
  previews: HeadCropPreview[];
  yoloMs: number;
  chcMs: number;
};

type DetectionBox = {
  classId: number;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type SelectedDetections = {
  head?: DetectionBox;
  eyes: DetectionBox[];
  mouth?: DetectionBox;
};

type LoadedDemoImage = {
  entry: DemoImageEntry;
  image: HTMLImageElement;
  width: number;
  height: number;
};

type HeadCropPreview = {
  imageName: string;
  dataUrl?: string;
  score?: number;
  scores?: Array<{ name: string; value: string }>;
};

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) {
  throw new Error('Missing #app root');
}

app.innerHTML = `
  <main class="shell">
    <header class="header">
      <div>
        <h1>CHC Web Benchmark</h1>
        <p>Browser renderer benchmark for ONNX Runtime Web and LiteRT.js with WASM and WebGPU.</p>
      </div>
      <div class="status" id="status">Loading manifest...</div>
    </header>

    <section class="panel controls">
      <label>
        <span>Input Mode</span>
        <select id="input-mode">
          <option value="synthetic_tensor">Synthetic Tensor</option>
          <option value="demo_image">Demo Image</option>
        </select>
      </label>
      <label>
        <span>Runtime</span>
        <select id="runtime">
          <option value="onnxruntime-web">onnxruntime-web</option>
          <option value="litert">LiteRT.js</option>
        </select>
      </label>
      <label>
        <span>Model</span>
        <select id="model"></select>
      </label>
      <label>
        <span>Backend</span>
        <select id="backend">
          <option value="wasm">WASM</option>
          <option value="webgpu">WebGPU</option>
        </select>
      </label>
      <label>
        <span>Warmup</span>
        <input id="warmup" type="number" min="0" step="1" value="5" />
      </label>
      <label>
        <span>Runs</span>
        <input id="runs" type="number" min="1" step="1" value="50" />
      </label>
      <button id="run" type="button">Run Benchmark</button>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Summary</h2>
        <div id="summary" class="summary muted">No benchmark has run yet.</div>
      </div>
      <div class="panel">
        <h2>Outputs</h2>
        <div id="outputs" class="muted">No output metadata yet.</div>
      </div>
    </section>

    <section class="panel samples-panel">
      <div class="panel-heading">
        <h2 id="samples-title">Inference Time Samples (ms)</h2>
        <button id="samples-toggle" class="secondary-button" type="button" aria-expanded="false" aria-controls="samples">Show</button>
      </div>
      <pre id="samples">[]</pre>
    </section>

    <section class="panel crop-panel" id="crop-panel" hidden>
      <h2>Head Crops</h2>
      <div id="head-crops" class="crop-grid"></div>
    </section>
  </main>
`;

const statusEl = document.querySelector<HTMLDivElement>('#status')!;
const inputModeSelect = document.querySelector<HTMLSelectElement>('#input-mode')!;
const runtimeSelect = document.querySelector<HTMLSelectElement>('#runtime')!;
const modelSelect = document.querySelector<HTMLSelectElement>('#model')!;
const backendSelect = document.querySelector<HTMLSelectElement>('#backend')!;
const warmupInput = document.querySelector<HTMLInputElement>('#warmup')!;
const runsInput = document.querySelector<HTMLInputElement>('#runs')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const summaryEl = document.querySelector<HTMLDivElement>('#summary')!;
const outputsEl = document.querySelector<HTMLDivElement>('#outputs')!;
const samplesTitleEl = document.querySelector<HTMLHeadingElement>('#samples-title')!;
const samplesToggleButton = document.querySelector<HTMLButtonElement>('#samples-toggle')!;
const samplesEl = document.querySelector<HTMLPreElement>('#samples')!;
const cropPanelEl = document.querySelector<HTMLElement>('#crop-panel')!;
const headCropsEl = document.querySelector<HTMLDivElement>('#head-crops')!;

let manifest: Manifest = { generatedAt: '', models: [], detectors: [], demoImages: [] };
let webGpuAvailable = false;
let liteRtModulePromise: Promise<LiteRtModule> | undefined;
let liteRtRuntimePromise: Promise<unknown> | undefined;
let liteRtJspiSupportedPromise: Promise<boolean> | undefined;
let demoImagesPromise: Promise<LoadedDemoImage[]> | undefined;
let demoImageReloadToken = '';
let samplesExpanded = false;
const outputOrder = [
  'prob_bg_plain',
  'prob_masked',
  'prob_sunglass',
  'prob_hat',
  'prob_eye_open',
  'prob_mouth_open',
  'quality_score',
];
const outputAliases: Record<string, string> = {
  prob_background_plain: 'prob_bg_plain',
  prob_sunglasses: 'prob_sunglass',
  prob_wearing_hat: 'prob_hat',
};
const detectorScoreThreshold = 0.35;
const detectorNmsThreshold = 0.45;
const yoloWidth = 640;
const yoloHeight = 480;
const imageNetMean = [0.485, 0.456, 0.406] as const;
const imageNetStd = [0.229, 0.224, 0.225] as const;

function setSamplesExpanded(expanded: boolean) {
  samplesExpanded = expanded;
  samplesEl.hidden = !expanded;
  samplesToggleButton.setAttribute('aria-expanded', String(expanded));
  samplesToggleButton.textContent = expanded ? 'Hide' : 'Show';
}

function seededFloat32Array(length: number, seed: number): Float32Array {
  const data = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < length; index += 1) {
    state = (1664525 * state + 1013904223) >>> 0;
    data[index] = state / 0xffffffff - 0.5;
  }
  return data;
}

function tensorSize(dims: readonly number[]): number {
  return dims.reduce((product, dim) => product * dim, 1);
}

function inputSeed(name: string): number {
  let seed = 2166136261;
  for (const char of name) {
    seed ^= char.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }
  return seed >>> 0;
}

async function loadOrt(backend: Backend): Promise<OrtModule> {
  if (backend === 'webgpu') {
    return (await import('onnxruntime-web/webgpu')) as unknown as OrtModule;
  }
  return await import('onnxruntime-web');
}

async function loadLiteRt(): Promise<LiteRtModule> {
  liteRtModulePromise ??= import('@litertjs/core');
  const liteRt = await liteRtModulePromise;
  liteRtJspiSupportedPromise ??= liteRt.supportsFeature('jspi');
  const jspiSupported = await liteRtJspiSupportedPromise;
  liteRtRuntimePromise ??= liteRt.loadLiteRt('/litert/wasm/', { threads: false, jspi: jspiSupported });
  await liteRtRuntimePromise;
  return liteRt;
}

async function createOnnxSession(model: Pick<ModelEntry, 'path'>, backend: Backend) {
  const ort = await loadOrt(backend);
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = 1;

  if (backend === 'webgpu' && !(await checkWebGpuAvailable())) {
    throw new Error('WebGPU is unsupported in this Chromium runtime or no GPU adapter is available.');
  }

  const loadStart = performance.now();
  let session: Awaited<ReturnType<OrtModule['InferenceSession']['create']>>;
  try {
    const createPromise = ort.InferenceSession.create(model.path, {
      executionProviders: backend === 'webgpu' ? ['webgpu'] : ['wasm'],
    });
    session =
      backend === 'webgpu'
        ? await Promise.race([
            createPromise,
            new Promise<never>((_resolve, reject) => {
              window.setTimeout(
                () => reject(new Error('Timed out while creating a WebGPU session.')),
                10000,
              );
            }),
          ])
        : await createPromise;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`${backend.toUpperCase()} session creation failed: ${message}`);
  }
  return { ort, session, loadMs: performance.now() - loadStart };
}

async function checkWebGpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu;
  if (!gpu) {
    return false;
  }
  if (!gpu.requestAdapter) {
    return true;
  }
  return (await gpu.requestAdapter()) !== null;
}

function makeOnnxFeeds(ort: OrtModule, session: OnnxSession) {
  const feeds: Record<string, InstanceType<OrtModule['Tensor']>> = {};
  for (const [index, input] of session.inputNames.entries()) {
    const metadata = session.inputMetadata[index];
    if (!metadata) {
      throw new Error(`Missing metadata for input ${input}`);
    }
    if (!metadata.isTensor) {
      throw new Error(`Input ${input} is not a tensor`);
    }
    const dims = metadata.shape.map((dim: unknown) => {
      if (typeof dim !== 'number' || dim <= 0) {
        throw new Error(`Input ${input} has non-static dimension ${String(dim)}`);
      }
      return dim;
    });
    feeds[input] = new ort.Tensor('float32', seededFloat32Array(tensorSize(dims), inputSeed(input)), dims);
  }
  return feeds;
}

async function createLiteRtModel(model: Pick<ModelEntry, 'path'>, backend: Backend) {
  if (backend === 'webgpu' && !(await checkWebGpuAvailable())) {
    throw new Error('WebGPU is unsupported in this Chromium runtime or no GPU adapter is available.');
  }

  const loadStart = performance.now();
  const liteRt = await loadLiteRt();
  let compiledModel: Awaited<ReturnType<LiteRtModule['loadAndCompile']>>;
  try {
    const compilePromise = liteRt.loadAndCompile(model.path, {
      accelerator: backend,
      cpuOptions: { numThreads: 1 },
    });
    compiledModel =
      backend === 'webgpu'
        ? await Promise.race([
            compilePromise,
            new Promise<never>((_resolve, reject) => {
              window.setTimeout(
                () => reject(new Error('Timed out while creating a LiteRT.js WebGPU model.')),
                10000,
              );
            }),
          ])
        : await compilePromise;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`LiteRT.js ${backend.toUpperCase()} model creation failed: ${message}`);
  }
  return { liteRt, compiledModel, loadMs: performance.now() - loadStart };
}

function makeLiteRtFeeds(
  liteRt: LiteRtModule,
  compiledModel: LiteRtCompiledModel,
): Record<string, InstanceType<LiteRtModule['Tensor']>> {
  const feeds: Record<string, InstanceType<LiteRtModule['Tensor']>> = {};
  for (const input of compiledModel.getInputDetails()) {
    if (input.dtype !== 'float32') {
      throw new Error(`Input ${input.name} has unsupported dtype ${input.dtype}`);
    }
    const dims = Array.from(input.shape, (dim) => {
      if (dim <= 0) {
        throw new Error(`Input ${input.name} has non-static dimension ${String(dim)}`);
      }
      return dim;
    });
    feeds[input.name] = new liteRt.Tensor(seededFloat32Array(tensorSize(dims), inputSeed(input.name)), dims);
  }
  return feeds;
}

function deleteLiteRtTensors(tensors: Iterable<{ delete(): void }>) {
  for (const tensor of tensors) {
    tensor.delete();
  }
}

function sortOutputMetadata(outputs: Array<{ name: string; dims: readonly number[]; type: string }>) {
  return outputs.sort((a, b) => {
    const aIndex = outputOrderIndex(a.name);
    const bIndex = outputOrderIndex(b.name);
    return (aIndex === -1 ? Number.MAX_SAFE_INTEGER : aIndex) - (bIndex === -1 ? Number.MAX_SAFE_INTEGER : bIndex);
  });
}

function canonicalOutputName(name: string): string {
  return outputAliases[name] ?? name;
}

function outputOrderIndex(name: string): number {
  return outputOrder.indexOf(canonicalOutputName(name));
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1);
  return sorted[index];
}

function makeStats(samplesMs: number[]) {
  const sorted = [...samplesMs].sort((a, b) => a - b);
  const sum = samplesMs.reduce((total, sample) => total + sample, 0);
  return {
    avgMs: sum / samplesMs.length,
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    minMs: sorted[0] ?? 0,
    maxMs: sorted.at(-1) ?? 0,
  };
}

async function runOnnxBenchmark(
  model: ModelEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkRun> {
  const { ort, session, loadMs } = await createOnnxSession(model, backend);
  const feeds = makeOnnxFeeds(ort, session);

  for (let index = 0; index < warmup; index += 1) {
    await session.run(feeds);
  }

  const samplesMs: number[] = [];
  let lastOutputs: Awaited<ReturnType<typeof session.run>> | undefined;
  for (let index = 0; index < runs; index += 1) {
    const start = performance.now();
    lastOutputs = await session.run(feeds);
    samplesMs.push(performance.now() - start);
  }

  const outputs = Object.entries(lastOutputs ?? {}).map(([name, tensor]) => ({
    name,
    dims: tensor.dims,
    type: tensor.type,
  }));

  await session.release();

  return {
    result: {
      mode: 'synthetic_tensor',
      modelName: model.name,
      runtime: 'onnxruntime-web',
      backend,
      warmup,
      runs,
      loadMs,
      ...makeStats(samplesMs),
      samplesMs,
      outputs: sortOutputMetadata(outputs),
    },
  };
}

async function runLiteRtBenchmark(
  model: ModelEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkRun> {
  const { liteRt, compiledModel, loadMs } = await createLiteRtModel(model, backend);
  const feeds = makeLiteRtFeeds(liteRt, compiledModel);

  try {
    for (let index = 0; index < warmup; index += 1) {
      const outputs = await compiledModel.run(feeds);
      deleteLiteRtTensors(Object.values(outputs));
    }

    const samplesMs: number[] = [];
    let outputMetadata: Array<{ name: string; dims: readonly number[]; type: string }> = compiledModel
      .getOutputDetails()
      .map((output) => ({
        name: output.name,
        dims: Array.from(output.shape),
        type: output.dtype,
      }));

    for (let index = 0; index < runs; index += 1) {
      const start = performance.now();
      const outputs = await compiledModel.run(feeds);
      samplesMs.push(performance.now() - start);
      outputMetadata = Object.entries(outputs).map(([name, tensor]) => ({
        name,
        dims: Array.from(tensor.type.layout.dimensions),
        type: tensor.type.dtype,
      }));
      deleteLiteRtTensors(Object.values(outputs));
    }

    return {
      result: {
        mode: 'synthetic_tensor',
        modelName: model.name,
        runtime: 'litert',
        backend,
        warmup,
        runs,
        loadMs,
        ...makeStats(samplesMs),
        samplesMs,
        outputs: sortOutputMetadata(outputMetadata),
      },
    };
  } finally {
    deleteLiteRtTensors(Object.values(feeds));
    compiledModel.delete();
  }
}

function canvas2d(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    throw new Error('Failed to create a 2D canvas context.');
  }
  return [canvas, context];
}

function drawImageData(
  image: HTMLImageElement,
  width: number,
  height: number,
  box?: DetectionBox,
  margin = { top: 0, bottom: 0, left: 0, right: 0 },
): ImageData | undefined {
  const [canvas, context] = canvas2d(width, height);
  if (!box) {
    return undefined;
  }

  const x1 = Math.max(0, Math.floor(box.x1 - margin.left));
  const y1 = Math.max(0, Math.floor(box.y1 - margin.top));
  const x2 = Math.min(image.naturalWidth, Math.ceil(box.x2 + margin.right));
  const y2 = Math.min(image.naturalHeight, Math.ceil(box.y2 + margin.bottom));
  if (x2 <= x1 || y2 <= y1) {
    return undefined;
  }

  context.drawImage(image, x1, y1, x2 - x1, y2 - y1, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

function drawFullImageData(image: HTMLImageElement, width: number, height: number): ImageData {
  const [, context] = canvas2d(width, height);
  context.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

function writeImageDataToTensor(
  target: Float32Array,
  imageData: ImageData | undefined,
  width: number,
  height: number,
  layout: TensorLayout,
  batchIndex: number,
  batchSize: number,
  normalizeImageNet = false,
) {
  if (batchSize <= batchIndex) {
    throw new Error('Invalid tensor batch index.');
  }
  if (!imageData) {
    return;
  }
  const pixels = imageData.data;
  const imageSize = width * height;
  if (layout === 'nchw') {
    const batchOffset = batchIndex * 3 * imageSize;
    for (let index = 0; index < imageSize; index += 1) {
      const pixelOffset = index * 4;
      const red = pixels[pixelOffset] / 255;
      const green = pixels[pixelOffset + 1] / 255;
      const blue = pixels[pixelOffset + 2] / 255;
      target[batchOffset + index] = normalizeImageNet ? (red - imageNetMean[0]) / imageNetStd[0] : red;
      target[batchOffset + imageSize + index] = normalizeImageNet
        ? (green - imageNetMean[1]) / imageNetStd[1]
        : green;
      target[batchOffset + 2 * imageSize + index] = normalizeImageNet
        ? (blue - imageNetMean[2]) / imageNetStd[2]
        : blue;
    }
    return;
  }

  const batchOffset = batchIndex * imageSize * 3;
  for (let index = 0; index < imageSize; index += 1) {
    const pixelOffset = index * 4;
    const targetOffset = batchOffset + index * 3;
    const red = pixels[pixelOffset] / 255;
    const green = pixels[pixelOffset + 1] / 255;
    const blue = pixels[pixelOffset + 2] / 255;
    target[targetOffset] = normalizeImageNet ? (red - imageNetMean[0]) / imageNetStd[0] : red;
    target[targetOffset + 1] = normalizeImageNet ? (green - imageNetMean[1]) / imageNetStd[1] : green;
    target[targetOffset + 2] = normalizeImageNet ? (blue - imageNetMean[2]) / imageNetStd[2] : blue;
  }
}

function makeYoloInput(image: HTMLImageElement, layout: TensorLayout): Float32Array {
  const data = new Float32Array(1 * yoloHeight * yoloWidth * 3);
  writeImageDataToTensor(data, drawFullImageData(image, yoloWidth, yoloHeight), yoloWidth, yoloHeight, layout, 0, 1);
  return data;
}

function makeCropPreview(image: HTMLImageElement, box?: DetectionBox): string | undefined {
  if (!box) {
    return undefined;
  }
  const [canvas, context] = canvas2d(96, 96);
  const x1 = Math.max(0, Math.floor(box.x1));
  const y1 = Math.max(0, Math.floor(box.y1));
  const x2 = Math.min(image.naturalWidth, Math.ceil(box.x2));
  const y2 = Math.min(image.naturalHeight, Math.ceil(box.y2));
  if (x2 <= x1 || y2 <= y1) {
    return undefined;
  }
  context.drawImage(image, x1, y1, x2 - x1, y2 - y1, 0, 0, 96, 96);
  return canvas.toDataURL('image/png');
}

function iou(a: DetectionBox, b: DetectionBox): number {
  const interX1 = Math.max(a.x1, b.x1);
  const interY1 = Math.max(a.y1, b.y1);
  const interX2 = Math.min(a.x2, b.x2);
  const interY2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  if (interW === 0 || interH === 0) {
    return 0;
  }
  const interArea = interW * interH;
  const aArea = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const bArea = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = aArea + bArea - interArea;
  return union > 0 ? interArea / union : 0;
}

function nms(boxes: DetectionBox[], threshold: number): DetectionBox[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: DetectionBox[] = [];
  for (const box of sorted) {
    if (kept.every((keptBox) => iou(box, keptBox) <= threshold)) {
      kept.push(box);
    }
  }
  return kept;
}

function decodeYoloDetections(data: Float32Array, dims: readonly number[], image: LoadedDemoImage): SelectedDetections {
  const channels = dims.at(-2) ?? 32;
  const anchors = dims.at(-1) ?? 6300;
  if (channels < 32) {
    throw new Error(`Unexpected YOLO output channel count: ${channels}`);
  }

  const byClass = new Map<number, DetectionBox[]>();
  for (const classId of [7, 17, 19]) {
    byClass.set(classId, []);
  }

  const scaleX = image.width / yoloWidth;
  const scaleY = image.height / yoloHeight;
  for (let anchor = 0; anchor < anchors; anchor += 1) {
    const cx = data[anchor] * scaleX;
    const cy = data[anchors + anchor] * scaleY;
    const width = data[2 * anchors + anchor] * scaleX;
    const height = data[3 * anchors + anchor] * scaleY;
    for (const classId of [7, 17, 19]) {
      const score = data[(4 + classId) * anchors + anchor];
      if (score < detectorScoreThreshold) {
        continue;
      }
      byClass.get(classId)!.push({
        classId,
        score,
        x1: Math.max(0, cx - width / 2),
        y1: Math.max(0, cy - height / 2),
        x2: Math.min(image.width, cx + width / 2),
        y2: Math.min(image.height, cy + height / 2),
      });
    }
  }

  const head = nms(byClass.get(7) ?? [], detectorNmsThreshold)[0];
  const eyes = nms(byClass.get(17) ?? [], detectorNmsThreshold)
    .slice(0, 2)
    .sort((a, b) => a.x1 - b.x1);
  const mouth = nms(byClass.get(19) ?? [], detectorNmsThreshold)[0];
  return { head, eyes, mouth };
}

function tensorSpecFromOnnxInput(session: OnnxSession, inputName: string) {
  const index = session.inputNames.indexOf(inputName);
  const metadata = session.inputMetadata[index];
  if (!metadata || !metadata.isTensor) {
    throw new Error(`Missing ONNX input metadata for ${inputName}`);
  }
  const dims = metadata.shape.map((dim) => {
    if (typeof dim !== 'number' || dim <= 0) {
      throw new Error(`Input ${inputName} has non-static dimension ${String(dim)}`);
    }
    return dim;
  });
  return { dims, layout: 'nchw' as const, batch: dims[0], width: dims[3], height: dims[2] };
}

function tensorSpecFromLiteRtInput(input: ReturnType<LiteRtCompiledModel['getInputDetails']>[number]) {
  const dims = Array.from(input.shape, (dim) => {
    if (dim <= 0) {
      throw new Error(`Input ${input.name} has non-static dimension ${String(dim)}`);
    }
    return dim;
  });
  return { dims, layout: 'nhwc' as const, batch: dims[0], width: dims[2], height: dims[1] };
}

function fillChcInputTensor(
  image: HTMLImageElement,
  detections: SelectedDetections,
  inputName: string,
  spec: { dims: number[]; layout: TensorLayout; batch: number; width: number; height: number },
): Float32Array {
  const data = new Float32Array(tensorSize(spec.dims));
  if (inputName === 'eye_images_24x40') {
    detections.eyes.slice(0, spec.batch).forEach((box, index) => {
      writeImageDataToTensor(
        data,
        drawImageData(image, spec.width, spec.height, box),
        spec.width,
        spec.height,
        spec.layout,
        index,
        spec.batch,
      );
    });
    return data;
  }

  const box = inputName === 'mouth_image_30x48' ? detections.mouth : detections.head;
  const margin =
    inputName === 'mouth_image_30x48'
      ? { top: 2, bottom: 6, left: 2, right: 2 }
      : { top: 0, bottom: 0, left: 0, right: 0 };
  writeImageDataToTensor(
    data,
    drawImageData(image, spec.width, spec.height, box, margin),
    spec.width,
    spec.height,
    spec.layout,
    0,
    spec.batch,
    inputName === 'head_image_352x352',
  );
  return data;
}

function onnxOutputMetadata(outputs: Awaited<ReturnType<OnnxSession['run']>> | undefined) {
  return sortOutputMetadata(
    Object.entries(outputs ?? {}).map(([name, tensor]) => ({
      name,
      dims: tensor.dims,
      type: tensor.type,
    })),
  );
}

function formatScore(values: readonly number[]): string {
  if (values.length === 0) {
    return '-';
  }
  if (values.length === 1) {
    return values[0].toFixed(6);
  }
  return `[${values.map((value) => value.toFixed(6)).join(', ')}]`;
}

function orderedOutputScores(scoresByName: Map<string, string>, extraScores: Array<{ name: string; value: string }>) {
  const maskedScore = Number(scoresByName.get('prob_masked'));
  if (Number.isFinite(maskedScore) && maskedScore >= 0.5) {
    scoresByName.set('prob_mouth_open', '-');
  }

  const orderedScores = outputOrder
    .filter((name) => scoresByName.has(name) || name === 'prob_hat')
    .map((name) => ({
      name,
      value: scoresByName.get(name) ?? '-',
    }));
  return [...orderedScores, ...extraScores.sort((a, b) => a.name.localeCompare(b.name))];
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function onnxOutputScores(outputs: Awaited<ReturnType<OnnxSession['run']>> | undefined) {
  const scoresByName = new Map<string, string>();
  const extraScores: Array<{ name: string; value: string }> = [];
  for (const [name, tensor] of Object.entries(outputs ?? {})) {
    const values = tensor.data instanceof Float32Array ? Array.from(tensor.data) : [];
    const canonicalName = canonicalOutputName(name);
    if (outputOrder.includes(canonicalName)) {
      scoresByName.set(canonicalName, formatScore(values));
    } else {
      extraScores.push({ name, value: formatScore(values) });
    }
  }
  return orderedOutputScores(scoresByName, extraScores);
}

async function liteRtOutputMetadata(outputs: Record<string, InstanceType<LiteRtModule['Tensor']>>) {
  return sortOutputMetadata(
    Object.entries(outputs).map(([name, tensor]) => ({
      name,
      dims: Array.from(tensor.type.layout.dimensions),
      type: tensor.type.dtype,
    })),
  );
}

async function liteRtOutputScores(outputs: Record<string, InstanceType<LiteRtModule['Tensor']>>) {
  const scoresByName = new Map<string, string>();
  const extraScores: Array<{ name: string; value: string }> = [];
  for (const [name, tensor] of Object.entries(outputs)) {
    const data = await tensor.data();
    const values = data instanceof Float32Array ? Array.from(data) : [];
    const canonicalName = canonicalOutputName(name);
    if (outputOrder.includes(canonicalName)) {
      scoresByName.set(canonicalName, formatScore(values));
    } else {
      extraScores.push({ name, value: formatScore(values) });
    }
  }
  return orderedOutputScores(scoresByName, extraScores);
}

async function loadDemoImages(): Promise<LoadedDemoImage[]> {
  if (demoImagesPromise) {
    return demoImagesPromise;
  }
  const entries = manifest.demoImages ?? [];
  if (entries.length === 0) {
    throw new Error('No demo images found in demo_images/.');
  }

  demoImagesPromise = Promise.all(
    entries.map(
      (entry) =>
        new Promise<LoadedDemoImage>((resolve, reject) => {
          const image = new Image();
          image.onload = () => {
            resolve({
              entry,
              image,
              width: image.naturalWidth,
              height: image.naturalHeight,
            });
          };
          image.onerror = () => reject(new Error(`Failed to load demo image ${entry.name}`));
          image.src = `${entry.path}?reload=${demoImageReloadToken}`;
        }),
    ),
  );
  return demoImagesPromise;
}

function selectedDetector(runtime: Runtime): DetectorEntry {
  const detector = (manifest.detectors ?? []).find((entry) => entry.runtime === runtime);
  if (!detector) {
    throw new Error(`No YOLOMiT detector is available for ${runtime}.`);
  }
  return detector;
}

async function runOnnxDemoImageBenchmark(
  model: ModelEntry,
  detector: DetectorEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkRun> {
  const images = await loadDemoImages();
  const detectorSession = await createOnnxSession(detector, backend);
  const chcSession = await createOnnxSession(model, backend);
  const loadMs = detectorSession.loadMs + chcSession.loadMs;
  let lastOutputs: Awaited<ReturnType<OnnxSession['run']>> | undefined;
  let lastHeadCrops: HeadCropPreview[] = [];

  const runPipelineOnce = async (capturePreview: boolean): Promise<DemoPipelineRun> => {
    const previews: HeadCropPreview[] = [];
    let yoloMs = 0;
    let chcMs = 0;
    for (const image of images) {
      const detectorInputName = detectorSession.session.inputNames[0];
      const detectorFeeds = {
        [detectorInputName]: new detectorSession.ort.Tensor(
          'float32',
          makeYoloInput(image.image, 'nchw'),
          [1, 3, yoloHeight, yoloWidth],
        ),
      };
      const yoloStart = performance.now();
      const detectorOutputs = await detectorSession.session.run(detectorFeeds);
      yoloMs += performance.now() - yoloStart;
      const detectorTensor = detectorOutputs[detectorSession.session.outputNames[0]];
      if (!detectorTensor || !(detectorTensor.data instanceof Float32Array)) {
        throw new Error('YOLOMiT ONNX output is not a float32 tensor.');
      }
      const detections = decodeYoloDetections(detectorTensor.data, detectorTensor.dims, image);
      const chcFeeds: Record<string, InstanceType<OrtModule['Tensor']>> = {};
      for (const inputName of chcSession.session.inputNames) {
        const spec = tensorSpecFromOnnxInput(chcSession.session, inputName);
        chcFeeds[inputName] = new chcSession.ort.Tensor(
          'float32',
          fillChcInputTensor(image.image, detections, inputName, spec),
          spec.dims,
        );
      }
      const chcStart = performance.now();
      lastOutputs = await chcSession.session.run(chcFeeds);
      chcMs += performance.now() - chcStart;
      if (capturePreview) {
        previews.push({
          imageName: image.entry.name,
          dataUrl: makeCropPreview(image.image, detections.head),
          score: detections.head?.score,
          scores: onnxOutputScores(lastOutputs),
        });
      }
    }
    return { previews, yoloMs, chcMs };
  };

  try {
    for (let index = 0; index < warmup; index += 1) {
      await runPipelineOnce(false);
    }

    const samplesMs: number[] = [];
    const yoloSamplesMs: number[] = [];
    const chcSamplesMs: number[] = [];
    for (let index = 0; index < runs; index += 1) {
      const start = performance.now();
      const pipelineRun = await runPipelineOnce(index === runs - 1);
      lastHeadCrops = pipelineRun.previews;
      yoloSamplesMs.push(pipelineRun.yoloMs);
      chcSamplesMs.push(pipelineRun.chcMs);
      samplesMs.push(performance.now() - start);
    }

    return {
      result: {
        mode: 'demo_image',
        modelName: model.name,
        detectorName: detector.name,
        imageCount: images.length,
        runtime: 'onnxruntime-web',
        backend,
        warmup,
        runs,
        loadMs,
        ...makeStats(samplesMs),
        avgYoloMs: makeStats(yoloSamplesMs).avgMs,
        avgChcMs: makeStats(chcSamplesMs).avgMs,
        samplesMs,
        outputs: onnxOutputMetadata(lastOutputs),
      },
      headCrops: lastHeadCrops,
    };
  } finally {
    await detectorSession.session.release();
    await chcSession.session.release();
  }
}

async function runLiteRtDetector(
  liteRt: LiteRtModule,
  detectorModel: LiteRtCompiledModel,
  image: LoadedDemoImage,
): Promise<SelectedDetections> {
  const input = detectorModel.getInputDetails()[0];
  const feeds = {
    [input.name]: new liteRt.Tensor(makeYoloInput(image.image, 'nhwc'), [1, yoloHeight, yoloWidth, 3]),
  };
  let outputs: Record<string, InstanceType<LiteRtModule['Tensor']>> | undefined;
  try {
    outputs = await detectorModel.run(feeds);
    const outputTensor = Object.values(outputs)[0];
    const data = await outputTensor.data();
    if (!(data instanceof Float32Array)) {
      throw new Error('YOLOMiT LiteRT.js output is not a float32 tensor.');
    }
    return decodeYoloDetections(data, Array.from(outputTensor.type.layout.dimensions), image);
  } finally {
    deleteLiteRtTensors(Object.values(feeds));
    if (outputs) {
      deleteLiteRtTensors(Object.values(outputs));
    }
  }
}

async function runLiteRtChc(
  liteRt: LiteRtModule,
  chcModel: LiteRtCompiledModel,
  image: LoadedDemoImage,
  detections: SelectedDetections,
) {
  const feeds: Record<string, InstanceType<LiteRtModule['Tensor']>> = {};
  for (const input of chcModel.getInputDetails()) {
    const spec = tensorSpecFromLiteRtInput(input);
    feeds[input.name] = new liteRt.Tensor(fillChcInputTensor(image.image, detections, input.name, spec), spec.dims);
  }
  try {
    return await chcModel.run(feeds);
  } finally {
    deleteLiteRtTensors(Object.values(feeds));
  }
}

async function runLiteRtDemoImageBenchmark(
  model: ModelEntry,
  detector: DetectorEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkRun> {
  const images = await loadDemoImages();
  const detectorModel = await createLiteRtModel(detector, backend);
  const chcModel = await createLiteRtModel(model, backend);
  const loadMs = detectorModel.loadMs + chcModel.loadMs;
  let lastOutputMetadata: Array<{ name: string; dims: readonly number[]; type: string }> = [];
  let lastHeadCrops: HeadCropPreview[] = [];

  const runPipelineOnce = async (capturePreview: boolean): Promise<DemoPipelineRun> => {
    const previews: HeadCropPreview[] = [];
    let yoloMs = 0;
    let chcMs = 0;
    for (const image of images) {
      const yoloStart = performance.now();
      const detections = await runLiteRtDetector(detectorModel.liteRt, detectorModel.compiledModel, image);
      yoloMs += performance.now() - yoloStart;
      const chcStart = performance.now();
      const outputs = await runLiteRtChc(chcModel.liteRt, chcModel.compiledModel, image, detections);
      chcMs += performance.now() - chcStart;
      lastOutputMetadata = await liteRtOutputMetadata(outputs);
      if (capturePreview) {
        previews.push({
          imageName: image.entry.name,
          dataUrl: makeCropPreview(image.image, detections.head),
          score: detections.head?.score,
          scores: await liteRtOutputScores(outputs),
        });
      }
      deleteLiteRtTensors(Object.values(outputs));
    }
    return { previews, yoloMs, chcMs };
  };

  try {
    for (let index = 0; index < warmup; index += 1) {
      await runPipelineOnce(false);
    }

    const samplesMs: number[] = [];
    const yoloSamplesMs: number[] = [];
    const chcSamplesMs: number[] = [];
    for (let index = 0; index < runs; index += 1) {
      const start = performance.now();
      const pipelineRun = await runPipelineOnce(index === runs - 1);
      lastHeadCrops = pipelineRun.previews;
      yoloSamplesMs.push(pipelineRun.yoloMs);
      chcSamplesMs.push(pipelineRun.chcMs);
      samplesMs.push(performance.now() - start);
    }

    return {
      result: {
        mode: 'demo_image',
        modelName: model.name,
        detectorName: detector.name,
        imageCount: images.length,
        runtime: 'litert',
        backend,
        warmup,
        runs,
        loadMs,
        ...makeStats(samplesMs),
        avgYoloMs: makeStats(yoloSamplesMs).avgMs,
        avgChcMs: makeStats(chcSamplesMs).avgMs,
        samplesMs,
        outputs: lastOutputMetadata,
      },
      headCrops: lastHeadCrops,
    };
  } finally {
    detectorModel.compiledModel.delete();
    chcModel.compiledModel.delete();
  }
}

async function runBenchmark(
  model: ModelEntry,
  backend: Backend,
  warmup: number,
  runs: number,
  mode: InputMode,
): Promise<BenchmarkRun> {
  if (mode === 'demo_image') {
    const detector = selectedDetector(model.runtime);
    if (model.runtime === 'litert') {
      return await runLiteRtDemoImageBenchmark(model, detector, backend, warmup, runs);
    }
    return await runOnnxDemoImageBenchmark(model, detector, backend, warmup, runs);
  }
  if (model.runtime === 'litert') {
    return await runLiteRtBenchmark(model, backend, warmup, runs);
  }
  return await runOnnxBenchmark(model, backend, warmup, runs);
}

function formatMs(value: number): string {
  return `${value.toFixed(3)} ms`;
}

function formatDemoAverage(result: BenchmarkResult): string {
  const yolo = typeof result.avgYoloMs === 'number' ? formatMs(result.avgYoloMs) : '-';
  const chc = typeof result.avgChcMs === 'number' ? formatMs(result.avgChcMs) : '-';
  return `${formatMs(result.avgMs)} / ${yolo} / ${chc}`;
}

function renderHeadCrops(crops: HeadCropPreview[] | undefined) {
  if (!crops || crops.length === 0) {
    cropPanelEl.hidden = true;
    headCropsEl.innerHTML = '';
    return;
  }

  cropPanelEl.hidden = false;
  headCropsEl.innerHTML = crops
    .map((crop) => {
      const score = typeof crop.score === 'number' ? `score ${crop.score.toFixed(3)}` : 'zero-filled';
      const media = crop.dataUrl
        ? `<img src="${crop.dataUrl}" alt="Head crop for ${escapeHtml(crop.imageName)}" />`
        : '<div class="crop-placeholder">No Head</div>';
      const scores =
        crop.scores && crop.scores.length > 0
          ? `<dl class="crop-scores">${crop.scores
              .map(
                (item) =>
                  `<dt>${escapeHtml(item.name)}</dt><dd>${escapeHtml(item.value)}</dd>`,
              )
              .join('')}</dl>`
          : '';
      return `<figure>${media}<figcaption><strong>${escapeHtml(crop.imageName)}</strong><span>${score}</span>${scores}</figcaption></figure>`;
    })
    .join('');
}

function renderResult(run: BenchmarkRun) {
  const { result } = run;
  const isDemo = result.mode === 'demo_image';
  summaryEl.classList.remove('muted');
  summaryEl.innerHTML = `
    <dl>
      <dt>Input Mode</dt><dd>${isDemo ? 'Demo Image' : 'Synthetic Tensor'}</dd>
      <dt>Model</dt><dd>${result.modelName}</dd>
      ${result.detectorName ? `<dt>Detector</dt><dd>${result.detectorName}</dd>` : ''}
      ${result.imageCount ? `<dt>Images / Run</dt><dd>${result.imageCount}</dd>` : ''}
      <dt>Runtime</dt><dd>${result.runtime === 'litert' ? 'LiteRT.js' : 'onnxruntime-web'}</dd>
      <dt>Backend</dt><dd>${result.backend}</dd>
      <dt>Load</dt><dd>${formatMs(result.loadMs)}</dd>
      <dt>${isDemo ? 'Avg E2E / YOLO / CHC' : 'Avg Inference'}</dt><dd>${isDemo ? formatDemoAverage(result) : formatMs(result.avgMs)}</dd>
      <dt>Median ${isDemo ? 'End-to-End' : 'Inference'}</dt><dd>${formatMs(result.medianMs)}</dd>
      <dt>P95 ${isDemo ? 'End-to-End' : 'Inference'}</dt><dd>${formatMs(result.p95Ms)}</dd>
      <dt>Min / Max ${isDemo ? 'End-to-End' : 'Inference'}</dt><dd>${formatMs(result.minMs)} / ${formatMs(result.maxMs)}</dd>
    </dl>
  `;
  outputsEl.classList.remove('muted');
  outputsEl.innerHTML = result.outputs
    .map((output) => `<div class="output-row"><code>${output.name}</code><span>${output.type} [${output.dims.join(', ')}]</span></div>`)
    .join('');
  samplesTitleEl.textContent = isDemo ? 'End-to-End Pipeline Time Samples (ms/run)' : 'Inference Time Samples (ms)';
  samplesEl.textContent = JSON.stringify(result.samplesMs.map((sample) => Number(sample.toFixed(4))), null, 2);
  renderHeadCrops(run.headCrops);
}

function selectedModel(): ModelEntry {
  const model = manifest.models.find((entry) => entry.name === modelSelect.value && entry.runtime === runtimeSelect.value);
  if (!model) {
    throw new Error('No model selected.');
  }
  return model;
}

function updateModelOptions(preferredModel?: string) {
  const runtime = runtimeSelect.value as Runtime;
  const mode = inputModeSelect.value as InputMode;
  const models = manifest.models.filter((model) => model.runtime === runtime);
  modelSelect.innerHTML = models
    .map((model) => `<option value="${model.name}">${model.name} (${(model.bytes / 1024 / 1024).toFixed(2)} MiB)</option>`)
    .join('');

  if (preferredModel && models.some((model) => model.name === preferredModel)) {
    modelSelect.value = preferredModel;
  }

  const hasDetector = (manifest.detectors ?? []).some((detector) => detector.runtime === runtime);
  const hasDemoImages = (manifest.demoImages ?? []).length > 0;
  const canRun = models.length > 0 && (mode === 'synthetic_tensor' || (hasDetector && hasDemoImages));
  modelSelect.disabled = models.length === 0;
  runButton.disabled = !canRun;

  if (mode === 'demo_image' && !hasDetector) {
    statusEl.textContent = `No YOLOMiT detector is available for ${runtime}`;
  } else if (mode === 'demo_image' && !hasDemoImages) {
    statusEl.textContent = 'No demo image is available';
  } else {
    const imageCount = mode === 'demo_image' ? `, ${(manifest.demoImages ?? []).length} demo image(s)` : '';
    statusEl.textContent = `${models.length} ${runtime === 'litert' ? 'LiteRT.js' : 'ONNX'} model(s) ready${imageCount}`;
  }
}

async function runFromUi() {
  runButton.disabled = true;
  statusEl.textContent = 'Running...';
  try {
    const mode = inputModeSelect.value as InputMode;
    if (mode === 'demo_image') {
      await loadManifest({ preserveSelection: true, updateWebGpu: false });
      demoImagesPromise = undefined;
      demoImageReloadToken = String(Date.now());
    }
    const run = await runBenchmark(
      selectedModel(),
      backendSelect.value as Backend,
      Number(warmupInput.value),
      Number(runsInput.value),
      mode,
    );
    renderResult(run);
    statusEl.textContent = 'Complete';
  } catch (error) {
    statusEl.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    runButton.disabled = false;
  }
}

async function loadManifest(options: { preserveSelection?: boolean; updateWebGpu?: boolean } = {}) {
  const previousModel = modelSelect.value;
  const response = await fetch('/models/manifest.json', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load manifest: ${response.status}`);
  }
  manifest = (await response.json()) as Manifest;
  manifest.detectors ??= [];
  manifest.demoImages ??= [];
  updateModelOptions(options.preserveSelection ? previousModel : undefined);
  if (options.updateWebGpu === false) {
    return;
  }
  webGpuAvailable = await checkWebGpuAvailable();
  const webGpuOption = backendSelect.querySelector<HTMLOptionElement>('option[value="webgpu"]');
  if (webGpuOption && !webGpuAvailable) {
    webGpuOption.disabled = true;
    webGpuOption.textContent = 'WebGPU (unsupported)';
  }
}

async function autoRunIfRequested() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('auto') !== '1') {
    return;
  }

  const requestedRuntime = params.get('runtime') === 'litert' ? 'litert' : 'onnxruntime-web';
  runtimeSelect.value = requestedRuntime;
  inputModeSelect.value = params.get('mode') === 'demo_image' ? 'demo_image' : 'synthetic_tensor';
  const requestedModel = params.get('model') ?? (requestedRuntime === 'litert' ? 'chc_s_float32.tflite' : 'chc_s.onnx');
  updateModelOptions(requestedModel);
  backendSelect.value = params.get('backend') === 'webgpu' ? 'webgpu' : 'wasm';
  warmupInput.value = params.get('warmup') ?? '2';
  runsInput.value = params.get('runs') ?? '10';

  try {
    const run = await runBenchmark(
      selectedModel(),
      backendSelect.value as Backend,
      Number(warmupInput.value),
      Number(runsInput.value),
      inputModeSelect.value as InputMode,
    );
    renderResult(run);
    window.electronBenchmark?.finishCliBenchmark({ ok: true, result: run.result });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    statusEl.textContent = message;
    window.electronBenchmark?.finishCliBenchmark({ ok: false, error: message });
  }
}

runButton.addEventListener('click', () => {
  void runFromUi();
});

runtimeSelect.addEventListener('change', () => {
  updateModelOptions();
});

inputModeSelect.addEventListener('change', () => {
  updateModelOptions();
  if (inputModeSelect.value === 'synthetic_tensor') {
    renderHeadCrops(undefined);
  }
});

samplesToggleButton.addEventListener('click', () => {
  setSamplesExpanded(!samplesExpanded);
});

setSamplesExpanded(false);

loadManifest()
  .then(autoRunIfRequested)
  .catch((error: unknown) => {
    statusEl.textContent = error instanceof Error ? error.message : String(error);
  });
