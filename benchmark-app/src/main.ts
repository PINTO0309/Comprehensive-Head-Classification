import './styles.css';

type Backend = 'wasm' | 'webgpu';
type Runtime = 'onnxruntime-web' | 'litert';

type ModelEntry = {
  name: string;
  path: string;
  bytes: number;
  withFiqa: boolean;
  format: 'onnx' | 'tflite';
  runtime: Runtime;
};

type Manifest = {
  generatedAt: string;
  models: ModelEntry[];
};

type OrtModule = typeof import('onnxruntime-web');
type LiteRtModule = typeof import('@litertjs/core');

type BenchmarkResult = {
  modelName: string;
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
  samplesMs: number[];
  outputs: Array<{ name: string; dims: readonly number[]; type: string }>;
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

    <section class="panel">
      <h2>Inference Time Samples (ms)</h2>
      <pre id="samples">[]</pre>
    </section>
  </main>
`;

const statusEl = document.querySelector<HTMLDivElement>('#status')!;
const runtimeSelect = document.querySelector<HTMLSelectElement>('#runtime')!;
const modelSelect = document.querySelector<HTMLSelectElement>('#model')!;
const backendSelect = document.querySelector<HTMLSelectElement>('#backend')!;
const warmupInput = document.querySelector<HTMLInputElement>('#warmup')!;
const runsInput = document.querySelector<HTMLInputElement>('#runs')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const summaryEl = document.querySelector<HTMLDivElement>('#summary')!;
const outputsEl = document.querySelector<HTMLDivElement>('#outputs')!;
const samplesEl = document.querySelector<HTMLPreElement>('#samples')!;

let manifest: Manifest = { generatedAt: '', models: [] };
let webGpuAvailable = false;
let liteRtModulePromise: Promise<LiteRtModule> | undefined;
let liteRtRuntimePromise: Promise<unknown> | undefined;
let liteRtJspiSupportedPromise: Promise<boolean> | undefined;
const outputOrder = [
  'prob_background_plain',
  'prob_masked',
  'prob_sunglasses',
  'prob_eye_open',
  'prob_mouth_open',
  'quality_score',
];

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

async function createOnnxSession(model: ModelEntry, backend: Backend) {
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

function makeOnnxFeeds(ort: OrtModule, session: Awaited<ReturnType<typeof createOnnxSession>>['session']) {
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

async function createLiteRtModel(model: ModelEntry, backend: Backend) {
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
  compiledModel: Awaited<ReturnType<LiteRtModule['loadAndCompile']>>,
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
    const aIndex = outputOrder.indexOf(a.name);
    const bIndex = outputOrder.indexOf(b.name);
    return (aIndex === -1 ? Number.MAX_SAFE_INTEGER : aIndex) - (bIndex === -1 ? Number.MAX_SAFE_INTEGER : bIndex);
  });
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1);
  return sorted[index];
}

async function runOnnxBenchmark(
  model: ModelEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkResult> {
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

  const sorted = [...samplesMs].sort((a, b) => a - b);
  const sum = samplesMs.reduce((total, sample) => total + sample, 0);
  const outputs = Object.entries(lastOutputs ?? {}).map(([name, tensor]) => ({
    name,
    dims: tensor.dims,
    type: tensor.type,
  }));

  await session.release();

  return {
    modelName: model.name,
    runtime: 'onnxruntime-web',
    backend,
    warmup,
    runs,
    loadMs,
    avgMs: sum / samplesMs.length,
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    minMs: sorted[0] ?? 0,
    maxMs: sorted.at(-1) ?? 0,
    samplesMs,
    outputs: sortOutputMetadata(outputs),
  };
}

async function runLiteRtBenchmark(
  model: ModelEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkResult> {
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

    const sorted = [...samplesMs].sort((a, b) => a - b);
    const sum = samplesMs.reduce((total, sample) => total + sample, 0);

    return {
      modelName: model.name,
      runtime: 'litert',
      backend,
      warmup,
      runs,
      loadMs,
      avgMs: sum / samplesMs.length,
      medianMs: percentile(sorted, 0.5),
      p95Ms: percentile(sorted, 0.95),
      minMs: sorted[0] ?? 0,
      maxMs: sorted.at(-1) ?? 0,
      samplesMs,
      outputs: sortOutputMetadata(outputMetadata),
    };
  } finally {
    deleteLiteRtTensors(Object.values(feeds));
    compiledModel.delete();
  }
}

async function runBenchmark(
  model: ModelEntry,
  backend: Backend,
  warmup: number,
  runs: number,
): Promise<BenchmarkResult> {
  if (model.runtime === 'litert') {
    return await runLiteRtBenchmark(model, backend, warmup, runs);
  }
  return await runOnnxBenchmark(model, backend, warmup, runs);
}

function formatMs(value: number): string {
  return `${value.toFixed(3)} ms`;
}

function renderResult(result: BenchmarkResult) {
  summaryEl.classList.remove('muted');
  summaryEl.innerHTML = `
    <dl>
      <dt>Model</dt><dd>${result.modelName}</dd>
      <dt>Runtime</dt><dd>${result.runtime === 'litert' ? 'LiteRT.js' : 'onnxruntime-web'}</dd>
      <dt>Backend</dt><dd>${result.backend}</dd>
      <dt>Load</dt><dd>${formatMs(result.loadMs)}</dd>
      <dt>Avg Inference</dt><dd>${formatMs(result.avgMs)}</dd>
      <dt>Median Inference</dt><dd>${formatMs(result.medianMs)}</dd>
      <dt>P95 Inference</dt><dd>${formatMs(result.p95Ms)}</dd>
      <dt>Min / Max Inference</dt><dd>${formatMs(result.minMs)} / ${formatMs(result.maxMs)}</dd>
    </dl>
  `;
  outputsEl.classList.remove('muted');
  outputsEl.innerHTML = result.outputs
    .map((output) => `<div class="output-row"><code>${output.name}</code><span>${output.type} [${output.dims.join(', ')}]</span></div>`)
    .join('');
  samplesEl.textContent = JSON.stringify(result.samplesMs.map((sample) => Number(sample.toFixed(4))), null, 2);
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
  const models = manifest.models.filter((model) => model.runtime === runtime);
  modelSelect.innerHTML = models
    .map((model) => `<option value="${model.name}">${model.name} (${(model.bytes / 1024 / 1024).toFixed(2)} MiB)</option>`)
    .join('');

  if (preferredModel && models.some((model) => model.name === preferredModel)) {
    modelSelect.value = preferredModel;
  }

  modelSelect.disabled = models.length === 0;
  runButton.disabled = models.length === 0;
  statusEl.textContent = `${models.length} ${runtime === 'litert' ? 'LiteRT.js' : 'ONNX'} model(s) ready`;
}

async function runFromUi() {
  runButton.disabled = true;
  statusEl.textContent = 'Running...';
  try {
    const result = await runBenchmark(
      selectedModel(),
      backendSelect.value as Backend,
      Number(warmupInput.value),
      Number(runsInput.value),
    );
    renderResult(result);
    statusEl.textContent = 'Complete';
  } catch (error) {
    statusEl.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    runButton.disabled = false;
  }
}

async function loadManifest() {
  const response = await fetch('/models/manifest.json', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load manifest: ${response.status}`);
  }
  manifest = (await response.json()) as Manifest;
  updateModelOptions();
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
  const requestedModel = params.get('model') ?? 'chc_s.onnx';
  updateModelOptions(requestedModel);
  backendSelect.value = params.get('backend') === 'webgpu' ? 'webgpu' : 'wasm';
  warmupInput.value = params.get('warmup') ?? '2';
  runsInput.value = params.get('runs') ?? '10';

  try {
    const result = await runBenchmark(
      selectedModel(),
      backendSelect.value as Backend,
      Number(warmupInput.value),
      Number(runsInput.value),
    );
    renderResult(result);
    window.electronBenchmark?.finishCliBenchmark({ ok: true, result });
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

loadManifest()
  .then(autoRunIfRequested)
  .catch((error: unknown) => {
    statusEl.textContent = error instanceof Error ? error.message : String(error);
  });
