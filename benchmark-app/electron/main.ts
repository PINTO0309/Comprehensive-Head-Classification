import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';

type CliBenchmarkOptions = {
  auto: boolean;
  backend: 'wasm' | 'webgpu';
  model: string;
  warmup: number;
  runs: number;
};

function configureChromiumGpuSwitches() {
  const enableFeatures =
    process.platform === 'win32' || process.platform === 'darwin'
      ? 'WebGPU,WebGPUService'
      : 'Vulkan,WebGPU,WebGPUService';

  app.commandLine.appendSwitch('ignore-gpu-blocklist');
  app.commandLine.appendSwitch('enable-zero-copy');
  app.commandLine.appendSwitch('disable-gpu-sandbox');
  app.commandLine.appendSwitch('enable-unsafe-webgpu');
  app.commandLine.appendSwitch('enable-webgpu-developer-features');
  app.commandLine.appendSwitch('enable-features', enableFeatures);
  app.commandLine.appendSwitch('use-webgpu-adapter', 'default');
  app.commandLine.appendSwitch('disable-features', 'UseSkiaRenderer,UseChromeOSDirectVideoDecoder');
}

configureChromiumGpuSwitches();

function parseCliBenchmarkOptions(): CliBenchmarkOptions {
  const args = process.argv.slice(2);
  const readValue = (name: string, fallback: string): string => {
    const prefix = `--${name}=`;
    const found = args.find((arg) => arg.startsWith(prefix));
    return found ? found.slice(prefix.length) : fallback;
  };
  const backend = readValue('backend', 'wasm');
  return {
    auto: args.includes('--benchmark'),
    backend: backend === 'webgpu' ? 'webgpu' : 'wasm',
    model: readValue('model', 'chc_s.onnx'),
    warmup: Number(readValue('warmup', '2')),
    runs: Number(readValue('runs', '10')),
  };
}

function benchmarkUrl(baseUrl: string, options: CliBenchmarkOptions): string {
  if (!options.auto) {
    return baseUrl;
  }
  const url = new URL(baseUrl);
  url.searchParams.set('auto', '1');
  url.searchParams.set('backend', options.backend);
  url.searchParams.set('model', options.model);
  url.searchParams.set('warmup', String(options.warmup));
  url.searchParams.set('runs', String(options.runs));
  return url.toString();
}

async function createWindow() {
  const cliOptions = parseCliBenchmarkOptions();
  if (cliOptions.auto) {
    const timeoutMs = cliOptions.backend === 'webgpu' ? 20000 : 120000;
    setTimeout(() => {
      console.log(
        JSON.stringify(
          {
            ok: false,
            error: `Benchmark timed out after ${timeoutMs} ms. The selected backend may be unsupported.`,
          },
          null,
          2,
        ),
      );
      app.exit(1);
    }, timeoutMs);
  }
  const window = new BrowserWindow({
    width: 1180,
    height: 900,
    minHeight: 860,
    show: !cliOptions.auto,
    backgroundColor: '#f6f7f9',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const serverUrl = process.env.VITE_DEV_SERVER_URL ?? process.env.BENCHMARK_APP_URL;
  if (serverUrl) {
    await window.loadURL(benchmarkUrl(serverUrl, cliOptions));
    return;
  }

  await window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'), {
    query: cliOptions.auto
      ? {
          auto: '1',
          backend: cliOptions.backend,
          model: cliOptions.model,
          warmup: String(cliOptions.warmup),
          runs: String(cliOptions.runs),
        }
      : undefined,
  });
}

ipcMain.on('benchmark-finished', (_event, payload: unknown) => {
  console.log(JSON.stringify(payload, null, 2));
  const ok =
    typeof payload === 'object' &&
    payload !== null &&
    'ok' in payload &&
    (payload as { ok: unknown }).ok === true;
  app.exit(ok ? 0 : 1);
});

app.whenReady().then(createWindow).catch((error: unknown) => {
  console.error(error);
  app.exit(1);
});

app.on('window-all-closed', () => {
  app.quit();
});
