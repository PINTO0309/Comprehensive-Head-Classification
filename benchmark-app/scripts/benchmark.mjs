import { spawn } from 'node:child_process';
import http from 'node:http';
import net from 'node:net';

const requestedRuntime = process.argv[2] === 'litert' ? 'litert' : 'onnxruntime-web';
const backend = requestedRuntime === 'litert' ? (process.argv[3] ?? 'wasm') : (process.argv[2] ?? 'wasm');
const model =
  requestedRuntime === 'litert'
    ? (process.argv[4] ?? 'chc_s_float32.tflite')
    : (process.argv[3] ?? 'chc_s.onnx');
const children = [];

function run(command, args, options = {}) {
  const child = spawn(command, args, { stdio: 'inherit', shell: false, ...options });
  children.push(child);
  return child;
}

function stopAll() {
  for (const child of children) {
    if (!child.killed) {
      child.kill();
    }
  }
}

function electronEnv(extra = {}) {
  const { ELECTRON_RUN_AS_NODE: _electronRunAsNode, ...baseEnv } = process.env;
  return { ...baseEnv, ...extra };
}

function waitForExit(child) {
  return new Promise((resolve, reject) => {
    child.once('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${child.spawnfile} failed`))));
  });
}

function waitForUrl(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const poll = () => {
      const request = http.get(url, (response) => {
        response.resume();
        resolve();
      });
      request.on('error', () => {
        if (Date.now() > deadline) {
          reject(new Error(`Timed out waiting for ${url}`));
          return;
        }
        setTimeout(poll, 250);
      });
    };
    poll();
  });
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      server.close(() => {
        if (address && typeof address === 'object') {
          resolve(address.port);
        } else {
          reject(new Error('Failed to allocate a free port'));
        }
      });
    });
  });
}

try {
  const port = await getFreePort();
  const appUrl = `http://127.0.0.1:${port}`;
  await waitForExit(run('pnpm', ['build']));
  run('pnpm', ['exec', 'vite', 'preview', '--host', '127.0.0.1', '--port', String(port)]);
  await waitForUrl(appUrl);
  await waitForExit(
    run(
      'pnpm',
      [
        'exec',
        'electron',
        '.',
        '--benchmark',
        `--runtime=${requestedRuntime}`,
        `--backend=${backend}`,
        `--model=${model}`,
        '--warmup=2',
        '--runs=10',
      ],
      { env: electronEnv({ BENCHMARK_APP_URL: appUrl }) },
    ),
  );
} finally {
  stopAll();
}
