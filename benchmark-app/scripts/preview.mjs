import { spawn } from 'node:child_process';
import http from 'node:http';

const appUrl = 'http://127.0.0.1:4173';
const children = [];

function run(command, args, options = {}) {
  const child = spawn(command, args, { stdio: 'inherit', shell: false, ...options });
  children.push(child);
  child.on('exit', (code, signal) => {
    if (signal || code) {
      stopAll();
      process.exit(code ?? 1);
    }
  });
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

process.on('SIGINT', () => {
  stopAll();
  process.exit(130);
});
process.on('SIGTERM', () => {
  stopAll();
  process.exit(143);
});

run('pnpm', ['exec', 'vite', 'preview', '--host', '127.0.0.1', '--port', '4173']);
await waitForUrl(appUrl);
run('pnpm', ['exec', 'electron', '.'], {
  env: electronEnv({ BENCHMARK_APP_URL: appUrl }),
});
