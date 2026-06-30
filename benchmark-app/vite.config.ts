import { createReadStream } from 'node:fs';
import { copyFile, mkdir, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

const appDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(appDir, '..');
const require = createRequire(import.meta.url);
const ortSourceDir = path.dirname(require.resolve('onnxruntime-web'));
const liteRtWasmSourceDir = path.join(path.dirname(require.resolve('@litertjs/core')), '..', 'wasm');
const ortAssetPattern = /^ort-.*\.(wasm|mjs)$/;
const liteRtAssetPattern = /^litert_wasm_.*\.(wasm|js)$/;
const onnxModelPattern = /^chc_.+\.onnx$/;
const tfliteModelPattern = /^chc_.+_float32\.tflite$/;
const modelPattern = /^(chc_.+\.onnx|chc_.+_float32\.tflite)$/;
const detectorPattern = /^yolomit_t_wholebody28_1x3x480x640(_float32)?\.(onnx|tflite)$/;
const demoImagePattern = /\.(png|jpe?g)$/i;
const demoImageDir = path.join(repoRoot, 'demo_images');

type ModelEntry = {
  name: string;
  path: string;
  bytes: number;
  withFiqa: boolean;
  format: 'onnx' | 'tflite';
  runtime: 'onnxruntime-web' | 'litert';
};

type DetectorEntry = Omit<ModelEntry, 'withFiqa'>;

type DemoImageEntry = {
  name: string;
  path: string;
  bytes: number;
};

async function listOrtAssets(): Promise<string[]> {
  return (await readdir(ortSourceDir)).filter((file) => ortAssetPattern.test(file));
}

async function listLiteRtAssets(): Promise<string[]> {
  return (await readdir(liteRtWasmSourceDir)).filter((file) => liteRtAssetPattern.test(file));
}

async function listModels(): Promise<ModelEntry[]> {
  const entries = await readdir(repoRoot);
  const models: ModelEntry[] = [];
  for (const entry of entries) {
    if (!modelPattern.test(entry)) {
      continue;
    }
    const sourcePath = path.join(repoRoot, entry);
    const sourceStat = await stat(sourcePath);
    if (!sourceStat.isFile()) {
      continue;
    }
    models.push({
      name: entry,
      path: `/models/${entry}`,
      bytes: sourceStat.size,
      withFiqa: !entry.includes('_wo_fiqa'),
      format: onnxModelPattern.test(entry) ? 'onnx' : 'tflite',
      runtime: onnxModelPattern.test(entry) ? 'onnxruntime-web' : 'litert',
    });
  }
  return models.sort((a, b) => a.runtime.localeCompare(b.runtime) || a.name.localeCompare(b.name));
}

async function listDetectors(): Promise<DetectorEntry[]> {
  const entries = await readdir(repoRoot);
  const detectors: DetectorEntry[] = [];
  for (const entry of entries) {
    if (!detectorPattern.test(entry)) {
      continue;
    }
    const sourcePath = path.join(repoRoot, entry);
    const sourceStat = await stat(sourcePath);
    if (!sourceStat.isFile()) {
      continue;
    }
    const format = entry.endsWith('.onnx') ? 'onnx' : 'tflite';
    detectors.push({
      name: entry,
      path: `/models/${entry}`,
      bytes: sourceStat.size,
      format,
      runtime: format === 'onnx' ? 'onnxruntime-web' : 'litert',
    });
  }
  return detectors.sort((a, b) => a.runtime.localeCompare(b.runtime) || a.name.localeCompare(b.name));
}

async function listDemoImages(): Promise<DemoImageEntry[]> {
  let entries: string[];
  try {
    entries = await readdir(demoImageDir);
  } catch {
    return [];
  }

  const images: DemoImageEntry[] = [];
  for (const entry of entries) {
    if (!demoImagePattern.test(entry)) {
      continue;
    }
    const sourcePath = path.join(demoImageDir, entry);
    const sourceStat = await stat(sourcePath);
    if (!sourceStat.isFile()) {
      continue;
    }
    images.push({
      name: entry,
      path: `/demo-images/${entry}`,
      bytes: sourceStat.size,
    });
  }
  return images.sort((a, b) => a.name.localeCompare(b.name));
}

async function modelManifest(): Promise<string> {
  const [models, detectors, demoImages] = await Promise.all([listModels(), listDetectors(), listDemoImages()]);
  return `${JSON.stringify({ generatedAt: new Date().toISOString(), models, detectors, demoImages }, null, 2)}\n`;
}

function contentTypeFor(fileName: string): string {
  if (fileName.endsWith('.wasm')) {
    return 'application/wasm';
  }
  if (fileName.endsWith('.onnx')) {
    return 'application/octet-stream';
  }
  if (fileName.endsWith('.tflite')) {
    return 'application/octet-stream';
  }
  if (fileName.endsWith('.json')) {
    return 'application/json; charset=utf-8';
  }
  if (fileName.endsWith('.png')) {
    return 'image/png';
  }
  if (fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) {
    return 'image/jpeg';
  }
  return 'text/javascript; charset=utf-8';
}

function benchmarkAssetsPlugin(): Plugin {
  return {
    name: 'chc-benchmark-assets',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;
        if (pathname === '/models/manifest.json') {
          try {
            const body = await modelManifest();
            res.setHeader('Content-Type', contentTypeFor('manifest.json'));
            res.setHeader('Content-Length', Buffer.byteLength(body));
            res.setHeader('Cache-Control', 'no-cache');
            res.end(body);
          } catch (error) {
            next(error);
          }
          return;
        }

        if (pathname.startsWith('/models/')) {
          const fileName = path.basename(decodeURIComponent(pathname));
          if (!modelPattern.test(fileName) && !detectorPattern.test(fileName)) {
            res.statusCode = 404;
            res.end('Not found');
            return;
          }
          const sourcePath = path.join(repoRoot, fileName);
          try {
            const sourceStat = await stat(sourcePath);
            if (!sourceStat.isFile()) {
              res.statusCode = 404;
              res.end('Not found');
              return;
            }
            res.setHeader('Content-Type', contentTypeFor(fileName));
            res.setHeader('Content-Length', sourceStat.size);
            res.setHeader('Cache-Control', 'no-cache');
            createReadStream(sourcePath).pipe(res);
          } catch (error) {
            next(error);
          }
          return;
        }

        if (pathname.startsWith('/demo-images/')) {
          const fileName = path.basename(decodeURIComponent(pathname));
          if (!demoImagePattern.test(fileName)) {
            res.statusCode = 404;
            res.end('Not found');
            return;
          }
          const sourcePath = path.join(demoImageDir, fileName);
          try {
            const sourceStat = await stat(sourcePath);
            if (!sourceStat.isFile()) {
              res.statusCode = 404;
              res.end('Not found');
              return;
            }
            res.setHeader('Content-Type', contentTypeFor(fileName));
            res.setHeader('Content-Length', sourceStat.size);
            res.setHeader('Cache-Control', 'no-cache');
            createReadStream(sourcePath).pipe(res);
          } catch (error) {
            next(error);
          }
          return;
        }

        if (pathname.startsWith('/litert/wasm/')) {
          const fileName = path.basename(decodeURIComponent(pathname));
          if (!liteRtAssetPattern.test(fileName)) {
            res.statusCode = 404;
            res.end('Not found');
            return;
          }
          const sourcePath = path.join(liteRtWasmSourceDir, fileName);
          try {
            const sourceStat = await stat(sourcePath);
            if (!sourceStat.isFile()) {
              res.statusCode = 404;
              res.end('Not found');
              return;
            }
            res.setHeader('Content-Type', contentTypeFor(fileName));
            res.setHeader('Content-Length', sourceStat.size);
            res.setHeader('Cache-Control', 'no-cache');
            createReadStream(sourcePath).pipe(res);
          } catch (error) {
            next(error);
          }
          return;
        }

        if (!pathname.startsWith('/ort/')) {
          next();
          return;
        }

        const fileName = path.basename(decodeURIComponent(pathname));
        if (!ortAssetPattern.test(fileName)) {
          res.statusCode = 404;
          res.end('Not found');
          return;
        }
        const sourcePath = path.join(ortSourceDir, fileName);
        try {
          const sourceStat = await stat(sourcePath);
          if (!sourceStat.isFile()) {
            res.statusCode = 404;
            res.end('Not found');
            return;
          }
          res.setHeader('Content-Type', contentTypeFor(fileName));
          res.setHeader('Content-Length', sourceStat.size);
          res.setHeader('Cache-Control', 'no-cache');
          createReadStream(sourcePath).pipe(res);
        } catch (error) {
          next(error);
        }
      });
    },
    async writeBundle() {
      const ortTargetDir = path.join(appDir, 'dist', 'ort');
      const ortFiles = await listOrtAssets();
      await rm(ortTargetDir, { recursive: true, force: true });
      await mkdir(ortTargetDir, { recursive: true });
      await Promise.all(
        ortFiles.map((file) => copyFile(path.join(ortSourceDir, file), path.join(ortTargetDir, file))),
      );

      const liteRtTargetDir = path.join(appDir, 'dist', 'litert', 'wasm');
      const liteRtFiles = await listLiteRtAssets();
      await rm(liteRtTargetDir, { recursive: true, force: true });
      await mkdir(liteRtTargetDir, { recursive: true });
      await Promise.all(
        liteRtFiles.map((file) => copyFile(path.join(liteRtWasmSourceDir, file), path.join(liteRtTargetDir, file))),
      );

      const modelTargetDir = path.join(appDir, 'dist', 'models');
      const [models, detectors, demoImages] = await Promise.all([listModels(), listDetectors(), listDemoImages()]);
      await rm(modelTargetDir, { recursive: true, force: true });
      await mkdir(modelTargetDir, { recursive: true });
      await Promise.all(
        [...models, ...detectors].map((model) =>
          copyFile(path.join(repoRoot, model.name), path.join(modelTargetDir, model.name)),
        ),
      );
      await writeFile(path.join(modelTargetDir, 'manifest.json'), await modelManifest());

      const demoImageTargetDir = path.join(appDir, 'dist', 'demo-images');
      await rm(demoImageTargetDir, { recursive: true, force: true });
      await mkdir(demoImageTargetDir, { recursive: true });
      await Promise.all(
        demoImages.map((image) => copyFile(path.join(demoImageDir, image.name), path.join(demoImageTargetDir, image.name))),
      );
    },
  };
}

export default defineConfig({
  plugins: [benchmarkAssetsPlugin()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
  },
});
