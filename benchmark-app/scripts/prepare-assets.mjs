import { copyFile, mkdir, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(appDir, '..');
const modelDir = path.join(appDir, 'public', 'models');
const ortDir = path.join(appDir, 'public', 'ort');
const demoImageDir = path.join(appDir, 'public', 'demo-images');
const repoDemoImageDir = path.join(repoRoot, 'demo_images');
const modelPattern = /^(chc_.+\.onnx|chc_.+_float32\.tflite)$/;
const detectorPattern = /^yolomit_t_wholebody28_1x3x480x640(_float32)?\.(onnx|tflite)$/;
const demoImagePattern = /\.(png|jpe?g)$/i;

async function copyModels() {
  await rm(modelDir, { recursive: true, force: true });
  await mkdir(modelDir, { recursive: true });

  const entries = await readdir(repoRoot);
  const models = [];
  const detectors = [];
  for (const entry of entries) {
    if (!modelPattern.test(entry) && !detectorPattern.test(entry)) {
      continue;
    }
    const source = path.join(repoRoot, entry);
    const sourceStat = await stat(source);
    if (!sourceStat.isFile()) {
      continue;
    }
    const target = path.join(modelDir, entry);
    await copyFile(source, target);
    const format = entry.endsWith('.onnx') ? 'onnx' : 'tflite';
    const runtime = format === 'onnx' ? 'onnxruntime-web' : 'litert';
    if (detectorPattern.test(entry)) {
      detectors.push({
        name: entry,
        path: `/models/${entry}`,
        bytes: sourceStat.size,
        format,
        runtime,
      });
    } else {
      models.push({
        name: entry,
        path: `/models/${entry}`,
        bytes: sourceStat.size,
        withFiqa: !entry.includes('_wo_fiqa'),
        format,
        runtime,
      });
    }
  }

  models.sort((a, b) => a.runtime.localeCompare(b.runtime) || a.name.localeCompare(b.name));
  detectors.sort((a, b) => a.runtime.localeCompare(b.runtime) || a.name.localeCompare(b.name));
  const demoImages = await copyDemoImages();
  await writeFile(
    path.join(modelDir, 'manifest.json'),
    `${JSON.stringify({ generatedAt: new Date().toISOString(), models, detectors, demoImages }, null, 2)}\n`,
  );

  if (models.length === 0) {
    console.warn('No chc_*.onnx or chc_*_float32.tflite models found in the repository root.');
  } else {
    console.log(`Copied ${models.length} model(s) to ${path.relative(repoRoot, modelDir)}`);
  }
}

async function copyDemoImages() {
  await rm(demoImageDir, { recursive: true, force: true });
  await mkdir(demoImageDir, { recursive: true });

  let entries = [];
  try {
    entries = await readdir(repoDemoImageDir);
  } catch {
    return [];
  }

  const demoImages = [];
  for (const entry of entries) {
    if (!demoImagePattern.test(entry)) {
      continue;
    }
    const source = path.join(repoDemoImageDir, entry);
    const sourceStat = await stat(source);
    if (!sourceStat.isFile()) {
      continue;
    }
    await copyFile(source, path.join(demoImageDir, entry));
    demoImages.push({
      name: entry,
      path: `/demo-images/${entry}`,
      bytes: sourceStat.size,
    });
  }
  return demoImages.sort((a, b) => a.name.localeCompare(b.name));
}

await rm(ortDir, { recursive: true, force: true });
await copyModels();
