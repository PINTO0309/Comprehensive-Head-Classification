import { copyFile, mkdir, readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(appDir, '..');
const modelDir = path.join(appDir, 'public', 'models');
const ortDir = path.join(appDir, 'public', 'ort');

async function copyModels() {
  await rm(modelDir, { recursive: true, force: true });
  await mkdir(modelDir, { recursive: true });

  const entries = await readdir(repoRoot);
  const models = [];
  for (const entry of entries) {
    if (!/^chc_.+\.onnx$/.test(entry)) {
      continue;
    }
    const source = path.join(repoRoot, entry);
    const sourceStat = await stat(source);
    if (!sourceStat.isFile()) {
      continue;
    }
    const target = path.join(modelDir, entry);
    await copyFile(source, target);
    models.push({
      name: entry,
      path: `/models/${entry}`,
      bytes: sourceStat.size,
      withFiqa: !entry.includes('_wo_fiqa'),
    });
  }

  models.sort((a, b) => a.name.localeCompare(b.name));
  await writeFile(
    path.join(modelDir, 'manifest.json'),
    `${JSON.stringify({ generatedAt: new Date().toISOString(), models }, null, 2)}\n`,
  );

  if (models.length === 0) {
    console.warn('No chc_*.onnx models found in the repository root.');
  } else {
    console.log(`Copied ${models.length} model(s) to ${path.relative(repoRoot, modelDir)}`);
  }
}

await rm(ortDir, { recursive: true, force: true });
await copyModels();
