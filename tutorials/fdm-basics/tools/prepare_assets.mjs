import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repository = path.resolve(root, "../..");

if (process.platform !== "darwin") {
  throw new Error(
    "Asset generation requires macOS with the Tingting voice and PingFang SC font.",
  );
}
for (const command of ["uv", "ffmpeg", "ffprobe"]) {
  const result = spawnSync(
    command,
    [command === "uv" ? "--version" : "-version"],
    {
      stdio: "ignore",
    },
  );
  if (result.error || result.status !== 0) {
    throw new Error(`Install ${command} before preparing tutorial assets.`);
  }
}
for (const file of [
  path.join(root, "assets/optical-test.jpg"),
  ...["encoder", "decoder"].map((part) =>
    path.join(
      repository,
      `runtime/segment-anything/edge_sam_3x/edge_sam_3x_${part}.onnx`,
    ),
  ),
]) {
  if (!fs.existsSync(file)) throw new Error(`Required input missing: ${file}`);
}

const run = (command, args, cwd) => {
  console.log(`\nPreparing: ${args.at(-1)}`);
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
};

for (const script of ["capture_fdm.py", "capture_series.py"]) {
  run(
    "uv",
    ["run", "--no-sync", "python", path.join(root, "tools", script)],
    repository,
  );
}
for (const script of ["build_series_narration.mjs", "prepare_font.mjs"]) {
  run(process.execPath, [path.join(root, "tools", script)], root);
}
console.log(
  "\nAssets ready. Run npm run lint, then npm run dev or npm run render:series.",
);
