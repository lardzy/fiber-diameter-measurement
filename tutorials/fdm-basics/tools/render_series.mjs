import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import {
  getCompositions,
  openBrowser,
  renderStill,
  renderMedia,
} from "@remotion/renderer";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const data = JSON.parse(
  fs.readFileSync(path.join(root, "src/series/data.json"), "utf8"),
);
const outputDir = path.join(root, "out/v2");
const selected = process.argv.filter((x) => x.startsWith("FDM-"));
const videos = data.filter((v) => !selected.length || selected.includes(v.id));
for (const v of videos)
  for (const s of v.scenes)
    for (const b of s.beats) {
      for (const step of b.guide.steps) {
        if (step.card !== undefined) continue;
        const file = step.screen?.file ?? b.guide.screen.file;
        if (!fs.existsSync(path.join(root, "public", file)))
          throw new Error("Missing guided shot " + file);
      }
    }
const previews = process.argv.includes("--previews");
const sceneFilter = process.argv
  .find((x) => x.startsWith("--scenes="))
  ?.slice(9)
  .split(",");
const dir = path.join(outputDir, "previews");
fs.mkdirSync(dir, { recursive: true });
const serveUrl = await bundle({
  entryPoint: path.join(root, "src/index.ts"),
  outDir: path.join(
    outputDir,
    previews ? "preview-bundle" : `render-${videos.map((v) => v.id).join("-")}`,
  ),
  rspack: true,
});
const browser = await openBrowser("chrome", {
  browserExecutable:
    process.env.REMOTION_BROWSER_EXECUTABLE ||
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
});
try {
  const compositions = await getCompositions(serveUrl, {
    puppeteerInstance: browser,
  });
  const records = [];
  for (const v of videos) {
    const composition = compositions.find((c) => c.id === v.id);
    if (!composition) throw new Error("Missing composition " + v.id);
    await renderStill({
      serveUrl,
      composition,
      frame: Math.min(100, composition.durationInFrames - 1),
      output: path.join(outputDir, `${v.id}-封面.png`),
      imageFormat: "png",
      puppeteerInstance: browser,
    });
    if (previews) {
      for (const s of v.scenes.filter(
        (s) => !sceneFilter || sceneFilter.includes(s.id),
      )) {
        // Check every cue, not just the first target in a spoken sentence.
        for (let i = 0; i < s.beats.length; i++) {
          const beat = s.beats[i];
          const span = (beat.duration + beat.hold) / beat.guide.steps.length;
          for (let cue = 0; cue < beat.guide.steps.length; cue++) {
            const frame = Math.floor(
              s.from +
                beat.from +
                cue * span +
                Math.min(span - 4, Math.max(42, span * 0.65)),
            );
            const output = path.join(dir, `${s.id}-${i + 1}.png`);
            const cueOutput = output.replace(/\.png$/, `-${cue + 1}.png`);
            await renderStill({
              serveUrl,
              composition,
              frame,
              output: cueOutput,
              imageFormat: "png",
              puppeteerInstance: browser,
            });
            records.push({
              video: v.id,
              scene: s.id,
              beat: i + 1,
              cue: cue + 1,
              label: beat.guide.steps[cue].label,
              action: beat.guide.steps[cue].action,
              frame,
              file: cueOutput,
            });
          }
        }
        console.log("PREVIEW", s.id);
      }
    } else {
      const outputLocation = path.join(outputDir, v.filename);
      let last = -1;
      const started = Date.now();
      await renderMedia({
        serveUrl,
        composition,
        outputLocation,
        codec: "h264",
        crf: 18,
        audioBitrate: "192k",
        pixelFormat: "yuv420p",
        imageFormat: "jpeg",
        jpegQuality: 92,
        concurrency: 4,
        puppeteerInstance: browser,
        onProgress: ({ progress }) => {
          const pct = Math.floor(progress * 20) * 5;
          if (pct !== last) {
            last = pct;
            console.log(
              v.id,
              pct + "%",
              Math.round((Date.now() - started) / 1000) + "s",
            );
          }
        },
      });
      console.log("DONE", v.id, outputLocation);
    }
  }
  if (previews)
    fs.writeFileSync(
      path.join(dir, "manifest.json"),
      JSON.stringify(records, null, 2),
    );
} finally {
  await browser.close({ silent: true });
}
