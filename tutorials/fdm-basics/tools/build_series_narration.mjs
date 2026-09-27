import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { videos as commonVideos } from "./series_storyboard.mjs";
import { basics } from "./basics_storyboard.mjs";
import { refineStoryboards, resolveGuidance } from "./guidance.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const videos = refineStoryboards([basics, ...commonVideos]);
const captures = {
  ...JSON.parse(fs.readFileSync(path.join(root, "public/series-capture.json")))
    .screens,
  ...Object.fromEntries(
    Object.entries(
      JSON.parse(
        fs.readFileSync(path.join(root, "public/capture-manifest.json")),
      ).screens,
    ).map(([k, v]) => [`base/${k}`, v]),
  ),
};
const audit = videos.flatMap((video) => resolveGuidance(video, captures));
const output = path.join(root, "out/v2");
const audio = path.join(root, "public/audio/guided-v2");
const scripts = path.join(root, "narration/guided-v2");
fs.mkdirSync(audio, { recursive: true });
fs.mkdirSync(scripts, { recursive: true });
fs.mkdirSync(path.join(root, "src/series"), { recursive: true });
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(
  path.join(output, "guidance-audit.json"),
  JSON.stringify(audit, null, 2),
);
const stamp = (frame) => {
  const n = Math.round((frame / 30) * 1000);
  return `${String(Math.floor(n / 3600000)).padStart(2, "0")}:${String(Math.floor(n / 60000) % 60).padStart(2, "0")}:${String(Math.floor(n / 1000) % 60).padStart(2, "0")},${String(n % 1000).padStart(3, "0")}`;
};
for (const video of videos) {
  let offset = 0;
  const subtitles = [];
  for (const scene of video.scenes) {
    let at = 24;
    for (let i = 0; i < scene.beats.length; i++) {
      const beat = scene.beats[i];
      const stem = `${scene.id}-${i + 1}`;
      const txt = path.join(scripts, `${stem}.txt`);
      const aiff = path.join(scripts, `${stem}.aiff`);
      const wav = path.join(audio, `${stem}.wav`);
      const hash = crypto
        .createHash("sha256")
        .update(`${beat.text}|Tingting|200|-18|-1.5`)
        .digest("hex");
      const hashfile = wav + ".sha256";
      fs.writeFileSync(txt, beat.text + "\n");
      if (
        !fs.existsSync(wav) ||
        !fs.existsSync(hashfile) ||
        fs.readFileSync(hashfile, "utf8") !== hash
      ) {
        execFileSync("say", [
          "-v",
          "Tingting",
          "-r",
          "200",
          "-o",
          aiff,
          "-f",
          txt,
        ]);
        execFileSync("ffmpeg", [
          "-hide_banner",
          "-loglevel",
          "error",
          "-y",
          "-i",
          aiff,
          "-af",
          "highpass=f=70,loudnorm=I=-18:TP=-1.5:LRA=9",
          "-ar",
          "48000",
          "-ac",
          "1",
          wav,
        ]);
        fs.writeFileSync(hashfile, hash);
        fs.unlinkSync(aiff);
      }
      const seconds = Number(
        execFileSync(
          "ffprobe",
          [
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            wav,
          ],
          { encoding: "utf8" },
        ).trim(),
      );
      beat.from = at;
      beat.duration = Math.ceil(seconds * 30);
      beat.audio = `audio/guided-v2/${stem}.wav`;
      beat.hold = Math.max(36, beat.guide.steps.length * 55 - beat.duration);
      subtitles.push({
        from: offset + at,
        to: offset + at + beat.duration + 20,
        text: beat.text,
      });
      at += beat.duration + beat.hold;
    }
    scene.from = offset;
    scene.duration = at + 24;
    offset += scene.duration;
    console.log(video.id, scene.id, `${(scene.duration / 30).toFixed(1)}s`);
  }
  video.duration = offset;
  video.captions = subtitles.map((c) => ({
    text: c.text,
    startMs: (c.from / 30) * 1000,
    endMs: (c.to / 30) * 1000,
    timestampMs: null,
    confidence: null,
  }));
  fs.writeFileSync(
    path.join(output, `${video.id}-captions.json`),
    JSON.stringify(video.captions, null, 2),
  );
  const srt = subtitles
    .map((c, i) => `${i + 1}\n${stamp(c.from)} --> ${stamp(c.to)}\n${c.text}\n`)
    .join("\n");
  fs.writeFileSync(
    path.join(output, video.filename.replace(".mp4", ".zh-CN.srt")),
    srt,
  );
  console.log("TOTAL", video.id, `${(offset / 30).toFixed(1)}s`);
}
fs.writeFileSync(
  path.join(root, "src/series/data.json"),
  JSON.stringify(videos, null, 2),
);
