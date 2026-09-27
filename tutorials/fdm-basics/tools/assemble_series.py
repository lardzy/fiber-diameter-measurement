"""Join the five finished episodes without recompressing their video frames."""

import json
from pathlib import Path
import subprocess

from validate_series import validate_media


def assemble(videos, out: Path):
    filename = "FDM-全套教程-引导增强版-v2.mp4"
    concat = out / "compilation.ffconcat"
    # Relative names avoid embedding workstation paths in the generated manifest.
    concat.write_text("ffconcat version 1.0\n" + "".join(
        f"file '{v['filename']}'\nduration {v['duration'] / 30:.9f}\n" for v in videos
    ))
    metadata = [";FFMETADATA1", "title=FDM 全套教程 · V2 引导增强版"]
    offset = 0
    scenes, captions = [], []
    for v in videos:
        for s in v["scenes"]:
            start = offset + s["from"]
            title = f"{v['number']} · {s['title'].replace(chr(10), '')}"
            metadata.extend(["[CHAPTER]", "TIMEBASE=1/30", f"START={start}", f"END={start + s['duration']}", f"title={title}"])
            scenes.append({"title": title, "chapter": s["chapter"], "time": start / 30})
        for caption in v["captions"]:
            captions.append({**caption, "startMs": caption["startMs"] + offset / 30 * 1000, "endMs": caption["endMs"] + offset / 30 * 1000})
        offset += v["duration"]
    meta = out / "compilation.ffmetadata"
    meta.write_text("\n".join(metadata) + "\n")
    command = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat)]
    for v in videos:
        command.extend(["-i", str(out / v["filename"])])
    command.extend(["-f", "ffmetadata", "-i", str(meta)])
    # Trim/pad each decoded track to the exact episode duration, avoiding AAC
    # encoder-padding gaps at joins while preserving all authored picture frames.
    filters = [f"[{i + 1}:a]apad,atrim=duration={v['duration'] / 30:.9f},asetpts=PTS-STARTPTS[a{i}]" for i, v in enumerate(videos)]
    filters.append("".join(f"[a{i}]" for i in range(len(videos))) + f"concat=n={len(videos)}:v=0:a=1[audio]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[audio]", "-map_metadata", str(len(videos) + 1), "-map_chapters", str(len(videos) + 1), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out / filename)])
    with (out / "compilation.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    report = validate_media(out / filename, offset, chapters=len(scenes))
    (out / "compilation-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    def stamp(ms):
        n = round(ms)
        return f"{n // 3600000:02d}:{n // 60000 % 60:02d}:{n // 1000 % 60:02d},{n % 1000:03d}"
    (out / filename.replace(".mp4", ".zh-CN.srt")).write_text("\n".join(
        f"{i}\n{stamp(c['startMs'])} --> {stamp(c['endMs'])}\n{c['text']}\n" for i, c in enumerate(captions, 1)
    ))
    (out / "FDM-All-captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2) + "\n")
    print("COMPILATION VERIFIED", report, flush=True)
    return {"id": "FDM-All", "number": "全套", "title": "五集连续播放", "filename": filename, "duration": offset, "scenes": scenes}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    assemble(json.loads((root / "src/series/data.json").read_text()), root / "out/v2")
