"""Check the actual delivered media against the authored timeline."""
from pathlib import Path
import json
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
def validate_media(file: Path, frames: int, chapters: int = 0):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-show_chapters", "-of", "json", str(file)]))
    picture = next(s for s in probe["streams"] if s["codec_type"] == "video")
    sound = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (picture["width"], picture["height"]) == (1920, 1080)
    assert picture["codec_name"] == "h264" and picture["pix_fmt"] in {"yuv420p", "yuvj420p"}
    assert picture["r_frame_rate"] == "30/1"
    assert int(picture["nb_frames"]) == frames
    assert sound["codec_name"] == "aac"
    assert abs(float(probe["format"]["duration"]) - frames / 30) < .15
    assert len(probe.get("chapters", [])) == chapters
    decode = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(file), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], capture_output=True, text=True)
    file.with_suffix(".decode.log").write_text(decode.stderr)
    assert decode.returncode == 0 and not decode.stderr.strip(), decode.stderr
    level = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(file), "-vn", "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    assert level.returncode == 0
    peak = float(re.search(r"max_volume: ([-\d.]+) dB", level.stderr).group(1))
    assert peak < 0, peak
    return {"file": file.name, "seconds": float(probe["format"]["duration"]), "frames": int(picture["nb_frames"]), "width":1920,"height":1080,"fps":30,"video_codec":"h264","pixel_format":picture["pix_fmt"],"audio_codec":"aac","audio_sample_rate":sound["sample_rate"],"audio_channels":sound["channels"],"max_volume_dbfs":peak,"bytes":file.stat().st_size,"chapters": chapters,"full_decode":"passed"}


if __name__ == "__main__":
    videos = json.loads((root / "src/series/data.json").read_text())
    report_path = root / "out/v2/series-validation.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for video in videos:
        if sys.argv[1:] and video["id"] not in sys.argv[1:]:
            continue
        report[video["id"]] = validate_media(root / "out/v2" / video["filename"], video["duration"])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("VERIFIED", video["id"], report[video["id"]], flush=True)
