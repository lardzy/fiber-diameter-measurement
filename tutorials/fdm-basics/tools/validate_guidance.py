"""Audit cue coverage, target geometry, timeline, captions and rendered previews."""

from collections import Counter
import json
import math
from pathlib import Path


root = Path(__file__).resolve().parents[1]
out = root / "out/v2"
videos = json.loads((root / "src/series/data.json").read_text())
audit = json.loads((out / "guidance-audit.json").read_text())
previews = json.loads((out / "previews/manifest.json").read_text())
expected = set()
actions = Counter()
reports = {}
assert [v["id"] for v in videos] == [
    "FDM-01", "FDM-02", "FDM-03", "FDM-Measurements", "FDM-09",
]
assert videos[3]["chapters"] == ["04 分类复核", "05 连续测量", "08 快速测径", "06 面积计数", "07 魔棒扩选"]

for video in videos:
    scene_end = 0
    beats = targets = 0
    caption_index = 0
    for scene in video["scenes"]:
        assert scene["from"] == scene_end, scene["id"]
        scene_end += scene["duration"]
        beat_end = 0
        for bi, beat in enumerate(scene["beats"], 1):
            beats += 1
            assert beat["from"] >= beat_end
            beat_end = beat["from"] + beat["duration"] + beat["hold"]
            assert beat_end <= scene["duration"]
            assert (root / "public" / beat["audio"]).is_file()
            steps = beat["guide"]["steps"]
            assert steps, (scene["id"], bi)
            assert (beat["duration"] + beat["hold"]) / len(steps) >= 55
            caption = video["captions"][caption_index]
            caption_index += 1
            assert caption["text"] == beat["text"]
            assert abs(caption["startMs"] - (scene["from"] + beat["from"]) / 30 * 1000) < .01
            assert caption["startMs"] < caption["endMs"] <= scene_end / 30 * 1000
            for ci, step in enumerate(steps, 1):
                targets += 1
                key = (video["id"], scene["id"], bi, ci)
                expected.add(key)
                assert step["label"].strip()
                assert step["action"] in {"inspect", "click", "select", "type", "drag", "path"}
                actions[step["action"]] += 1
                if "card" in step:
                    assert 0 <= step["card"] < len(scene["cards"])
                    continue
                screen = step.get("screen", beat["guide"]["screen"])
                assert (root / "public" / screen["file"]).is_file(), key
                x, y, w, h = step["rect"]
                assert all(math.isfinite(n) for n in (x, y, w, h))
                assert 0 <= x and 0 <= y and w > 0 and h > 0, key
                assert x + w <= screen["width"] + .01 and y + h <= screen["height"] + .01, key
                if step["action"] in {"drag", "path"}:
                    assert len(step["points"]) >= 2, key
                    for px, py in step["points"]:
                        assert x - .01 <= px <= x + w + .01 and y - .01 <= py <= y + h + .01, key
        assert scene["duration"] - beat_end >= 24
    assert scene_end == video["duration"]
    assert caption_index == len(video["captions"])
    assert all(a["endMs"] <= b["startMs"] for a, b in zip(video["captions"], video["captions"][1:]))
    reports[video["id"]] = {"scenes": len(video["scenes"]), "spoken_beats": beats, "guided_targets": targets, "seconds": video["duration"] / 30}

actual = {(p["video"], p["scene"], p["beat"], p["cue"]) for p in previews}
assert len(actual) == len(previews) and actual == expected, {"missing": sorted(expected - actual), "extra": sorted(actual - expected)}
assert all(Path(p["file"]).is_file() for p in previews)
assert len(audit) == sum(v["spoken_beats"] for v in reports.values())
assert sum(a["targets"] for a in audit) == len(expected)
report = {"status": "passed", "videos": reports, "spoken_beats": len(audit), "guided_targets": len(expected), "preview_frames": len(previews), "actions": dict(actions), "checks": ["all narration beats have targets", "target and gesture geometry inside capture", "cue minimum duration", "timeline continuity", "caption alignment", "referenced assets exist", "every cue has a rendered preview"]}
(out / "guidance-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
