"""Package the unified V2 episodes, continuous edition and practice files."""

import json
from pathlib import Path
import zipfile

from assemble_series import assemble

root = Path(__file__).resolve().parents[1]
out = root / "out/v2"
videos = json.loads((root / "src/series/data.json").read_text())
checks = json.loads((root / "public/series-capture.json").read_text())["checks"]
validation = json.loads((out / "series-validation.json").read_text())
guidance = json.loads((out / "guidance-validation.json").read_text())
assert guidance["status"] == "passed"
for v in videos:
    assert validation[v["id"]]["frames"] == v["duration"]
    assert validation[v["id"]]["bytes"] == (out / v["filename"]).stat().st_size
    assert validation[v["id"]]["full_decode"] == "passed"


def write_json(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stamp(seconds):
    n = int(seconds + .5)
    return f"{n // 60:02d}:{n % 60:02d}"


fixtures = [
    ("demo", "入门测量示例.fdmproj", 1, 3),
    ("demo-series", "02-多图片项目.fdmproj", 2, 3),
    ("demo-series", "03-标定与预设.fdmproj", 2, 1),
    ("demo-series", "04-05-08-06-07-测量操作.fdmproj", 4, 18),
    ("demo-series", "09-比例尺标注导出.fdmproj", 1, 3),
]
fixture_report = {}
for folder, filename, images, measurements in fixtures:
    base = root / folder
    documents = json.loads((base / filename).read_text())["documents"]
    counts = len(documents), sum(len(d["measurements"]) for d in documents)
    assert counts == (images, measurements), (filename, counts)
    for document in documents:
        image = Path(document["path"])
        assert not image.is_absolute() and (base / image).is_file(), image
    fixture_report[filename] = {"documents": counts[0], "measurements": counts[1], "image_paths_relative": True}
fixture_report["01_export_file_count"] = sum(p.is_file() for p in (root / "demo/导出结果").iterdir())
fixture_report["09_export_file_count"] = sum(p.is_file() for p in (root / "demo-series/09-导出结果").iterdir())
assert fixture_report["01_export_file_count"] == 5
assert fixture_report["09_export_file_count"] == 7
write_json("series-fixture-validation.json", fixture_report)

practice = """# FDM 全套教程跟练包 · V2

先完整解压，再在 FDM 中打开相应的 `.fdmproj`。保留项目旁的图片、assets 目录和标定文件。

- `01-入门/入门测量示例.fdmproj`：1 张合成图、3 条手动线段，配套 Excel、CSV 和叠加图。
- `常用功能/02-多图片项目.fdmproj`：2 张合成图，分别含 1、2 条测量。
- `常用功能/03-标定与预设.fdmproj`：同成像条件的两个合成样本，演示 µm 与 mm。预设保存在软件用户设置中；`03-预设参数.json` 供照填参数，项目不会自动安装预设。
- `常用功能/04-05-08-06-07-测量操作.fdmproj`：分类、连续折线、快速测径、面积孔洞、自由圈选、计数、魔棒与同类扩选。
- `常用功能/09-比例尺标注导出.fdmproj`：3 条测量、文字和箭头。`09-导出结果/` 包含实际导出的 7 个文件。

从零练习时，可把原图复制到新文件夹，不要同时复制同名 `.fdm.json`；该文件可能带入已有标定和测量。项目示例用于比对完成后的状态。

黄色框表示当前讲解目标；跟随鼠标看点击位置、拖动方向和选点次序。左侧会区分“单击”“输入”“拖动”“观察核对”。每完成一步可以暂停，在软件中跟练。

合成图仅用于操作练习。光学测试图没有已知标尺，保留原水印，结果使用 px / px²。智能工具采用实际本地 Edge SAM 3X 推理。显示比例尺和导出范围分别核对。
"""
practice_path = out / "FDM-全套教程-跟练素材-v2.zip"
with zipfile.ZipFile(practice_path, "w", zipfile.ZIP_DEFLATED) as z:
    prefix = Path("FDM-全套教程-跟练素材-v2")
    z.writestr(str(prefix / "跟练说明.md"), practice)
    for source, dest in [("demo", "01-入门"), ("demo-series", "常用功能")]:
        for p in sorted((root / source).rglob("*")):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root / source).parts) and p.name not in {"跟练说明.md", "03-预设参数.json"}:
                z.write(p, prefix / dest / p.relative_to(root / source))
    z.writestr(str(prefix / "常用功能/03-预设参数.json"), json.dumps(checks["03"]["presets"], ensure_ascii=False, indent=2))
with zipfile.ZipFile(practice_path) as z:
    assert z.testzip() is None

compilation = assemble(videos, out)
player_data = []
for v in videos:
    cover = f"{v['id']}-封面.png"
    assert (out / cover).is_file()
    scenes = [{"title": s["title"].replace("\n", ""), "chapter": s["chapter"], "time": s["from"] / 30, "stamp": stamp(s["from"] / 30)} for s in v["scenes"]]
    player_data.append({"id": v["id"], "number": v["number"], "title": v["title"], "file": v["filename"], "cover": cover, "duration": stamp(v["duration"] / 30), "scenes": scenes})
player_data.append({"id": compilation["id"], "number": compilation["number"], "title": compilation["title"], "file": compilation["filename"], "cover": "FDM-01-封面.png", "duration": stamp(compilation["duration"] / 30), "scenes": [{**s, "stamp": stamp(s["time"])} for s in compilation["scenes"]]})
write_json("章节索引.json", player_data)
template = (root / "tools/player.html").read_text()
(out / "教程目录.html").write_text(template.replace("__DATA__", json.dumps(player_data, ensure_ascii=False).replace("<", "\\u003c")))

lines = ["# FDM 全套操作教程 · V2 引导增强版", "", "五集独立视频和一支全套连播视频，均为 1920×1080、30 fps、H.264 / AAC，含中文旁白与画内字幕。原 v1 入门篇已重做为 01，与其余视频使用相同的指引与节奏。", "", "| 教程 | 时长 | 成片 |", "|---|---:|---|"]
for v in player_data:
    lines.append(f"| {v['number']} {v['title']} | {v['duration']} | [{v['file']}]({v['file']}) |")
lines += ["", "## 观看与跟练", "", "打开同目录的 [教程目录.html](教程目录.html)，选择单集或全套连播，点击章节跳转。支持 0.75× / 1× / 1.25× / 1.5× 播放。也可直接用本地播放器打开 MP4。", "", f"全套 {guidance['spoken_beats']} 句旁白配置了 {guidance['guided_targets']} 处引导。黄色目标框、局部放大、编号标记和连线指向讲解区域；鼠标光圈演示点击，轨迹演示拖动，编号点演示多点选择。左侧说明当前动作；观察结果时不模拟点击。每步留有停顿，方便暂停跟练。", "", "[跟练素材 ZIP](FDM-全套教程-跟练素材-v2.zip) 包含 01 和其余四集的完整项目、原图、标定文件与导出结果。先解压整个文件夹，再打开项目。", "", "每支视频提供同名 `.zh-CN.srt`；字幕已烧录在 MP4 中，无需另外加载。连播版含 42 个内嵌章节，支持章节的播放器可直接跳转。", "", "## 章节时间点"]
for v in player_data[:5]:
    lines.extend(["", f"### {v['number']} {v['title']}", ""])
    lines.extend(f"- {s['stamp']} — {s['chapter']} · {s['title']}" for s in v["scenes"])
lines += ["", "## 素材与验证", "", "画面来自 FDM 0.4.7 的真实 Qt 控件与实际操作状态，以两倍像素密度采集，再由 Remotion 编排镜头、鼠标、标记、字幕与旁白。教程为分步动画，使用 macOS 源码界面；Windows 安装包界面未在本次验证范围内。", "", "合成练习图在画面中明确标注。快速测径与标准魔棒使用本地光学测试图片，保留原水印，没有虚构物理标定；同类扩选的六候选示例使用合成练习图，候选来自实际模型与搜索流程。教程验证操作流程，不作为显微计量精度验证。", "", f"已生成 {guidance['preview_frames']} 个引导预览帧，覆盖全部目标，并抽查关键操作画面；五集及连播 MP4 均通过完整解码、画面帧数、时长、编码和音轨峰值检查。五个跟练项目的相对图片路径与测量记录数量已核对。", "", "- `guidance-audit.json`：逐句旁白与目标对照。", "- `guidance-validation.json`：引导覆盖、坐标与时间轴检查。", "- `series-validation.json` / `compilation-validation.json`：成片检查。", "- `series-fixture-validation.json`：跟练项目检查。", "", "## 可编辑工程", "", "源工程位于 FDM 仓库的 `tutorials/fdm-basics/`，编辑方法与重建命令见工程 README。01 文稿在 `tools/basics_storyboard.mjs`，其余文稿在 `tools/series_storyboard.mjs`，统一指引在 `tools/guidance.mjs`。生成的视频、音频、截图、时间轴和交付包均留在本地，不加入 Git。旧版 MP4 和 ZIP 保留在上一级 `out/`。"]
(out / "全套教程说明.md").write_text("\n".join(lines) + "\n")

delivery = ["教程目录.html", "全套教程说明.md", "章节索引.json", "series-validation.json", "compilation-validation.json", "series-fixture-validation.json", "guidance-audit.json", "guidance-validation.json", practice_path.name]
for v in videos + [compilation]:
    delivery.extend([v["filename"], v["filename"].replace(".mp4", ".zh-CN.srt"), f"{v['id']}-captions.json"])
delivery.extend(f"{v['id']}-封面.png" for v in videos)
bundle = out / "FDM-全套教程-完整教程包-v2.zip"
with zipfile.ZipFile(bundle, "w") as z:
    for name in delivery:
        z.write(out / name, Path("FDM-全套教程-v2") / name, compress_type=zipfile.ZIP_STORED if name.endswith((".mp4", ".zip", ".png")) else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(delivery)
write_json("package-validation.json", {"status": "passed", "archive": bundle.name, "bytes": bundle.stat().st_size, "members": len(delivery), "crc": "passed", "practice_archive_crc": "passed", "episodes": 5, "continuous_edition": True})
print("PLAYER", out / "教程目录.html", flush=True)
print("COMPLETE BUNDLE", bundle, bundle.stat().st_size, "bytes", flush=True)
