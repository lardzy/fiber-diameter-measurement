#!/usr/bin/env python3
"""Build PERFORMANCE.md only from existing final JSON; no workloads or third-party imports."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("performance_audit", ROOT / "scripts/validation/audit_cu_live_performance.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
PLATFORMS = {"macos": "macOS ARM64", "windows": "Windows ARM64／x64 仿真"}
STAGES = {"snapshot_copy": "整帧快照复制", "native_service": "原生测量服务（局部 ROI）",
          "snapshot_plus_native_service": "快照复制＋原生服务", "cached_summary": "缓存摘要读取",
          "append_summary_undo": "追加＋摘要＋撤销", "edit_summary_undo": "修改＋摘要＋撤销"}
CHECKS = {"draft_p95_ms": "草稿 P95", "draft_p99_ms": "草稿 P99", "result_visible_p95_ms": "结果可见 P95",
          "result_visible_p99_ms": "结果可见 P99", "gui_publish_max_ms": "GUI 发布最大值",
          "formal_failures": "正式尝试未全部成功", "all_rejection_events_including_warmup": "存在拒收（含预热）",
          "background_errors": "后台存在未完成／异常状态", "distinct_painted_fps": "不同帧绘制 FPS",
          "frame_budget_peak_bytes": "帧预算", "producer_errors": "输入线程异常"}


def f(value, decimals=3):
    return "—" if value is None else f"{value:.{decimals}f}" if isinstance(value, (float, int)) else str(value)


def four(metric):
    values = [metric.get(key) for key in ("p50_ms", "p95_ms", "p99_ms")]
    values.append(metric.get("exact_max_ms") if metric.get("exact_max_ms") is not None else metric.get("max_ms"))
    return " / ".join(f(value) for value in values)


def memory(resources):
    return " / ".join(f(resources.get(key) / 2**20, 1) if resources.get(key) is not None else "—"
                      for key in ("sampled_max_rss_bytes", "sampled_max_children_rss_bytes", "sampled_max_total_rss_bytes"))


def counts(values):
    return "；".join(f"{key} × {value}" for key, value in values.items()) or "无"


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "／").replace("\n", " ") for value in row) + " |")
    lines.append("")


def link(name):
    return f"[{name}]({name})"


def report_name(platform, suffix, revision):
    # The unchanged automatic algorithm keeps its original R1 build identity.
    prefix = 'final-' if revision == 'r1' or suffix.startswith('automatic-') else f'final-{revision}-'
    return f'{prefix}{platform}-{suffix}.json'


def thread_configuration(doc, suffix):
    """Keep each runner's requested/reported settings distinct, including null."""
    env = doc.get('environment', doc)  # motion records these at the root
    if 'opencv_requested_threads' in env:
        request = ('默认（未请求）' if env['opencv_requested_threads'] is None
                   else str(env['opencv_requested_threads']))
    elif suffix == 'manual-statistics' or suffix.startswith('automatic-'):
        request = '1（脚本固定；JSON未单列请求字段）'
    else:
        request = '未记录'
    reported = env.get('opencv_reported_threads', env.get('opencv_threads', '未记录'))
    framework = env.get('opencv_parallel_framework', '未记录')
    if isinstance(framework, str):
        framework = framework.replace('Parallel framework:', '').strip()
    variables = env.get('thread_environment')
    return request, reported, framework, json.dumps(variables, ensure_ascii=False) if variables is not None else '未记录'


def load(revision):
    docs, invalid = {}, {}
    selected = []
    for path in HERE.glob('*.json'):
        name = path.name
        if revision == 'r1':
            include = name.startswith(('final-macos-', 'final-windows-'))
        else:
            include = (name.startswith(tuple(f'final-{revision}-{p}-' for p in PLATFORMS))
                       or name in {report_name(p, f'automatic-{size}', revision)
                                   for p in PLATFORMS for size in (2048, 4096, 6144)}
                       or name == f'cu-live-workspace-hour-windows-final-{revision}.json')
        if include:
            selected.append(path)
    for path in sorted(selected):
        try:
            doc = json.loads(path.read_text())
            if isinstance(doc, dict) and any(key in doc for key in ("manual", "statistics", "automatic", "records", "cases", "latency_by_fifteen_minutes")):
                docs[path.name] = doc
        except (OSError, ValueError) as error:
            invalid[path.name] = str(error)
    return docs, invalid


def generate(revision='r1'):
    docs, invalid = load(revision)
    filename = lambda platform, suffix: report_name(platform, suffix, revision)
    output_name = 'PERFORMANCE.md' if revision == 'r1' else f'{revision.upper()}_PERFORMANCE.md'
    revision_label = revision.upper()
    hour_name = f'cu-live-workspace-hour-windows-final-{revision}.json'
    ui_rows, ui_raw = [], {}
    for name, doc in docs.items():
        if "cases" in doc:
            ui_rows.extend(audit.ui_cases(HERE / name, doc))
            for case in doc["cases"]:
                ui_raw[(name, case["width"], case["save_images"])] = case
    audit.compare_loaded(ui_rows)
    identity = ("**源码身份：**以下 macOS／Windows 九组短程结果属于首轮冻结快照。采样后新增的 [后台组距配置修正](configuration-rebuild-overlay.json) 及后续运动优化必须另列源码和复测证据；这些原始 JSON 保持不变，不能将旧快照的成绩写成后续修正版已经通过。" if revision == 'r1' else
        f"**源码身份：**本页独立读取 `final-{revision}-{{macos,windows}}-*.json` 的每平台六组新版短程结果及 `{hour_name}`。首轮 [PERFORMANCE.md](PERFORMANCE.md)、[R2_PERFORMANCE.md](R2_PERFORMANCE.md) 与原 JSON 保留；新版结果不覆盖旧失败。R2 尚无正式采样时保持待验收，不从 R1 或 R3 借用数据填充。三个尺寸的独立全图算法未改，单列引用 R1 的六份报告，明确是原构建证据，不伪称重新跑过 {revision_label} 全图。配置修正见 [overlay](configuration-rebuild-overlay.json)，各运行的实际源码／原生摘要以原始 JSON 为准。")
    if revision == 'r4':
        identity += (" R4 相对 [R3_PERFORMANCE.md](R3_PERFORMANCE.md) 仅调整原生运动峰值扫描的有限数分类，"
                     "复用已有位分类 helper；PNG 编码、预算、界面与统计实现沿用 R3。"
                     "R3 报告及原始 JSON 完整保留，R4 的配对验证、正式性能和一小时结果独立判定；"
                     "缺少 R4 证据时不填写收益，也不将 R3 成绩记作 R4 已通过。")
    conclusion = ("**当前已出现未达标条件，不能宣称整体性能通过。**" if revision == 'r1' or any(r['overall_status'] == 'FAIL' for r in ui_rows) else
                  "**新版完整验收尚待所有分项、失败分母与一小时证据共同核对。**")
    memory_scope = (
        "R3 的 512 MiB 逻辑预算包含去重后的保留帧、来源声明的 CPU／GPU 采集预留，以及 PNG 编码工作区。"
        "生产 `png_workspace_bytes(width, channels)` 为 2 MiB 编码 allowance＋两行缓冲，即 `2*width*channels+1` 字节；"
        "原帧租约与编码预留原子准入，同一串行 writer 组取尚存任务预留的最大值，不同组相加。"
        "生产逐行 SUB／Z_RLE 编码不建立完整 RGB 图像或完整压缩 PNG 数组；原帧在编码结束后释放该 writer 的租约，"
        "其他消费者仍持有的像素继续计费。该 allowance 是逻辑工作区预留，**不是进程 RSS 或驱动内存的硬上限**；"
        "不能因账本低于512 MiB就称全部内存已达标。R1 的完整 `cv2.imencode` 输出未纳入账本的历史缺口不应被改写为R3仍使用同一路径。"
        if revision in ('r3', 'r4') else
        "512 MiB 是去重后的保留帧及来源预留预算（含明确的 CPU／GPU 帧载荷），不是进程 RSS 上限。"
        "R1 ledger 不覆盖 `cv2.imencode` 的压缩输出缓冲、Qt／驱动全部分配，原计划的总缓冲预算未完整落实。"
        "R2／R3 的覆盖范围须按各版源码及原始报告另核对；不能沿用旧边界推断新版，也不能因任一账本低于512 MiB就称总内存已达标。"
    )
    if revision == 'r4':
        memory_scope = "R4 沿用 R3 的 PNG 与预算实现，以下为共同边界。" + memory_scope
    memory_scope += ("RSS 另含 Python、Qt、算法、SDK、PNG 与子进程工作区；GPU 驱动不透明开销也不能由 RSS 完整覆盖。"
                     "资源表按每个条件约100 ms采样的父／子／合计峰值列出，短于采样间隔的峰值可能遗漏；"
                     "三者各自取峰，不要求父峰＋子峰等于同时观测到的合计峰。")
    lines = ["# CU 实时试验" + (f" {revision_label}" if revision != 'r1' else "") + " 正式性能报告", "",
        "本报告只读取已经到达且符合本页版本命名的 JSON。缺失项明确列为待验收；不使用早期 `performance-*.json`，也不把早期无真实画布的一小时控制器测试算作最终 UI 联合验收。",
        "", identity,
        "", conclusion + " 逐项成功、失败与未完成以以下表格为准；脚本退出 0、快速原生内核或静止画面通过，均不能替代动态画面和联合负载验收。",
        "", "## 固定口径与覆盖", "",
        "硬门槛以 [采样前固定规则](performance-acceptance-rules.md) 为准。每个短程条件预热 5 次、正式尝试 30 次；不补测凑满成功数。成功延迟只以实际成功且绘制的样本计算，失败分母始终保留 30 次。预热拒收单列。`—` 表示没有有效样本或未测，不是零耗时。",
        "", "短程 UI 的 `interval_ms=100` 是每次结果成功／失败后再等待 100 ms，属于串行连续测量，**不是固定每秒 10 条**。输入请求 30 Hz；不同帧绘制 FPS 与重复重绘次数分开。结束区间边界计数可能使短窗口 FPS 略高于 30，不代表请求速率提高。",
        "", "表内延迟单位均为 ms，四元组依次为 **P50 / P95 / P99 / 最大值**。30 次中的 P99 仅为本批估计；失败越多，可用分位样本越少，不能解释为长期尾延迟保证。",
        "", "GUI 发布计时覆盖 `_accept` 和同步 UI 信号，每次最大值须 ≤ 4 ms；不能用接受后重绘时间替代。端到端结果可见由实际 QTest 第二击计到包含该记录的 `paintEvent` 完成。"]
    table(lines, ["项目", "固定门槛"], [
        ["第一击草稿可见", "P95 ≤ 16.7；P99 ≤ 33"], ["GUI 同步发布", "每次 ≤ 4；报告所有超限次数"],
        ["第二击结果可见 2K／4K", "P95 ≤ 50；P99 ≤ 100"], ["第二击结果可见 6K", "P95 ≤ 100；P99 ≤ 150"],
        ["预览", "请求 30 Hz；实际不同帧绘制 ≥ 27 FPS"],
        ["换场可见清除", "2K／4K P95 ≤ 150；6K P95 ≤ 250"],
        ["PNG＋后台全图相对影响", "匹配无 PNG、无后台基线；结果可见 P95 增量 ≤ max(基线×20%, 10)，绝对门槛仍须通过"],
        ["稳定输入中的拒收／超时", "任何一次都记失败；不可删除失败后只比较成功延迟"],
        ["真实 Qt 一小时", "≥3600 s、≥10000 实际入账；输入30 Hz、尝试4 Hz、PNG开启、传统全图每10秒提交"]])
    expected = [filename(p, suffix) for p in PLATFORMS for suffix in
                ("manual-statistics", "motion", "automatic-2048", "automatic-4096", "automatic-6144", "ui-stable", "ui-dynamic", "ui-joint-stable", "ui-joint-dynamic")]
    if revision != 'r1':
        expected.append(hour_name)
    table(lines, ["正式报告", "到达情况"], [[link(name), "已读取" if name in docs else "待验收／尚未到达"] for name in expected])
    if invalid:
        lines.extend(["暂不能读取的结果（可能尚在写入）：" + counts(invalid), ""])
    lines.extend(["覆盖边界：静止输入与逐帧唯一标记的低幅噪声动态输入分列；静止输入能命中像素相等缓存，不能代表相机每帧变化。短程联合为 PNG＋**持续至多一个 multiscale 任务**，一小时为 PNG＋**每10秒 traditional**；二者不是同等负载。短程 PNG＋持续 traditional、长期持续 multiscale、真实相机／采集卡与显微录像均不得由已有组合推定通过。", "", "## 平台和资源口径", "",
        "macOS UI 为真实 QWidget／QPainter／事件循环的 offscreen 渲染，不包含物理屏幕呈现。Windows UI 必须为 `qt_platform=windows`。本设备为 Parallels Windows ARM64、4 个虚拟 CPU、约16 GiB内存；x64 Python／原生扩展经仿真运行，不代表原生 Windows x64 用户设备。窗口尺寸、画布、DPI 按各 JSON 独立列出。"])
    env_rows = []
    for platform, label in PLATFORMS.items():
        doc = docs.get(filename(platform, "manual-statistics")) or docs.get(filename(platform, "ui-stable"))
        if not doc:
            env_rows.append([label, "正式环境待到达", "—", "—", "—"])
            continue
        env = doc["environment"]
        env_rows.append([label, env["platform"], env["python"].splitlines()[0],
            "; ".join(f"{k} {v}" for k, v in env.get("dependencies", {}).items()),
            f"逻辑 CPU {env.get('cpu_count')}；版本来源：" + filename(platform, 'manual-statistics' if docs.get(filename(platform, 'manual-statistics')) else 'ui-stable')])
    table(lines, ["平台", "系统", "Python", "依赖", "基础版本来源（线程逐批另列）"], env_rows)
    thread_rows = []
    for platform, label in PLATFORMS.items():
        for suffix in ('manual-statistics', 'motion', 'ui-stable', 'ui-dynamic', 'ui-joint-stable', 'ui-joint-dynamic',
                       'automatic-2048', 'automatic-4096', 'automatic-6144'):
            name = filename(platform, suffix)
            doc = docs.get(name)
            version = 'R1沿用' if revision != 'r1' and suffix.startswith('automatic-') else revision.upper()
            thread_rows.append([label, f'{version} {link(name)}', *(thread_configuration(doc, suffix) if doc else ('待到达', '—', '—', '—'))])
    table(lines, ["平台", "采样报告", "OpenCV请求", "该脚本报告值", "并行框架", "该报告记录的环境变量"], thread_rows)
    lines.extend(["UI／运动的默认 OpenCV 配置是这两类场景的主验收：`requested=null` 表示未主动调用线程限制，不能解释为1。手工／统计及独立全图使用另一采样脚本，其入口显式请求 `cv2.setNumThreads(1)`；旧JSON未单列请求字段，上表已区分代码依据与实际报告值。该脚本 SHA256 `ee3f3f8837b82ff2ac1712361c35cb1a83d46c7f16181c0f7703f7e44cbfa3fb` 与首轮905文件清单一致；新报告若改变脚本，应另核对该设置。", "",
        "不能将某一行请求或报告值扩展到全场：首轮 Windows 手工／统计报告1，运动与四种UI报告默认4／Concurrency；首轮 macOS UI／运动为默认18／GCD，手工脚本即使请求1也报告18。报告线程数不等于实际同时运行的线程数；主进程环境记录也不能证明全图子进程采用同一个线程数。OMP／BLAS 等环境变量另列，不把其值1等同于OpenCV单线程。未到达的新版条件不沿用其他版本配置猜测。源码摘要、输入摘要、原生编译器／ABI 位于原始 JSON，不能只凭 Git HEAD 判断未提交工作树一致。",
        "", memory_scope, "", "## 手工原生服务与增量统计", "",
        "这些是非 UI 分段计时：完整 BGR 快照复制、局部 ROI 服务及两者合计分别测量，不能替代第二击至结果可见门槛。已完成的原生手工样本要求固定已知宽度 19 px；复制阶段不产出测量状态。统计样本覆盖 1／100／1000／10000 条历史，追加和修改包含摘要与撤销；不包含界面表格或导出。此处没有另行设立事后提速阈值。"])
    for platform, label in PLATFORMS.items():
        name = filename(platform, "manual-statistics")
        doc = docs.get(name)
        lines.extend([f"### {label}", "", link(name) if doc else "待验收：正式报告尚未到达。"])
        if not doc:
            continue
        table(lines, ["宽×高", "阶段", "样本数／状态", "P50 / P95 / P99 / 最大", "采样 RSS 父／子／合计 MiB"],
            [[f"{r['shape'][1]}×{r['shape'][0]}", STAGES[r["stage"]], f"{r['timing']['count']}/30；{counts(r['statuses'])}", four(r["timing"]), memory(r["resources"])] for r in doc["manual"]])
        table(lines, ["历史条数", "操作", "样本数／最终摘要状态", "P50 / P95 / P99 / 最大"],
            [[r["history_count"], STAGES[r["stage"]], f"{r['timing']['count']}/30；{r['status']}", four(r["timing"])] for r in doc["statistics"]])
        lines.extend(["手工／统计脚本预热执行但未逐条保留预热状态，不能据此证明预热零拒收；UI 拒收另有完整计数。", ""])
    lines.extend(["## 运动核验分段", "", "含缩小与运动比较，不含 GUI、捕获、分配和 PNG。基线关闭缓存并用原双精度归一化与 NumPy 峰值搜索；生产路径使用等价查表和 C++ 峰值扫描。结果差异数单列。P99／最大值从每行原始30个样本直接复算；分段未另设硬阈值，端到端门槛仍见 UI。"])
    motion_rows = []
    for platform, label in PLATFORMS.items():
        name = filename(platform, "motion")
        doc = docs.get(name)
        if not doc:
            motion_rows.append([label, "待验收", "—", "—", "—", "—"])
            continue
        lines.extend([link(name), ""])
        for r in doc["records"]:
            def motion_dist(key):
                values = r[key]["samples"]
                return {"p50_ms": audit.percentile(values,.5), "p95_ms": audit.percentile(values,.95), "p99_ms": audit.percentile(values,.99), "max_ms": max(values)}
            motion_rows.append([label, "×".join(map(str,r["size"])), "静止相同帧" if r["scenario"] == "identical" else "动态独立低幅噪声",
                f"{r['samples']}；差异 {r['result_disagreements']}", four(motion_dist("baseline_no_cache_ms")), four(motion_dist("production_ms"))])
    table(lines,["平台","宽×高","场景","样本／差异","基线 P50 / P95 / P99 / 最大","生产 P50 / P95 / P99 / 最大"],motion_rows)
    lines.extend(["## 独立全图服务", "", "所有行分别预热5次、正式30次；返回状态 `ok` 和 `empty` 都表示算法正常结束，空结果是按恢复规则未保留候选，不能当作运行错误。错误、超时、内存拒绝另列。worker 往返包含传递与等待，worker 内服务是诊断分段；相减不同分位数不能作为精确复制成本。30秒硬超时、2 GiB算法预算是资源保护，不是交互延迟验收目标。"])
    if revision != 'r1':
        lines.extend(["", f"**本节全部为 R1 原构建证据。** 全图算法代码未改，因此保留三个尺寸的原独立采样，不将其源码／原生摘要替换成 {revision_label}。新版端到端联合 UI 的全图负载另见本页 {revision_label} 条件；这两类证据不互相冒充。", ""])
    for platform,label in PLATFORMS.items():
        lines.extend([f"### {label}", ""])
        for large in (False, True):
            rows=[]
            for size in (2048,4096,6144):
                name=filename(platform, f"automatic-{size}")
                doc=docs.get(name)
                if not doc:
                    if large:rows.append([f"{size} 批", "待验收", "—", "—", "—", "—", "—", "—"])
                    continue
                for r in doc["automatic"]:
                    if (r["case"] == "large_sparse8") != large:continue
                    candidates=sorted(set(r["candidate_counts"]))
                    rows.append([f"[{size}批]({name})", f"{r['case']} {r['shape'][1]}×{r['shape'][0]}",r["algorithm"],
                        f"{counts(r['statuses'])} / {r['worker_roundtrip']['count']}；错误 {len(r['errors'])}；候选 {candidates}",
                        four(r["worker_roundtrip"]),four(r["service_inside_worker"]),memory(r["resources"]),r["resources"].get("sampled_max_native_threads","—")])
            lines.extend(["大图固定八根样本：" if large else "小图标准样本（每个大图批次单独重跑，不能合并成一批30样本）："])
            table(lines,["独立批次","输入","算法","状态／返回分母","往返 P50 / P95 / P99 / 最大","内服务 P50 / P95 / P99 / 最大","采样 RSS 父／子／合计 MiB","父线程峰"],rows or [["待验收"]+ ["—"]*7])
    lines.extend(["## 真实 Qt 短程交互", "", "以下只用正式30次；成功须既接受又在画布绘制。拒收／未完成与成功分位数并列，预热拒收不混入正式分母。`PASS` 仅适用于本行固定合成条件；静止／动态、PNG开关与后台条件不可互相替代。"])
    for platform,label in PLATFORMS.items():
        lines.extend([f"### {label}", ""])
        platform_rows=[r for r in ui_rows if Path(r["file"]).name in {filename(platform, f"ui-{kind}") for kind in ("stable","dynamic","joint-stable","joint-dynamic")}]
        if not platform_rows:
            lines.extend(["待验收：正式 UI 报告尚未到达。", ""])
            continue
        for kind in ("stable","dynamic","joint-stable","joint-dynamic"):
            name=filename(platform, f"ui-{kind}")
            selected=[r for r in platform_rows if Path(r["file"]).name==name]
            if not selected:continue
            doc=docs[name]
            lines.extend([f"#### {kind}", "", link(name), "",
                f"Qt={doc['qt_platform']}；输入={doc['parameters']['motion_input']}；后台={doc['parameters']['background_automatic'] or '无'}。"])
            table(lines,["宽／PNG","成功并绘制／尝试；失败","预热／未归属拒收","结果可见 P50 / P95 / P99 / 最大","固定规则审计","不通过／缺失项"],
                [[f"{r['width']}／{'开' if r['png'] else '关'}",f"{r['attempts']['success_fraction']}；{r['attempts']['failed_or_unfinished']}",r['attempts']['warmup_or_unassigned_rejection_events'],four(r['metrics']['result_visible']),r['overall_status'],
                    "、".join(CHECKS.get(c['name'],c['name']) for c in r['checks'] if c['status']!='PASS') or "无"] for r in selected])
            details=[]
            for r in selected:
                c=ui_raw[(name,r['width'],r['png'])];gui=c['metrics']['gui_publish_ms'];display=c['display']
                details.append([f"{r['width']}／{'开' if r['png'] else '关'}",four(r['metrics']['draft']),four(r['metrics']['gui_publish']),
                    f"{sum(x>4 for x in gui.get('samples_ms',[]))}/{gui['count']}",f"{f(r['fps']['generated'],2)} / {f(r['fps']['painted'],2)}",
                    f(c['frame_budget_peak_bytes']/2**20,1),memory(c['resources']),
                    f"{display['window_size']}／{display['canvas_size']}；DPR {display['device_pixel_ratio']}；DPI {f(display['logical_dpi'],1)}"])
            table(lines,["宽／PNG","草稿 P50 / P95 / P99 / 最大","GUI P50 / P95 / P99 / 最大","GUI >4ms／样本","生成／不同帧 FPS","帧预算峰 MiB","采样 RSS 父／子／合计 MiB","窗口／画布／DPI"],details)
            for r in selected:
                if r['attempts']['rejection_reasons_all']:
                    lines.extend([f"{r['width']}／PNG {'开' if r['png'] else '关'} 拒收原因（含预热）："+counts(r['attempts']['rejection_reasons_all'])+"。正式失败分类："+counts(r['attempts']['failure_reasons'])+"。", ""])
    lines.extend(["### PNG＋持续 multiscale 的相对影响", "", "仅与同平台、尺寸、输入、窗口／画布／DPI、源码和环境一致的 PNG关闭／无后台基线比较。相对P95通过不代表绝对门槛通过；任何一边有失败，禁止用成功样本比较宣称通过。"])
    relative=[]
    for r in ui_rows:
        if not r['background']:continue
        c=ui_raw[(Path(r['file']).name,r['width'],r['png'])]; rel=r['relative_png_auto']
        relative.append([Path(r['file']).stem.replace('final-',''),r['width'],counts(Counter(x.get('status') for x in c['background_results'])),
            rel['status'],f"{f(rel.get('baseline_p95_ms'))} → {f(rel.get('loaded_p95_ms'))}",f"{f(rel.get('p95_delta_ms'))} / {f(rel.get('p95_allowed_delta_ms'))}",
            f(rel.get('p99_delta_ms')),rel.get('reason','绝对门槛与后台完整性仍见上表')])
    table(lines,["报告","宽","后台返回状态","相对判定","基线→联合P95","增量／允许增量","P99增量（诊断）","说明"],relative or [["待验收"]+["—"]*7])
    background_rows = [r for r in ui_rows if r['background']]
    table(lines,["报告／宽","预热正常完成","正式正常完成（旧版未分期）","收尾正常完成","阶段响应数"],
        [[Path(r['file']).stem.replace('final-','')+'／'+str(r['width'])] +
         [r.get('background_completion',{}).get(key,'—') for key in
            ('warmup_normal_completed','normal_completed','teardown_normal_completed')] +
         [counts(r.get('background_completion',{}).get('phase_counts',{})) if r.get('background_completion',{}).get('phase_trace_valid') is not None else '旧版无阶段记录']
            for r in background_rows] or [["待验收"]+["—"]*4])
    table(lines,["报告／宽","取消总数","显式收尾取消","推断收尾取消","未归因取消","后台错误","阶段轨迹有效"],
        [[Path(r['file']).stem.replace('final-','')+'／'+str(r['width'])] + [r.get('background_completion',{}).get(key,'—') for key in
            ('cancelled_total','teardown_cancel_explicit','teardown_cancel_inferred','unexplained_cancelled','background_errors')] +
         [r.get('background_completion',{}).get('phase_trace_valid') if r.get('background_completion',{}).get('phase_trace_valid') is not None else '旧版无阶段记录']
            for r in background_rows] or [["待验收"]+["—"]*6])
    for row in background_rows:
        errors = row.get('background_completion', {}).get('phase_trace_errors', [])
        if errors:
            lines.extend([f"{Path(row['file']).name}／{row['width']} 阶段轨迹错误：" + '；'.join(errors), ''])
    phase_versions = "R2／R3／R4" if revision == 'r4' else "R2／R3"
    lines.extend([f"后台 `empty` 是正常无候选返回。`cancelled` 是未完成任务，始终保留计数，不改成成功。{phase_versions} 的预热结果单独存储；每条响应记录 `phase`、`task_id` 和接收单调时间，审计先核验阶段边界、时间顺序及任务 ID，再核对收尾取消是否属于明确请求关闭的任务。只有有效轨迹匹配时才列为**显式收尾取消**；正式正常完成只计 formal，预热与收尾正常返回另列。轨迹缺项／冲突不得退回旧版推断来豁免失败。", "",
        "R1 无阶段记录，仍只在报告中harness源码SHA与已核查版本完全一致、整批唯一取消且为末项、显式 `error:null` 与 `candidates:0`、此前至少一次正常返回时，单列为**源码约束下推断的收尾取消**。已核查旧代码在 `finally` 关闭后台worker，后续 `wait_for` 仍可能向返回对象共享的列表追加取消；这不是旧JSON的运行时阶段证明。R1 的正常完成数没有预热／正式／收尾分期，不能冒充新版 formal 数。所有未归因取消仍失败；两种归因都不改变手工测量失败、超时或背压分母。", "", "## 一小时真实工作区联合运行", "",
        f"只接收本页版本加载清单中包含 `counters` 和 `latency_by_fifteen_minutes` 的真实 Qt 报告；新版文件按版本固定为 `{hour_name}`（本页未列入加载清单的报告不引用）。早期 `soak-one-hour-macos.json` 不进入本节。每15分钟独立看尾延迟与发布最大值；至少10000条实际入账、零串场、提交／接受／终结守恒、PNG摘要／像素、账本往返、撤销历史和关闭资源均是门槛。"])
    soaks=[(name,doc) for name,doc in docs.items() if 'counters' in doc and 'latency_by_fifteen_minutes' in doc]
    if not soaks:lines.extend(["**待验收：最终真实 Qt 一小时报告尚未到达。** 不以早期控制器测试或短程通过代替。", ""])
    for name,doc in soaks:
        row=audit.soak_case(HERE/name,doc)
        lines.extend([f"### {name}", "", link(name), "",f"运行 {f(row['active_seconds'],1)} 秒；脚本 passed={row['script_passed_flag']}；固定规则判定 **{row['overall_status']}**。"])
        table(lines,["分母／状态","计数"],[[key,value] for key,value in row['attempts'].items()])
        table(lines,["时段","草稿 P50 / P95 / P99 / 最大","结果可见 P50 / P95 / P99 / 最大","GUI P50 / P95 / P99 / 最大","判定"],
            [[f"{q['start_seconds']/60:g}–{q['end_seconds']/60:g} min",four(q['metrics']['draft']),four(q['metrics']['result_visible']),four(q['metrics']['gui_publish']),q['status']] for q in row['quarters']])
        table(lines,["指标","结果"],[["不同帧绘制 FPS",f(row['fps']['painted'],2)],["实际生成 FPS",f(row['fps']['generated'],2)],
            ["输入新场景→清场可见 P50/P95/P99/max",four(row['metrics']['scene_to_clear_paint'])],
            ["帧预算峰 MiB",f(row['memory']['frame_budget_peak_bytes']/2**20,1)],
            ["采样 RSS 父／子／合计 MiB",memory(doc.get('resources',{}))],["队列峰值",json.dumps(row['queue_max'],ensure_ascii=False)],
            ["不通过／未完成检查","；".join(x['name']+'='+x['status'] for x in row['checks'] if x['status']!='PASS') or '无']])
        lines.extend(["本节读取报告已记录的原图／账本一致性检查，不再次读取或哈希PNG，也不启动工作区。", ""])
    lines.extend(["## 未验收事项与复算", "",
        "缺失的平台／尺寸／负载组合维持待验收；单独内核、静止相等缓存、正常空候选和安全拒收有不同含义。不能通过降低运动阈值、放宽最新已显示帧核验、停止实时画面，或剔除失败样本换取通过。真实相机／采集卡、真实CU画面、曝光变化／失焦／振动和物理计量均需独立验证。",
        "", "本页由同目录 `summarize_final_performance.py` 仅读取已有JSON生成，UI／一小时判定复用 `scripts/validation/audit_cu_live_performance.py`；运行聚合不会启动测试、Qt、基准、子进程或网络。新正式文件到达后重新生成，并保留所有失败。", "", "```sh", "python docs/validation/cu-live-measurement-2026-10-15/summarize_final_performance.py --revision " + revision, "```", "", "输入文件摘要（保证本页所引用的是这一批结果）："])
    table(lines,["文件","SHA256"],[[link(name),hashlib.sha256((HERE/name).read_bytes()).hexdigest()] for name in docs])
    formatted = []
    for line in lines:
        if line.startswith('#') and formatted and formatted[-1]:
            formatted.append('')
        formatted.append(line)
    (HERE/output_name).write_text('\n'.join(formatted)+'\n')
    print(json.dumps({'output':str(HERE/output_name),'revision':revision,'final_reports':len(docs),'ui_conditions':len(ui_rows),'real_qt_soaks':len(soaks),'invalid_files':invalid},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', choices=('r1', 'r2', 'r3', 'r4'), default='r1')
    generate(parser.parse_args().revision)
