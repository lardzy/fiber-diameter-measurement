// Every spoken beat has an explicit, auditable visual target. Coordinates are
// resolved from captured Qt controls or image-space geometry before rendering.
const cue = (label, target, action = "inspect", extra = {}) => ({
  label,
  target,
  action,
  ...extra,
});
const B = (label, button, action = "click") => cue(label, { button }, action);
const R = (label, box, action = "inspect") => cue(label, { box }, action);
const D = (label, dialog, action = "inspect") => cue(label, { dialog }, action);
const I = (label, image, action = "inspect") => cue(label, { image }, action);
const C = (label, control, action = "select") =>
  cue(label, { control }, action);
const P = (label, path, action = "drag", extra = {}) =>
  cue(label, { path }, action, extra);
const S = (label = "检查标定状态") => R(label, [1400, 110, 186, 44]);
const rows = (label) => R(label, [12, 697, 708, 109]);
const list = (label) => R(label, [12, 245, 238, 195]);
const outer = [
  [220, 160],
  [490, 130],
  [620, 300],
  [570, 520],
  [360, 570],
  [200, 430],
  [220, 160],
];
const hole = [
  [345, 265],
  [430, 250],
  [468, 330],
  [395, 382],
  [335, 335],
  [345, 265],
];
const ellipse = Array.from({ length: 25 }, (_, i) => [
  900 + 150 * Math.cos((i * 2 * Math.PI) / 24),
  310 + 130 * Math.sin((i * 2 * Math.PI) / 24),
]);
const seed = (label, action = "inspect") =>
  cue(label, { point: "seed", size: 140 }, action);
const object = (label, index = 0) => cue(label, { object: index });
const guides = {
  "01-intro": [
    [I("目标框指向讲解位置", [200, 190, 840, 230])],
    [rows("这里核对测量结果")],
  ],
  "01-open": [
    [B("单击顶部“打开”", "打开")],
    [
      D("选择练习图片", [135, 80, 610, 370], "select"),
      B("打开所选图片", "Open"),
    ],
    [list("确认当前图片"), S("未标定：先完成比例标定")],
  ],
  "01-calibrate": [
    [B("单击“标定”工具", "标定"), I("找到已知比例尺", [780, 720, 445, 83])],
    [
      P("按住左键：从左端拖到右端", [
        [800, 744],
        [1200, 744],
      ]),
    ],
    [
      D("输入真实长度 100", [120, 8, 139, 36], "type"),
      D("单位选择 µm", [120, 49, 139, 35], "select"),
      B("确认标定", "OK"),
    ],
    [S("状态已变为“已标定”")],
  ],
  "01-measure": [
    [
      B("选择“手动线段”", "手动线段"),
      I("找到同一根纤维的两侧", [195, 215, 115, 75]),
    ],
    [
      P("垂直跨过纤维：按住并拖动", [
        [212, 250],
        [288, 250],
      ]),
    ],
    [
      object("复核第一条测量线", 0),
      object("复核其他测量位置", 1),
      object("检查数值标签", 2),
    ],
  ],
  "01-review": [
    [B("打开“结果”面板", "结果")],
    [rows("同时核对类别、数值和单位")],
    [
      R("切换“统计”页", [71, 623, 50, 35], "click"),
      R("查看数量、均值和离散程度", [14, 670, 1000, 142]),
    ],
  ],
  "01-save": [
    [B("单击保存项目", "保存")],
    [R("确认项目已保存", [400, 20, 176, 38]), list("原图与项目一起保留")],
  ],
  "01-export": [
    [B("单击“导出”", "导出")],
    [
      B("勾选 Excel 表格", "Excel 文档", "select"),
      B("勾选 CSV 明细", "CSV 文档", "select"),
    ],
    [
      B("选择测量叠加图", "测量叠加图", "select"),
      B("确认当前图片范围", "当前图片", "select"),
      B("保存导出文件", "保存"),
    ],
  ],
  "01-outro": [
    [
      S("先确认标定"),
      I("再复核测量位置", [200, 190, 840, 230]),
      B("完成后保存项目", "保存", "inspect"),
    ],
  ],
  "02-intro": [
    [list("多张图片集中管理")],
    [I("每张图片分别保存测量", [190, 190, 850, 330])],
  ],
  "02-open": [
    [B("从顶部打开图片", "打开")],
    [D("多选这一批图片", [135, 80, 680, 400], "select"), B("确认打开", "Open")],
    [
      list("左侧是图片清单"),
      R("也可单击顶部标签切换", [275, 162, 460, 35], "click"),
    ],
  ],
  "02-switch": [
    [R("选择样本 A", [13, 259, 231, 22], "click"), rows("样本 A：一条记录")],
    [R("切换样本 B", [13, 280, 231, 23], "click"), rows("样本 B：两条记录")],
  ],
  "02-save": [
    [
      C("填写项目名称", { name: "fileNameEdit" }, "type"),
      B("保存 .fdmproj 项目", "Save"),
    ],
    [R("查看保存状态", [400, 20, 176, 38]), S("标定与记录随项目保存")],
  ],
  "02-reopen": [
    [list("重新打开后检查图片清单")],
    [rows("再核对恢复的测量记录")],
  ],
  "02-backup": [
    [cue("项目与原图一起备份", { card: 0 }), cue("保留图片关联", { card: 1 })],
    [cue("配套 assets 目录一起保留", { card: 2 })],
  ],
  "02-outro": [[list("确认当前图片"), rows("核对恢复后的结果")]],
  "03-intro": [
    [S("物理尺寸来自有效标定")],
    [D("预设名称记录成像条件", [144, 10, 202, 34])],
  ],
  "03-calibrate": [
    [
      D("真实长度：100", [120, 8, 139, 36], "type"),
      D("单位：µm", [120, 49, 139, 35], "select"),
      B("确认当前图标定", "OK"),
    ],
    [S("确认已标定状态")],
  ],
  "03-preset": [
    [B("标定面板：新增预设", "新增预设")],
    [D("填写相机与分辨率", [144, 10, 202, 34], "type")],
    [
      D("像素距离：400", [144, 48, 202, 34]),
      D("实际距离：100 µm", [144, 86, 202, 73]),
      B("保存预设", "OK"),
    ],
  ],
  "03-current": [
    [
      C("选择同条件的预设", { text: "教学相机", type: "Combo" }),
      B("单击“应用预设”", "应用预设"),
    ],
    [B("只应用到当前图片", "当前图片")],
    [S("查看已标定状态"), object("核对换算后的测量", 0)],
  ],
  "03-project": [
    [B("应用到项目所有图片", "项目所有图片")],
    [S("各图片重新换算"), list("仅用于相同成像条件")],
  ],
  "03-units": [
    [
      D("像素距离仍是 400", [144, 48, 202, 34]),
      D("实际距离改为 0.1", [144, 86, 202, 34], "type"),
      D("单位切换为 mm", [144, 124, 202, 34], "select"),
    ],
    [rows("数值和 mm 单位一起读取")],
  ],
  "03-outro": [
    [
      cue("更换条件后重新验证", { card: 0 }),
      cue("确认输出分辨率", { card: 1 }),
    ],
    [cue("最后核对范围与单位", { card: 2 })],
  ],
  "m-intro": [
    [R("先从类别管理开始", [12, 506, 241, 226])],
    [C("已有对象也要核对类别", { text: "复核样本", type: "Combo" })],
  ],
  "m-category": [
    [
      D("填写类别名称", [64, 10, 126, 34], "type"),
      D("选择类别颜色", [64, 49, 126, 34], "select"),
      B("确认新增类别", "OK"),
    ],
    [
      C("新测量归入这个类别", { name: "quickMeasurementGroup" }),
      object("这条新测量使用新类别", -1),
    ],
    [
      C("修改当前对象自己的类别", {
        text: "复核样本",
        type: "Combo",
        last: true,
      }),
    ],
  ],
  "m-review": [
    [
      C("输入类别名称筛选", { text: "复核样本", type: "LineEdit" }, "type"),
      rows("只复核筛选后的记录"),
    ],
    [
      R("单击一条结果记录", [16, 725, 700, 28], "click"),
      object("回到对应的测量位置", 1),
    ],
    [
      R("切换统计页", [72, 623, 47, 35], "click"),
      R("数量、均值与离散程度", [14, 670, 1000, 142]),
    ],
    [
      R("查看分布页", [120, 623, 48, 35], "click"),
      R("异常值仍需回到原图复核", [65, 670, 1200, 151]),
    ],
  ],
  "m-polyline": [
    [
      R("手动线段旁的工具菜单", [103, 66, 118, 36], "select"),
      P(
        "沿路径逐点单击",
        [
          [420, 120],
          [420, 285],
          [465, 395],
          [445, 545],
        ],
        "path",
      ),
    ],
    [
      B("单击“完成”确认路径", "完成"),
      I("核对整条折线与累计长度", [390, 95, 110, 490]),
    ],
  ],
  "m-snap": [
    [
      B("选择“边缘吸附”", "边缘吸附"),
      P(
        "在两侧边缘各单击一次",
        [
          [550, 520],
          [650, 520],
        ],
        "path",
      ),
    ],
    [I("检查吸附后的两端", [520, 485, 160, 90])],
    [B("选错时单击“撤回”", "撤回")],
  ],
  "m-endpoint": [
    [
      B("切到浏览模式", "浏览"),
      P("按住端点，拖动修正", [
        [288, 250],
        [303, 250],
      ]),
    ],
    [B("撤回刚才的端点修改", "撤回"), object("检查恢复后的测量线", 0)],
  ],
  "m-quick": [
    [B("选择“快速测径”", "快速测径"), seed("在纤维内部单击", "click")],
    [seed("检查预览是否贴合目标")],
    [B("单击“完成”确认测量", "完成"), object("复核代表测量线", 0)],
  ],
  "m-quick-review": [
    [seed("检查交叉和粘连位置")],
    [rows("核对测量记录"), S("未标定时只能按像素解释")],
  ],
  "m-area": [
    [B("选择多边形面积", "多边形面积"), P("沿外轮廓逐点单击", outer, "path")],
    [
      B("单击“完成”闭合区域", "完成"),
      I("检查闭合边界和面积", [180, 110, 460, 485]),
    ],
  ],
  "m-hole": [
    [
      I("先选中已有面积对象", [180, 110, 460, 485], "click"),
      B("切换为“剔除”", "剔除(T)"),
    ],
    [
      P("沿孔洞边缘逐点圈选", hole, "path"),
      I("核对孔洞和净面积", [315, 230, 177, 170]),
    ],
  ],
  "m-freehand": [
    [
      R("当前工具：自由形状面积", [226, 66, 141, 36]),
      P("按住鼠标沿边缘描绘", ellipse),
    ],
    [I("松开后检查闭合轮廓", [730, 155, 345, 315])],
    [
      B("切换“计数”工具", "计数"),
      P(
        "每个对象只单击一次",
        [
          [400, 450],
          [900, 310],
          [900, 610],
        ],
        "path",
        { connect: false },
      ),
    ],
  ],
  "m-magic": [
    [B("选择“标准魔棒”", "标准魔棒"), seed("目标内部添加正点", "click")],
    [
      B("切换到负采样", "负采样(R)"),
      cue("在误选区域添加负点", { point: "negative", size: 115 }, "click"),
    ],
    [B("确认正确的分割轮廓", "完成"), object("查看生成的面积对象", 1)],
  ],
  "m-reference": [
    [B("选择“同类扩选”", "同类扩选"), seed("点击已确认对象作为参考", "click")],
    [
      I("查看六个候选轮廓", [115, 100, 1000, 540]),
      B("核对后单击“加入”", "加入"),
    ],
    [rows("批量加入后逐条复核")],
  ],
  "m-outro": [
    [
      cue("先复核类别与单位", { card: 0 }),
      cue("再检查位置与轮廓", { card: 1 }),
    ],
    [cue("确认之后保存项目", { card: 2 })],
  ],
  "09-intro": [
    [I("交付图包含测量与标注", [210, 95, 1060, 250])],
    [S("开始前确认标定有效")],
  ],
  "09-scale": [
    [
      C("自定义长度：50", { text: "50", type: "LineEdit" }, "type"),
      C("位置：右上角", { text: "右上角", type: "Combo" }),
    ],
    [
      C("选择实心条样式", { text: "实心条", type: "Combo" }),
      C("设置线宽", { text: "8 px", type: "LineEdit" }, "type"),
      C("设置字号", { text: "28 px", type: "LineEdit" }, "type"),
    ],
    [
      B("确认比例尺编辑", "完成编辑"),
      I("检查比例尺最终位置", [1020, 4, 255, 98]),
    ],
  ],
  "09-annotations": [
    [
      { ...B("先选择文字工具", "文字"), shot: "09-scale-finished" },
      D("输入简短的文字说明", [10, 32, 246, 180], "type"),
      B("确认文字内容", "OK"),
    ],
    [
      B("选择箭头工具", "箭头"),
      P("按住并拖向说明位置", [
        [520, 185],
        [300, 260],
      ]),
    ],
  ],
  "09-export": [
    [
      B("勾选 Excel", "Excel 文档", "select"),
      B("勾选 CSV", "CSV 文档", "select"),
    ],
    [
      B("选择测量叠加图", "测量叠加图", "select"),
      B("选择测量与比例尺组合图", "测量 + 比例尺叠加图", "select"),
      B("勾选包含标注", "包含标注", "select"),
    ],
    [
      B("确认导出范围", "当前图片", "select"),
      C("确认完整分辨率", { text: "完整分辨率", type: "Combo" }),
      B("保存导出文件", "保存"),
    ],
  ],
  "09-output": [
    [
      I("核对 50 µm 比例尺", [1040, 8, 235, 78]),
      I("核对文字与箭头", [280, 90, 270, 205]),
      I("核对测量标签", [190, 190, 850, 220]),
    ],
    [I("图片中的测量应与表格一致", [190, 190, 850, 220])],
  ],
  "09-outro": [
    [
      B("项目留作继续编辑", "保存", "inspect"),
      B("图表用于交付表达", "导出", "inspect"),
      S("输出前再核对单位"),
    ],
  ],
};

export function refineStoryboards(videos) {
  const scene = (id) =>
    videos.flatMap((v) => v.scenes).find((s) => s.id === id);
  const open = scene("02-open");
  open.beats.unshift({
    text: "先在顶部工具栏单击打开，开始导入这一批图片。",
    shot: "02-image-b",
    focus: "top",
    target: null,
  });
  open.beats[1].text = "在文件窗口中多选这一批图片，然后确认打开。";
  const preset = scene("03-preset");
  preset.beats.unshift({
    text: "展开右侧标定区域，单击新增预设。",
    shot: "03-preset-panel",
    focus: "right",
    target: null,
  });
  preset.beats[1].text = "在名称中记录相机和输出分辨率等条件。";
  scene("03-current").beats[0].text =
    "切到同条件的另一张图片，在右侧选择预设，再单击应用预设。";
  scene("09-scale").beats[2].shot = "09-scale-style";
  scene("09-scale").beats[2].text =
    "单击完成编辑，再检查最终位置。显示范围和导出图片范围需要分别确认。";
  scene("m-quick").beats[2].shot = "08-quick-contour";
  scene("m-quick").beats[2].text =
    "轮廓合适时单击完成，形成测量记录；随后到记录列表复核。";
  guides["m-quick"][2][1] = {
    ...object("复核生成的代表测量线", 0),
    shot: "08-quick-result",
  };
  scene("m-magic").beats[2].shot = "07-negative-preview";
  guides["m-magic"][2][1] = {
    ...object("查看确认后的面积对象", 1),
    shot: "07-area-result",
  };
  guides["m-polyline"][1][0].shot = "05-polyline-preview";
  guides["m-area"][1][0].shot = "06-polygon-preview";
  guides["09-scale"][2][1].shot = "09-scale-finished";
  for (const v of videos) {
    v.version = 2;
    v.filename = v.filename.replace(/\.mp4$/, "-v2.mp4");
    v.scenes.forEach((s, index) => {
      s.chapterIndex ??=
        v.id === "FDM-Measurements"
          ? Math.max(
              0,
              v.chapters.findIndex((c) => c.startsWith(s.chapter)),
            )
          : Math.min(v.chapters.length - 1, Math.max(0, index - 1));
      if (!guides[s.id] || guides[s.id].length !== s.beats.length)
        throw new Error(`Missing guidance plan: ${s.id}`);
      s.beats.forEach((b, i) => {
        b.guidance = guides[s.id][i];
      });
    });
  }
  return videos;
}

export function resolveGuidance(video, captures) {
  const audits = [];
  for (const s of video.scenes)
    for (const [i, b] of s.beats.entries()) {
      const capture = captures[b.shot];
      if (!capture) throw new Error(`Unknown captured shot ${b.shot}`);
      const artifact = s.kind === "artifact";

      const boxOf = (points, pad = 12) => {
        const xs = points.map((p) => p[0]),
          ys = points.map((p) => p[1]);
        return [
          Math.min(...xs) - pad,
          Math.min(...ys) - pad,
          Math.max(1, Math.max(...xs) - Math.min(...xs)) + 2 * pad,
          Math.max(1, Math.max(...ys) - Math.min(...ys)) + 2 * pad,
        ];
      };
      const steps = b.guidance.map((g) => {
        const shot = g.shot ?? b.shot;
        const c = captures[shot];
        const map = (p) =>
          artifact
            ? p
            : [
                c.image_transform.origin[0] + p[0] * c.image_transform.scale,
                c.image_transform.origin[1] + p[1] * c.image_transform.scale,
              ];
        const t = g.target;
        let rect, points;
        if (t.card !== undefined) {
          if (t.card >= s.cards.length) throw new Error(`Missing card ${s.id}`);
          return { label: g.label, action: g.action, card: t.card };
        }
        if (t.button)
          rect = c.dialog_buttons?.[t.button] ?? c.buttons?.[t.button];
        if (t.box) rect = t.box;
        if (t.dialog) {
          if (!c.dialog) throw new Error(`No dialog: ${b.shot}`);
          rect = [
            c.dialog[0] + t.dialog[0],
            c.dialog[1] + t.dialog[1],
            t.dialog[2],
            t.dialog[3],
          ];
        }
        if (t.control) {
          const q = t.control;
          const matches = (c.controls ?? []).filter(
            (x) =>
              (!q.type || x.type.includes(q.type)) &&
              (!q.name || x.name === q.name) &&
              (!q.text || x.text.includes(q.text)) &&
              x.rect[2] > 0 &&
              x.rect[1] >= 0 &&
              x.rect[1] + x.rect[3] <= 900,
          );
          rect = (q.last ? matches.at(-1) : matches[0])?.rect;
        }
        if (t.image) {
          const [x, y, w, h] = t.image;
          rect = boxOf([map([x, y]), map([x + w, y + h])], 0);
        }
        if (t.point) {
          const p = c.image_points?.[t.point];
          if (p) rect = [p[0] - t.size / 2, p[1] - t.size / 2, t.size, t.size];
        }
        if (t.path) {
          points = t.path.map(map);
          rect = boxOf(points, 22);
        }
        if (t.object !== undefined) {
          const o = c.objects?.at(t.object);
          if (o) {
            const [x, y, w, h] = o.bounds;
            rect = boxOf([map([x, y]), map([x + w, y + h])], 25);
          }
        }
        if (!rect)
          throw new Error(
            `Unresolved guide ${s.id}/${i + 1} ${g.label}: ${JSON.stringify(t)}`,
          );
        const [x, y, w, h] = rect;
        const W = artifact ? 1280 : 1600,
          H = artifact ? 820 : 900;
        if (x < 0 || y < 0 || x + w > W + 1 || y + h > H + 1)
          throw new Error(`Out of bounds ${s.id}/${i + 1}: ${g.label} ${rect}`);
        const screen = {
          file: artifact
            ? s.artifact
            : `screens/${shot.startsWith("base/") ? shot.slice(5) : `series/${shot}`}.png`,
          width: artifact ? 1280 : 1600,
          height: artifact ? 820 : 900,
        };
        return {
          label: g.label,
          action: g.action,
          rect,
          points,
          connect: g.connect ?? true,
          screen,
        };
      });
      b.guide = {
        steps,
        screen: {
          file: artifact
            ? s.artifact
            : `screens/${b.shot.startsWith("base/") ? b.shot.slice(5) : `series/${b.shot}`}.png`,
          width: artifact ? 1280 : 1600,
          height: artifact ? 820 : 900,
        },
      };
      delete b.guidance;
      audits.push({
        scene: s.id,
        beat: i + 1,
        shot: b.shot,
        text: b.text,
        targets: steps.length,
        labels: steps.map((s) => s.label),
      });
    }
  return audits;
}
