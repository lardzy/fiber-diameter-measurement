import type { Caption } from "@remotion/captions";
export type Rect = [number, number, number, number];
export type XY = [number, number];
export type GuideScreen = { file: string; width: number; height: number };
export type GuideStep = {
  label: string;
  action: string;
  rect?: Rect;
  points?: XY[];
  connect?: boolean;
  card?: number;
  screen?: GuideScreen;
};
export type Beat = {
  text: string;
  shot: string;
  from: number;
  duration: number;
  hold: number;
  audio: string;
  guide: { steps: GuideStep[]; screen: GuideScreen };
};
export type SceneData = {
  id: string;
  chapter: string;
  chapterIndex: number;
  title: string;
  subtitle: string;
  notes: string[];
  tip: string;
  beats: Beat[];
  from: number;
  duration: number;
  kind?: string;
  source?: string;
  artifact?: string;
  cards?: { label: string; title: string; detail: string }[];
};
export type VideoData = {
  id: string;
  number: string;
  title: string;
  filename: string;
  chapters: string[];
  scenes: SceneData[];
  duration: number;
  captions: Caption[];
  version: number;
};
export const actions: Record<string, string> = {
  inspect: "观察核对",
  click: "单击左键",
  select: "选择选项",
  type: "输入内容",
  drag: "按住并拖动",
  path: "逐点单击",
};
export function guideAt(scene: SceneData, frame: number) {
  const beatIndex = Math.max(
    0,
    scene.beats.findLastIndex((b) => b.from <= frame),
  );
  const beat = scene.beats[beatIndex];
  const span = beat.duration + beat.hold;
  const slice = span / beat.guide.steps.length;
  const stepIndex = Math.min(
    beat.guide.steps.length - 1,
    Math.floor(Math.max(0, frame - beat.from) / slice),
  );
  const step = beat.guide.steps[stepIndex];
  const previous =
    stepIndex > 0
      ? beat.guide.steps[stepIndex - 1]
      : scene.beats[beatIndex - 1]?.guide.steps.at(-1);
  return {
    beat,
    beatIndex,
    step,
    stepIndex,
    previous,
    local: Math.max(0, frame - beat.from - stepIndex * slice),
    span: slice,
  };
}
