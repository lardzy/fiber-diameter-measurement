import { Audio } from "@remotion/media";
import {
  AbsoluteFill,
  Sequence,
  Series,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
} from "remotion";
import source from "./data.json";
import { GuidedScreen } from "./GuidedScreen";
import { GuidedCaptions } from "./GuidedCaptions";
import { actions, guideAt, type SceneData, type VideoData } from "./types";
export const tutorials = source as VideoData[];
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const ink = "#163c33",
  muted = "#59756a",
  accent = "#238b73";
const Board: React.FC<{ scene: SceneData }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { step } = guideAt(scene, frame);
  return (
    <div
      style={{
        position: "absolute",
        left: 492,
        top: 151,
        width: 1348,
        height: 743,
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}
    >
      {scene.cards?.map((card, i) => {
        const active = step.card === i;
        const p = spring({
          frame: frame - i * 7,
          fps: 30,
          config: { damping: 200 },
        });
        return (
          <div
            key={card.label}
            style={{
              height: 207,
              boxSizing: "border-box",
              background: active ? "#fff7e7" : "#f5f9f6",
              border: `${active ? 3 : 1}px solid ${active ? "#e9aa32" : "#c5d8cd"}`,
              borderRadius: 22,
              display: "flex",
              alignItems: "center",
              padding: "28px 46px",
              gap: 38,
              opacity: p * (active ? 1 : 0.64),
              translate: `0px ${(1 - p) * 20}px`,
            }}
          >
            <div
              style={{
                width: 89,
                height: 89,
                borderRadius: 50,
                background: active ? "#f7c35f" : "#d0e5d8",
                color: ink,
                display: "grid",
                placeItems: "center",
                fontSize: 38,
                fontWeight: 800,
              }}
            >
              {card.label}
            </div>
            <div>
              <div style={{ fontSize: 40, fontWeight: 750, marginBottom: 13 }}>
                {card.title}
              </div>
              <div style={{ fontSize: 28, color: muted }}>{card.detail}</div>
            </div>
            {active && (
              <span
                style={{
                  marginLeft: "auto",
                  fontSize: 26,
                  fontWeight: 750,
                  color: "#986510",
                }}
              >
                当前重点
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
};
export const TutorialScene: React.FC<{
  scene: SceneData;
  video: VideoData;
  index: number;
}> = ({ scene, video, index }) => {
  const frame = useCurrentFrame();
  const state = guideAt(scene, frame);
  const { beat, step, stepIndex } = state;
  const opacity = interpolate(
    frame,
    [0, 9, scene.duration - 8, scene.duration - 1],
    [0, 1, 1, 0],
    clamp,
  );
  const isReal = scene.source === "real";
  return (
    <AbsoluteFill
      style={{
        background: "#edf3ef",
        backgroundImage:
          "radial-gradient(ellipse at 85% 12%,#dceee5 0%,transparent 58%)",
        fontFamily: "FDM Noto, PingFang SC, sans-serif",
        color: ink,
      }}
    >
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 41,
          fontSize: 30,
          fontWeight: 850,
          letterSpacing: -1,
        }}
      >
        FDM.
      </div>
      <div
        style={{
          position: "absolute",
          left: 178,
          top: 48,
          width: 1,
          height: 29,
          background: "#b8cec2",
        }}
      />
      <div
        style={{
          position: "absolute",
          left: 203,
          top: 46,
          fontSize: 22,
          color: muted,
        }}
      >
        操作教程 · V2 引导增强版
      </div>
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 37,
          border: "1px solid #a9c9b8",
          borderRadius: 40,
          padding: "11px 24px",
          fontSize: 20,
          color: "#2b6856",
        }}
      >
        {video.number} · {video.title}
      </div>
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 112,
          width: 1760,
          height: 1,
          background: "#c6d8ce",
        }}
      />
      <AbsoluteFill style={{ opacity }}>
        <div
          style={{
            position: "absolute",
            left: 82,
            top: 164,
            width: 350,
            fontSize: 23,
            color: accent,
            fontWeight: 700,
          }}
        >
          步骤 {String(index + 1).padStart(2, "0")} /{" "}
          {String(video.scenes.length).padStart(2, "0")}{" "}
          <span style={{ color: "#88a094", fontWeight: 400 }}>
            · {scene.chapter}
          </span>
        </div>
        <div
          style={{
            position: "absolute",
            left: 78,
            top: 239,
            width: 365,
            fontSize: 58,
            lineHeight: 1.23,
            fontWeight: 850,
            letterSpacing: -1.5,
            whiteSpace: "pre-line",
            translate: `0px ${(1 - spring({ frame, fps: 30, config: { damping: 200 } })) * 12}px`,
          }}
        >
          {scene.title}
        </div>
        <div
          style={{
            position: "absolute",
            left: 82,
            top: 413,
            width: 350,
            fontSize: 26,
            lineHeight: 1.6,
            color: muted,
          }}
        >
          {scene.subtitle}
        </div>
        <div
          style={{
            position: "absolute",
            left: 80,
            top: 546,
            width: 345,
            minHeight: 204,
            boxSizing: "border-box",
            padding: "19px 22px 22px",
            border: "1px solid #e4bf72",
            borderRadius: 18,
            background: "#fff8e9",
            boxShadow: "0 8px 20px #7953160b",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: 20,
              color: "#986a17",
              marginBottom: 15,
            }}
          >
            <span>
              {actions[step.action]}
              {scene.kind === "board" ? " · 要点" : " · 现在看这里"}
            </span>
            <b>
              {stepIndex + 1}/{beat.guide.steps.length}
            </b>
          </div>
          <div
            style={{
              fontSize: 30,
              lineHeight: 1.4,
              fontWeight: 780,
              color: "#4a340e",
            }}
          >
            {step.label}
          </div>
          <div
            style={{
              height: 3,
              background: "#ead8b3",
              marginTop: 20,
              borderRadius: 4,
            }}
          >
            <div
              style={{
                height: 3,
                width: `${Math.min(100, (state.local / state.span) * 100)}%`,
                background: "#dc9a25",
                borderRadius: 4,
              }}
            />
          </div>
        </div>
        <div
          style={{
            position: "absolute",
            left: 82,
            top: 794,
            width: 343,
            borderLeft: "3px solid #8cb69e",
            paddingLeft: 17,
            boxSizing: "border-box",
            fontSize: 22,
            lineHeight: 1.6,
            color: muted,
          }}
        >
          {scene.tip}
        </div>
        {scene.kind === "board" ? (
          <Board scene={scene} />
        ) : (
          <GuidedScreen scene={scene} />
        )}
        <div
          style={{
            position: "absolute",
            left: 503,
            top: 914,
            width: 1336,
            display: "flex",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          {video.chapters.map((chapter, i) => (
            <div
              key={chapter}
              style={{
                display: "flex",
                gap: 9,
                alignItems: "center",
                fontSize: 20,
                color: i === scene.chapterIndex ? "#176b56" : "#829b8d",
                fontWeight: i === scene.chapterIndex ? 750 : 450,
              }}
            >
              <span
                style={{
                  width: 24,
                  height: 24,
                  borderRadius: 15,
                  display: "grid",
                  placeItems: "center",
                  background: i === scene.chapterIndex ? accent : "#dce7df",
                  color: i === scene.chapterIndex ? "white" : "#8ea699",
                  fontSize: 15,
                }}
              >
                {i + 1}
              </span>
              {chapter}
            </div>
          ))}
        </div>
      </AbsoluteFill>
      <GuidedCaptions captions={video.captions} frame={scene.from + frame} />
      <div
        style={{
          position: "absolute",
          left: 82,
          bottom: 24,
          fontSize: 16,
          color: "#748b80",
        }}
      >
        FDM 0.4.7 ·{" "}
        {isReal
          ? "显微测试图 · 未标定，单位 px / px²"
          : "合成教学图像 · 非实测样品"}
      </div>
      <div
        style={{
          position: "absolute",
          right: 80,
          bottom: 31,
          width: 260,
          height: 3,
          background: "#cadbd1",
        }}
      >
        <div
          style={{
            width: `${(100 * (scene.from + frame)) / video.duration}%`,
            height: 3,
            background: accent,
          }}
        />
      </div>
      {scene.beats.map((b) => (
        <Sequence
          key={b.audio}
          from={b.from}
          durationInFrames={b.duration}
          layout="none"
          name={b.text}
        >
          <Audio src={staticFile(b.audio)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
export const Tutorial: React.FC<{ videoId: string }> = ({ videoId }) => {
  const video = tutorials.find((v) => v.id === videoId);
  if (!video) throw new Error(`Unknown tutorial ${videoId}`);
  return (
    <Series>
      {video.scenes.map((scene, index) => (
        <Series.Sequence
          key={scene.id}
          durationInFrames={scene.duration}
          name={`${scene.chapter} ${scene.title.replace("\n", "")}`}
        >
          <TutorialScene scene={scene} video={video} index={index} />
        </Series.Sequence>
      ))}
    </Series>
  );
};
