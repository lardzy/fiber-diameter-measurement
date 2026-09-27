import {
  CanvasImage,
  Easing,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import {
  actions,
  guideAt,
  type GuideStep,
  type Rect,
  type SceneData,
  type XY,
} from "./types";
const W = 1348,
  H = 758;
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const limit = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
function camera(step: GuideStep, iw: number, ih: number) {
  const r = step.rect ?? [0, 0, iw, ih];
  const zoom = limit(Math.min(iw / (r[2] + 400), ih / (r[3] + 250)), 1, 2.05);
  const fit = Math.min(W / iw, H / ih),
    scale = fit * zoom;
  const halfX = W / scale / 2,
    halfY = H / scale / 2;
  return {
    scale,
    x: halfX >= iw / 2 ? iw / 2 : limit(r[0] + r[2] / 2, halfX, iw - halfX),
    y: halfY >= ih / 2 ? ih / 2 : limit(r[1] + r[3] / 2, halfY, ih - halfY),
  };
}
const center = (r: Rect): XY => [r[0] + r[2] / 2, r[1] + r[3] / 2];
export const GuidedScreen: React.FC<{ scene: SceneData }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { beat, step, stepIndex, previous, local, span } = guideAt(
    scene,
    frame,
  );
  const screen = step.screen ?? beat.guide.screen;
  const rect = step.rect!;
  const to = camera(step, screen.width, screen.height);
  const from =
    previous?.rect && previous.screen?.file === screen.file
      ? camera(previous, screen.width, screen.height)
      : {
          scale: Math.min(W / screen.width, H / screen.height),
          x: screen.width / 2,
          y: screen.height / 2,
        };
  const travel = interpolate(local, [0, 22], [0, 1], {
    ...clamp,
    easing: Easing.inOut(Easing.cubic),
  });
  const scale = from.scale + (to.scale - from.scale) * travel;
  const ox = W / 2 - (from.x + (to.x - from.x) * travel) * scale;
  const oy = H / 2 - (from.y + (to.y - from.y) * travel) * scale;
  const map = (p: XY): XY => [ox + p[0] * scale, oy + p[1] * scale];
  const r: Rect = [
    ox + rect[0] * scale,
    oy + rect[1] * scale,
    rect[2] * scale,
    rect[3] * scale,
  ];
  const reveal = interpolate(local, [12, 24], [0, 1], clamp);
  const path = step.points;
  const draw = interpolate(local, [24, Math.max(44, span - 23)], [0, 1], clamp);
  let point: XY = center(rect),
    clickAt = 34;
  let completed: XY[] = [];
  if (path?.length) {
    const segment = Math.min(
      path.length - 2,
      Math.floor(draw * (path.length - 1)),
    );
    const t = draw === 1 ? 1 : draw * (path.length - 1) - segment;
    point = [
      path[segment][0] + (path[segment + 1][0] - path[segment][0]) * t,
      path[segment][1] + (path[segment + 1][1] - path[segment][1]) * t,
    ];
    completed = [...path.slice(0, segment + 1), point];
    if (step.action === "path") {
      const near = Math.round(draw * (path.length - 1));
      clickAt = 24 + (near / (path.length - 1)) * Math.max(20, span - 47);
    }
  } else if (step.action === "inspect")
    point = [rect[0] + Math.min(24, rect[2] * 0.2), rect[1] + rect[3] * 0.55];
  const target = map(point);
  const arrive = interpolate(local, [14, 32], [0, 1], {
    ...clamp,
    easing: Easing.inOut(Easing.cubic),
  });
  const cursor: XY = path?.length
    ? target
    : [target[0] + (1 - arrive) * 75, target[1] + (1 - arrive) * 70];
  const click =
    (step.action === "click" ||
      step.action === "select" ||
      step.action === "path") &&
    local >= clickAt &&
    local < clickAt + 25;
  const activeBox: Rect = [
    limit(r[0] - 7, 5, W - 20),
    limit(r[1] - 7, 5, H - 20),
    Math.min(r[2] + 14, W - 10),
    Math.min(r[3] + 14, H - 10),
  ];
  activeBox[2] = Math.min(activeBox[2], W - activeBox[0] - 5);
  activeBox[3] = Math.min(activeBox[3], H - activeBox[1] - 5);
  const [x, y, w, h] = activeBox;
  const labelW = Math.min(470, Math.max(260, step.label.length * 24 + 66));
  const labelH = step.label.length * 23 > labelW - 58 ? 78 : 48;
  const labelX = limit(x + w / 2 - labelW / 2, 18, W - labelW - 18);
  const above = y > labelH + 48;
  const labelY = above
    ? y - labelH - 22
    : limit(y + h + 22, 18, H - labelH - 40);
  const anchorX = limit(x + w / 2, labelX + 25, labelX + labelW - 25);
  const anchorY = above ? y : y + h;
  const pointerVisible = frame >= beat.from + 10;
  return (
    <div
      style={{
        position: "absolute",
        left: 492,
        top: 136,
        width: W,
        height: H,
        overflow: "hidden",
        borderRadius: 18,
        background: "#edf1ee",
        border: "1px solid #afc6b9",
        boxShadow: "0 18px 42px #193b3020",
      }}
    >
      <CanvasImage
        src={staticFile(screen.file)}
        style={{
          position: "absolute",
          left: ox,
          top: oy,
          width: screen.width,
          height: screen.height,
          scale,
          transformOrigin: "0 0",
        }}
      />
      <svg
        width={W}
        height={H}
        style={{
          position: "absolute",
          inset: 0,
          opacity: reveal,
          pointerEvents: "none",
        }}
      >
        <path
          d={`M0 0H${W}V${H}H0Z M${x} ${y}H${x + w}V${y + h}H${x}Z`}
          fill="#10251e"
          fillRule="evenodd"
          opacity={0.29}
        />
        <rect
          x={x}
          y={y}
          width={w}
          height={h}
          rx={10}
          fill="#ffc649"
          fillOpacity={0.025}
          stroke="white"
          strokeWidth={8}
        />
        <rect
          x={x}
          y={y}
          width={w}
          height={h}
          rx={10}
          fill="none"
          stroke="#efa820"
          strokeWidth={4}
        />
        {path?.length && (
          <>
            {step.connect !== false && (
              <polyline
                points={path.map((p) => map(p).join(",")).join(" ")}
                fill="none"
                stroke="#fff"
                strokeWidth={6}
                strokeDasharray="8 7"
                opacity={0.8}
              />
            )}
            {step.connect !== false && (
              <polyline
                points={completed.map((p) => map(p).join(",")).join(" ")}
                fill="none"
                stroke="#dc8713"
                strokeWidth={4}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}
            {path
              .filter(
                (_, i) =>
                  step.action === "path" || i === 0 || i === path.length - 1,
              )
              .map((p, i) => {
                const pt = map(p);
                return (
                  <g key={i}>
                    <circle
                      cx={pt[0]}
                      cy={pt[1]}
                      r={8}
                      fill="#efa820"
                      stroke="white"
                      strokeWidth={3}
                    />
                    {step.action === "path" && (
                      <text
                        x={pt[0] + 13}
                        y={pt[1] - 12}
                        fill="#163c33"
                        stroke="white"
                        strokeWidth={4}
                        paintOrder="stroke"
                        fontSize={23}
                        fontWeight={800}
                      >
                        {i + 1}
                      </text>
                    )}
                  </g>
                );
              })}
          </>
        )}
        <path
          d={`M${labelX + labelW / 2} ${above ? labelY + labelH : labelY}L${anchorX} ${anchorY}`}
          stroke="#eda621"
          strokeWidth={3}
          fill="none"
        />
        <circle
          cx={anchorX}
          cy={anchorY}
          r={5}
          fill="#eda621"
          stroke="white"
          strokeWidth={2}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          left: labelX,
          top: labelY,
          width: labelW,
          minHeight: labelH,
          padding: "9px 14px",
          boxSizing: "border-box",
          display: "flex",
          alignItems: "center",
          gap: 10,
          background: "#fffaf0",
          border: "2px solid #e6a42a",
          borderRadius: 12,
          color: "#593908",
          fontSize: 23,
          lineHeight: 1.3,
          fontWeight: 750,
          opacity: reveal,
          boxShadow: "0 5px 18px #1a2d3430",
        }}
      >
        <span
          style={{
            background: "#edaa2b",
            color: "#332000",
            borderRadius: 20,
            minWidth: 30,
            height: 30,
            display: "grid",
            placeItems: "center",
            fontSize: 20,
          }}
        >
          {stepIndex + 1}
        </span>
        <span>{step.label}</span>
      </div>
      {pointerVisible && (
        <>
          <div
            style={{
              position: "absolute",
              left: cursor[0] - 31,
              top: cursor[1] - 31,
              width: 62,
              height: 62,
              borderRadius: "50%",
              border: "2px solid #f4aa25",
              background: "#ffd14b38",
              boxShadow: "0 0 0 7px #ffdc6420",
              opacity: reveal,
              scale: 1 + 0.045 * Math.sin(local / 13),
            }}
          />
          {click && (
            <div
              style={{
                position: "absolute",
                left: cursor[0] - 31,
                top: cursor[1] - 31,
                width: 62,
                height: 62,
                borderRadius: "50%",
                border: "4px solid #f5aa22",
                scale: interpolate(
                  local,
                  [clickAt, clickAt + 25],
                  [0.6, 1.8],
                  clamp,
                ),
                opacity: interpolate(
                  local,
                  [clickAt, clickAt + 25],
                  [1, 0],
                  clamp,
                ),
              }}
            />
          )}
          <svg
            width={49}
            height={62}
            viewBox="0 0 38 46"
            style={{
              position: "absolute",
              left: cursor[0] - 6,
              top: cursor[1] - 4,
              filter: "drop-shadow(1px 3px 3px #0008)",
              opacity: reveal,
            }}
          >
            <path
              d="M5 3L5 34L13 27L20 42L26 39L19 24L31 24Z"
              fill={step.action === "drag" ? "#efa820" : "#163c33"}
              stroke="white"
              strokeWidth={2.5}
              strokeLinejoin="round"
            />
          </svg>
        </>
      )}
      <div
        style={{
          position: "absolute",
          right: 12,
          bottom: 11,
          padding: "6px 12px",
          background: "#183d32e8",
          color: "white",
          fontSize: 18,
          borderRadius: 7,
        }}
      >
        {scene.kind === "artifact" ? "实际导出图片" : "真实界面 · 操作指引示意"}{" "}
        · {actions[step.action]}
      </div>
    </div>
  );
};
