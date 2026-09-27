import type { Caption } from "@remotion/captions";
export const GuidedCaptions: React.FC<{
  captions: Caption[];
  frame: number;
}> = ({ captions, frame }) => {
  const ms = (frame / 30) * 1000;
  const current = captions.find((c) => ms >= c.startMs && ms < c.endMs);
  return (
    <div
      style={{
        position: "absolute",
        left: 90,
        top: 956,
        width: 1740,
        height: 80,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        fontSize: 32,
        lineHeight: 1.35,
        fontWeight: 550,
        color: "#153d31",
      }}
    >
      {current?.text}
    </div>
  );
};
