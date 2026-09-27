import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const walk = (dir) =>
  fs
    .readdirSync(dir, { withFileTypes: true })
    .flatMap((e) =>
      e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)],
    );
const source = [
  ...walk(path.join(root, "src")),
  ...walk(path.join(root, "narration")),
]
  .filter((f) => /\.(tsx?|json|txt)$/.test(f))
  .map((f) => fs.readFileSync(f, "utf8"))
  .join("\n");
const chinese = source.match(/[\u3000-\u9fff\uff00-\uffefµ²✓·→—…“”]/gu) ?? [];
const ascii = Array.from({ length: 95 }, (_, i) => String.fromCharCode(i + 32));
const glyphs = [...new Set([...chinese, ...ascii])];
const codes = glyphs.map((char) => char.codePointAt(0));
const packageDir = path.join(
  root,
  "node_modules/@fontsource-variable/noto-sans-sc",
);
const css = fs.readFileSync(path.join(packageDir, "index.css"), "utf8");
const dir = path.join(root, "public/fonts");
fs.mkdirSync(dir, { recursive: true });
const subsets = [];
let bytes = 0;
const covered = new Set();
for (const block of css.matchAll(/@font-face\s*\{([^}]+)\}/g)) {
  const file = block[1].match(/url\(\.\/files\/([^)]+)\)/)?.[1];
  const unicodeRange = block[1].match(/unicode-range:\s*([^;]+)/)?.[1];
  if (!file || !unicodeRange) continue;
  const ranges = unicodeRange.split(",").map((r) =>
    r
      .trim()
      .replace(/^U\+/i, "")
      .split("-")
      .map((n) => parseInt(n, 16)),
  );
  const matched = codes.filter((c) =>
    ranges.some(([lo, hi]) => c >= lo && c <= (hi ?? lo)),
  );
  if (!matched.length) continue;
  matched.forEach((c) => covered.add(c));
  const binary = fs.readFileSync(path.join(packageDir, "files", file));
  fs.writeFileSync(path.join(dir, file), binary);
  bytes += binary.length;
  subsets.push({ file, unicodeRange });
}
const missing = codes.filter((c) => !covered.has(c));
if (missing.length)
  throw new Error(
    `Font missing glyphs: ${missing.map((c) => String.fromCodePoint(c)).join("")}`,
  );
fs.writeFileSync(
  path.join(dir, "subsets.json"),
  JSON.stringify(subsets, null, 2),
);
fs.copyFileSync(path.join(packageDir, "LICENSE"), path.join(dir, "OFL.txt"));
fs.writeFileSync(
  path.join(dir, "source.txt"),
  "Noto Sans SC, weight 100–900.\nSource: @fontsource-variable/noto-sans-sc@5.3.0\nhttps://www.npmjs.com/package/@fontsource-variable/noto-sans-sc\nhttps://github.com/notofonts/noto-cjk\nSIL Open Font License; see OFL.txt.\nOnly Unicode subsets used by this tutorial are bundled.\n",
);
const partial = path.join(dir, "NotoSansSC-VF.ttf");
if (fs.existsSync(partial)) fs.unlinkSync(partial);
console.log(
  JSON.stringify({
    font: "Noto Sans SC",
    glyphs: glyphs.length,
    subsets: subsets.length,
    bytes,
  }),
);
