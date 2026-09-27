import { loadFont } from "@remotion/fonts";
import {
  cancelRender,
  continueRender,
  delayRender,
  staticFile,
} from "remotion";
import subsets from "../public/fonts/subsets.json";

const handle = delayRender("Loading bundled Chinese font");
Promise.all(
  subsets.map((subset) =>
    loadFont({
      family: "FDM Noto",
      url: staticFile(`fonts/${subset.file}`),
      format: "woff2",
      weight: "100 900",
      unicodeRange: subset.unicodeRange,
    }),
  ),
)
  .then(() => continueRender(handle))
  .catch(cancelRender);
