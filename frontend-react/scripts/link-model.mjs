import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = path.join(root, "public");
const target = path.join(publicDir, "web_model.json");
const source = path.resolve(root, "../models/web_model.json");

if (!fs.existsSync(source)) {
  console.warn(
    "web_model.json not found. Run from repo root:\n  python -m src.export_web",
  );
  process.exit(0);
}

fs.mkdirSync(publicDir, { recursive: true });
try {
  fs.unlinkSync(target);
} catch {
  /* absent */
}
fs.symlinkSync(source, target);
console.log("linked public/web_model.json ->", source);
