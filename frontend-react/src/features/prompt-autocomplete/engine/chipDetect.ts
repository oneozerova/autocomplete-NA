import type { WebModel } from "../types";
import { colorParts, nearestStem, SOFT_HARD } from "./colors";
import type {
  AngleHit,
  ChipHit,
  ColorHit,
  CountHit,
  PercentHit,
  PresetHit,
} from "../types";

export const NUM_M = [
  "ноль", "один", "два", "три", "четыре", "пять",
  "шесть", "семь", "восемь", "девять", "десять",
];
export const NUM_F = [
  "ноль", "одна", "две", "три", "четыре", "пять",
  "шесть", "семь", "восемь", "девять", "десять",
];

export const WORD2NUM: Record<string, number> = {};
NUM_M.forEach((w, i) => {
  WORD2NUM[w] = i;
});
NUM_F.forEach((w, i) => {
  WORD2NUM[w] = i;
});

export const PRESETS = {
  res: ["720p", "1080p", "2k", "4k", "8k"],
  style: [
    "в стиле 1999",
    "в стиле 2004",
    "в стиле 2010",
    "ретро",
    "неон",
    "винтаж",
  ],
} as const;

export function detectChips(text: string): ChipHit[] {
  const hits: ChipHit[] = [];
  let m: RegExpExecArray | null;
  let re: RegExp;

  re = /(\d+)\s*%/g;
  while ((m = re.exec(text))) {
    hits.push({
      s: m.index,
      e: m.index + m[0].length,
      type: "percent",
      val: +m[1],
    } satisfies PercentHit);
  }

  re = /(\d+)\s*(?:°|градус(?:ов|а)?)/gi;
  while ((m = re.exec(text))) {
    hits.push({
      s: m.index,
      e: m.index + m[0].length,
      type: "angle",
      val: +m[1],
    } satisfies AngleHit);
  }

  re = /\b(720p|1080p|2k|4k|8k|hd|full\s?hd)\b/gi;
  while ((m = re.exec(text))) {
    hits.push({
      s: m.index,
      e: m.index + m[0].length,
      type: "preset",
      kind: "res",
    } satisfies PresetHit);
  }

  re = /в\s+стиле\s+((?:19|20)\d{2})/gi;
  while ((m = re.exec(text))) {
    hits.push({
      s: m.index,
      e: m.index + m[0].length,
      type: "preset",
      kind: "style",
    } satisfies PresetHit);
  }

  re = /[а-яё]+(?:-[а-яё]+)*/gi;
  while ((m = re.exec(text))) {
    const p = colorParts(m[0]);
    if (p) {
      hits.push({
        s: m.index,
        e: m.index + m[0].length,
        type: "color",
        ...p,
      } satisfies ColorHit);
    }
  }

  const numAlt = "\\d{1,3}|" + Object.keys(WORD2NUM).join("|");
  re = new RegExp("(?<![а-яёa-z0-9])(" + numAlt + ")(?=\\s+[а-яё]{2,})", "gi");
  while ((m = re.exec(text))) {
    hits.push({
      s: m.index,
      e: m.index + m[0].length,
      type: "count",
      word: m[0],
    } satisfies CountHit);
  }

  hits.sort((a, b) => a.s - b.s || b.e - b.s - (a.e - a.s));
  const out: ChipHit[] = [];
  let last = -1;
  for (const h of hits) {
    if (h.s >= last) {
      out.push(h);
      last = h.e;
    }
  }
  return out;
}

export type ColorIndex = Record<string, Record<string, [form: string, hasInterfix: boolean]>>;

export function buildColorIndex(model: WebModel): ColorIndex {
  const colorIdx: ColorIndex = {};
  for (const [w] of model.words) {
    const code = model.morph[w];
    if (code && code[0] !== "A") continue;
    const p = colorParts(w);
    if (!p || p.mod) continue;
    const slot = colorIdx[p.stem] || (colorIdx[p.stem] = {});
    const prev = slot[p.ending];
    if (!prev || (prev[1] && !p.interfix)) {
      slot[p.ending] = [w, !!p.interfix];
    }
  }
  return colorIdx;
}

export function recolorWord(
  word: string,
  hex: string,
  colorIdx: ColorIndex,
): string | null {
  const p = colorParts(word);
  if (!p) return null;
  const forms = colorIdx[nearestStem(hex)] || {};
  let loose: string | null = null;
  for (const e of [p.ending, ...(SOFT_HARD[p.ending] || [])]) {
    const f = forms[e];
    if (!f) continue;
    if (!f[1]) return p.mod + f[0];
    if (!loose) loose = p.mod + f[0];
  }
  return loose;
}
