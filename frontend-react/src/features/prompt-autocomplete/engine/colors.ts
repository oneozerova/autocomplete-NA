export const COLORS: [stem: string, hex: string][] = [
  ["красн", "#e23b3b"],
  ["син", "#3167e0"],
  ["голуб", "#49b6f0"],
  ["зелён", "#33a852"],
  ["зелен", "#33a852"],
  ["жёлт", "#f3c62a"],
  ["желт", "#f3c62a"],
  ["оранжев", "#f5842a"],
  ["фиолетов", "#8a3ff0"],
  ["розов", "#f06fae"],
  ["чёрн", "#222222"],
  ["черн", "#222222"],
  ["бел", "#ececec"],
  ["сер", "#9aa0a6"],
  ["коричнев", "#96613b"],
  ["бирюзов", "#1ec5c5"],
  ["золот", "#d4af37"],
  ["серебр", "#c3c7cc"],
  ["пурпурн", "#a12bb0"],
  ["бордов", "#8c2231"],
];

export const MOD_RE = /^(?:тёмно-|темно-|светло-|ярко-|бледно-)/;
export const ADJ_TAIL =
  /^(ян|ист|оват)?(ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ыми|ими|ым|им|ом|ем|ую|юю|ых|их)$/;

export const SOFT_HARD: Record<string, string[]> = {
  ий: ["ый", "ой"],
  ый: ["ий", "ой"],
  ой: ["ый", "ий"],
  яя: ["ая"],
  ая: ["яя"],
  ее: ["ое"],
  ое: ["ее"],
  ие: ["ые"],
  ые: ["ие"],
  его: ["ого"],
  ого: ["его"],
  ему: ["ому"],
  ому: ["ему"],
  юю: ["ую"],
  ую: ["юю"],
  им: ["ым"],
  ым: ["им"],
  ем: ["ом"],
  ом: ["ем"],
  их: ["ых"],
  ых: ["их"],
  ими: ["ыми"],
  ыми: ["ими"],
};

export function shade(hex: string, amt: number): string {
  let n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) & 255;
  let g = (n >> 8) & 255;
  let b = n & 255;
  const f = amt < 0 ? 1 + amt : 1;
  const add = amt > 0 ? amt * 255 : 0;
  r = Math.max(0, Math.min(255, Math.round(r * f + add)));
  g = Math.max(0, Math.min(255, Math.round(g * f + add)));
  b = Math.max(0, Math.min(255, Math.round(b * f + add)));
  return (
    "#" +
    ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)
  );
}

export function colorParts(word: string): {
  mod: string;
  stem: string;
  interfix: string;
  ending: string;
} | null {
  const w = word.toLowerCase();
  const mod = (w.match(MOD_RE) || [""])[0];
  const base = w.slice(mod.length);
  for (const [stem] of COLORS) {
    if (!base.startsWith(stem)) continue;
    const m = base.slice(stem.length).match(ADJ_TAIL);
    if (m) return { mod, stem, interfix: m[1] || "", ending: m[2] };
  }
  return null;
}

export function colorHex(word: string): string {
  const p = colorParts(word);
  if (!p) return "#888888";
  const entry = COLORS.find((c) => c[0] === p.stem);
  let base = entry?.[1] ?? "#888888";
  const w = word.toLowerCase();
  if (/^(тёмно|темно)/.test(w)) base = shade(base, -0.28);
  else if (/^(светло|бледно)/.test(w)) base = shade(base, 0.3);
  else if (/^ярко/.test(w)) base = shade(base, 0.1);
  return base;
}

export function nearestStem(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  let best: string | null = null;
  let bestD = Infinity;
  for (const [stem, h] of COLORS) {
    const m = parseInt(h.slice(1), 16);
    const d =
      ((m >> 16 & 255) - r) ** 2 +
      ((m >> 8 & 255) - g) ** 2 +
      ((m & 255) - b) ** 2;
    if (d < bestD) {
      bestD = d;
      best = stem;
    }
  }
  return best!;
}
