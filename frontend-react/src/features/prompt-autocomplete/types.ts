export type WordEntry = [word: string, freq: number];

export interface WebModelCfg {
  alpha: number;
  gamma: number;
  suf_len: number;
  word_w?: number;
  agr_w?: number;
}

export interface WebModelCtx {
  suf_len: number;
  window: number;
  decay: number;
  word_w?: number;
  adj: Record<string, Record<string, number>>;
  adj_total: Record<string, number>;
  tri: Record<string, Record<string, number>>;
  tri_total: Record<string, number>;
  wadj?: Record<string, Record<string, number>>;
  wadj_total?: Record<string, number>;
  bigram: Record<string, Record<string, Record<string, number>>>;
  prev_total: Record<string, Record<string, number>>;
  uni: Record<string, number>;
  uni_total: number;
}

export interface WebModelAgr {
  alpha: number;
  gov_window: number;
  vnum: Record<string, Record<string, number>>;
  adj_gn: Record<string, Record<string, number>>;
  noun_gn: Record<string, Record<string, number>>;
}

export interface WebModel {
  words: WordEntry[];
  ctx: WebModelCtx;
  morph: Record<string, string>;
  agr: WebModelAgr | null;
  stop: string[];
  cfg: WebModelCfg;
}

export type ChipType = "count" | "percent" | "angle" | "color" | "preset";

export interface ChipHitBase {
  s: number;
  e: number;
  type: ChipType;
}

export interface CountHit extends ChipHitBase {
  type: "count";
  word: string;
}

export interface PercentHit extends ChipHitBase {
  type: "percent";
  val: number;
}

export interface AngleHit extends ChipHitBase {
  type: "angle";
  val: number;
}

export interface ColorHit extends ChipHitBase {
  type: "color";
  mod: string;
  stem: string;
  interfix: string;
  ending: string;
}

export interface PresetHit extends ChipHitBase {
  type: "preset";
  kind: "res" | "style";
}

export type ChipHit = CountHit | PercentHit | AngleHit | ColorHit | PresetHit;

export interface GhostResult {
  ending: string;
  partial: boolean;
}

export interface NextWordResponse {
  ok: boolean;
  suggestion?: string;
  reason?: string;
  ttft_ms?: number;
}

export interface PromptAutocompleteProps {
  value?: string;
  onChange?: (value: string) => void;
  model: WebModel;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  nextWordUrl?: string;
  onLatency?: (ms: number) => void;
  label?: string;
  showHint?: boolean;
  colorChipStyle?: "droplet" | "label";
  angleChipStyle?: "protractor" | "minimal";
}
