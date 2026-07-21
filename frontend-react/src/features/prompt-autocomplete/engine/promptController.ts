import { buildChip } from "./chipDom";
import { detectChips } from "./chipDetect";
import {
  appendText,
  getCaret,
  serialize,
  setCaret,
} from "./domUtils";
import { GhostEngine, shouldSuggestNextWord } from "./ghostEngine";
import { closeColorPicker } from "./colorPicker";
import { closeAngleOrbit } from "./anglePicker";
import type { WebModel } from "../types";

const CHIP_DEBOUNCE_MS = 140;
const LLM_DEBOUNCE_MS = 400;

export interface PromptControllerOptions {
  el: HTMLElement;
  model: WebModel;
  onChange?: (text: string) => void;
  onGhostLatency?: (ms: number) => void;
  onLlmLatency?: (ms: number) => void;
  nextWordUrl?: string;
  colorChipStyle?: "droplet" | "label";
  angleChipStyle?: "protractor" | "minimal";
}

export class PromptController {
  private ghostEngine: GhostEngine;
  private ghost = "";
  private ghostPartial = false;
  private phrase = "";
  private composing = false;
  private debChip: ReturnType<typeof setTimeout> | null = null;
  private debPhrase: ReturnType<typeof setTimeout> | null = null;
  private inflight: AbortController | null = null;
  private llmEnabled = true;
  private phraseCache = new Map<string, string>();
  private sig = "";
  private readonly onOutsideClick = (e: MouseEvent) => {
    const t = e.target as Element;
    if (!t.closest(".pa-chip") && !t.closest(".pa-cpick") && !t.closest(".pa-angle-orbit")) {
      document
        .querySelectorAll(".pa-pop.pa-open")
        .forEach((p) => p.classList.remove("pa-open"));
      document
        .querySelectorAll(".pa-chip.pa-angle--open")
        .forEach((chip) => chip.classList.remove("pa-angle--open", "pa-angle--orbit-open"));
      const wrap = this.opts.el.closest(".pa-field-wrap") as HTMLElement | null;
      if (wrap) {
        wrap.classList.remove("pa-field-wrap--percent-glow");
        wrap.style.removeProperty("--pa-glow");
        const glow = wrap.querySelector(".pa-field-glow") as HTMLElement | null;
        glow?.style.removeProperty("opacity");
      }
      closeColorPicker();
      closeAngleOrbit();
    }
  };

  constructor(private readonly opts: PromptControllerOptions) {
    this.ghostEngine = new GhostEngine(opts.model);
    this.bind();
    document.addEventListener("mousedown", this.onOutsideClick);
  }

  setDisabled(disabled: boolean): void {
    this.opts.el.contentEditable = disabled ? "false" : "true";
  }

  setText(text: string): void {
    this.opts.el.textContent = text;
    this.refresh(true);
  }

  destroy(): void {
    if (this.debChip) clearTimeout(this.debChip);
    this.clearPhrase();
    closeColorPicker();
    closeAngleOrbit();
    document.removeEventListener("mousedown", this.onOutsideClick);
  }

  private bind(): void {
    const { el } = this.opts;

    el.addEventListener("input", () => {
      this.paintGhost();
      if (this.debChip) clearTimeout(this.debChip);
      this.debChip = setTimeout(() => this.refresh(false), CHIP_DEBOUNCE_MS);
    });

    el.addEventListener("keydown", (e) => this.onKeyDown(e));
    el.addEventListener("paste", (e) => this.onPaste(e));
    el.addEventListener("compositionstart", () => {
      this.composing = true;
    });
    el.addEventListener("compositionend", () => {
      this.composing = false;
      this.refresh(false);
    });
  }

  private emit(): void {
    const text = serialize(this.opts.el);
    this.opts.onChange?.(text);
  }

  private stripAux(): void {
    this.opts.el
      .querySelectorAll(".ghost,.phrase")
      .forEach((n) => n.remove());
  }

  private appendAux(cls: string, text: string): void {
    const s = document.createElement("span");
    s.className = cls;
    s.contentEditable = "false";
    s.textContent = text;
    this.opts.el.append(s);
  }

  private clearPhrase(): void {
    this.phrase = "";
    this.opts.el.querySelectorAll(".phrase").forEach((n) => n.remove());
    if (this.inflight) {
      this.inflight.abort();
      this.inflight = null;
    }
    if (this.debPhrase) {
      clearTimeout(this.debPhrase);
      this.debPhrase = null;
    }
  }

  private paintGhost(): void {
    this.clearPhrase();
    this.stripAux();
    this.ghost = "";
    this.ghostPartial = false;

    const el = this.opts.el;
    const T = serialize(el);
    const caret = getCaret(el);
    const t0 = performance.now();

    if (caret != null && caret === T.length) {
      const g = this.ghostEngine.ghostFor(T);
      if (g.ending) {
        this.ghost = g.ending;
        this.ghostPartial = g.partial;
        this.appendAux("ghost", g.ending);
      }
      const ms = performance.now() - t0;
      if (g.ending) this.opts.onGhostLatency?.(ms);
    }

    if (
      this.llmEnabled &&
      this.opts.nextWordUrl &&
      caret === T.length &&
      !this.ghost &&
      shouldSuggestNextWord(T)
    ) {
      this.debPhrase = setTimeout(() => this.requestPhrase(T), LLM_DEBOUNCE_MS);
    }
  }

  private async requestPhrase(text: string): Promise<void> {
    if (!this.opts.nextWordUrl) return;
    if (this.phraseCache.has(text)) {
      this.setPhrase(this.phraseCache.get(text)!, text);
      return;
    }
    const ctrl = new AbortController();
    this.inflight = ctrl;
    try {
      const res = await fetch(this.opts.nextWordUrl, {
        method: "POST",
        signal: ctrl.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (text !== serialize(this.opts.el)) return;
      if (data.reason === "no_key") {
        this.llmEnabled = false;
        return;
      }
      const s = data.ok ? (data.suggestion as string) : "";
      this.phraseCache.set(text, s);
      this.setPhrase(s, text, data.ttft_ms as number | undefined);
    } catch {
      /* ignore */
    }
  }

  private setPhrase(s: string, forText: string, ttft?: number): void {
    if (forText !== serialize(this.opts.el)) return;
    this.phrase = s || "";
    if (this.phrase) {
      this.appendAux("phrase", this.phrase);
      if (ttft != null) this.opts.onLlmLatency?.(ttft);
    }
  }

  private chipify(force: boolean): void {
    if (this.composing) return;
    this.stripAux();

    const el = this.opts.el;
    const text = serialize(el);
    const caret = getCaret(el);
    const hits = detectChips(text).filter(
      (h) => force || caret == null || caret < h.s || caret > h.e,
    );
    const sig = hits.map((h) => h.type + h.s + h.e).join(",");
    if (!force && sig === this.sig) return;
    this.sig = sig;

    const emit = () => this.emit();
    const relayout = () => {
      closeColorPicker();
      closeAngleOrbit();
      this.emit();
      this.sig = "";
      requestAnimationFrame(() => this.chipify(true));
    };
    const frag = document.createDocumentFragment();
    let i = 0;
    for (const h of hits) {
      if (h.s > i) appendText(frag, text.slice(i, h.s));
      frag.append(
        buildChip(h, text, this.opts.model, emit, relayout, {
          colorChipStyle: this.opts.colorChipStyle ?? "droplet",
          angleChipStyle: this.opts.angleChipStyle ?? "protractor",
        }),
      );
      i = h.e;
    }
    if (i < text.length) appendText(frag, text.slice(i));

    el.replaceChildren(frag);
    setCaret(el, caret);
    this.emit();
    this.paintGhost();
  }

  refresh(force: boolean): void {
    this.chipify(force);
  }

  private onKeyDown(e: KeyboardEvent): void {
    const el = this.opts.el;
    if (e.key === "Tab" && (this.ghost || this.phrase)) {
      e.preventDefault();
      const add =
        (this.ghost ? this.ghost + (this.ghostPartial ? "" : " ") : this.phrase + " ");
      this.clearPhrase();
      this.stripAux();
      this.ghost = "";
      const last = el.lastChild;
      if (last?.nodeType === Node.TEXT_NODE) {
        last.textContent = (last.textContent ?? "") + add;
      } else {
        el.append(document.createTextNode(add));
      }
      const r = document.createRange();
      r.selectNodeContents(el);
      r.collapse(false);
      const sel = window.getSelection();
      sel?.removeAllRanges();
      sel?.addRange(r);
      this.refresh(false);
      return;
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      this.stripAux();
      this.clearPhrase();
      this.ghost = "";
      document.execCommand("insertLineBreak");
      setTimeout(() => this.refresh(false), 0);
      return;
    }

    if (e.key === " " || e.key === ",") {
      setTimeout(() => this.refresh(false), 0);
    }
  }

  private onPaste(e: ClipboardEvent): void {
    e.preventDefault();
    const t = e.clipboardData?.getData("text/plain") ?? "";
    document.execCommand("insertText", false, t.replace(/\r\n?/g, "\n"));
  }
}
