import type { ChipHit, WebModel } from "../types";
import {
  buildColorIndex,
  NUM_F,
  NUM_M,
  PRESETS,
  WORD2NUM,
  recolorWord,
  type ColorIndex,
} from "./chipDetect";
import { colorHex } from "./colors";
import {
  closeColorPicker,
  isColorPickerOpenFor,
  openColorPicker,
} from "./colorPicker";
import {
  closeAngleOrbit,
  createAnglePickerUi,
  isAngleOrbitOpenFor,
  openAngleOrbit,
} from "./anglePicker";
import "./anglePicker.css";
import { createColorIcon, setColorIconFill } from "./colorIcon";

type EmitFn = () => void;

const FIELD_GLOW_AT_100 = 0.35;

function setPercentFieldGlow(chip: HTMLElement, val: number | null): void {
  const wrap = chip.closest(".pa-field-wrap") as HTMLElement | null;
  if (!wrap) return;
  const glow = wrap.querySelector(".pa-field-glow") as HTMLElement | null;
  if (val == null) {
    wrap.classList.remove("pa-field-wrap--percent-glow");
    wrap.style.removeProperty("--pa-glow");
    glow?.style.removeProperty("opacity");
    return;
  }
  const t = Math.max(0, Math.min(100, val)) / 100;
  const opacity = String(FIELD_GLOW_AT_100 * t);
  wrap.style.setProperty("--pa-glow", opacity);
  wrap.classList.add("pa-field-wrap--percent-glow");
  if (glow) glow.style.opacity = opacity;
}

function chipEl(cls: string, dataText: string): HTMLSpanElement {
  const c = document.createElement("span");
  c.className = "pa-chip " + cls;
  c.contentEditable = "false";
  c.dataset.text = dataText;
  return c;
}

function makeCount(h: Extract<ChipHit, { type: "count" }>, emit: EmitFn, relayout: EmitFn) {
  const isWord = Number.isNaN(+h.word);
  const fem = isWord && NUM_F.includes(h.word.toLowerCase());
  let val = isWord ? WORD2NUM[h.word.toLowerCase()] : +h.word;
  const c = chipEl("pa-count", h.word);

  const display = () =>
    isWord ? (fem ? NUM_F : NUM_M)[Math.min(val, 10)] : String(val);

  const paint = () => {
    const disp = display();
    c.dataset.text = disp;
    const b = c.querySelector("b");
    if (b) b.textContent = disp;
  };

  const bump = (delta: number) => {
    const prev = c.dataset.text ?? "";
    if (isWord) {
      val = Math.max(0, Math.min(10, val + delta));
    } else {
      val = Math.max(0, Math.min(999, val + delta));
    }
    paint();
    const next = c.dataset.text ?? "";
    if (prev.length !== next.length) relayout();
    else emit();
  };

  const b = document.createElement("b");
  b.textContent = display();

  const stepper = document.createElement("span");
  stepper.className = "pa-stepper-col";
  const up = document.createElement("button");
  up.type = "button";
  up.className = "pa-step-vert pa-step-vert--up";
  up.setAttribute("aria-label", "Увеличить");
  up.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    bump(1);
  });
  const down = document.createElement("button");
  down.type = "button";
  down.className = "pa-step-vert pa-step-vert--down";
  down.setAttribute("aria-label", "Уменьшить");
  down.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    bump(-1);
  });
  stepper.append(up, down);
  c.append(b, stepper);
  paint();
  return c;
}

function makePercent(h: Extract<ChipHit, { type: "percent" }>, emit: EmitFn) {
  let val = h.val;
  const c = chipEl("pa-percent", val + "%");
  const b = document.createElement("b");
  const pop = document.createElement("span");
  pop.className = "pa-pop";
  const range = document.createElement("input");
  range.type = "range";
  range.min = "0";
  range.max = "100";
  range.value = String(val);
  const lbl = document.createElement("span");
  const paint = () => {
    b.textContent = val + "%";
    lbl.textContent = val + "%";
    c.dataset.text = val + "%";
    if (pop.classList.contains("pa-open")) {
      setPercentFieldGlow(c, val);
    }
  };
  range.addEventListener("input", (e) => {
    e.stopPropagation();
    val = +(e.target as HTMLInputElement).value;
    paint();
    emit();
  });
  pop.append(range, lbl);
  c.append(b, pop);
  paint();
  c.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    closeColorPicker();
    document.querySelectorAll(".pa-pop.pa-open").forEach((p) => {
      if (p !== pop) p.classList.remove("pa-open");
    });
    const opening = !pop.classList.contains("pa-open");
    pop.classList.toggle("pa-open");
    if (opening) setPercentFieldGlow(c, val);
    else setPercentFieldGlow(c, null);
  });
  pop.addEventListener("mousedown", (e) => e.stopPropagation());
  return c;
}

function makeAngle(
  h: Extract<ChipHit, { type: "angle" }>,
  emit: EmitFn,
  style: "protractor" | "minimal" = "protractor",
) {
  let val = h.val % 360;
  const isOrbit = style === "protractor";
  const c = chipEl("pa-angle pa-angle--minimal", val + "°");
  const b = document.createElement("b");

  let pop: HTMLSpanElement | null = null;
  let picker: ReturnType<typeof createAnglePickerUi> | null = null;

  if (!isOrbit) {
    pop = document.createElement("span");
    pop.className = "pa-pop pa-pop--angle-semicircle";
    picker = createAnglePickerUi(val, (deg) => {
      val = deg;
      paint();
      emit();
    }, "minimal");
    pop.append(picker.root);
  }

  const paint = () => {
    b.textContent = val + "°";
    c.dataset.text = val + "°";
    if (pop?.classList.contains("pa-open")) {
      picker?.setValue(val);
    }
  };

  c.append(b);
  if (pop) c.append(pop);
  paint();

  c.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    closeColorPicker();
    document.querySelectorAll(".pa-pop.pa-open").forEach((p) => {
      p.classList.remove("pa-open");
    });

    if (isOrbit) {
      if (isAngleOrbitOpenFor(c)) {
        closeAngleOrbit();
        return;
      }
      closeAngleOrbit();
      c.classList.add("pa-angle--open");
      openAngleOrbit({
        anchor: c,
        value: val,
        onChange: (deg) => {
          val = deg;
          paint();
          emit();
        },
        onClose: () => {
          c.classList.remove("pa-angle--open", "pa-angle--orbit-open");
        },
      });
      return;
    }

    closeAngleOrbit();

    if (!pop) return;
    const opening = !pop.classList.contains("pa-open");
    pop.classList.toggle("pa-open");
    c.classList.toggle("pa-angle--open", pop.classList.contains("pa-open"));
    if (opening) picker?.setValue(val);
  });

  pop?.addEventListener("mousedown", (e) => e.stopPropagation());
  return c;
}

function makeColor(
  word: string,
  colorIdx: ColorIndex,
  emit: EmitFn,
  style: "droplet" | "label" = "droplet",
) {
  let cur = word;
  const isLabel = style === "label";
  const c = chipEl(
    isLabel ? "pa-color pa-color--label" : "pa-color pa-color--icon",
    cur,
  );
  let iconHex = colorHex(cur);
  c.setAttribute("role", "button");
  c.setAttribute("aria-label", `${cur} — выбрать цвет`);

  const label = isLabel ? document.createElement("b") : null;
  const icon = isLabel ? null : createColorIcon(iconHex, "droplet");

  const syncMeta = () => {
    c.dataset.text = cur;
    c.dataset.iconHex = iconHex;
    c.title = cur;
    c.setAttribute("aria-label", `${cur} — выбрать цвет`);
    if (label) {
      label.textContent = cur;
      label.style.color = iconHex;
    }
    if (icon) setColorIconFill(icon, iconHex);
  };

  const applyHex = (hex: string) => {
    const form = recolorWord(cur, hex, colorIdx);
    if (form) {
      cur = form;
      iconHex = isLabel ? colorHex(form) : hex;
    } else {
      iconHex = hex;
    }
    syncMeta();
    emit();
  };

  if (label) c.append(label);
  if (icon) c.append(icon);
  syncMeta();

  c.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    document.querySelectorAll(".pa-pop.pa-open").forEach((p) => {
      p.classList.remove("pa-open");
    });
    if (isColorPickerOpenFor(c)) {
      closeColorPicker();
      return;
    }
    c.classList.add("pa-color--open");
    openColorPicker({
      anchor: c,
      value: iconHex,
      onChange: applyHex,
      onClose: () => {
        c.classList.remove("pa-color--open");
      },
    });
  });

  return c;
}

function makePreset(h: Extract<ChipHit, { type: "preset" }>, word: string, relayout: EmitFn) {
  const list: string[] = [...(PRESETS[h.kind] || [])];
  let i = list.findIndex((v) => v.toLowerCase() === word.toLowerCase());
  if (i < 0) {
    list.unshift(word);
    i = 0;
  }
  const c = chipEl("pa-preset", word);
  const b = document.createElement("b");
  const caret = document.createElement("span");
  caret.textContent = "▸";
  caret.style.opacity = "0.55";
  const paint = () => {
    b.textContent = list[i];
    c.dataset.text = list[i];
  };
  c.append(b, caret);
  paint();
  c.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    i = (i + 1) % list.length;
    paint();
    relayout();
  });
  return c;
}

let cachedColorIdx: ColorIndex | null = null;
let cachedModel: WebModel | null = null;

function getColorIndex(model: WebModel): ColorIndex {
  if (cachedColorIdx && cachedModel === model) return cachedColorIdx;
  cachedModel = model;
  cachedColorIdx = buildColorIndex(model);
  return cachedColorIdx;
}

export interface BuildChipOptions {
  colorChipStyle?: "droplet" | "label";
  angleChipStyle?: "protractor" | "minimal";
}

export function buildChip(
  h: ChipHit,
  text: string,
  model: WebModel,
  emit: EmitFn,
  relayout: EmitFn,
  opts: BuildChipOptions = {},
): HTMLElement {
  const raw = text.slice(h.s, h.e);
  switch (h.type) {
    case "count":
      return makeCount(h, emit, relayout);
    case "percent":
      return makePercent(h, emit);
    case "angle":
      return makeAngle(h, emit, opts.angleChipStyle ?? "protractor");
    case "color":
      return makeColor(
        raw,
        getColorIndex(model),
        emit,
        opts.colorChipStyle ?? "droplet",
      );
    case "preset":
      return makePreset(h, raw, relayout);
  }
}
