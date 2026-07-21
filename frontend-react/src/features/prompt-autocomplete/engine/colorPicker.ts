import {
  clamp,
  hexToHsv,
  hsvToHex,
  hsvToRgb,
  rgbToHsv,
} from "./colorMath";
import "./colorPicker.css";

export type ColorFormat = "hex" | "rgb" | "hsb";

export interface ColorPickerOptions {
  anchor: HTMLElement;
  value: string;
  onChange: (hex: string) => void;
  onClose?: () => void;
}

let activePicker: HTMLElement | null = null;
let activeAnchor: HTMLElement | null = null;
let activeClose: (() => void) | null = null;

export function closeColorPicker(): void {
  activeClose?.();
}

export function isColorPickerOpenFor(anchor: HTMLElement): boolean {
  return activeAnchor === anchor;
}

function positionPicker(picker: HTMLElement, anchor: HTMLElement): void {
  const r = anchor.getBoundingClientRect();
  const pw = picker.offsetWidth || 280;
  const ph = picker.offsetHeight || 320;
  let left = r.left + r.width / 2 - pw / 2;
  let top = r.top - ph - 14;
  if (top < 12) top = r.bottom + 14;
  left = clamp(left, 12, window.innerWidth - pw - 12);
  picker.style.left = `${left}px`;
  picker.style.top = `${top}px`;
}

export function openColorPicker(opts: ColorPickerOptions): () => void {
  closeColorPicker();

  let hsv = hexToHsv(opts.value);
  let format: ColorFormat = "hex";

  const root = document.createElement("div");
  root.className = "pa-cpick";
  root.addEventListener("mousedown", (e) => e.stopPropagation());

  const sv = document.createElement("div");
  sv.className = "pa-cpick-sv";
  const svBg = document.createElement("div");
  svBg.className = "pa-cpick-sv-bg";
  const svWhite = document.createElement("div");
  svWhite.className = "pa-cpick-sv-white";
  const svBlack = document.createElement("div");
  svBlack.className = "pa-cpick-sv-black";
  const svThumb = document.createElement("div");
  svThumb.className = "pa-cpick-thumb";
  sv.append(svBg, svWhite, svBlack, svThumb);

  const hueRow = document.createElement("div");
  hueRow.className = "pa-cpick-hue";
  const hueTrack = document.createElement("div");
  hueTrack.className = "pa-cpick-hue-track";
  const hueThumb = document.createElement("div");
  hueThumb.className = "pa-cpick-hue-thumb";
  hueRow.append(hueTrack, hueThumb);

  const bottom = document.createElement("div");
  bottom.className = "pa-cpick-bottom";

  const formatCol = document.createElement("div");
  formatCol.className = "pa-cpick-formats";

  const formats: ColorFormat[] = ["rgb", "hex", "hsb"];
  const formatBtns: Record<ColorFormat, HTMLButtonElement> = {} as never;
  for (const f of formats) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pa-cpick-fmt";
    btn.dataset.fmt = f;
    btn.textContent = f.toUpperCase();
    formatCol.append(btn);
    formatBtns[f] = btn;
  }

  const inputWrap = document.createElement("div");
  inputWrap.className = "pa-cpick-input-wrap";
  const input = document.createElement("input");
  input.className = "pa-cpick-input";
  input.spellcheck = false;
  inputWrap.append(input);

  bottom.append(formatCol, inputWrap);
  root.append(sv, hueRow, bottom);
  document.body.append(root);

  const emit = () => {
    const hex = hsvToHex(hsv.h, hsv.s, hsv.v);
    opts.onChange(hex);
  };

  const paintSv = () => {
    svBg.style.background = `hsl(${hsv.h}, 100%, 50%)`;
    svThumb.style.left = `${hsv.s}%`;
    svThumb.style.top = `${100 - hsv.v}%`;
  };

  const paintHue = () => {
    hueThumb.style.left = `${(hsv.h / 360) * 100}%`;
  };

  const paintFormats = () => {
    for (const f of formats) {
      formatBtns[f].classList.toggle("pa-cpick-fmt--active", f === format);
    }
  };

  const paintInput = () => {
    const { r, g, b } = hsvToRgb(hsv.h, hsv.s, hsv.v);
    input.dataset.fmt = format;
    inputWrap.classList.toggle("pa-cpick-input-wrap--hex", format === "hex");
    if (format === "hex") {
      input.value = hsvToHex(hsv.h, hsv.s, hsv.v).slice(1).toUpperCase();
      input.placeholder = "6040FF";
    } else if (format === "rgb") {
      input.value = `${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}`;
      input.placeholder = "96, 64, 255";
    } else {
      input.value = `${Math.round(hsv.h)}, ${Math.round(hsv.s)}, ${Math.round(hsv.v)}`;
      input.placeholder = "250, 75, 100";
    }
  };

  const paint = () => {
    paintSv();
    paintHue();
    paintFormats();
    paintInput();
  };

  const setHsv = (next: Partial<typeof hsv>, fire = true) => {
    hsv = {
      h: next.h ?? hsv.h,
      s: clamp(next.s ?? hsv.s, 0, 100),
      v: clamp(next.v ?? hsv.v, 0, 100),
    };
    paint();
    if (fire) emit();
  };

  const drag = (
    el: HTMLElement,
    onMove: (e: MouseEvent, rect: DOMRect) => void,
  ) => {
    el.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const rect = el.getBoundingClientRect();
      const move = (ev: MouseEvent) => onMove(ev, rect);
      move(e);
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  };

  drag(sv, (e, rect) => {
    const s = clamp(((e.clientX - rect.left) / rect.width) * 100, 0, 100);
    const v = clamp(100 - ((e.clientY - rect.top) / rect.height) * 100, 0, 100);
    setHsv({ s, v });
  });

  drag(hueRow, (e, rect) => {
    const h = clamp(((e.clientX - rect.left) / rect.width) * 360, 0, 360);
    setHsv({ h });
  });

  input.addEventListener("input", (e) => {
    e.stopPropagation();
    const raw = input.value.trim();
    if (format === "hex") {
      const clean = raw.replace(/^#/, "");
      if (/^[0-9a-f]{6}$/i.test(clean)) {
        hsv = hexToHsv("#" + clean);
        paintSv();
        paintHue();
        emit();
      }
    } else if (format === "rgb") {
      const m = raw.match(/(\d+)\D+(\d+)\D+(\d+)/);
      if (m) {
        hsv = rgbToHsv(+m[1], +m[2], +m[3]);
        paintSv();
        paintHue();
        emit();
      }
    } else {
      const m = raw.match(/(\d+)\D+(\d+)\D+(\d+)/);
      if (m) {
        setHsv({ h: +m[1], s: +m[2], v: +m[3] }, true);
      }
    }
  });

  input.addEventListener("keydown", (e) => e.stopPropagation());

  for (const f of formats) {
    formatBtns[f].addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      format = f;
      paintInput();
      paintFormats();
      input.focus();
      input.select();
    });
  }

  const close = () => {
    root.remove();
    document.removeEventListener("mousedown", onOutside, true);
    window.removeEventListener("resize", onReposition);
    window.removeEventListener("scroll", onReposition, true);
    if (activePicker === root) {
      activePicker = null;
      activeAnchor = null;
      activeClose = null;
    }
    opts.onClose?.();
  };

  const onOutside = (e: MouseEvent) => {
    const t = e.target as Node;
    if (!root.contains(t) && !opts.anchor.contains(t)) close();
  };

  const onReposition = () => positionPicker(root, opts.anchor);

  paint();
  positionPicker(root, opts.anchor);
  requestAnimationFrame(() => positionPicker(root, opts.anchor));

  document.addEventListener("mousedown", onOutside, true);
  window.addEventListener("resize", onReposition);
  window.addEventListener("scroll", onReposition, true);

  activePicker = root;
  activeAnchor = opts.anchor;
  activeClose = close;
  return close;
}
