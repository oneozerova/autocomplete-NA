/** Angle chip pickers (orbit + semicircle). */

const CX = 74;
const CY = 74;
const R = 58;
const CARDINAL = [0, 90, 180, 270] as const;

function norm360(v: number): number {
  return ((Math.round(v) % 360) + 360) % 360;
}

/** 0° = up, 90° = right. */
function armRad(deg: number): number {
  return ((deg - 90) * Math.PI) / 180;
}

function pointOnCircle(deg: number, r = R): { x: number; y: number } {
  const a = armRad(deg);
  return { x: CX + r * Math.cos(a), y: CY + r * Math.sin(a) };
}

function angleFromPointer(clientX: number, clientY: number, rect: DOMRect): number {
  const dx = clientX - (rect.left + rect.width / 2);
  const dy = clientY - (rect.top + rect.height / 2);
  return norm360((Math.atan2(dy, dx) * 180) / Math.PI + 90);
}

export type AnglePickerStyle = "protractor" | "minimal";

export interface AnglePickerUi {
  root: HTMLElement;
  setValue: (deg: number) => void;
}

function isCoarsePointer(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches;
}

function hapticTick(): void {
  if (!isCoarsePointer()) return;
  try {
    navigator.vibrate?.(7);
  } catch {
    /* ignore */
  }
}

function bindPointerDrag(
  el: Element,
  onDelta: (delta: number, e: PointerEvent) => void,
): void {
  el.addEventListener("pointerdown", (e) => {
    const pe = e as PointerEvent;
    pe.preventDefault();
    pe.stopPropagation();
    el.setPointerCapture(pe.pointerId);
    const move = (ev: Event) => {
      onDelta(0, ev as PointerEvent);
    };
    const up = (ev: Event) => {
      const upe = ev as PointerEvent;
      el.releasePointerCapture(upe.pointerId);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  });
}

export function createAnglePickerUi(
  initial: number,
  onChange: (deg: number) => void,
  style: AnglePickerStyle = "protractor",
): AnglePickerUi {
  if (style === "minimal") {
    return createSemicircleAnglePickerUi(initial, onChange);
  }
  return createProtractorAnglePickerUi(initial, onChange);
}

/* Semicircle angle picker */
const SC_W = 256;
const SC_H = 148;
const SC_CX = SC_W / 2;
const SC_CY = SC_H - 14;
const SC_R = 100;
const SC_TICK_IN = SC_R - 20;
const SC_TICK_OUT = SC_R - 3;
const SC_LABEL_R = SC_R - 34;
const SC_HALF = 90;
const MARKER_Y = SC_CY - SC_R + 8;

function dialPoint(deg: number, r = SC_R): { x: number; y: number } {
  const a = armRad(deg);
  return { x: SC_CX + r * Math.cos(a), y: SC_CY + r * Math.sin(a) };
}

function screenDegFromOffset(offset: number): number {
  return ((offset % 360) + 360) % 360;
}

function pointerAngle(clientX: number, clientY: number, rect: DOMRect): number {
  const sx = rect.width / SC_W;
  const sy = rect.height / SC_H;
  const cx = rect.left + SC_CX * sx;
  const cy = rect.top + SC_CY * sy;
  const dx = clientX - cx;
  const dy = clientY - cy;
  return norm360((Math.atan2(dy, dx) * 180) / Math.PI + 90);
}

function createSemicircleAnglePickerUi(
  initial: number,
  onChange: (deg: number) => void,
): AnglePickerUi {
  let val = norm360(initial);
  let lastHapticVal = val;
  let lastPtrAngle = 0;

  const root = document.createElement("div");
  root.className = "pa-angle-semicircle";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${SC_W} ${SC_H}`);
  svg.setAttribute("class", "pa-angle-semicircle-svg");
  svg.setAttribute("aria-hidden", "true");

  const defs = document.createElementNS(svg.namespaceURI, "defs");
  const trackGrad = document.createElementNS(svg.namespaceURI, "linearGradient");
  trackGrad.setAttribute("id", "pa-sc-track-grad");
  trackGrad.setAttribute("x1", "0%");
  trackGrad.setAttribute("y1", "0%");
  trackGrad.setAttribute("x2", "100%");
  trackGrad.setAttribute("y2", "0%");
  const tg1 = document.createElementNS(svg.namespaceURI, "stop");
  tg1.setAttribute("offset", "0%");
  tg1.setAttribute("stop-color", "#6b7cff");
  tg1.setAttribute("stop-opacity", "0.45");
  const tg2 = document.createElementNS(svg.namespaceURI, "stop");
  tg2.setAttribute("offset", "50%");
  tg2.setAttribute("stop-color", "#b06cff");
  tg2.setAttribute("stop-opacity", "0.55");
  const tg3 = document.createElementNS(svg.namespaceURI, "stop");
  tg3.setAttribute("offset", "100%");
  tg3.setAttribute("stop-color", "#ff9a6c");
  tg3.setAttribute("stop-opacity", "0.45");
  trackGrad.append(tg1, tg2, tg3);
  defs.append(trackGrad);
  svg.append(defs);

  const glass = document.createElementNS(svg.namespaceURI, "path");
  glass.setAttribute("class", "pa-angle-semicircle-glass");
  const gLeft = dialPoint(270, SC_R - 8);
  const gRight = dialPoint(90, SC_R - 8);
  glass.setAttribute(
    "d",
    `M ${SC_CX - SC_R - 4} ${SC_CY} L ${gLeft.x} ${gLeft.y} A ${SC_R - 8} ${SC_R - 8} 0 0 1 ${gRight.x} ${gRight.y} L ${SC_CX + SC_R + 4} ${SC_CY} Z`,
  );

  const track = document.createElementNS(svg.namespaceURI, "path");
  track.setAttribute("class", "pa-angle-semicircle-track");
  const tLeft = dialPoint(270);
  const tRight = dialPoint(90);
  track.setAttribute(
    "d",
    `M ${tLeft.x} ${tLeft.y} A ${SC_R} ${SC_R} 0 0 1 ${tRight.x} ${tRight.y}`,
  );
  track.setAttribute("stroke", "url(#pa-sc-track-grad)");

  const baseline = document.createElementNS(svg.namespaceURI, "line");
  baseline.setAttribute("class", "pa-angle-semicircle-base");
  baseline.setAttribute("x1", String(SC_CX - SC_R - 4));
  baseline.setAttribute("y1", String(SC_CY));
  baseline.setAttribute("x2", String(SC_CX + SC_R + 4));
  baseline.setAttribute("y2", String(SC_CY));

  const ticksG = document.createElementNS(svg.namespaceURI, "g");
  ticksG.setAttribute("class", "pa-angle-semicircle-ticks");

  const marker = document.createElementNS(svg.namespaceURI, "path");
  marker.setAttribute("class", "pa-angle-semicircle-marker");
  marker.setAttribute(
    "d",
    `M ${SC_CX} ${MARKER_Y} L ${SC_CX - 7} ${MARKER_Y + 14} L ${SC_CX + 7} ${MARKER_Y + 14} Z`,
  );

  const markerRing = document.createElementNS(svg.namespaceURI, "circle");
  markerRing.setAttribute("class", "pa-angle-semicircle-marker-ring");
  markerRing.setAttribute("cx", String(SC_CX));
  markerRing.setAttribute("cy", String(MARKER_Y + 18));
  markerRing.setAttribute("r", "3");

  svg.append(glass, track, baseline, ticksG, marker, markerRing);

  const valLbl = document.createElement("span");
  valLbl.className = "pa-angle-semicircle-val";

  const maybeHaptic = (next: number) => {
    if (Math.abs(next - lastHapticVal) >= 4) {
      hapticTick();
      lastHapticVal = next;
    }
  };

  const paintScale = () => {
    while (ticksG.firstChild) ticksG.removeChild(ticksG.firstChild);

    const start = val - SC_HALF;
    const end = val + SC_HALF;
    const first = Math.ceil(start / 5) * 5;

    for (let a = first; a <= end + 0.001; a += 5) {
      const offset = a - val;
      if (offset < -SC_HALF || offset > SC_HALF) continue;

      const screenDeg = screenDegFromOffset(offset);
      const angleVal = norm360(a);
      const major = angleVal % 45 === 0;
      const mid = angleVal % 15 === 0;

      const p1 = dialPoint(
        screenDeg,
        major ? SC_TICK_IN : mid ? SC_TICK_IN + 3 : SC_TICK_IN + 6,
      );
      const p2 = dialPoint(screenDeg, major ? SC_TICK_OUT : SC_TICK_OUT - 2);
      const tick = document.createElementNS(svg.namespaceURI, "line");
      tick.setAttribute("x1", String(p1.x));
      tick.setAttribute("y1", String(p1.y));
      tick.setAttribute("x2", String(p2.x));
      tick.setAttribute("y2", String(p2.y));
      tick.setAttribute(
        "class",
        major
          ? "pa-angle-semicircle-tick pa-angle-semicircle-tick--major"
          : "pa-angle-semicircle-tick",
      );
      ticksG.append(tick);

      if (major) {
        const lp = dialPoint(screenDeg, SC_LABEL_R);
        const lbl = document.createElementNS(svg.namespaceURI, "text");
        lbl.setAttribute("class", "pa-angle-semicircle-label");
        lbl.setAttribute("x", String(lp.x));
        lbl.setAttribute("y", String(lp.y));
        lbl.setAttribute("text-anchor", "middle");
        lbl.setAttribute("dominant-baseline", "middle");
        lbl.textContent = String(angleVal);
        ticksG.append(lbl);
      }
    }
  };

  const paint = (fire: boolean) => {
    val = ((val % 360) + 360) % 360;
    const shown = Math.round(val) % 360;
    valLbl.textContent = shown + "°";
    paintScale();
    if (fire) {
      maybeHaptic(shown);
      onChange(shown);
    }
  };

  const apply = (next: number, fire = true) => {
    val = ((next % 360) + 360) % 360;
    paint(fire);
  };

  bindPointerDrag(svg, (_zero, e) => {
    const rect = svg.getBoundingClientRect();
    const a = pointerAngle(e.clientX, e.clientY, rect);
    let delta = a - lastPtrAngle;
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    lastPtrAngle = a;
    apply(val - delta);
  });

  svg.addEventListener(
    "pointerdown",
    (e) => {
      const rect = svg.getBoundingClientRect();
      lastPtrAngle = pointerAngle(e.clientX, e.clientY, rect);
    },
    true,
  );

  root.append(svg, valLbl);
  paint(false);

  return {
    root,
    setValue: (deg: number) => {
      val = norm360(deg);
      lastHapticVal = val;
      paint(false);
    },
  };
}

function createProtractorAnglePickerUi(
  initial: number,
  onChange: (deg: number) => void,
): AnglePickerUi {
  let val = norm360(initial);

  const root = document.createElement("div");
  root.className = "pa-angle-tool";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 148 148");
  svg.setAttribute("class", "pa-angle-dial");
  svg.setAttribute("aria-hidden", "true");

  const ring = document.createElementNS(svg.namespaceURI, "circle");
  ring.setAttribute("cx", String(CX));
  ring.setAttribute("cy", String(CY));
  ring.setAttribute("r", String(R));
  ring.setAttribute("class", "pa-angle-dial-ring");

  const ticksG = document.createElementNS(svg.namespaceURI, "g");
  for (let d = 0; d < 360; d += 45) {
    const p1 = pointOnCircle(d, R - 8);
    const p2 = pointOnCircle(d, R);
    const tick = document.createElementNS(svg.namespaceURI, "line");
    tick.setAttribute("x1", String(p1.x));
    tick.setAttribute("y1", String(p1.y));
    tick.setAttribute("x2", String(p2.x));
    tick.setAttribute("y2", String(p2.y));
    tick.setAttribute("class", "pa-angle-dial-tick");
    ticksG.append(tick);
  }

  const arm = document.createElementNS(svg.namespaceURI, "line");
  arm.setAttribute("class", "pa-angle-dial-arm");
  arm.setAttribute("x1", String(CX));
  arm.setAttribute("y1", String(CY));

  const hub = document.createElementNS(svg.namespaceURI, "circle");
  hub.setAttribute("class", "pa-angle-dial-hub");
  hub.setAttribute("cx", String(CX));
  hub.setAttribute("cy", String(CY));
  hub.setAttribute("r", "4");

  const handle = document.createElementNS(svg.namespaceURI, "circle");
  handle.setAttribute("class", "pa-angle-dial-handle");
  handle.setAttribute("r", "6");

  const dialVal = document.createElementNS(svg.namespaceURI, "text");
  dialVal.setAttribute("class", "pa-angle-dial-val");
  dialVal.setAttribute("x", String(CX));
  dialVal.setAttribute("y", String(CY + 1));
  dialVal.setAttribute("text-anchor", "middle");
  dialVal.setAttribute("dominant-baseline", "middle");

  svg.append(ring, ticksG, arm, hub, handle, dialVal);

  const rulerWrap = document.createElement("div");
  rulerWrap.className = "pa-angle-beam";
  const rulerTrack = document.createElement("div");
  rulerTrack.className = "pa-angle-beam-track";
  const rulerSweep = document.createElement("div");
  rulerSweep.className = "pa-angle-beam-sweep";
  const rulerNeedle = document.createElement("div");
  rulerNeedle.className = "pa-angle-beam-needle";
  rulerTrack.append(rulerSweep, rulerNeedle);

  const rulerMarks = document.createElement("div");
  rulerMarks.className = "pa-angle-beam-marks";
  for (const d of CARDINAL) {
    const mark = document.createElement("span");
    mark.className = "pa-angle-beam-mark";
    mark.textContent = String(d);
    mark.style.left = `${(d / 360) * 100}%`;
    mark.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      apply(d, true);
    });
    rulerMarks.append(mark);
  }

  rulerWrap.append(rulerTrack, rulerMarks);
  root.append(svg, rulerWrap);

  const paint = (fire: boolean) => {
    val = norm360(val);
    dialVal.textContent = `${val}°`;
    const tip = pointOnCircle(val);
    arm.setAttribute("x2", String(tip.x));
    arm.setAttribute("y2", String(tip.y));
    handle.setAttribute("cx", String(tip.x));
    handle.setAttribute("cy", String(tip.y));
    const t = (val / 360) * 100;
    rulerSweep.style.width = `${t}%`;
    rulerNeedle.style.left = `${t}%`;
    for (const mark of rulerMarks.querySelectorAll(".pa-angle-beam-mark")) {
      const d = +(mark.textContent || "0");
      mark.classList.toggle("pa-angle-beam-mark--active", d === val);
    }
    if (fire) onChange(val);
  };

  const apply = (next: number, fire = true) => {
    val = norm360(next);
    paint(fire);
  };

  const dragDial = (e: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    apply(angleFromPointer(e.clientX, e.clientY, rect));
  };

  svg.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragDial(e);
    const move = (ev: MouseEvent) => dragDial(ev);
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });

  const dragRuler = (e: MouseEvent) => {
    const rect = rulerTrack.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    apply((x / rect.width) * 360);
  };

  rulerTrack.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragRuler(e);
    const move = (ev: MouseEvent) => dragRuler(ev);
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });

  paint(false);

  return {
    root,
    setValue: (deg: number) => apply(deg, false),
  };
}

/* Orbit angle picker */
const O_SIZE = 224;
const O_C = O_SIZE / 2;
const O_R = 88;
const O_BAND = 12;
const O_TICK_OUT = O_R - 1;
const O_TICK_IN = O_R - O_BAND + 2;

let activeOrbitClose: (() => void) | null = null;
let activeOrbitAnchor: HTMLElement | null = null;

export function closeAngleOrbit(): void {
  activeOrbitClose?.();
}

export function isAngleOrbitOpenFor(anchor: HTMLElement): boolean {
  return activeOrbitAnchor === anchor;
}

export interface AngleOrbitOptions {
  anchor: HTMLElement;
  value: number;
  onChange: (deg: number) => void;
  onClose?: () => void;
}

function orbitPoint(deg: number, r = O_R): { x: number; y: number } {
  const a = armRad(deg);
  return { x: O_C + r * Math.cos(a), y: O_C + r * Math.sin(a) };
}

function positionOrbit(root: HTMLElement, anchor: HTMLElement): void {
  const r = anchor.getBoundingClientRect();
  root.style.left = `${r.left + r.width / 2}px`;
  root.style.top = `${r.top + r.height / 2}px`;
}

export function openAngleOrbit(opts: AngleOrbitOptions): () => void {
  closeAngleOrbit();

  let val = norm360(opts.value);

  const root = document.createElement("div");
  root.className = "pa-angle-orbit";
  root.style.setProperty("--pa-orbit-r", `${O_SIZE / 2}px`);
  const fieldWrap = opts.anchor.closest(".pa-field-wrap") as HTMLElement | null;
  const anchorRect = opts.anchor.getBoundingClientRect();
  const holeR = Math.max(anchorRect.width, anchorRect.height) / 2 + 18;
  root.style.setProperty("--pa-orbit-hole", `${holeR}px`);
  root.style.setProperty("--pa-orbit-outer", `${O_R + 5}px`);
  if (fieldWrap) {
    const fieldBg = getComputedStyle(fieldWrap).backgroundColor;
    root.style.setProperty("--pa-orbit-center-bg", fieldBg);
  }

  const blur = document.createElement("div");
  blur.className = "pa-angle-orbit-blur";

  const glass = document.createElement("div");
  glass.className = "pa-angle-orbit-glass";

  const center = document.createElement("div");
  center.className = "pa-angle-orbit-center";

  const chipFace = document.createElement("div");
  chipFace.className = "pa-angle-orbit-chip";
  chipFace.textContent = `${Math.round(val)}°`;
  const anchorStyle = getComputedStyle(opts.anchor);
  chipFace.style.minWidth = `${anchorRect.width}px`;
  chipFace.style.minHeight = `${anchorRect.height}px`;
  chipFace.style.fontSize = anchorStyle.fontSize;
  chipFace.style.padding = anchorStyle.padding;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${O_SIZE} ${O_SIZE}`);
  svg.setAttribute("class", "pa-angle-orbit-svg");
  svg.setAttribute("aria-hidden", "true");

  const defs = document.createElementNS(svg.namespaceURI, "defs");

  const sweepGrad = document.createElementNS(svg.namespaceURI, "linearGradient");
  sweepGrad.setAttribute("id", "pa-orbit-sweep-grad");
  sweepGrad.setAttribute("gradientUnits", "userSpaceOnUse");
  sweepGrad.setAttribute("x1", String(O_C));
  sweepGrad.setAttribute("y1", String(O_C - O_R));
  sweepGrad.setAttribute("x2", String(O_C + O_R));
  sweepGrad.setAttribute("y2", String(O_C));
  const sg1 = document.createElementNS(svg.namespaceURI, "stop");
  sg1.setAttribute("offset", "0%");
  sg1.setAttribute("stop-color", "#6b7cff");
  sg1.setAttribute("stop-opacity", "0.14");
  const sg2 = document.createElementNS(svg.namespaceURI, "stop");
  sg2.setAttribute("offset", "55%");
  sg2.setAttribute("stop-color", "#b06cff");
  sg2.setAttribute("stop-opacity", "0.1");
  const sg3 = document.createElementNS(svg.namespaceURI, "stop");
  sg3.setAttribute("offset", "100%");
  sg3.setAttribute("stop-color", "#ff9a6c");
  sg3.setAttribute("stop-opacity", "0.08");
  sweepGrad.append(sg1, sg2, sg3);

  const handleGrad = document.createElementNS(svg.namespaceURI, "linearGradient");
  handleGrad.setAttribute("id", "pa-orbit-handle-grad");
  handleGrad.setAttribute("gradientUnits", "userSpaceOnUse");
  handleGrad.setAttribute("x1", "0%");
  handleGrad.setAttribute("y1", "0%");
  handleGrad.setAttribute("x2", "100%");
  handleGrad.setAttribute("y2", "100%");
  const hg1 = document.createElementNS(svg.namespaceURI, "stop");
  hg1.setAttribute("offset", "0%");
  hg1.setAttribute("stop-color", "#6b7cff");
  const hg2 = document.createElementNS(svg.namespaceURI, "stop");
  hg2.setAttribute("offset", "48%");
  hg2.setAttribute("stop-color", "#b06cff");
  const hg3 = document.createElementNS(svg.namespaceURI, "stop");
  hg3.setAttribute("offset", "100%");
  hg3.setAttribute("stop-color", "#ff9a6c");
  handleGrad.append(hg1, hg2, hg3);

  const armGrad = document.createElementNS(svg.namespaceURI, "linearGradient");
  armGrad.setAttribute("id", "pa-orbit-arm-grad");
  armGrad.setAttribute("gradientUnits", "userSpaceOnUse");
  armGrad.setAttribute("x1", String(O_C));
  armGrad.setAttribute("y1", String(O_C));
  armGrad.setAttribute("x2", String(O_C));
  armGrad.setAttribute("y2", String(O_C - O_R));
  const ag1 = document.createElementNS(svg.namespaceURI, "stop");
  ag1.setAttribute("offset", "0%");
  ag1.setAttribute("stop-color", "rgba(255,255,255,0.9)");
  const ag2 = document.createElementNS(svg.namespaceURI, "stop");
  ag2.setAttribute("offset", "70%");
  ag2.setAttribute("stop-color", "rgba(107,124,255,0.55)");
  const ag3 = document.createElementNS(svg.namespaceURI, "stop");
  ag3.setAttribute("offset", "100%");
  ag3.setAttribute("stop-color", "#b06cff");
  armGrad.append(ag1, ag2, ag3);

  defs.append(sweepGrad, handleGrad, armGrad);
  svg.append(defs);

  const sweep = document.createElementNS(svg.namespaceURI, "path");
  sweep.setAttribute("class", "pa-angle-orbit-sweep");
  sweep.setAttribute("fill", "url(#pa-orbit-sweep-grad)");

  const innerRing = document.createElementNS(svg.namespaceURI, "circle");
  innerRing.setAttribute("cx", String(O_C));
  innerRing.setAttribute("cy", String(O_C));
  innerRing.setAttribute("r", String(O_R - O_BAND));
  innerRing.setAttribute("class", "pa-angle-orbit-ring pa-angle-orbit-ring--inner");

  const outerRing = document.createElementNS(svg.namespaceURI, "circle");
  outerRing.setAttribute("cx", String(O_C));
  outerRing.setAttribute("cy", String(O_C));
  outerRing.setAttribute("r", String(O_R));
  outerRing.setAttribute("class", "pa-angle-orbit-ring pa-angle-orbit-ring--outer");

  const track = document.createElementNS(svg.namespaceURI, "circle");
  track.setAttribute("cx", String(O_C));
  track.setAttribute("cy", String(O_C));
  track.setAttribute("r", String(O_R - O_BAND / 2));
  track.setAttribute("class", "pa-angle-orbit-track");

  const ticksG = document.createElementNS(svg.namespaceURI, "g");
  ticksG.setAttribute("class", "pa-angle-orbit-ticks");
  for (let d = 0; d < 360; d += 15) {
    const major = d % 45 === 0;
    const p1 = orbitPoint(d, major ? O_TICK_IN : O_TICK_IN + 2);
    const p2 = orbitPoint(d, major ? O_TICK_OUT : O_TICK_OUT - 1);
    const tick = document.createElementNS(svg.namespaceURI, "line");
    tick.setAttribute("x1", String(p1.x));
    tick.setAttribute("y1", String(p1.y));
    tick.setAttribute("x2", String(p2.x));
    tick.setAttribute("y2", String(p2.y));
    tick.setAttribute("class", major ? "pa-angle-orbit-tick pa-angle-orbit-tick--major" : "pa-angle-orbit-tick");
    ticksG.append(tick);
  }

  const rulerG = document.createElementNS(svg.namespaceURI, "g");
  rulerG.setAttribute("class", "pa-angle-orbit-ruler");

  const rulerArm = document.createElementNS(svg.namespaceURI, "line");
  rulerArm.setAttribute("class", "pa-angle-orbit-arm");
  rulerArm.setAttribute("x1", String(O_C));
  rulerArm.setAttribute("y1", String(O_C));
  rulerArm.setAttribute("stroke", "url(#pa-orbit-arm-grad)");

  const handle = document.createElementNS(svg.namespaceURI, "circle");
  handle.setAttribute("class", "pa-angle-orbit-handle");
  handle.setAttribute("r", "5.5");
  handle.setAttribute("fill", "url(#pa-orbit-handle-grad)");
  const handleRing = document.createElementNS(svg.namespaceURI, "circle");
  handleRing.setAttribute("class", "pa-angle-orbit-handle-ring");
  handleRing.setAttribute("r", "7.5");

  rulerG.append(rulerArm, handleRing, handle);
  svg.append(sweep, innerRing, outerRing, track, ticksG, rulerG);
  root.append(blur, glass, center, svg, chipFace);
  root.style.visibility = "hidden";
  document.body.append(root);

  const revealOrbit = () => {
    positionOrbit(root, opts.anchor);
    root.style.visibility = "";
  };

  const piePath = (deg: number): string => {
    if (deg <= 0) return "";
    const r = O_R - O_BAND / 2;
    const tip = orbitPoint(deg, r);
    const large = deg > 180 ? 1 : 0;
    return `M ${O_C} ${O_C} L ${O_C} ${O_C - r} A ${r} ${r} 0 ${large} 1 ${tip.x} ${tip.y} Z`;
  };

  const updateArmGrad = (tip: { x: number; y: number }) => {
    armGrad.setAttribute("x2", String(tip.x));
    armGrad.setAttribute("y2", String(tip.y));
  };

  const paint = (fire: boolean) => {
    val = norm360(val);
    const tip = orbitPoint(val, O_R - O_BAND / 2);
    sweep.setAttribute("d", piePath(val));
    rulerArm.setAttribute("x2", String(tip.x));
    rulerArm.setAttribute("y2", String(tip.y));
    updateArmGrad(tip);
    handle.setAttribute("cx", String(tip.x));
    handle.setAttribute("cy", String(tip.y));
    handleRing.setAttribute("cx", String(tip.x));
    handleRing.setAttribute("cy", String(tip.y));
    chipFace.textContent = `${Math.round(val)}°`;
    if (fire) opts.onChange(val);
  };

  const apply = (next: number, fire = true) => {
    val = norm360(next);
    paint(fire);
  };

  const drag = (e: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    apply(angleFromPointer(e.clientX, e.clientY, rect));
  };

  const startDrag = (e: Event) => {
    const me = e as MouseEvent;
    me.preventDefault();
    me.stopPropagation();
    drag(me);
    const move = (ev: Event) => drag(ev as MouseEvent);
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };

  svg.style.pointerEvents = "none";
  track.setAttribute("pointer-events", "stroke");
  ticksG.setAttribute("pointer-events", "none");
  sweep.setAttribute("pointer-events", "none");
  rulerG.setAttribute("pointer-events", "all");
  track.addEventListener("mousedown", startDrag);
  rulerG.addEventListener("mousedown", startDrag);

  const close = () => {
    root.remove();
    document.removeEventListener("mousedown", onOutside, true);
    window.removeEventListener("resize", onReposition);
    window.removeEventListener("scroll", onReposition, true);
    opts.anchor.classList.remove("pa-angle--orbit-open");
    fieldWrap?.classList.remove("pa-field-wrap--orbit");
    if (activeOrbitAnchor === opts.anchor) {
      activeOrbitAnchor = null;
      activeOrbitClose = null;
    }
    opts.onClose?.();
  };

  const onOutside = (e: MouseEvent) => {
    const t = e.target as Node;
    if (!root.contains(t) && !opts.anchor.contains(t)) close();
  };

  const onReposition = () => positionOrbit(root, opts.anchor);

  paint(false);
  opts.anchor.classList.add("pa-angle--orbit-open");
  fieldWrap?.classList.add("pa-field-wrap--orbit");
  revealOrbit();
  requestAnimationFrame(revealOrbit);

  document.addEventListener("mousedown", onOutside, true);
  window.addEventListener("resize", onReposition);
  window.addEventListener("scroll", onReposition, true);

  activeOrbitAnchor = opts.anchor;
  activeOrbitClose = close;

  return close;
}
