export type ColorIconStyle = "droplet" | "tile" | "ring";

const DROPLET =
  "M12 3.2C9.2 8.2 5.5 11.4 5.5 15.5a6.5 6.5 0 1 0 13 0c0-4.1-3.7-7.3-6.5-12.3z";

export function createColorIcon(
  hex: string,
  style: ColorIconStyle = "droplet",
): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "pa-color-icon");
  svg.setAttribute("aria-hidden", "true");

  if (style === "tile") {
    const r = document.createElementNS(svg.namespaceURI, "rect");
    r.setAttribute("x", "5");
    r.setAttribute("y", "5");
    r.setAttribute("width", "14");
    r.setAttribute("height", "14");
    r.setAttribute("rx", "4");
    r.setAttribute("fill", hex);
    r.setAttribute("stroke", "rgba(255,255,255,0.9)");
    r.setAttribute("stroke-width", "1.5");
    svg.append(r);
    return svg;
  }

  if (style === "ring") {
    const c = document.createElementNS(svg.namespaceURI, "circle");
    c.setAttribute("cx", "12");
    c.setAttribute("cy", "12");
    c.setAttribute("r", "7.5");
    c.setAttribute("fill", "none");
    c.setAttribute("stroke", hex);
    c.setAttribute("stroke-width", "3.5");
    svg.append(c);
    return svg;
  }

  const path = document.createElementNS(svg.namespaceURI, "path");
  path.setAttribute("d", DROPLET);
  path.setAttribute("fill", hex);
  path.setAttribute("stroke", "#fff");
  path.setAttribute("stroke-width", "1.5");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  return svg;
}

export function setColorIconFill(icon: SVGSVGElement, hex: string): void {
  const h = hex.startsWith("#") ? hex : `#${hex}`;
  const path = icon.querySelector("path");
  if (path) {
    path.setAttribute("fill", h);
    return;
  }
  const rect = icon.querySelector("rect");
  if (rect) {
    rect.setAttribute("fill", h);
    return;
  }
  const ring = icon.querySelector("circle[stroke]");
  if (ring) {
    ring.setAttribute("stroke", h);
  }
}
