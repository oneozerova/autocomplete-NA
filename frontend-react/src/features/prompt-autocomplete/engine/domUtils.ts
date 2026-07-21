const BLOCK = new Set(["DIV", "P", "LI"]);

export function isChip(node: Node): node is HTMLElement {
  return node instanceof HTMLElement && node.classList.contains("pa-chip");
}

export function isAux(node: Node): boolean {
  return (
    node instanceof HTMLElement &&
    (node.classList.contains("ghost") || node.classList.contains("phrase"))
  );
}

export function textLen(node: Node): number {
  if (node.nodeType === Node.TEXT_NODE) return node.textContent?.length ?? 0;
  if (isAux(node)) return 0;
  if (isChip(node)) return node.dataset.text?.length ?? 0;
  if (node instanceof HTMLBRElement) return 1;
  let n = 0;
  for (const c of node.childNodes) n += textLen(c);
  return n;
}

export function serialize(root: HTMLElement): string {
  let out = "";
  for (const c of root.childNodes) {
    if (c.nodeType === Node.TEXT_NODE) out += c.textContent ?? "";
    else if (isAux(c)) continue;
    else if (isChip(c)) out += c.dataset.text ?? "";
    else if (c instanceof HTMLBRElement) out += "\n";
    else if (c instanceof HTMLElement) {
      if (BLOCK.has(c.tagName) && out && !out.endsWith("\n")) out += "\n";
      out += serialize(c);
    }
  }
  return out;
}

export function getCaret(root: HTMLElement): number | null {
  const sel = window.getSelection();
  if (!sel?.rangeCount) return null;
  const r = sel.getRangeAt(0);
  if (!root.contains(r.endContainer)) return null;
  const pre = r.cloneRange();
  pre.selectNodeContents(root);
  pre.setEnd(r.endContainer, r.endOffset);
  const frag = pre.cloneContents();
  let n = 0;
  for (const c of frag.childNodes) n += textLen(c);
  return n;
}

export function setCaret(root: HTMLElement, off: number | null): void {
  if (off == null) return;
  let remaining = off;

  const walk = (node: Node): boolean => {
    for (const c of node.childNodes) {
      if (isAux(c)) continue;
      if (c.nodeType === Node.TEXT_NODE) {
        const len = c.textContent?.length ?? 0;
        if (remaining <= len) {
          const r = document.createRange();
          r.setStart(c, remaining);
          r.collapse(true);
          const sel = window.getSelection();
          sel?.removeAllRanges();
          sel?.addRange(r);
          return true;
        }
        remaining -= len;
      } else if (isChip(c)) {
        const len = c.dataset.text?.length ?? 0;
        if (remaining <= len) {
          const r = document.createRange();
          if (remaining === 0) r.setStartBefore(c);
          else r.setStartAfter(c);
          r.collapse(true);
          const sel = window.getSelection();
          sel?.removeAllRanges();
          sel?.addRange(r);
          return true;
        }
        remaining -= len;
      } else if (c instanceof HTMLBRElement) {
        if (remaining <= 1) {
          const r = document.createRange();
          r.setStartAfter(c);
          r.collapse(true);
          const sel = window.getSelection();
          sel?.removeAllRanges();
          sel?.addRange(r);
          return true;
        }
        remaining -= 1;
      } else if (walk(c)) return true;
    }
    return false;
  };

  if (!walk(root)) {
    const r = document.createRange();
    r.selectNodeContents(root);
    r.collapse(false);
    const sel = window.getSelection();
    sel?.removeAllRanges();
    sel?.addRange(r);
  }
}

export function appendText(frag: DocumentFragment, s: string): void {
  s.split("\n").forEach((part, i) => {
    if (i) frag.append(document.createElement("br"));
    if (part) frag.append(document.createTextNode(part));
  });
}
