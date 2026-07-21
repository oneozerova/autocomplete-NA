import type { WebModel } from "../types";

const CAP = 10;

interface TrieNode {
  c: Record<string, TrieNode>;
  top: [string, number][];
}

function makeTrie(words: [string, number][]): TrieNode {
  const root: TrieNode = { c: {}, top: [] };
  for (const [w, f] of words) {
    let node = root;
    for (const ch of w) {
      let nx = node.c[ch];
      if (!nx) nx = node.c[ch] = { c: {}, top: [] };
      node = nx;
      if (node.top.length < CAP) node.top.push([w, f]);
    }
  }
  return root;
}

function query(trie: TrieNode, prefix: string): [string, number][] {
  let node: TrieNode | undefined = trie;
  for (const ch of prefix) {
    node = node.c[ch];
    if (!node) return [];
  }
  return node.top;
}

const WORD = /[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*/g;
const TAIL = /[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*$/;

const GHOST_MIN_PREFIX = 3;
const GHOST_TEMP = 2.0;
const GHOST_P_HIGH = 0.55;
const GHOST_P_COVER = 0.80;
const GHOST_CAND_K = 8;

function softmax(scores: number[], temp: number): number[] {
  const m = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - m) / temp));
  const z = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / z);
}

function commonPrefix(strs: string[]): string {
  if (!strs.length) return "";
  let lo = strs[0];
  let hi = strs[0];
  for (const s of strs) {
    if (s < lo) lo = s;
    if (s > hi) hi = s;
  }
  let i = 0;
  while (i < lo.length && i < hi.length && lo[i] === hi[i]) i++;
  return lo.slice(0, i);
}

export function shouldSuggestNextWord(text: string): boolean {
  if (TAIL.test(text)) return false;
  return (text.match(WORD) || []).length >= 2;
}

export class GhostEngine {
  private trie: TrieNode;
  private stop: Set<string>;
  private ctx: WebModel["ctx"];
  private cfg: WebModel["cfg"];
  private morph: WebModel["morph"];
  private agr: WebModel["agr"];
  private readonly uniKeys: number;
  private readonly win: number;
  private readonly decay: number;
  private readonly adjW = 1.0;
  private readonly triW = 0.8;
  private readonly contentW = 0.6;
  private readonly wordW: number;
  private readonly agrW: number;
  private readonly hasAgr: boolean;
  private readonly wordBackoff = -9.0;

  constructor(model: WebModel) {
    this.trie = makeTrie(model.words);
    this.stop = new Set(model.stop || []);
    this.ctx = model.ctx;
    this.cfg = model.cfg;
    this.morph = model.morph || {};
    this.agr = model.agr;
    this.uniKeys = Object.keys(this.ctx.uni).length;
    this.win = this.ctx.window || 1;
    this.decay = this.ctx.decay ?? 0.6;
    this.wordW = this.cfg.word_w ?? 2.0;
    this.agrW = this.cfg.agr_w ?? 4.0;
    this.hasAgr = !!(this.agr && Object.keys(this.morph).length > 0);
  }

  private suf(w: string): string {
    return w.slice(-this.cfg.suf_len);
  }

  private logpWord(prevWord: string, cand: string): number {
    const t = this.ctx.wadj?.[prevWord];
    if (t && t[cand] != null) {
      return Math.log(t[cand] / (this.ctx.wadj_total?.[prevWord] ?? 1));
    }
    return this.wordBackoff;
  }

  private uniLogp(sb: string): number {
    return Math.log(
      ((this.ctx.uni[sb] || 0) + 1) / (this.ctx.uni_total + this.uniKeys + 1),
    );
  }

  private logpAdj(sa: string, sb: string): number {
    const t = this.ctx.adj?.[sa];
    if (t && t[sb] != null) return Math.log(t[sb] / this.ctx.adj_total[sa]);
    return Math.log(0.4) + this.uniLogp(sb);
  }

  private logpTri(sa2: string, sa1: string, sb: string): number {
    const key = sa2 + "|" + sa1;
    const t = this.ctx.tri?.[key];
    if (t && t[sb] != null) return Math.log(t[sb] / this.ctx.tri_total[key]);
    return Math.log(0.4) + this.logpAdj(sa1, sb);
  }

  private logpAt(d: number, sa: string, sb: string): number {
    const tbl = this.ctx.bigram[String(d)];
    const t = tbl?.[sa];
    if (t && t[sb] != null) {
      return Math.log(t[sb] / this.ctx.prev_total[String(d)][sa]);
    }
    return Math.log(0.4) + this.uniLogp(sb);
  }

  private ctxScore(
    rawPrev: string[],
    prevContent: string[],
    cand: string,
  ): number {
    const sb = this.suf(cand);
    let tot = 0;
    let used = false;
    if (rawPrev.length) {
      tot += this.adjW * this.logpAdj(this.suf(rawPrev[0]), sb);
      used = true;
      if (this.wordW) tot += this.wordW * this.logpWord(rawPrev[0], cand);
      if (rawPrev.length >= 2) {
        tot += this.triW * this.logpTri(
          this.suf(rawPrev[1]),
          this.suf(rawPrev[0]),
          sb,
        );
      }
    }
    const lim = Math.min(this.win, prevContent.length);
    for (let d = 1; d <= lim; d++) {
      tot +=
        this.contentW *
        Math.pow(this.decay, d - 1) *
        this.logpAt(d, this.suf(prevContent[d - 1]), sb);
      used = true;
    }
    return used ? tot : this.uniLogp(sb);
  }

  private gnKey(code: string): string {
    return code[2] + code[1];
  }

  private agrLogp(
    tbl: Record<string, Record<string, number>> | undefined,
    gov: string,
    cand: string,
  ): number | null {
    const row = tbl?.[gov];
    if (!row) return null;
    let total = 0;
    let v = 0;
    for (const k in row) {
      total += row[k];
      v++;
    }
    const a = this.agr!.alpha;
    return Math.log(((row[cand] || 0) + a) / (total + a * v));
  }

  private governors(leftCodes: (string | undefined)[]): [string | null, string | null, string | null] {
    let subj: string | null = null;
    let noun: string | null = null;
    let adj: string | null = null;
    for (const c of leftCodes) {
      if (!c) continue;
      if (subj === null && c[0] === "N" && c[3] === "1" && c[1] !== "-") subj = c[1];
      if (noun === null && c[0] === "N") noun = this.gnKey(c);
      if (adj === null && c[0] === "A") adj = this.gnKey(c);
    }
    return [subj, noun, adj];
  }

  private agrScore(
    govs: [string | null, string | null, string | null],
    code: string | undefined,
  ): number | null {
    if (!code || !this.agr) return null;
    const [subj, noun, adj] = govs;
    if (code[0] === "V" && code[1] !== "-" && subj) {
      return this.agrLogp(this.agr.vnum, subj, code[1]);
    }
    if (code[0] === "A" && noun) {
      return this.agrLogp(this.agr.adj_gn, noun, this.gnKey(code));
    }
    if (code[0] === "N" && adj) {
      return this.agrLogp(this.agr.noun_gn, adj, this.gnKey(code));
    }
    return null;
  }

  ghostFor(text: string): { ending: string; partial: boolean } {
    const m = text.match(TAIL);
    if (!m) return { ending: "", partial: false };
    const prefix = m[0].toLowerCase();
    if (prefix.length < GHOST_MIN_PREFIX) return { ending: "", partial: false };

    const left = text.slice(0, text.length - m[0].length);
    const all = (left.match(WORD) || []).map((w) => w.toLowerCase());
    const rawPrev = all.slice(-2).reverse();
    const content = all
      .filter((w) => !this.stop.has(w) && w.length >= 2)
      .reverse()
      .slice(0, this.win);

    const cands = query(this.trie, prefix).filter(([w]) => w !== prefix);
    if (!cands.length) return { ending: "", partial: false };

    let ctxNorm: number[] | null = null;
    if (rawPrev.length || content.length) {
      const cs = cands.map(([w]) => this.ctxScore(rawPrev, content, w));
      const lo = Math.min(...cs);
      const hi = Math.max(...cs);
      const rng = hi - lo || 1;
      ctxNorm = cs.map((v) => (v - lo) / rng);
    }

    let agrTerm: number[] | null = null;
    if (this.hasAgr && all.length && this.agr) {
      const win = this.agr.gov_window || 4;
      const govs = this.governors(
        all
          .slice(-win)
          .reverse()
          .map((w) => this.morph[w]),
      );
      if (govs[0] || govs[1] || govs[2]) {
        const raw = cands.map(([w]) => this.agrScore(govs, this.morph[w]));
        const app = raw.filter((v): v is number => v != null);
        if (app.length >= 2) {
          const mean = app.reduce((a, b) => a + b, 0) / app.length;
          agrTerm = raw.map((v) => (v == null ? 0 : v - mean));
        }
      }
    }

    const scored = cands
      .map(([w, f], i) => ({
        ending: w.slice(prefix.length),
        score:
          this.cfg.alpha * Math.log(f) +
          (ctxNorm ? this.cfg.gamma * ctxNorm[i] : 0) +
          (agrTerm ? this.agrW * agrTerm[i] : 0),
        ok: w.startsWith(prefix),
      }))
      .filter((c) => c.ok)
      .slice(0, GHOST_CAND_K);

    if (!scored.length) return { ending: "", partial: false };
    scored.sort((a, b) => b.score - a.score);

    const probs = softmax(
      scored.map((c) => c.score),
      GHOST_TEMP,
    );
    if (probs[0] >= GHOST_P_HIGH) {
      return { ending: scored[0].ending, partial: false };
    }

    let acc = 0;
    const group: string[] = [];
    for (let i = 0; i < scored.length; i++) {
      group.push(scored[i].ending);
      acc += probs[i];
      if (acc >= GHOST_P_COVER) break;
    }
    const safe = commonPrefix(group);
    if (!safe) return { ending: "", partial: false };
    return { ending: safe, partial: true };
  }
}
