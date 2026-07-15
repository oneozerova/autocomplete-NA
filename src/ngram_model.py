"""Character-level n-gram language model over Russian word forms.

This is the "Bayesian"/Markov core of the completer: it estimates
    P(next_char | previous up-to-(n-1) chars)
from the frequency-weighted vocabulary, using Katz-style *stupid backoff*
(a fast, well-known smoothing that degrades gracefully to shorter contexts).

Words are framed with a start marker "^" and an end marker "$" so the model
learns where words *end* — which is exactly the morphological-ending signal we
need (e.g. after the stem "красив" the model puts mass on -ый/-ая/-ое/-ые).

The model is deliberately tiny and pure-Python: no torch, instant load,
trivially embeddable as a feature in a larger service.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

BOS = "^"   # start-of-word
EOS = "$"   # end-of-word
BACKOFF = 0.4  # stupid-backoff discount


class CharNGram:
    def __init__(self, order: int = 6):
        self.order = order
        # context (str, len 0..order-1) -> {next_char: count}
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_totals: dict[str, int] = defaultdict(int)
        self.alphabet: set[str] = set()

    # ---- training -------------------------------------------------------
    def add_word(self, word: str, weight: int = 1) -> None:
        seq = BOS + word + EOS
        self.alphabet.update(word)
        n = self.order
        for i in range(1, len(seq)):
            nxt = seq[i]
            ctx_full = seq[max(0, i - (n - 1)):i]
            # store the full-length context AND every shorter suffix so that
            # backoff lookups are O(order) dict hits at inference time.
            for j in range(len(ctx_full) + 1):
                ctx = ctx_full[len(ctx_full) - j:] if j else ""
                self.counts[ctx][nxt] += weight
                self.context_totals[ctx] += weight

    @staticmethod
    def freq_weight(count: int) -> int:
        # Compress the huge dynamic range of corpus counts so that a handful of
        # ultra-frequent function words don't swamp the ending statistics,
        # while still letting frequency inform the model. ~ log2 scaling.
        return 1 + int(math.log2(count))

    def prune(self, min_count_by_len: dict[int, int]) -> None:
        """Drop rare (context,next_char) entries to keep the model tiny.

        `min_count_by_len` maps a context length to the minimum count required
        to keep an entry; longer contexts are pruned more aggressively since
        they are the memory hogs and least generalisable. Totals are recomputed
        from what survives (fine for stupid backoff)."""
        for ctx in list(self.counts.keys()):
            thr = min_count_by_len.get(len(ctx), 1)
            tbl = self.counts[ctx]
            kept = {c: n for c, n in tbl.items() if n >= thr}
            if kept:
                self.counts[ctx] = defaultdict(int, kept)
                self.context_totals[ctx] = sum(kept.values())
            else:
                del self.counts[ctx]
                self.context_totals.pop(ctx, None)

    # ---- scoring --------------------------------------------------------
    def logprob_next(self, context: str, ch: str) -> float:
        """log P(ch | context) with stupid backoff."""
        ctx = context[-(self.order - 1):]
        penalty = 0.0
        while True:
            table = self.counts.get(ctx)
            if table and ch in table:
                return math.log(table[ch] / self.context_totals[ctx]) + penalty
            if ctx == "":
                # unseen char: uniform over alphabet (+EOS) fallback
                return math.log(1.0 / (len(self.alphabet) + 1)) + penalty
            ctx = ctx[1:]
            penalty += math.log(BACKOFF)

    def next_char_dist(self, context: str):
        """Return {char: prob} for the current context (highest-order seen)."""
        ctx = context[-(self.order - 1):]
        while ctx not in self.counts and ctx != "":
            ctx = ctx[1:]
        table = self.counts.get(ctx, {})
        total = self.context_totals.get(ctx, 0)
        if not total:
            return {}
        return {c: n / total for c, n in table.items()}

    # ---- completion via beam search ------------------------------------
    def complete(self, prefix: str, k: int = 5, beam: int = 24, max_len: int = 14):
        """Generate up to k full-word completions of `prefix`.

        Returns list of (word, logprob) sorted by logprob desc. `word` includes
        the prefix. Search runs char-by-char until EOS.
        """
        start_ctx = BOS + prefix
        # beam entries: (accumulated_logprob, generated_suffix, finished)
        beams = [(0.0, "", False)]
        finished: list[tuple[float, str]] = []
        for _ in range(max_len):
            cand: list[tuple[float, str, bool]] = []
            for lp, suf, done in beams:
                if done:
                    finished.append((lp, suf))
                    continue
                ctx = start_ctx + suf
                dist = self.next_char_dist(ctx)
                if not dist:
                    finished.append((lp, suf))
                    continue
                # expand the most promising chars only
                for ch, p in sorted(dist.items(), key=lambda x: -x[1])[:beam]:
                    nlp = lp + math.log(p)
                    if ch == EOS:
                        finished.append((nlp, suf))
                    else:
                        cand.append((nlp, suf + ch, False))
            if not cand:
                break
            cand.sort(key=lambda x: -x[0])
            beams = cand[:beam]
        finished.extend((lp, suf) for lp, suf, _ in beams)
        # length-normalise so short/long endings compete fairly
        out = {}
        for lp, suf in finished:
            word = prefix + suf
            if word == prefix:
                continue
            norm = lp / max(1, len(suf))
            if word not in out or norm > out[word]:
                out[word] = norm
        return sorted(out.items(), key=lambda x: -x[1])[:k]

    # ---- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "alphabet": "".join(sorted(self.alphabet)),
            "counts": {ctx: tbl for ctx, tbl in self.counts.items()},
            "context_totals": dict(self.context_totals),
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharNGram":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(order=d["order"])
        m.alphabet = set(d["alphabet"])
        m.counts = defaultdict(lambda: defaultdict(int),
                               {c: defaultdict(int, t) for c, t in d["counts"].items()})
        m.context_totals = defaultdict(int, d["context_totals"])
        return m
