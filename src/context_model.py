"""Context model for agreement — windowed, для сложных предложений.

Agreement is learned statistically (no parser, no morphology library). The naive
version looked only at the *immediately* preceding word, which breaks in long
sentences: across conjunctions and participial / adverbial phrases the word that
governs gender/number/case sits several tokens back:

    девушка, идущая по мосту, красивая     (красивая agrees with «девушка», not «мосту»)
    старик и его верн[ый] пёс              (across the conjunction «и»)

The model combines **two complementary signals**:
  1. **Adjacency** — `P(suffix(w) | suffix(immediate previous word))`, over *raw*
     tokens. This keeps the strong local signal, including prepositions that
     govern case (в лес[у], с друзь[ями]) and adjacent agreement (красив[ая]
     девушк[а]).
  2. **Content window** — the same, but over the last K *content* words with
     function words (conjunctions / prepositions / particles) filtered out and a
     distance decay. This lets agreement reach *past* an inserted participial /
     adverbial phrase or a conjunction to the real head word:
         девушка, идущая по мосту, красив[ая]   (agrees with «девушка»)

Both are plain Bayesian/Markov suffix statistics — tiny, fast, no parser.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*")
BACKOFF = 0.4

# Function words that do NOT govern agreement of a following content word.
# Filtered out so the window spans real content words. (Deliberately excludes
# nouns/adjectives/verbs, which DO carry agreement.)
STOPWORDS = set("""
и а но или да ни же бы б ли ль не ведь вот как то это эти эта этот тот та те такой
такая такое такие который которая которое которые чей что чтоб чтобы если когда пока
хотя потому оттого зато однако либо тоже также лишь только даже уж уже ещё вон там тут
здесь очень весьма почти совсем совершенно
в во на над под подо за перед передо при о об обо от ото до из изо к ко с со у про для
без через сквозь между меж около возле вокруг вдоль поперёк среди ради насчёт кроме
кругом мимо
я ты он она оно мы вы они меня тебя него неё нас вас них мне тебе ему ей нам вам им себя
себе собой свой своя своё свои мой моя моё мои твой твоя твоё твои наш наша наше наши
ваш ваша ваше ваши его её их кто кого кому чем чём
где куда откуда зачем почему отчего сколько столько всегда никогда иногда теперь тогда
сейчас затем потом опять снова вновь
""".split())


def suffix(word: str, n: int = 3) -> str:
    return word[-n:]


ADJ_W = 1.0       # weight of the raw adjacency (bigram) signal
TRI_W = 0.8       # weight of the raw suffix-trigram signal (2-word history)
CONTENT_W = 0.6   # weight of the content-window reach (corrective)

# Lexical collocation signal: a *whole-word* immediate-adjacency bigram
# P(word | previous raw word). The suffix signals above capture agreement
# (gender/number/case) but are blind to *which word* fits — "на изображении",
# "чёткий фокус", "масляная живопись" are lexical, not morphological. This adds
# that, keeping everything an n-gram (O(1) dict lookups, no parser). Trained over
# raw adjacent pairs so case-governing prepositions ("на"→"изображение") count,
# even though they are stop-words for the content window.
WORD_W = 2.0          # weight of the lexical collocation signal (tuned, eval_context:
                      # top-1 36.5%→40.2% vs off; saturates beyond ~2.0)
WORD_BACKOFF = -9.0   # log-score floor for an unseen (prev, cand) pair
WORD_CAP = 30         # keep at most this many successors per previous word


class ContextModel:
    def __init__(self, suf_len: int = 3, window: int = 4, decay: float = 0.6,
                 word_w: float = WORD_W):
        self.suf_len = suf_len
        self.window = window
        self.decay = decay
        self.tri_w = TRI_W
        self.word_w = word_w
        # signal 3: lexical whole-word adjacency bigram  prev_word -> {cur_word: n}
        self.wadj: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.wadj_total: dict[str, int] = defaultdict(int)
        # signal 1: raw immediate-previous-word suffix bigram
        self.adj: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.adj_total: dict[str, int] = defaultdict(int)
        # signal 1b: raw suffix trigram P(w | w-2, w-1) — captures preposition +
        # adjective -> noun case government ("в тёмн[ом] лес[у]")
        self.tri: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.tri_total: dict[str, int] = defaultdict(int)
        # signal 2: content-word skip-grams, distance d -> {prev_suf: {cur_suf: n}}
        self.bigram: dict[int, dict[str, dict[str, int]]] = {}
        self.prev_total: dict[int, dict[str, int]] = {}
        for d in range(1, window + 1):
            self.bigram[d] = defaultdict(lambda: defaultdict(int))
            self.prev_total[d] = defaultdict(int)
        self.uni: dict[str, int] = defaultdict(int)   # curr_suffix -> count
        self.uni_total: int = 0

    # ---- training -------------------------------------------------------
    @staticmethod
    def content_words(tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

    def add_sentence(self, tokens: list[str], lexical: bool = True) -> None:
        """Accumulate context statistics from one sentence.

        `lexical=False` skips the whole-word collocation bigram (signal 3). Use it
        for the *synthetic* prompt corpus: its templated adjective→noun pairs would
        flood wadj with a handful of stock nouns and evict real collocations
        ("красивые девушки") learned from natural text. The suffix signals still
        benefit from it (they generalise over endings, not specific words)."""
        sl = self.suf_len
        # signal 1: raw adjacency (captures prepositions & adjacent agreement)
        raw = [suffix(t, sl) for t in tokens]
        for i in range(1, len(raw)):
            self.adj[raw[i - 1]][raw[i]] += 1
            self.adj_total[raw[i - 1]] += 1
            if i >= 2:                        # signal 1b: raw suffix trigram
                key = raw[i - 2] + "|" + raw[i - 1]
                self.tri[key][raw[i]] += 1
                self.tri_total[key] += 1
            # signal 3: whole-word collocation. cur must be a content word (the
            # kind of word we actually complete); prev can be anything, incl. a
            # case-governing preposition. Real text only (see `lexical`).
            cur = tokens[i]
            if lexical and len(cur) >= 2 and cur not in STOPWORDS:
                self.wadj[tokens[i - 1]][cur] += 1
                self.wadj_total[tokens[i - 1]] += 1
        # signal 2: content-word window (reaches past function words / inserts)
        sufs = [suffix(w, sl) for w in self.content_words(tokens)]
        for i, sb in enumerate(sufs):
            self.uni[sb] += 1
            self.uni_total += 1
            for d in range(1, self.window + 1):
                if i - d >= 0:
                    self.bigram[d][sufs[i - d]][sb] += 1
                    self.prev_total[d][sufs[i - d]] += 1

    @classmethod
    def from_sentences(cls, sentences, suf_len: int = 3, window: int = 4,
                       min_count: int = 3):
        m = cls(suf_len=suf_len, window=window)
        for toks in sentences:
            if len(toks) >= 2:
                m.add_sentence(toks)
        m.prune(min_count)
        return m

    def prune(self, min_count: int) -> None:
        for a in list(self.adj.keys()):
            kept = {b: c for b, c in self.adj[a].items() if c >= min_count}
            if kept:
                self.adj[a] = defaultdict(int, kept)
                self.adj_total[a] = sum(kept.values())
            else:
                del self.adj[a]
                self.adj_total.pop(a, None)
        for a in list(self.tri.keys()):     # trigram is sparser -> prune a bit harder
            kept = {b: c for b, c in self.tri[a].items() if c >= min_count + 1}
            if kept:
                self.tri[a] = defaultdict(int, kept)
                self.tri_total[a] = sum(kept.values())
            else:
                del self.tri[a]
                self.tri_total.pop(a, None)
        # content skip-grams: prune harder at larger distances (sparser, noisier)
        for d in range(1, self.window + 1):
            thr = min_count + (d - 1)
            big, tot = self.bigram[d], self.prev_total[d]
            for a in list(big.keys()):
                kept = {b: c for b, c in big[a].items() if c >= thr}
                if kept:
                    big[a] = defaultdict(int, kept)
                    tot[a] = sum(kept.values())
                else:
                    del big[a]
                    tot.pop(a, None)
        # lexical word-bigram: prune rare pairs, then cap successors per prev word
        # (keep the most frequent collocates) to bound artifact size.
        wc = min_count + 2   # word bigrams are sparser than suffix bigrams
        for a in list(self.wadj.keys()):
            kept = {b: c for b, c in self.wadj[a].items() if c >= wc}
            if len(kept) > WORD_CAP:
                kept = dict(sorted(kept.items(), key=lambda kv: -kv[1])[:WORD_CAP])
            if kept:
                self.wadj[a] = defaultdict(int, kept)
                self.wadj_total[a] = sum(kept.values())
            else:
                del self.wadj[a]
                self.wadj_total.pop(a, None)

    # ---- scoring --------------------------------------------------------
    def _uni_logp(self, sb: str) -> float:
        return math.log((self.uni.get(sb, 0) + 1) /
                        (self.uni_total + len(self.uni) + 1))

    def _logp_adj(self, sa: str, sb: str) -> float:
        tbl = self.adj.get(sa)
        if tbl and sb in tbl:
            return math.log(tbl[sb] / self.adj_total[sa])
        return math.log(BACKOFF) + self._uni_logp(sb)

    def _logp_tri(self, sa2: str, sa1: str, sb: str) -> float:
        tbl = self.tri.get(sa2 + "|" + sa1)
        if tbl and sb in tbl:
            return math.log(tbl[sb] / self.tri_total[sa2 + "|" + sa1])
        return math.log(BACKOFF) + self._logp_adj(sa1, sb)   # backoff to bigram

    def _logp_at(self, d: int, sa: str, sb: str) -> float:
        tbl = self.bigram[d].get(sa)
        if tbl and sb in tbl:
            return math.log(tbl[sb] / self.prev_total[d][sa])
        return math.log(BACKOFF) + self._uni_logp(sb)

    def _logp_word(self, prev_word: str, cand_word: str) -> float:
        """Lexical collocation: log P(cand_word | prev_word), whole words.
        Unseen pairs get a constant floor, so a known collocate is lifted
        relative to the rest once the completer normalises the context score."""
        tbl = self.wadj.get(prev_word)
        if tbl and cand_word in tbl:
            return math.log(tbl[cand_word] / self.wadj_total[prev_word])
        return WORD_BACKOFF

    def context_logscore(self, raw_prev, content_prev: list[str], cand: str) -> float:
        """Blend adjacency + suffix-trigram + content-window agreement evidence.

        `raw_prev`     = up to 2 previous RAW words, NEAREST FIRST (or a single
                         string / None for backward compatibility).
        `content_prev` = preceding *content* words, NEAREST FIRST (distance 1..).
        """
        if isinstance(raw_prev, str):
            raw_prev = [raw_prev]
        raw_prev = raw_prev or []
        sb = suffix(cand, self.suf_len)
        total, used = 0.0, False
        if raw_prev:
            total += ADJ_W * self._logp_adj(suffix(raw_prev[0], self.suf_len), sb)
            used = True
            if self.word_w:                    # signal 3: lexical collocation
                total += self.word_w * self._logp_word(raw_prev[0], cand)
            if len(raw_prev) >= 2 and self.tri_w:
                total += self.tri_w * self._logp_tri(
                    suffix(raw_prev[1], self.suf_len),
                    suffix(raw_prev[0], self.suf_len), sb)
        for d, pw in enumerate(content_prev[: self.window], start=1):
            total += CONTENT_W * (self.decay ** (d - 1)) * \
                self._logp_at(d, suffix(pw, self.suf_len), sb)
            used = True
        return total if used else self._uni_logp(sb)

    # backward-compatible single-word helper (adjacency only)
    def logprob(self, prev_word, cand_word: str) -> float:
        return self.context_logscore(prev_word, [], cand_word)

    # ---- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "suf_len": self.suf_len, "window": self.window, "decay": self.decay,
            "word_w": self.word_w,
            "adj": {a: t for a, t in self.adj.items()},
            "adj_total": dict(self.adj_total),
            "tri": {a: t for a, t in self.tri.items()},
            "tri_total": dict(self.tri_total),
            "wadj": {a: t for a, t in self.wadj.items()},
            "wadj_total": dict(self.wadj_total),
            "bigram": {str(d): {a: t for a, t in self.bigram[d].items()}
                       for d in self.bigram},
            "prev_total": {str(d): dict(self.prev_total[d]) for d in self.prev_total},
            "uni": dict(self.uni), "uni_total": self.uni_total,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ContextModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(suf_len=d["suf_len"], window=d.get("window", 1),
                decay=d.get("decay", 0.6), word_w=d.get("word_w", WORD_W))
        m.adj = defaultdict(lambda: defaultdict(int),
                            {a: defaultdict(int, t) for a, t in d.get("adj", {}).items()})
        m.adj_total = defaultdict(int, d.get("adj_total", {}))
        m.tri = defaultdict(lambda: defaultdict(int),
                            {a: defaultdict(int, t) for a, t in d.get("tri", {}).items()})
        m.tri_total = defaultdict(int, d.get("tri_total", {}))
        m.wadj = defaultdict(lambda: defaultdict(int),
                             {a: defaultdict(int, t) for a, t in d.get("wadj", {}).items()})
        m.wadj_total = defaultdict(int, d.get("wadj_total", {}))
        for ds, tables in d["bigram"].items():
            di = int(ds)
            m.bigram[di] = defaultdict(lambda: defaultdict(int),
                                       {a: defaultdict(int, t) for a, t in tables.items()})
        for ds, tot in d["prev_total"].items():
            m.prev_total[int(ds)] = defaultdict(int, tot)
        m.uni = defaultdict(int, d["uni"])
        m.uni_total = d["uni_total"]
        return m


# ---- corpus reader ------------------------------------------------------
def tatoeba_sentences(path: Path):
    """Yield token lists from a Tatoeba tsv(.bz2) export (id \\t lang \\t text)."""
    import bz2
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            text = parts[-1].lower()
            toks = WORD_RE.findall(text)
            if toks:
                yield toks
