"""High-level Russian word-ending autocompleter.

Public API (stable — this is what the larger project embeds):

    comp = Completer.load()                 # loads bundled artifacts in models/
    comp.complete("красивый пейза", k=5)    # -> list[Suggestion]

A Suggestion tells the caller the whole word, the *ending* to append after what
the user has already typed, a normalised score and where it came from.

Design: a hybrid of two cheap signals.
  * FreqTrie  — exact, high-precision completion of *known* word forms, ranked
                by corpus frequency. Handles the bulk of real usage.
  * CharNGram — generative char model that supplies morphologically-plausible
                endings, generalises to out-of-vocabulary stems (rare/domain
                prompt words), and provides a smoothing score for ranking.

The two candidate sets are merged and re-ranked by a blended score so that real
frequent words win, but plausible unseen inflections still surface.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .context_model import ContextModel
from .ngram_model import CharNGram
from .trie import FreqTrie

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# a "current word" = trailing run of Russian letters / hyphen the user is typing
TOKEN_RE = re.compile(r"[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*$")
# any Russian word (used to find the left-context word)
ANY_WORD_RE = re.compile(r"[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*")
GAMMA = 9.0  # weight of the agreement (context) signal; tuned in eval_context.py

# Inline ghost text is shown only when the top completion is *confident* (à la
# Gmail Smart Compose's triggering thresholds): a short, ambiguous prefix should
# not commit grey text to a guess. The top-k list (complete()) is unaffected —
# it still offers candidates from 2 characters; this only gates the single inline
# suggestion, trading recall for precision so we stop confidently showing the
# wrong word ("из" → "извините").
GHOST_MIN_PREFIX = 3   # never ghost on a 1-2 char stub

AGR_W = 4.0   # weight of the factored grammatical-agreement signal (морфология),
              # tuned on eval_agreement.py. Applied as a centered log-prob so it
              # only re-orders candidates that share a governor (subject/head).

# --- inline-ghost confidence (softmax over blended scores) + safe completion ---
# A Russian ending is genuinely ambiguous mid-word (после «молод» валидны
# -ой/-ого/-ая/-ые). Rather than always commit the top-1 ending (which is wrong
# ~28% of shown chars — see eval_ghost_chars.py), we turn the blended scores into
# a P(word) via a tempered softmax and act on it two ways: commit the full ending
# only when one form dominates, else show only the leading chars all plausible
# endings share (shell-style "expand as far as is unambiguous"). Constants tuned
# against the wrong-char-rate ↔ coverage curve in eval_ghost_chars.py.
GHOST_TEMP = 2.0     # softmax temperature. Higher ⇒ less peaked ⇒ close-frequency
                     # inflections split their mass and trigger the safe-partial path.
GHOST_P_HIGH = 0.55  # one ending owning ≥ this mass ⇒ commit its FULL completion.
GHOST_P_COVER = 0.80 # else: fewest leaders covering this mass define the "plausible
                     # set"; show only the ending prefix ALL of them agree on.
                     # 0.55/0.80 chosen from eval_ghost_chars.py: wrong-char-rate
                     # 28.0%→16.8% at unchanged usefulness (saved 1.23→1.22) and
                     # higher coverage (61.9%→65.4%). Raise p_high for fewer wrong
                     # endings at some usefulness cost (0.65→13.1%, 0.75→8.9%).
GHOST_CAND_K = 8     # candidates considered for the mass / common-prefix decision.


def _softmax(scores: list[float], temp: float) -> list[float]:
    """Tempered softmax; `scores` are blended, not calibrated, so temp sets scale."""
    m = max(scores)
    exps = [math.exp((s - m) / temp) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _common_prefix(strings: list[str]) -> str:
    """Longest common prefix of a set (== LCP of its lexicographic min and max)."""
    if not strings:
        return ""
    lo, hi = min(strings), max(strings)
    i = 0
    for a, b in zip(lo, hi):
        if a != b:
            break
        i += 1
    return lo[:i]


@dataclass
class Suggestion:
    word: str        # full completed word (lowercased stem + ending)
    ending: str      # characters to append after the typed prefix
    score: float     # blended score, higher = better
    source: str      # "vocab" | "model" | "vocab+model"


class Completer:
    def __init__(self, ngram: CharNGram, trie: FreqTrie,
                 vocab_counts: dict[str, int], total: int,
                 context: ContextModel | None = None, gamma: float = GAMMA,
                 morph=None, agreement=None, agr_w: float = AGR_W):
        self.ngram = ngram
        self.trie = trie
        self.vocab_counts = vocab_counts
        self.total = total or 1
        self.context = context
        self.gamma = gamma
        self.morph = morph            # MorphTable | None
        self.agreement = agreement    # AgreementModel | None
        self.agr_w = agr_w

    # ---- prefix / context extraction -----------------------------------
    @staticmethod
    def current_prefix(text: str):
        """Return (prefix, start_index) of the word being typed, or ('', len)."""
        m = TOKEN_RE.search(text)
        if not m:
            return "", len(text)
        return m.group(0), m.start()

    @staticmethod
    def prev_word(text: str, start: int) -> str | None:
        """The completed Russian word immediately left of the one being typed."""
        left = text[:start]
        matches = ANY_WORD_RE.findall(left)
        return matches[-1].lower() if matches else None

    def prev_content(self, text: str, start: int) -> list[str]:
        """Preceding *content* words (function words filtered), NEAREST FIRST.

        This is what lets agreement reach past conjunctions and participial /
        adverbial inserts to the real head word in long sentences."""
        from .context_model import ContextModel as _CM
        words = [w.lower() for w in ANY_WORD_RE.findall(text[:start])]
        content = _CM.content_words(words)
        window = getattr(self.context, "window", 1) if self.context else 1
        return content[::-1][:window]   # nearest first, capped to the window

    def _agreement_terms(self, raw_words: list[str], cands) -> dict[str, float]:
        """Centered agreement log-prob per candidate (0 for inapplicable ones).

        Uses the morphology table for the governor search and the candidate's
        grammemes. Centering (subtracting the mean over applicable candidates)
        means the signal re-orders only the group that actually agrees with a
        governor, without nudging unrelated candidates."""
        if not (self.agreement and self.morph and raw_words and cands):
            return {}
        # nearest-first grammemes of the preceding tokens; resolve governors once
        left_feats = [self.morph.get(w) for w in reversed(raw_words[-self.agreement.gov_window:])]
        govs = self.agreement.governors(left_feats)
        if govs == (None, None, None):
            return {}
        mget = self.morph.get
        raw = {}
        for w in cands:
            v = self.agreement.score_cand(govs, mget(w))
            if v is not None:
                raw[w] = v
        if len(raw) < 2:
            return {}
        mean = sum(raw.values()) / len(raw)
        return {w: v - mean for w, v in raw.items()}

    # ---- main API -------------------------------------------------------
    def complete(self, text: str, k: int = 5, min_prefix: int = 2,
                 use_context: bool = True) -> list[Suggestion]:
        raw_prefix, start = self.current_prefix(text)
        prefix = raw_prefix.lower()
        if len(prefix) < min_prefix:
            return []
        raw_words = [w.lower() for w in ANY_WORD_RE.findall(text[:start])] \
            if use_context else []
        if use_context and self.context:
            raw_prev = raw_words[::-1][:2]           # last 2 raw words, nearest first
            content_prev = self.prev_content(text, start)
        else:
            raw_prev, content_prev = [], []

        # 1) known word forms from the trie
        vocab_hits = self.trie.query(prefix)  # [(word, count)], desc by count
        vocab_score = {}
        for w, c in vocab_hits:
            if w == prefix:  # already a complete word; still offer inflections
                continue
            vocab_score[w] = math.log(c)

        # 2) generative n-gram completions (OOV + smoothing)
        model_hits = self.ngram.complete(prefix, k=max(k * 3, 12))
        model_score = {w: lp for w, lp in model_hits if w != prefix}

        # 3) blend. Real words get a strong base; the model score refines the
        #    order and lets unseen-but-plausible endings compete.
        alpha = 1.6   # weight on corpus evidence
        beta = 1.0    # weight on model plausibility
        # normalise model scores to ~[0,1] so scales are comparable
        if model_score:
            lo = min(model_score.values())
            hi = max(model_score.values())
            rng = (hi - lo) or 1.0
        cands = set(vocab_score) | set(model_score)

        # 3a) agreement re-ranking over the window of preceding content words;
        #     normalise across candidates so the bonus reflects *relative*
        #     agreement, not raw magnitude. Applied only when we have context.
        ctx_norm = {}
        if (raw_prev or content_prev) and cands:
            cs = {w: self.context.context_logscore(raw_prev, content_prev, w)
                  for w in cands}
            clo, chi = min(cs.values()), max(cs.values())
            crng = (chi - clo) or 1.0
            ctx_norm = {w: (v - clo) / crng for w, v in cs.items()}

        # 3b) factored grammatical agreement (number/gender) via morphology.
        #     Centered log-prob so it only re-orders candidates that share a
        #     governor (subject/head), leaving inapplicable candidates untouched.
        agr_term = self._agreement_terms(raw_words, cands) if use_context else {}

        scored = []
        for w in cands:
            in_vocab = w in vocab_score
            in_model = w in model_score
            vs = vocab_score.get(w, 0.0)
            ms = (model_score[w] - lo) / rng if in_model else 0.0
            score = (alpha * vs + beta * ms + self.gamma * ctx_norm.get(w, 0.0)
                     + self.agr_w * agr_term.get(w, 0.0))
            if in_vocab and in_model:
                src = "vocab+model"
            elif in_vocab:
                src = "vocab"
            else:
                src = "model"
                score -= 3.0  # prefer attested forms over invented ones
            scored.append((score, w, src))

        scored.sort(key=lambda x: -x[0])
        out: list[Suggestion] = []
        seen = set()
        for score, w, src in scored:
            if w in seen or not w.startswith(prefix):
                continue
            seen.add(w)
            out.append(Suggestion(word=w, ending=w[len(prefix):], score=score, source=src))
            if len(out) >= k:
                break
        return out

    # ---- confident inline suggestion (ghost text) -----------------------
    def ghost(self, text: str, min_prefix: int = GHOST_MIN_PREFIX,
              p_high: float = GHOST_P_HIGH, p_cover: float = GHOST_P_COVER,
              temp: float = GHOST_TEMP) -> Suggestion | None:
        """The single inline completion to show as grey ghost text, or None.

        A Russian ending is genuinely ambiguous mid-word, so committing the top-1
        ending on every keystroke shows a *wrong* tail ~28% of the time. Instead we
        read the blended scores as a P(word) via a tempered softmax and stay safe
        two ways:

          * CONFIDENT — one inflection owns ≥ `p_high` of the mass (frequency +
            context + agreement all point one way): commit its FULL ending.
            (`изображ→ение`)
          * SAFE PARTIAL — the mass is split: take the fewest leaders covering
            `p_cover`, and show only the leading chars ALL of them share — nothing
            past the point they diverge (shell-style unambiguous expansion). If they
            diverge at once, show nothing. (`молод→` stays silent, never `→ого`;
            `изображ…е/…я→` still offers the shared `ени`.)

        Thresholds tuned against the wrong-char-rate ↔ coverage curve in
        eval_ghost_chars.py. `source="lcp"` marks a safe-partial (non-word) tail."""
        raw_prefix, _ = self.current_prefix(text)
        if len(raw_prefix) < min_prefix:
            return None
        cands = self.complete(text, k=GHOST_CAND_K, min_prefix=min_prefix)
        if not cands:
            return None
        probs = _softmax([c.score for c in cands], temp)
        # CONFIDENT: one ending dominates → its full completion is safe to show.
        if probs[0] >= p_high:
            return cands[0]
        # SAFE PARTIAL: fewest leaders covering p_cover mass; show their shared prefix.
        acc = 0.0
        group: list[str] = []
        for c, p in zip(cands, probs):
            group.append(c.ending)
            acc += p
            if acc >= p_cover:
                break
        safe = _common_prefix(group)
        if not safe:
            return None                       # endings diverge at once — stay silent
        prefix = raw_prefix.lower()
        return Suggestion(word=prefix + safe, ending=safe,
                          score=cands[0].score, source="lcp")

    # ---- loading --------------------------------------------------------
    @classmethod
    def load(cls, models_dir: Path = MODELS, trie_cap: int = 30) -> "Completer":
        """Load a ready-to-serve completer from artifacts in `models_dir`.

        On-disk footprint is small (ngram.json + vocab.tsv, ~23 MB); the
        frequency trie is rebuilt in memory at load (~1-2 s) rather than shipped,
        which keeps the artifact bundle tiny and easy to version."""
        models_dir = Path(models_dir)
        ngram = CharNGram.load(models_dir / "ngram.json")
        items: list[tuple[str, int]] = []
        with (models_dir / "vocab.tsv").open(encoding="utf-8") as f:
            for line in f:
                w, c = line.rstrip("\n").split("\t")
                items.append((w, int(c)))
        trie = FreqTrie.build(items, cap=trie_cap)
        counts = dict(items)
        total = sum(c for _, c in items)
        ctx_path = models_dir / "context.json"
        context = ContextModel.load(ctx_path) if ctx_path.exists() else None
        # optional morphology-aware agreement (graceful if artifacts absent)
        morph = agreement = None
        morph_path = models_dir / "morph.tsv.gz"
        agr_path = models_dir / "agreement.json"
        if morph_path.exists() and agr_path.exists():
            from .agreement import AgreementModel
            from .morph import MorphTable
            morph = MorphTable.load(morph_path)
            agreement = AgreementModel.load(agr_path)
        return cls(ngram, trie, counts, total, context=context,
                   morph=morph, agreement=agreement)
