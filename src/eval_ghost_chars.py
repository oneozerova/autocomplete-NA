"""Character-level *safety* benchmark for the inline ghost.

`eval_ghost.py` scores the ghost at *word* granularity: a shown completion is
either the exact word the user typed or a miss. But the ghost is grey text the
user reads mid-word, and what actually annoys ("сбивает") is a **wrong ending** —
characters that don't match what they go on to type and have to be mentally
discarded. A completion that shares the first two chars of the true ending and
diverges after is *partly* useful, not a flat miss; and one that shows three
wrong chars is worse than one that shows one. Word-precision can't see either.

So we measure the ghost at *character* granularity. At each keystroke the ghost
offers an ending string `g`; the truth is the rest of the word the user typed,
`t = target[plen:]`. Split `g` at its longest common prefix with `t`:

    correct = len(commonprefix(g, t))     # chars the user keeps
    wrong   = len(g) - correct            # chars they must ignore/delete

Aggregated over all fired keystrokes:

  * wrong_char_rate   — wrong / shown. THE "сбивает" number: of the grey chars we
                        put on screen, what fraction was misleading. Lower better.
  * clean_ghost_rate  — fraction of fired ghosts with zero wrong chars. The share
                        of suggestions that never mislead at all. Higher better.
  * chars_saved/ks    — correct / n_keystrokes. Usefulness: correct chars we
                        pre-filled per keystroke. Higher better.
  * coverage          — fired / n_keystrokes. How often we say anything.

`evaluate(comp, cases, ghost_fn)` takes the ghost as a plug so the same harness
scores today's `Completer.ghost` (baseline) and, later, the softmax+LCP variant —
which is how step 3 will sweep its thresholds against this curve.

    python -m src.eval_ghost_chars
"""
from __future__ import annotations

import os
import random

from .completer import Completer, GHOST_MIN_PREFIX
from .context_model import ContextModel, tatoeba_sentences
from .eval_context import load_items
from .eval_ghost import keystroke_cases
from .ngram_model import CharNGram
from .train import SENTENCES, build_ngram
from .trie import FreqTrie

SEED = 7
N_TEST_SENTENCES = 5000


def lcp_len(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def ghost_margin(comp: Completer, text: str, margin: float = 1.4) -> str:
    """Replica of the OLD gate (top-1 full ending iff it beats top-2 by `margin`).

    Kept self-contained so before/after is measured on identical cases in one run,
    independent of whatever Completer.ghost currently does."""
    raw_prefix, _ = comp.current_prefix(text)
    if len(raw_prefix) < GHOST_MIN_PREFIX:
        return ""
    top = comp.complete(text, k=2, min_prefix=GHOST_MIN_PREFIX)
    if not top:
        return ""
    if len(top) == 1 or (top[0].score - top[1].score) >= margin:
        return top[0].ending
    return ""


def make_softmax_ghost(p_high: float, p_cover: float, temp: float):
    """A ghost_fn for the NEW softmax+LCP path with the given thresholds."""
    def fn(comp: Completer, text: str) -> str:
        g = comp.ghost(text, p_high=p_high, p_cover=p_cover, temp=temp)
        return g.ending if g else ""
    return fn


def evaluate(comp: Completer, cases, ghost_fn) -> dict:
    """Character-level safety metrics for `ghost_fn` over the keystroke cases.

    `ghost_fn(comp, text) -> str` returns the ending to show ("" = stay silent).
    """
    n = len(cases)
    fired = clean = 0
    shown = correct = 0
    for text, target, plen in cases:
        g = ghost_fn(comp, text)
        if not g:
            continue
        truth = target[plen:]          # what the user actually goes on to type
        c = lcp_len(g, truth)
        fired += 1
        shown += len(g)
        correct += c
        if c == len(g):                # every shown char matched
            clean += 1
    wrong = shown - correct
    return {
        "n": n,
        "fired": fired,
        "coverage": 100 * fired / n if n else 0.0,
        "shown_chars": shown,
        "correct_chars": correct,
        "wrong_chars": wrong,
        "wrong_char_rate": 100 * wrong / shown if shown else 0.0,
        "clean_ghost_rate": 100 * clean / fired if fired else 0.0,
        "chars_saved_per_ks": correct / n if n else 0.0,
    }


def print_report(title: str, m: dict) -> None:
    print(f"\n{title}")
    print(f"  coverage           {m['coverage']:>6.1f}%   ({m['fired']} / {m['n']} keystrokes)")
    print(f"  wrong_char_rate    {m['wrong_char_rate']:>6.1f}%   "
          f"({m['wrong_chars']} / {m['shown_chars']} shown chars misleading)  <- lower better")
    print(f"  clean_ghost_rate   {m['clean_ghost_rate']:>6.1f}%   (fired ghosts with 0 wrong chars)")
    print(f"  chars_saved / ks   {m['chars_saved_per_ks']:>6.3f}    (correct chars pre-filled per keystroke)")


# Thresholds swept in step 3. Temp fixed (it only rescales the same axis as p_high);
# p_high trades confident-full-ending coverage, p_cover trades safe-partial safety.
SWEEP_TEMP = 2.0
SWEEP_P_HIGH = (0.55, 0.65, 0.75, 0.85)
SWEEP_P_COVER = (0.80, 0.90, 0.95)


def build_comp_and_cases():
    """Shared setup: same corpus/split/seed as eval_ghost, so the two are comparable."""
    rng = random.Random(SEED)
    items = load_items()
    ngram = build_ngram(items)
    trie = FreqTrie.build(items, cap=30)

    all_sents = list(tatoeba_sentences(SENTENCES))
    rng.shuffle(all_sents)
    test_sents, train_sents = all_sents[:N_TEST_SENTENCES], all_sents[N_TEST_SENTENCES:]
    print(f"training context on {len(train_sents)} sentences ...")
    ctx = ContextModel.from_sentences(train_sents, window=4)
    comp = Completer(ngram, trie, dict(items), sum(c for _, c in items), context=ctx)
    cases = keystroke_cases(test_sents, rng)
    return comp, cases


def main():
    comp, cases = build_comp_and_cases()
    print(f"keystroke positions (prefix >= {GHOST_MIN_PREFIX}): {len(cases)}")

    base = evaluate(comp, cases, ghost_margin)
    print_report("BASELINE — old margin gate (top-1 full ending, margin=1.4)", base)

    print("\n\nSWEEP — softmax + LCP safe completion  (temp = %.1f)" % SWEEP_TEMP)
    print(f"  {'p_high':>6} {'p_cover':>7} | {'cover':>6} {'wrong%':>7} "
          f"{'clean%':>7} {'saved/ks':>9}")
    print("  " + "-" * 54)
    for ph in SWEEP_P_HIGH:
        for pc in SWEEP_P_COVER:
            m = evaluate(comp, cases, make_softmax_ghost(ph, pc, SWEEP_TEMP))
            print(f"  {ph:>6.2f} {pc:>7.2f} | {m['coverage']:>5.1f}% "
                  f"{m['wrong_char_rate']:>6.1f}% {m['clean_ghost_rate']:>6.1f}% "
                  f"{m['chars_saved_per_ks']:>9.3f}")
    print("\n  baseline for reference: "
          f"cover {base['coverage']:.1f}%  wrong {base['wrong_char_rate']:.1f}%  "
          f"clean {base['clean_ghost_rate']:.1f}%  saved {base['chars_saved_per_ks']:.3f}")


if __name__ == "__main__":
    main()
