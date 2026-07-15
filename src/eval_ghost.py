"""Precision–coverage benchmark for the inline-ghost *trigger* threshold.

Showing a completion on every keystroke is not the goal — showing a *correct*
one when we speak, and staying silent when unsure, is (cf. Gmail Smart Compose,
arXiv:1906.00080, which trains a separate trigger to decide whether to surface a
suggestion at all). We gate the ghost on the leader's confidence — the blended
score gap between top-1 and top-2 — and here we pick that gap threshold from data.

For held-out Tatoeba sentences we replay typing keystroke by keystroke (from the
minimum ghost prefix up to the last char) and, per candidate threshold θ, measure

  * coverage  — % of keystrokes where the ghost fires
  * precision — of fired ghosts, % whose shown word is the word actually typed
  * useful    — coverage × precision (correct ghosts per keystroke; the quantity
                a real accept-rate tracks)

Higher θ ⇒ fewer, safer suggestions. We choose the smallest θ that reaches a
precision target, keeping coverage as high as possible.

    python -m src.eval_ghost
"""
from __future__ import annotations

import random

from .completer import GHOST_MIN_PREFIX, Completer
from .context_model import ContextModel, tatoeba_sentences
from .eval_context import load_items
from .ngram_model import CharNGram
from .train import SENTENCES, build_ngram
from .trie import FreqTrie

SEED = 7
N_TEST_SENTENCES = 5000
MIN_TARGET_LEN = 5
THRESHOLDS = (0.0, 0.5, 0.8, 1.1, 1.4, 1.8, 2.2, 3.0)
PRECISION_TARGET = 70.0   # pick the smallest θ that reaches this precision.
                          # Note: precision here is *exact* word match; at short
                          # prefixes several inflections are valid, so true useful
                          # precision is higher than the number shown.


def keystroke_cases(test_sents, rng):
    """(text, target, plen) for every keystroke from GHOST_MIN_PREFIX onward."""
    cases = []
    for toks in test_sents:
        idxs = [i for i in range(1, len(toks)) if len(toks[i]) >= MIN_TARGET_LEN]
        if not idxs:
            continue
        i = rng.choice(idxs)
        target = toks[i]
        left = " ".join(toks[:i])
        for plen in range(GHOST_MIN_PREFIX, len(target)):
            cases.append((left + " " + target[:plen], target, plen))
    return cases


def snapshot(comp, cases):
    """Precompute, per keystroke, the top-2 blended scores and the leader word.
    Returns rows (gap, leader_word, target); gap=None means a sole candidate."""
    rows = []
    for text, target, _ in cases:
        top = comp.complete(text, k=2, min_prefix=GHOST_MIN_PREFIX)
        if not top:
            rows.append((-1.0, None, target))            # nothing to show
        elif len(top) == 1:
            rows.append((float("inf"), top[0].word, target))
        else:
            rows.append((top[0].score - top[1].score, top[0].word, target))
    return rows


def main():
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
    rows = snapshot(comp, cases)
    n = len(rows)
    print(f"keystroke positions (prefix >= {GHOST_MIN_PREFIX}): {n}\n")

    print(f"  {'theta':>6} {'coverage':>9} {'precision':>10} {'useful':>8}")
    chosen = None
    for th in THRESHOLDS:
        fired = correct = 0
        for gap, word, target in rows:
            if word is None or gap < th:
                continue
            fired += 1
            if word == target:
                correct += 1
        cov = 100 * fired / n
        prec = 100 * correct / fired if fired else 0.0
        useful = 100 * correct / n
        star = ""
        if chosen is None and prec >= PRECISION_TARGET:
            chosen = th
            star = "  <- chosen (smallest θ ≥ target precision)"
        print(f"  {th:>6.1f} {cov:>8.1f}% {prec:>9.1f}% {useful:>7.1f}%{star}")
    print(f"\nprecision target: {PRECISION_TARGET:.0f}%  ->  GHOST_MARGIN = "
          f"{chosen if chosen is not None else 'n/a'}")


if __name__ == "__main__":
    main()
