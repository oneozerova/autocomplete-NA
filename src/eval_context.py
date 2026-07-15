"""Context/agreement benchmark on held-out running text (Tatoeba sentences).

Unlike the frequency-list benchmark, here every target word has *real left
context*, so we can measure whether agreement re-ranking actually helps. We
sweep the agreement weight gamma (gamma=0 == context off) and report top-k / MRR.

    python -m src.eval_context
"""
from __future__ import annotations

import random
from itertools import islice
from pathlib import Path

from .completer import Completer
from .context_model import STOPWORDS, ContextModel, tatoeba_sentences
from .ngram_model import CharNGram
from .train import SENTENCES, build_ngram
from .trie import FreqTrie
from .preprocess import OUT as VOCAB_TSV, clean

SEED = 7
PREFIX_FRAC = 0.5
MIN_TARGET_LEN = 5
N_TEST_SENTENCES = 6000
K = 5


def load_items():
    if not VOCAB_TSV.exists():
        clean()
    items = []
    with VOCAB_TSV.open(encoding="utf-8") as f:
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            items.append((w, int(c)))
    return items


def build_cases(test_sents, rng):
    """(text, target, is_hard) per usable sentence.

    is_hard = the word immediately left of the target is a function word, i.e.
    the naive (window=1) model would try to agree with a conjunction / preposition
    / particle — exactly the long-sentence failure the window is meant to fix."""
    cases = []
    for toks in test_sents:
        idxs = [i for i in range(1, len(toks)) if len(toks[i]) >= MIN_TARGET_LEN]
        if not idxs:
            continue
        i = rng.choice(idxs)
        target = toks[i]
        plen = max(2, int(len(target) * PREFIX_FRAC))
        if plen >= len(target):
            continue
        left = " ".join(toks[:i])
        is_hard = toks[i - 1] in STOPWORDS
        cases.append((left + " " + target[:plen], target, is_hard))
    return cases


def score(comp, cases):
    n = 0; hits = {1: 0, 3: 0, 5: 0}; mrr = 0.0
    for text, target, _ in cases:
        words = [s.word for s in comp.complete(text, k=K)]
        n += 1
        rank = words.index(target) + 1 if target in words else 0
        if rank:
            for kk in hits:
                if rank <= kk:
                    hits[kk] += 1
            mrr += 1.0 / rank
    if not n:
        return None
    return {"n": n, "top1": 100*hits[1]/n, "top3": 100*hits[3]/n,
            "top5": 100*hits[5]/n, "mrr": mrr/n}


def _completer(items, ctx):
    ngram = build_ngram(items)
    trie = FreqTrie.build(items, cap=30)
    return Completer(ngram, trie, dict(items), sum(c for _, c in items), context=ctx)


def _row(label, r):
    if r is None:
        return
    print(f"  {label:<22} {r['n']:>6} {r['top1']:6.1f}% {r['top3']:6.1f}% "
          f"{r['top5']:6.1f}% {r['mrr']:6.3f}")


def main():
    rng = random.Random(SEED)
    items = load_items()

    all_sents = list(tatoeba_sentences(SENTENCES))
    rng.shuffle(all_sents)
    test_sents, train_sents = all_sents[:N_TEST_SENTENCES], all_sents[N_TEST_SENTENCES:]

    print(f"training context models on {len(train_sents)} sentences "
          f"(naive window=1 vs windowed window=4) ...")
    ctx_off = None
    ctx_naive = ContextModel.from_sentences(train_sents, window=1)
    ctx_win = ContextModel.from_sentences(train_sents, window=4)

    cases = build_cases(test_sents, rng)
    hard = [c for c in cases if c[2]]
    print(f"eval cases: {len(cases)} total, {len(hard)} 'hard' "
          f"(immediate left word is a function word)\n")

    comp_off = _completer(items, ctx_off)      # context disabled (gamma path off)
    comp_off.context = None
    comp_naive = _completer(items, ctx_naive)
    comp_win = _completer(items, ctx_win)

    print(f"  {'model':<22} {'cases':>6} {'top1':>7} {'top3':>7} {'top5':>7} {'MRR':>7}")
    print("  -- all cases --")
    _row("context off", score(comp_off, cases))
    _row("naive (window=1)", score(comp_naive, cases))
    _row("windowed (window=4)", score(comp_win, cases))
    print("  -- HARD slice (long-distance across function word) --")
    _row("context off", score(comp_off, hard))
    _row("naive (window=1)", score(comp_naive, hard))
    _row("windowed (window=4)", score(comp_win, hard))


if __name__ == "__main__":
    main()
