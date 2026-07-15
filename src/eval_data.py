"""Data ablation: does a BIGGER training corpus improve the models?

Two learning curves:
  A) char n-gram + trie completer vs. training-vocabulary size
     -> exact-match top-5 on a FIXED held-out target set.
  B) context (agreement) model vs. number of training sentences
     -> agreement top-1 on FIXED held-out cases.

If a curve has plateaued, more data of the same kind won't help much.

    python -m src.eval_data
"""
from __future__ import annotations

import random

from .completer import Completer
from .context_model import STOPWORDS, ContextModel, tatoeba_sentences
from .ngram_model import CharNGram
from .train import SENTENCES, build_ngram
from .trie import FreqTrie
from .evaluate import load_vocab
from .eval_context import build_cases

SEED = 21
K = 5


def curve_vocab(items):
    rng = random.Random(SEED)
    # fixed held-out targets (frequency-weighted), incl. rare words
    pool = [(w, c) for w, c in items if len(w) >= 6]
    targets = rng.choices(pool, weights=[c for _, c in pool], k=4000)

    print("A) COMPLETER vs training-vocabulary size "
          "(fixed held-out targets, prefix=4/5)")
    print(f"   {'train_words':>12} {'top1':>7} {'top5':>7} {'ks-save':>8}")
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        n_words = int(len(items) * frac)
        sub = items[:n_words]                       # top-N by frequency
        comp = Completer(build_ngram(sub), FreqTrie.build(sub, cap=30),
                         dict(sub), sum(c for _, c in sub))
        comp.context = None
        n = hit1 = hit5 = saved = tot = 0
        for w, _ in targets:
            for plen in (4, 5):
                if len(w) <= plen:
                    continue
                words = [s.word for s in comp.complete(w[:plen], k=K)]
                n += 1
                if words and words[0] == w:
                    hit1 += 1; saved += len(w) - plen
                if w in words:
                    hit5 += 1
                tot += len(w) - plen
        print(f"   {n_words:>12} {100*hit1/n:6.1f}% {100*hit5/n:6.1f}% "
              f"{100*saved/tot:7.1f}%")


def curve_sentences(items):
    rng = random.Random(SEED)
    all_sents = list(tatoeba_sentences(SENTENCES))
    rng.shuffle(all_sents)
    test_sents, train_sents = all_sents[:6000], all_sents[6000:]
    cases = build_cases(test_sents, rng)

    ngram = build_ngram(items)
    trie = FreqTrie.build(items, cap=30)
    counts, total = dict(items), sum(c for _, c in items)

    print("\nB) CONTEXT (agreement) vs number of training sentences "
          f"(fixed {len(cases)} held-out cases)")
    print(f"   {'sentences':>12} {'top1':>7} {'top3':>7} {'MRR':>7}")
    for n_tr in (50_000, 150_000, 400_000, len(train_sents)):
        ctx = ContextModel.from_sentences(train_sents[:n_tr], window=4)
        comp = Completer(ngram, trie, counts, total, context=ctx)
        n = h1 = h3 = 0; mrr = 0.0
        for text, target, _ in cases:
            words = [s.word for s in comp.complete(text, k=K)]
            n += 1
            rank = words.index(target) + 1 if target in words else 0
            if rank:
                h1 += rank == 1
                h3 += rank <= 3
                mrr += 1.0 / rank
        print(f"   {n_tr:>12} {100*h1/n:6.1f}% {100*h3/n:6.1f}% {mrr/n:6.3f}")


def main():
    items = load_vocab()
    curve_vocab(items)
    curve_sentences(items)


if __name__ == "__main__":
    main()
