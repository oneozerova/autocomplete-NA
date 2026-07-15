"""Benchmark the completer on a prefix -> full-word completion task.

No canonical Russian *ending-autocomplete* benchmark exists, so we use the
accepted word/query-autocompletion methodology on open data (OpenSubtitles
frequency list): sample target words *frequency-weighted* (mirroring what users
type) and measure completion quality by typed-prefix length.

Two things are measured, because they answer different questions:

  1. EXACT-MATCH (in-vocab, realistic). Did we surface the *exact* word the user
     was typing, in top-1/3/5?  Plus MRR and keystroke-savings. This is the
     headline product number. Reported per absolute prefix length, because a
     3-char prefix is inherently more ambiguous than a 6-char one.

  2. SUGGESTION VALIDITY (generalisation). For stems held out of training, are
     the model's generated completions *real Russian words*? Exact-match is the
     wrong OOV metric — from a prefix, several inflections (-ый/-ая/-ое) are all
     valid, so penalising all-but-one is unfair. What matters is that we never
     offer garbage. We check generated words against the full vocabulary.

    python -m src.evaluate
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from .completer import Completer
from .train import build_ngram, load_vocab
from .trie import FreqTrie

SEED = 13
PREFIX_LENS = (3, 4, 5, 6)
MIN_WORD_LEN = 6
N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", "4000"))  # lower for fast sweeps
K = 5


def sample_targets(items, n, rng, min_len=MIN_WORD_LEN):
    pool = [(w, c) for w, c in items if len(w) >= min_len]
    weights = [c for _, c in pool]
    return rng.choices(pool, weights=weights, k=n)


def make_completer(items, cap=30):
    ngram = build_ngram(items)
    trie = FreqTrie.build(items, cap=cap)
    counts = dict(items)
    total = sum(c for _, c in items)
    return Completer(ngram, trie, counts, total)


def eval_exact(comp, targets, label):
    print(f"\n== EXACT-MATCH  [{label}] ==")
    print(f"  {'plen':>4} {'cases':>6} {'top1':>7} {'top3':>7} {'top5':>7} "
          f"{'MRR':>6} {'ks-save':>8}")
    agg = {"n": 0, 1: 0, 3: 0, 5: 0, "mrr": 0.0, "saved": 0, "tot": 0}
    for plen in PREFIX_LENS:
        n = 0; hits = {1: 0, 3: 0, 5: 0}; mrr = 0.0; saved = 0; tot = 0
        for word, _ in targets:
            if len(word) <= plen:
                continue
            prefix = word[:plen]
            words = [s.word for s in comp.complete(prefix, k=K)]
            n += 1
            rank = words.index(word) + 1 if word in words else 0
            if rank:
                for kk in hits:
                    if rank <= kk:
                        hits[kk] += 1
                mrr += 1.0 / rank
                if words[0] == word:
                    saved += len(word) - plen
            tot += len(word) - plen
        if not n:
            continue
        print(f"  {plen:>4} {n:>6} {100*hits[1]/n:6.1f}% {100*hits[3]/n:6.1f}% "
              f"{100*hits[5]/n:6.1f}% {mrr/n:6.3f} {100*saved/tot:7.1f}%")
        agg["n"] += n
        for kk in (1, 3, 5):
            agg[kk] += hits[kk]
        agg["mrr"] += mrr; agg["saved"] += saved; agg["tot"] += tot
    n = agg["n"]
    print(f"  {'ALL':>4} {n:>6} {100*agg[1]/n:6.1f}% {100*agg[3]/n:6.1f}% "
          f"{100*agg[5]/n:6.1f}% {agg['mrr']/n:6.3f} {100*agg['saved']/agg['tot']:7.1f}%")


def eval_validity(comp, targets, full_vocab, label):
    """% of *model-generated* suggestions that are real words in the full vocab.
    Measures whether the char model generalises to real inflections vs garbage.
    """
    total = 0; real = 0; offered_valid = 0; cases = 0
    for word, _ in targets:
        plen = max(3, len(word) // 2)
        if plen >= len(word):
            continue
        cases += 1
        sug = comp.complete(word[:plen], k=K)
        model_sugs = [s for s in sug if "model" in s.source]
        any_valid = False
        for s in model_sugs:
            total += 1
            if s.word in full_vocab:
                real += 1
                any_valid = True
        if any(s.word in full_vocab for s in sug):
            offered_valid += 1
    print(f"\n== SUGGESTION VALIDITY  [{label}] ==  ({cases} cases)")
    if total:
        print(f"  model-generated suggestions that are real words: "
              f"{100*real/total:5.1f}%  ({real}/{total})")
    print(f"  cases where >=1 suggestion is a real word         : "
          f"{100*offered_valid/cases:5.1f}%")


def main():
    rng = random.Random(SEED)
    items = load_vocab()
    full_vocab = set(w for w, _ in items)
    rng.shuffle(items)

    # hold out 10% of forms to test generalisation
    cut = len(items) // 10
    heldout, train_items = items[:cut], items[cut:]

    print("training full model (in-vocab regime) ...")
    comp_full = make_completer(items)
    print("training held-out model (generalisation regime) ...")
    comp_oov = make_completer(train_items)

    in_targets = sample_targets(items, N_TARGETS, rng)
    oov_targets = sample_targets(heldout, N_TARGETS, rng)

    eval_exact(comp_full, in_targets, "in-vocab / full model")
    eval_validity(comp_oov, oov_targets, full_vocab, "held-out stems")


if __name__ == "__main__":
    main()
