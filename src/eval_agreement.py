"""Benchmark the factored agreement signal on held-out running text.

We restrict to target words where agreement actually applies (a verb with a
resolvable subject, or an adjective/pronoun with a governing noun) — that is the
slice the morphology signal is meant to fix (ежик[и] стоя[т], женщина котор[ую]).
We sweep its weight `AGR_W` (0 = off) and report top-k / MRR on those targets.

    python -m src.eval_agreement
"""
from __future__ import annotations

import itertools
import random

from .agreement import AgreementModel
from .completer import Completer
from .context_model import ContextModel, tatoeba_sentences
from .eval_context import load_items
from .morph import MorphTable, is_adj
from .ngram_model import CharNGram
from .train import SENTENCES, build_ngram
from .trie import FreqTrie

SEED = 7
N_TEST_SENTENCES = 5000
N_TRAIN_SENTENCES = 400_000   # enough for stable agreement tables; keeps it quick
MIN_TARGET_LEN = 4
AGR_WEIGHTS = (0.0, 2.0, 4.0, 6.0)
K = 5


def applicable_cases(test_sents, morph, agr, rng):
    """(text, target) where the target is a verb/adj with a resolved governor."""
    cases = []
    for toks in test_sents:
        idxs = list(range(1, len(toks)))
        rng.shuffle(idxs)
        for i in idxs:
            target = toks[i]
            if len(target) < MIN_TARGET_LEN:
                continue
            cand = morph.get(target)
            if cand is None or (cand.pos != "VERB" and not is_adj(cand)):
                continue
            left = [morph.get(t) for t in toks[:i]][::-1]
            if agr.score_cand(agr.governors(left), cand) is None:
                continue
            plen = max(2, len(target) // 2)
            if plen >= len(target):
                continue
            cases.append((" ".join(toks[:i]) + " " + target[:plen], target))
            break
    return cases


def score(comp, cases):
    n = 0; hits = {1: 0, 3: 0, 5: 0}; mrr = 0.0
    for text, target in cases:
        words = [s.word for s in comp.complete(text, k=K)]
        n += 1
        if target in words:
            r = words.index(target) + 1
            for kk in hits:
                if r <= kk:
                    hits[kk] += 1
            mrr += 1.0 / r
    return {"n": n, "top1": 100*hits[1]/n, "top3": 100*hits[3]/n,
            "top5": 100*hits[5]/n, "mrr": mrr/n}


def main():
    rng = random.Random(SEED)
    items = load_items()
    ngram = build_ngram(items)
    trie = FreqTrie.build(items, cap=30)
    morph = MorphTable.load()

    all_sents = list(tatoeba_sentences(SENTENCES))
    rng.shuffle(all_sents)
    test_sents = all_sents[:N_TEST_SENTENCES]
    train_sents = all_sents[N_TEST_SENTENCES:N_TEST_SENTENCES + N_TRAIN_SENTENCES]

    print(f"training context + agreement on {len(train_sents)} sentences ...")
    ctx = ContextModel(window=4)
    for toks in train_sents:
        if len(toks) >= 2:
            ctx.add_sentence(toks)
    ctx.prune(3)
    agr = AgreementModel.from_sentences(
        [morph.get(t) for t in s] for s in train_sents if len(s) >= 2)

    comp = Completer(ngram, trie, dict(items), sum(c for _, c in items),
                     context=ctx, morph=morph, agreement=agr)
    cases = applicable_cases(test_sents, morph, agr, rng)
    print(f"applicable agreement cases: {len(cases)}\n")

    print(f"  {'AGR_W':>6} {'top1':>7} {'top3':>7} {'top5':>7} {'MRR':>7}")
    for w in AGR_WEIGHTS:
        comp.agr_w = w
        r = score(comp, cases)
        print(f"  {w:>6.1f} {r['top1']:6.1f}% {r['top3']:6.1f}% "
              f"{r['top5']:6.1f}% {r['mrr']:6.3f}")


if __name__ == "__main__":
    main()
