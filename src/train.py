"""Build and serialise the deployed model artifacts into models/.

    python -m src.train

Produces:
    models/ngram.json         char n-gram language model
    models/trie.json          frequency prefix index
    models/vocab_counts.json  {counts: {word: count}, total: int}
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .context_model import ContextModel, tatoeba_sentences
from .ngram_model import CharNGram
from .preprocess import OUT as VOCAB_TSV, clean

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SENTENCES = ROOT / "data" / "rus_sentences.tsv.bz2"

ORDER = 6
# keep the model small & generalisable: prune long, rare contexts hard
PRUNE = {2: 1, 3: 2, 4: 3, 5: 4}
PROMPT_REPEAT = 3   # how many times to fold in the domain prompt corpus


def load_vocab(path: Path = VOCAB_TSV) -> list[tuple[str, int]]:
    if not path.exists():
        clean()
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            items.append((w, int(c)))
    return items


def build_ngram(items: list[tuple[str, int]], order: int = ORDER) -> CharNGram:
    m = CharNGram(order=order)
    for w, c in items:
        m.add_word(w, weight=CharNGram.freq_weight(c))
    m.prune(PRUNE)
    return m


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    items = load_vocab()
    print(f"vocab: {len(items)} forms")

    print("building n-gram ...")
    ngram = build_ngram(items)
    ngram.save(MODELS / "ngram.json")
    print(f"  contexts: {len(ngram.counts)}  size: "
          f"{(MODELS / 'ngram.json').stat().st_size/1e6:.1f} MB")

    # ship the vocabulary; the trie is rebuilt in memory at load time
    shutil.copyfile(VOCAB_TSV, MODELS / "vocab.tsv")
    print(f"  vocab.tsv: {(MODELS / 'vocab.tsv').stat().st_size/1e6:.1f} MB")

    # build-time morphology: tag the vocabulary once (needs pymorphy3, a build
    # dependency only). Runtime uses the shipped table with zero morphology cost.
    morph = None
    try:
        from . import morph as morph_mod
        n = morph_mod.build(VOCAB_TSV, morph_mod.OUT)
        morph = morph_mod.MorphTable.load(morph_mod.OUT)
        print(f"  {morph_mod.OUT.name}: {n} tagged forms, "
              f"{morph_mod.OUT.stat().st_size/1e6:.1f} MB")
    except ImportError:
        print("  (skip morphology: pymorphy3 not installed)")

    if SENTENCES.exists():
        print("building context (agreement) model (Tatoeba + prompt corpus) ...")
        from .prompt_corpus import sentences as prompt_sentences

        # Natural text trains every signal (incl. the lexical word-bigram). The
        # synthetic prompt corpus trains only the suffix/agreement signals
        # (lexical=False) so its stock template nouns don't pollute collocations.
        ctx = ContextModel(window=4)
        for toks in tatoeba_sentences(SENTENCES):
            if len(toks) >= 2:
                ctx.add_sentence(toks, lexical=True)
        for _ in range(PROMPT_REPEAT):            # weight the domain corpus
            for toks in prompt_sentences():
                if len(toks) >= 2:
                    ctx.add_sentence(toks, lexical=False)
        ctx.prune(3)
        ctx.save(MODELS / "context.json")
        n_ctx = sum(len(ctx.bigram[d]) for d in ctx.bigram)
        print(f"  window: {ctx.window}  suffix-contexts: {n_ctx}  size: "
              f"{(MODELS / 'context.json').stat().st_size/1e6:.1f} MB")

        # factored agreement model: learn number/gender agreement over grammemes
        # from real text (Tatoeba), tagged via the build-time morphology table.
        if morph is not None:
            from .agreement import AgreementModel

            def tagged():
                for toks in tatoeba_sentences(SENTENCES):
                    if len(toks) >= 2:
                        yield [morph.get(t) for t in toks]

            agr = AgreementModel.from_sentences(tagged())
            agr.save(MODELS / "agreement.json")
            print(f"  agreement: vnum={len(agr.vnum)} adj_gn={len(agr.adj_gn)} "
                  f"noun_gn={len(agr.noun_gn)}  size: "
                  f"{(MODELS / 'agreement.json').stat().st_size/1e3:.0f} KB")
    else:
        print(f"  (skip context model: {SENTENCES.name} not found)")
    print("done.")


if __name__ == "__main__":
    main()
