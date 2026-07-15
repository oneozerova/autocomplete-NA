"""Clean the raw frequency list into a compact vocabulary of Russian word forms.

Input : data/ru_full.txt  (lines "<word> <count>", OpenSubtitles 2018)
Output: data/vocab.tsv    (lines "<word>\t<count>", cleaned & filtered)

The vocabulary is a list of *inflected word forms* with corpus counts. It is the
single source of truth for both the completion trie (exact known-word lookup)
and the character n-gram model (endings generalisation).
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "ru_full.txt"
OUT = ROOT / "data" / "vocab.tsv"

# Russian letters only (incl. ё), internal hyphen allowed (кто-то, из-за).
WORD_RE = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$")

MIN_COUNT = 5      # drop noise / one-off typos
MIN_LEN = 2
MAX_LEN = 24

# Domain-frequency interpolation (linear interpolation / count merging — the
# standard LM domain-adaptation recipe):
#     count_final(w) = count_general(w) + λ · N_general · P_domain(w)
# i.e. we take a fraction λ of the general corpus mass and redistribute it over
# the domain word distribution P_domain (curated prompt lexicon + synthetic
# prompt-corpus token frequencies). This lifts prompt vocabulary in the ranking
# without discarding the general prior. λ is deliberately small so general
# accuracy (evaluate.py) does not move; overridable via env for tuning.
LAMBDA_DOMAIN = float(os.environ.get("DOMAIN_LAMBDA", "0.1"))


def domain_counts() -> Counter:
    """Combined domain word frequency: curated lexicon forms + content-word
    token counts from the synthetic prompt corpus. This is the empirical
    P_domain (unnormalised) that the interpolation redistributes mass over."""
    from .context_model import STOPWORDS
    from .domain_lexicon import domain_forms
    from .prompt_corpus import word_counts

    d: Counter = Counter()
    for w, c in domain_forms().items():
        d[w] += c
    for w, c in word_counts().items():          # running-text prompt frequencies
        if w in STOPWORDS or len(w) < MIN_LEN:   # don't boost function words
            continue
        d[w] += c
    return d


def clean(raw_path: Path = RAW, out_path: Path = OUT) -> int:
    kept: dict[str, int] = {}
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            w, c = parts[0], parts[1]
            if not c.isdigit():
                continue
            c = int(c)
            if c < MIN_COUNT:
                continue
            w = w.replace("ё", "е") if False else w  # keep ё (meaningful in RU)
            if not (MIN_LEN <= len(w) <= MAX_LEN):
                continue
            if not WORD_RE.match(w):
                continue
            # merge duplicates (e.g. case-folded collisions upstream)
            kept[w] = kept.get(w, 0) + c

    # Interpolate the domain distribution into the general frequencies as a
    # *floor*: target(w) = λ · N_general · P_domain(w), and we lift w only up to
    # that floor — max(), not +=. This raises prompt vocabulary that OpenSubtitles
    # under-represents (изображение: rank ~6k → competitive) while deliberately
    # NOT inflating words that are already frequent in general text (девушка),
    # which would otherwise overpower morphological agreement ("красивые
    # девушк[и]"). Standard count-merging domain adaptation, floored so it only
    # ever corrects the under-representation it is meant to fix. The *context*
    # model (train.py) is what flips short ambiguous prefixes once left context is
    # present ("на из…" → "изображение").
    n_general = sum(kept.values())
    dom = domain_counts()
    n_domain = sum(dom.values()) or 1
    mass = LAMBDA_DOMAIN * n_general
    for w, c in dom.items():
        floor = int(round(mass * c / n_domain))
        if floor > kept.get(w, 0):
            kept[w] = floor

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for w, c in sorted(kept.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{w}\t{c}\n")
    return len(kept)


if __name__ == "__main__":
    n = clean()
    print(f"vocab written: {n} word forms -> {OUT}")
