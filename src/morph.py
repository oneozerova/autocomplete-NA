"""Build-time morphological tagging of the vocabulary.

Agreement (ежик[и] стоя[т], женщина ... котор[ую]) is a *grammatical* relation:
it depends on part of speech + number / gender / case of the words involved, not
on their surface endings (which are syncretic — «-и» is plural-nominative OR
singular-genitive). We therefore tag every vocabulary form once, at build time,
with a real morphological analyzer (pymorphy3, an OpenCorpora dictionary engine),
and ship a compact `form -> (POS, number, gender, case)` table.

At *run time* nothing morphological runs — the completer just does O(1) lookups
in this table (pure stdlib, no pymorphy, no latency). This is the "factored"
half of a factored-language-model approach: the expensive analysis is offline;
the online model is a few dict lookups over grammemes.

    python -m src.morph          # build models/morph.tsv from data/vocab.tsv
"""
from __future__ import annotations

import gzip
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "data" / "vocab.tsv"
# gzip-compressed (stdlib): the tag table is very repetitive, ~12 MB → ~3 MB.
OUT = ROOT / "models" / "morph.tsv.gz"

# The four grammemes agreement needs. Stored as short pymorphy strings ("" =
# unknown/not-applicable), which are already compact.
Feat = namedtuple("Feat", ["pos", "number", "gender", "case"])

# POS we actually reason about; everything else is tagged but only these drive
# agreement (kept as-is from pymorphy: NOUN, VERB, INFN, ADJF, ADJS, PRTF, NPRO).
NOUNISH = {"NOUN", "NPRO"}
ADJISH = {"ADJF", "ADJS", "PRTF"}   # который/красивый/причастия — agree with a noun
VERBISH = {"VERB", "PRTF", "GRND"}  # finite verb we care about is VERB


def _g(tag, attr: str) -> str:
    v = getattr(tag, attr)
    return str(v) if v else ""


def build(vocab_path: Path = VOCAB, out_path: Path = OUT) -> int:
    """Tag each vocabulary form (top parse) and write `form\\tPOS\\tnum\\tgen\\tcase`."""
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with vocab_path.open(encoding="utf-8") as f, \
            gzip.open(out_path, "wt", encoding="utf-8") as w:
        for line in f:
            form = line.split("\t", 1)[0]
            p = morph.parse(form)[0]              # best (most frequent) parse
            t = p.tag
            pos = _g(t, "POS")
            if not pos:
                continue
            w.write(f"{form}\t{pos}\t{_g(t,'number')}\t{_g(t,'gender')}\t{_g(t,'case')}\n")
            n += 1
    return n


class MorphTable:
    """Runtime grammeme lookup — pure stdlib, loaded once."""

    def __init__(self, feats: dict[str, Feat]):
        self.feats = feats

    def get(self, word: str) -> Feat | None:
        return self.feats.get(word)

    @classmethod
    def load(cls, path: Path = OUT) -> "MorphTable":
        feats: dict[str, Feat] = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 5:
                    form, pos, num, gen, case = parts
                    feats[form] = Feat(pos, num, gen, case)
        return cls(feats)


# ---- feature helpers shared by training and scoring ----------------------
def is_noun(ft: Feat | None) -> bool:
    return ft is not None and ft.pos in NOUNISH


def is_adj(ft: Feat | None) -> bool:
    return ft is not None and ft.pos in ADJISH


def is_verb(ft: Feat | None) -> bool:
    return ft is not None and ft.pos == "VERB"


def gn_key(ft: Feat) -> str:
    """Gender+number key for adjective/noun agreement (gender is None in plural,
    so number carries the signal there)."""
    return f"{ft.gender}|{ft.number}"


if __name__ == "__main__":
    print(f"morph forms tagged: {build()} -> {OUT}")
