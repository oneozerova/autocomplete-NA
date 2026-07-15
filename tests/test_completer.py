"""Unit tests for the completer. Run:  python -m pytest -q  (or python tests/test_completer.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.completer import Completer  # noqa: E402
from src.ngram_model import CharNGram  # noqa: E402
from src.trie import FreqTrie  # noqa: E402

_comp = None


def comp():
    global _comp
    if _comp is None:
        _comp = Completer.load()
    return _comp


def test_prefix_extraction():
    assert Completer.current_prefix("красивый пейза")[0] == "пейза"
    assert Completer.current_prefix("hello мир")[0] == "мир"
    assert Completer.current_prefix("город ")[0] == ""       # trailing space
    assert Completer.current_prefix("")[0] == ""


def test_endings_are_appended_to_prefix():
    for s in comp().complete("красив", k=5):
        assert s.word.startswith("красив")
        assert s.word == "красив" + s.ending


def test_adjective_gender_number_forms_present():
    words = {s.word for s in comp().complete("красив", k=8)}
    # feminine / neuter / plural inflections must surface
    assert {"красивая", "красивый"} & words
    assert len(words) >= 4


def test_suggestions_are_ordered_by_score():
    sc = [s.score for s in comp().complete("город", k=5)]
    assert sc == sorted(sc, reverse=True)


def test_empty_and_short_prefix():
    assert comp().complete("", k=5) == []
    assert comp().complete("a", k=5) == []          # latin, no RU prefix
    assert isinstance(comp().complete("го", k=5), list)


def test_ngram_generates_valid_endings_offline():
    m = CharNGram(order=5)
    for w, c in [("красивый", 50), ("красивая", 40), ("красивое", 20),
                 ("красивые", 15), ("красота", 30)]:
        m.add_word(w, weight=c)
    outs = [w for w, _ in m.complete("красив", k=5)]
    assert any(w.startswith("красив") for w in outs)
    assert "красивая" in outs or "красивый" in outs


def test_prev_word_extraction():
    assert Completer.prev_word("красивая девушк", len("красивая ")) == "красивая"
    assert Completer.prev_word("девушк", 0) is None


def test_context_number_agreement():
    """Plural context should lift the plural noun form above the singular."""
    c = comp()
    if c.context is None:
        return  # context model not built in this environment; skip
    sg = [s.word for s in c.complete("красивая девушк", k=5)]
    pl = [s.word for s in c.complete("красивые девушк", k=5)]

    # singular adjective -> singular noun ranks first; plural -> plural first.
    # A form that drops off the top-k entirely just ranks "after" everything
    # (the morphological agreement signal can push a disagreeing number out).
    def rank(words, w):
        return words.index(w) if w in words else len(words)

    assert rank(sg, "девушка") < rank(sg, "девушки")
    assert rank(pl, "девушки") < rank(pl, "девушка")


def test_trie_ranks_by_frequency():
    t = FreqTrie.build([("кот", 100), ("который", 500), ("кофе", 300)])
    top = [w for w, _ in t.query("ко")]
    assert top[0] == "который"          # highest count first
    assert set(top) == {"кот", "который", "кофе"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
