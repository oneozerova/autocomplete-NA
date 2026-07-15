"""Domain regression probe for prompt-style completions + latency.

Not a general benchmark — it pins the specific failure mode this project cares
about: at short prefixes the completer must not confidently surface a
conversational word (e.g. "извините") over the domain word the user means
("изображение"). Each case lists the prefix and the word(s) we expect to lead.

    python -m scripts.domain_probe
"""
from __future__ import annotations

import time

from src.completer import Completer

# (input_text, {acceptable top-1 words})  — any of the set winning top-1 is a pass.
# Sets (not a single word) because several inflections are legitimately valid for
# a short prefix; we only assert the domain-correct ones, never a conversational
# hijack like "извините"/"из-за".
CASES = [
    ("на из", {"изображение"}),
    ("смотрит на из", {"изображение"}),
    ("на изоб", {"изображение"}),
    ("детали", {"детализация", "детализированный"}),
    ("реалист", {"реалистичный"}),
    ("пейз", {"пейзаж"}),
    ("портре", {"портрет", "портретный"}),
    ("акварел", {"акварельный"}),
    ("минимал", {"минималистичный"}),
    ("футурист", {"футуристический"}),
    ("закат", {"закат", "закатный"}),
    ("в тёмном лес", {"лесу"}),
    ("красивая девушк", {"девушка"}),
    ("красивые девушк", {"девушки"}),
    # grammatical agreement (morphology signal): number of verb, gender of pronoun
    ("ежики сто", {"стоят", "стояли", "стойте", "стоите"}),   # plural subject → plural verb
    ("ежик сто", {"стоит", "стой", "стоило", "стоишь"}),      # singular subject → singular verb
    ("дети игра", {"играют", "играли"}),
    ("женщина котор", {"которую", "которая", "которой"}),     # femn antecedent
    ("мужчина котор", {"который", "которого", "котором"}),    # masc antecedent
    ("река котор", {"которая", "которой", "которую"}),
]


def run(comp: Completer) -> tuple[int, int]:
    ok = 0
    print(f"  {'input':<22} {'top-1':<16} {'expected':<24} hit  top-5")
    for text, expected in CASES:
        sug = comp.complete(text, k=5)
        top1 = sug[0].word if sug else "—"
        words = [s.word for s in sug]
        hit = top1 in expected
        ok += hit
        mark = "✓" if hit else "✗"
        exp = "|".join(sorted(expected))
        print(f"  {text:<22} {top1:<16} {exp:<24} {mark}   {words}")
    print(f"  domain top-1: {ok}/{len(CASES)}")
    return ok, len(CASES)


def latency(comp: Completer, n: int = 2000) -> None:
    probes = ["красивый пейза", "смотрит на изоб", "детализирован", "в тёмном лес"]
    # warm
    for p in probes:
        comp.complete(p, k=5)
    t0 = time.perf_counter()
    for i in range(n):
        comp.complete(probes[i % len(probes)], k=5)
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  latency: {dt:.3f} ms/call  ({n} calls)")


# (input, should_ghost_fire) — the confidence gate must SUPPRESS grey text on a
# short ambiguous stub and FIRE once the word is unambiguous.
GHOST_CASES = [
    ("из", False),            # 2-char stub: below min prefix
    ("на из", False),         # still ambiguous (из-за/известно/…)
    ("на изоб", True),        # now clearly изображение
    ("красивый пейза", True),
    ("детализирован", True),
    ("в", False),
]


def ghost_check(comp: Completer) -> None:
    print(f"  {'input':<18} {'fires?':<7} {'ghost':<16} ok")
    ok = 0
    for text, expect_fire in GHOST_CASES:
        g = comp.ghost(text)
        fired = g is not None
        good = fired == expect_fire
        ok += good
        shown = (g.word if g else "—")
        print(f"  {text:<18} {str(fired):<7} {shown:<16} {'✓' if good else '✗'}")
    print(f"  ghost gate: {ok}/{len(GHOST_CASES)}")


if __name__ == "__main__":
    comp = Completer.load()
    print("== DOMAIN PROBE (top-1) ==")
    run(comp)
    print("== GHOST CONFIDENCE GATE ==")
    ghost_check(comp)
    print("== LATENCY ==")
    latency(comp)
