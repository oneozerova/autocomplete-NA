"""Benchmark the optional OpenRouter next-word layer against the local core.

Answers the two questions directly, with numbers on *your* example prompts:
  1) LATENCY — is the LLM "as fast"?  We print time-to-first-token (the number
     that decides if it clears the ~500 ms inline-suggestion budget) and total
     time, next to the local completer's per-call time.
  2) QUALITY — does it predict the *next word* better?  For each prompt we show
     the LLM's semantic next-word/phrase and the local model's ending completion,
     so you can see they solve different halves of the task.

Run:
    export OPENROUTER_API_KEY=sk-or-...          # required for the LLM half
    export OPENROUTER_MODEL=openai/gpt-4o-mini   # optional, this is the default
    python scripts/bench_llm.py

With no key set, it still runs the local half and explains what the LLM half
would add — so it's safe to run before you have a key.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.completer import Completer
from src.llm_next import LLMNext

# Mid-word cases (what the local model is built for) + between-word cases
# (where an LLM's semantic next-word prediction is the added value).
PROMPTS = [
    "детализированный портрет молод",      # mid-word: -> молодой/молодой человек
    "закат над горами, масляная живопис",  # mid-word: -> живопись
    "красивая девушка в красном платье на фоне ",  # between words: -> заката?
    "фотореалистичный кот сидит на ",      # between words: -> подоконнике?
    "в тёмном лесу, ",                      # between words: -> туман / атмосферный
]


def bench_local(comp: LLMNext, prompts, reps=200):
    core = Completer.load()
    print("=" * 72)
    print("ЛОКАЛЬНОЕ ЯДРО (offline, stdlib) — дополнение окончания текущего слова")
    print("=" * 72)
    for p in prompts:
        # time it
        t0 = time.perf_counter()
        for _ in range(reps):
            res = core.complete(p, k=3)
        dt_ms = (time.perf_counter() - t0) / reps * 1000
        tops = ", ".join(f"{s.word}" for s in res) or "—(между словами, ждёт LLM)"
        print(f"\n  {p!r}")
        print(f"    top-3: {tops}")
        print(f"    latency: {dt_ms:.3f} ms/call")


def bench_llm(prompts):
    llm = LLMNext.from_env()
    print("\n" + "=" * 72)
    print(f"LLM ЧЕРЕЗ OPENROUTER — предсказание следующего слова/фразы")
    print(f"model = {llm.model}")
    print("=" * 72)
    if not llm.api_key:
        print("\n  ⚠  OPENROUTER_API_KEY не задан — LLM-часть пропущена.")
        print("     Задай ключ и перезапусти, чтобы увидеть качество и задержку:")
        print("       export OPENROUTER_API_KEY=sk-or-...")
        return
    ttfts, totals = [], []
    for p in prompts:
        res = llm.predict(p)
        print(f"\n  {p!r}")
        if res.ok:
            print(f"    next -> {res.suggestion!r}")
            print(f"    TTFT: {res.time_to_first_token*1000:.0f} ms | "
                  f"total: {res.total_time*1000:.0f} ms")
            ttfts.append(res.time_to_first_token * 1000)
            totals.append(res.total_time * 1000)
        else:
            print(f"    ✗ {res.reason}")
    if ttfts:
        print("\n  --- сводка задержки LLM ---")
        print(f"    TTFT : min {min(ttfts):.0f} / avg {sum(ttfts)/len(ttfts):.0f} "
              f"/ max {max(ttfts):.0f} ms")
        print(f"    total: min {min(totals):.0f} / avg {sum(totals)/len(totals):.0f} "
              f"/ max {max(totals):.0f} ms")
        print(f"\n  Сравни: локальное ядро ~0.5 ms/вызов. Бюджет inline-подсказки"
              f" ~500 ms.")


if __name__ == "__main__":
    bench_local(None, PROMPTS)
    bench_llm(PROMPTS)
