"""Test the latency hypotheses for the LLM next-word layer with real numbers.

The metric that decides inline-suggestion feel is TTFT (time-to-first-token),
not total time — the ghost text can start rendering the moment the first token
lands. Each experiment prints p50/p90 TTFT so a slow tail is visible.

Hypotheses (ranked by expected impact):
  H1  provider/model is the dominant lever — Groq/Cerebras vs gpt-4o-mini.
  H2  reusing one TCP/TLS connection removes the per-call handshake.
  H3  a shorter system prompt lowers prefill → lower TTFT.
  H4  max_tokens does NOT change TTFT (only total) — a control.

Run:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/bench_latency.py            # all experiments
    python scripts/bench_latency.py h1         # just one (h1/h2/h3/h4)

No key → prints the plan and exits 0 (offline self-check still runs).
"""
import http.client
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_next import SYSTEM_PROMPT, LLMNext

API_KEY = os.environ.get("OPENROUTER_API_KEY")

# One representative between-word prompt (the state where the LLM layer fires).
PROMPT = "красивая девушка в красном платье на фоне "

REPS = 6           # measured calls per config (+1 warm-up, discarded)
MAX_TOKENS = 12

# --- H1 configs: (label, model, provider-routing dict) -----------------------
# Provider routing is passed straight through to OpenRouter. Bad model ids or
# unavailable providers surface as an http_ error row, not a crash — so the
# numbers speak and you can edit this list freely.
H1_CONFIGS = [
    ("gpt-4o-mini (baseline)",   "openai/gpt-4o-mini",                 None),
    ("gpt-4o-mini sort=latency", "openai/gpt-4o-mini",                 {"sort": "latency"}),
    ("gemini-flash-latest",      "google/gemini-flash-latest",         None),
    ("ministral-3b sort=lat",    "mistralai/ministral-3b-2512",        {"sort": "latency"}),
    ("llama-3.1-8b @Groq",       "meta-llama/llama-3.1-8b-instruct",   {"order": ["Groq"], "allow_fallbacks": False}),
    ("llama-3.3-70b @Groq",      "meta-llama/llama-3.3-70b-instruct",  {"order": ["Groq"], "allow_fallbacks": False}),
    ("llama-3.3-70b @SambaNova", "meta-llama/llama-3.3-70b-instruct",  {"order": ["SambaNova"], "allow_fallbacks": False}),
]

# A one-line replacement for the ~80-token system prompt, to test H3.
SHORT_PROMPT = "Продолжи русский промпт одним–двумя словами. Только продолжение."


def _pct(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else xs[0]


def _summary(label, ttfts, totals):
    if not ttfts:
        print(f"  {label:<26} — нет успешных вызовов")
        return
    print(f"  {label:<26} TTFT p50 {_pct(ttfts,50):4.0f}  p90 {_pct(ttfts,90):4.0f} ms"
          f"   | total p50 {_pct(totals,50):4.0f} ms  (n={len(ttfts)})")


# --- H2 transport: raw http.client so we can reuse ONE connection ------------
def _send_raw(conn: http.client.HTTPSConnection, model, api_key):
    """One streaming call over `conn` (kept alive between calls). Returns
    (ttft_ms, total_ms). Timing starts before request() so a fresh conn's
    TLS handshake is counted — that's the whole point of H2."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": MAX_TOKENS, "temperature": 0.2, "stream": True,
    })
    t0 = time.perf_counter()
    conn.request("POST", "/api/v1/chat/completions", body=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    })
    resp = conn.getresponse()
    ttft = None
    while True:
        line = resp.readline()
        if not line:
            break
        s = line.decode("utf-8", "replace").strip()
        if not s.startswith("data:"):
            continue
        data = s[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {})
        except json.JSONDecodeError:
            continue
        if delta.get("content") and ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
    resp.read()  # drain so the socket is reusable
    total = (time.perf_counter() - t0) * 1000
    return ttft, total


def _run_llm(model, provider=None, system_prompt=None, max_tokens=MAX_TOKENS):
    """Measured TTFT/total lists via the production LLMNext path (fresh conn each)."""
    # provider={} → no instance-default routing; H1 passes provider per call.
    llm = LLMNext(api_key=API_KEY, model=model, max_tokens=max_tokens,
                  timeout=8.0, provider={})
    ttfts, totals, sample, err = [], [], "", ""
    for i in range(REPS + 1):
        r = llm.predict(PROMPT, provider=provider, system_prompt=system_prompt)
        if i == 0:
            continue  # warm-up (routing decision / DNS), discarded
        if r.ok and r.time_to_first_token is not None:
            ttfts.append(r.time_to_first_token * 1000)
            totals.append(r.total_time * 1000)
            sample = r.suggestion
        else:
            err = r.reason
    return ttfts, totals, sample, err


def h1():
    print("\n" + "=" * 78)
    print("H1 — ПРОВАЙДЕР/МОДЕЛЬ (ожидаемо главный рычаг)")
    print("=" * 78)
    for label, model, provider in H1_CONFIGS:
        ttfts, totals, sample, err = _run_llm(model, provider=provider)
        _summary(label, ttfts, totals)
        if sample:
            print(f"  {'':<26}   → {sample!r}")
        elif err:
            print(f"  {'':<26}   ✗ {err[:90]}")


def h2():
    print("\n" + "=" * 78)
    print("H2 — ПЕРЕИСПОЛЬЗОВАНИЕ СОЕДИНЕНИЯ (убираем TLS-рукопожатие)")
    print("=" * 78)
    model = "openai/gpt-4o-mini"
    # fresh connection each call (what urllib does today)
    fresh = []
    for i in range(REPS + 1):
        conn = http.client.HTTPSConnection("openrouter.ai", timeout=8.0)
        try:
            t, _ = _send_raw(conn, model, API_KEY)
            if i and t:
                fresh.append(t)
        finally:
            conn.close()
    # one persistent connection, reused
    warm = []
    conn = http.client.HTTPSConnection("openrouter.ai", timeout=8.0)
    try:
        for i in range(REPS + 1):
            t, _ = _send_raw(conn, model, API_KEY)
            if i and t:
                warm.append(t)
    finally:
        conn.close()
    _summary("fresh conn / call", fresh, fresh)
    _summary("reused conn", warm, warm)
    if fresh and warm:
        print(f"\n  экономия на рукопожатии ≈ {_pct(fresh,50) - _pct(warm,50):.0f} ms/вызов (p50)")


def h3():
    print("\n" + "=" * 78)
    print("H3 — ДЛИНА SYSTEM-ПРОМПТА (prefill)")
    print("=" * 78)
    model = "openai/gpt-4o-mini"
    for label, sp in [("полный промпт", None), ("короткий промпт", SHORT_PROMPT)]:
        ttfts, totals, _, err = _run_llm(model, system_prompt=sp)
        _summary(label, ttfts, totals)
        if err:
            print(f"  {'':<26}   ✗ {err[:90]}")


def h4():
    print("\n" + "=" * 78)
    print("H4 — max_tokens (контроль: должен двигать total, не TTFT)")
    print("=" * 78)
    model = "openai/gpt-4o-mini"
    for mt in (4, 12, 64):
        ttfts, totals, _, err = _run_llm(model, max_tokens=mt)
        _summary(f"max_tokens={mt}", ttfts, totals)
        if err:
            print(f"  {'':<26}   ✗ {err[:90]}")


EXPERIMENTS = {"h1": h1, "h2": h2, "h3": h3, "h4": h4}


def _selfcheck():
    """Offline: the SSE line parser keeps only content deltas. Runs with no key."""
    line = 'data: {"choices":[{"delta":{"content":"заката"}}]}'
    s = line[5:].strip()
    delta = (json.loads(s).get("choices") or [{}])[0].get("delta", {})
    assert delta.get("content") == "заката"
    assert 'data: [DONE]'[5:].strip() == "[DONE]"


if __name__ == "__main__":
    _selfcheck()
    if not API_KEY:
        print("OPENROUTER_API_KEY не задан — тесты требуют сети и ключа.")
        print("Гипотезы к проверке:", ", ".join(EXPERIMENTS))
        print("Задай ключ и запусти: python scripts/bench_latency.py")
        sys.exit(0)
    which = sys.argv[1:] or list(EXPERIMENTS)
    for name in which:
        EXPERIMENTS[name]()
