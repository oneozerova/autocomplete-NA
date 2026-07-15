"""Optional online layer: predict the *next word / short phrase* via an LLM.

This is deliberately a **separate, opt-in layer** from the local `Completer`.
The local char-n-gram completes the *ending of the word being typed* in ~0.5 ms
offline; it does not (and should not) try to guess the next *word by meaning*
(`девушка в красном платье на фоне …` → `заката`). That semantic next-word
prediction is where an LLM helps — but it costs a network round-trip
(~100–900 ms TTFT), so it must never run on every keystroke.

Design constraints kept faithful to the project's ethos:
  * **stdlib only** — talks to OpenRouter over `urllib`, no `requests`/`openai`.
  * **degrades gracefully** — no key, offline, or timeout → returns a result with
    `ok=False` and a reason, never raises into the caller's hot path.
  * **cancellable + streaming** — a `threading.Event` stops an in-flight request
    the moment the user types again; `time_to_first_token` is measured so you can
    benchmark whether it clears your latency budget.

Usage (see `scripts/bench_llm.py` for a runnable benchmark):

    llm = LLMNext.from_env()                 # reads OPENROUTER_API_KEY
    res = llm.predict("закат над горами, масляная ")
    if res.ok:
        print(res.suggestion, res.time_to_first_token)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_TRAILING_WORD = re.compile(r"[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*$")
_ANY_WORD = re.compile(r"[а-яёА-ЯЁ]+")

# A small, fast model is the right tool for inline suggestion; a chat-grade model
# is too slow. Override with OPENROUTER_MODEL. (Any OpenRouter model id works.)
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# The task, pinned so the model returns a bare continuation, not a chat reply.
SYSTEM_PROMPT = (
    "Ты — движок автодополнения русских промптов для генерации изображений. "
    "Тебе дают НАЧАЛО промпта. Продолжи его самым вероятным СЛЕДУЮЩИМ словом "
    "(или коротким словосочетанием из 1–3 слов), которое естественно идёт дальше. "
    "Отвечай ТОЛЬКО продолжением, без кавычек, без пояснений, без точки в конце. "
    "Если начало обрывается на середине слова — дополни это слово."
)


@dataclass
class NextResult:
    ok: bool
    suggestion: str = ""
    reason: str = ""                      # why ok=False (no_key / timeout / http_error / cancelled)
    time_to_first_token: float | None = None   # seconds, TTFT — the latency that matters
    total_time: float | None = None            # seconds, full response
    model: str = ""
    tokens: list[str] = field(default_factory=list)


class LLMNext:
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL,
                 timeout: float = 2.5, max_tokens: int = 12,
                 temperature: float = 0.2):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_env(cls, **kw) -> "LLMNext":
        return cls(api_key=os.environ.get("OPENROUTER_API_KEY"), **kw)

    # ---- main API -------------------------------------------------------
    def predict(self, text: str, stop: threading.Event | None = None) -> NextResult:
        """Predict the next word/phrase for `text`. Streams so we can measure TTFT
        and abort early via `stop`. Never raises — failures come back as ok=False."""
        if not self.api_key:
            return NextResult(ok=False, reason="no_key", model=self.model)
        if not text.strip():
            return NextResult(ok=False, reason="empty_input", model=self.model)

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            OPENROUTER_URL, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Optional attribution headers OpenRouter recommends:
                "HTTP-Referer": "https://github.com/ru-prompt-autocomplete",
                "X-Title": "ru-prompt-autocomplete",
            },
        )

        t0 = time.perf_counter()
        ttft: float | None = None
        tokens: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    if stop is not None and stop.is_set():
                        return NextResult(ok=False, reason="cancelled",
                                          time_to_first_token=ttft,
                                          total_time=time.perf_counter() - t0,
                                          model=self.model, tokens=tokens)
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        tokens.append(piece)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            return NextResult(ok=False, reason=f"http_{e.code}: {detail}",
                              model=self.model)
        except (urllib.error.URLError, TimeoutError) as e:
            return NextResult(ok=False, reason=f"network: {e}", model=self.model)

        suggestion = "".join(tokens).strip().strip('"').strip()
        # keep it to a short continuation even if the model over-generates
        suggestion = _first_words(suggestion, 3)
        return NextResult(
            ok=bool(suggestion), suggestion=suggestion,
            reason="" if suggestion else "empty_output",
            time_to_first_token=ttft, total_time=time.perf_counter() - t0,
            model=self.model, tokens=tokens,
        )


def _first_words(s: str, n: int) -> str:
    """Trim a possibly chatty continuation down to its first n words / first line."""
    s = s.splitlines()[0] if s else s
    parts = s.split()
    return " ".join(parts[:n])


def should_suggest(text: str, min_words: int = 2) -> bool:
    """Gate: should we even spend an LLM round-trip on this state?

    Two rules, both learned from the benchmark:
      * NOT mid-word. If the text ends inside a word (`…живопис`), the local ghost
        owns it — and the LLM actively hurts there (it produced `живопис|ьная`, a
        non-word, because its tokenizer doesn't split a Russian stem cleanly).
        We fire only *between* words (trailing space / punctuation).
      * Enough context. Need ≥ `min_words` real words, or the next-word guess is
        meaningless and not worth ~1.2 s + a paid call.
    """
    if _TRAILING_WORD.search(text):
        return False
    return len(_ANY_WORD.findall(text)) >= min_words


class Debouncer:
    """Run a slow prediction only after the user pauses, cancelling stale calls.

    A UI layer (websocket handler / Streamlit) calls `submit(text, on_result)` on
    every keystroke. Flow, tuned to the measured ~1.2 s TTFT of the LLM:

      1. GATE — `should_suggest` drops mid-word and low-context states before any
         timer, so we never call the network where the local model already wins.
      2. CACHE — a state already answered (backspace / retyping) returns instantly
         from an LRU without a second paid round-trip.
      3. DEBOUNCE — wait `delay` s of quiet (default 0.4 s) to detect a real pause.
      4. CANCEL — a newer keystroke restarts the timer *and* signals the in-flight
         request to abort (`stop` event), so only the latest input hits the wire.
    """

    def __init__(self, llm: LLMNext, delay: float = 0.4,
                 min_words: int = 2, cache_size: int = 256):
        self.llm = llm
        self.delay = delay
        self.min_words = min_words
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, NextResult]" = OrderedDict()
        self._timer: threading.Timer | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def submit(self, text, on_result):
        with self._lock:
            # cancel a pending timer and abort any request already in flight
            if self._timer is not None:
                self._timer.cancel()
            self._stop.set()
            self._stop = threading.Event()

            if not should_suggest(text, self.min_words):
                return  # mid-word or too little context — local model owns it

            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                on_result(cached)
                return

            stop = self._stop

            def fire():
                res = self.llm.predict(text, stop=stop)
                if stop.is_set():
                    return
                if res.ok:
                    with self._lock:
                        self._cache[text] = res
                        if len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)
                on_result(res)

            self._timer = threading.Timer(self.delay, fire)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._stop.set()
