"""How to embed the autocompleter as a feature in a larger project.

The whole public surface is `Completer`: load once at startup, call `complete`
per keystroke. No torch, no network, ~26 MB artifacts, sub-millisecond calls.

    python examples/integration_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.completer import Completer

# 1) Load once (e.g. at service start / FastAPI lifespan / Streamlit cache).
completer = Completer.load()


# 2) Wrap it however your app needs. Example: a JSON-friendly endpoint handler.
def autocomplete(text: str, k: int = 5) -> list[dict]:
    """Return top-k completions for the word currently being typed in `text`."""
    return [
        {
            "word": s.word,       # full completed word
            "ending": s.ending,   # chars to append after what the user typed
            "score": round(s.score, 3),
            "source": s.source,   # vocab | model | vocab+model
        }
        for s in completer.complete(text, k=k)
    ]


if __name__ == "__main__":
    for prompt in [
        "детализированный портрет молод",
        "закат над горами, масляная живопис",
        "реалистичн",
    ]:
        print(f"\n{prompt!r}")
        for r in autocomplete(prompt):
            print(f"   +{r['ending']:10} -> {r['word']:22} [{r['source']}]")
