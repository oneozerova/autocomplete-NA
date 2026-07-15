"""Local demo server: one window — offline ghost + inline chips + LLM next-word.

Why a server at all? The browser page (`frontend/prompt.html`) does the word-ending
completion and the inline parameter chips entirely client-side — no network,
~0.5 ms. The *next-word* layer needs an LLM, and the OpenRouter API key must NEVER
ship to the browser. So this tiny stdlib server does two jobs, in one window:

    GET  /        -> the merged demo page (ghost + chips), web model baked in
    POST /next    -> {"text": "..."} -> proxied to LLMNext, key stays server-side

Run:
    export OPENROUTER_API_KEY=sk-or-...        # optional; without it only the
    python scripts/serve.py                    #   offline ghost works
    # open http://localhost:8000

No new dependencies (http.server from the stdlib). ThreadingHTTPServer so a slow
~1 s LLM call never blocks page loads or other requests.
"""
import functools
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_next import LLMNext, should_suggest

WEB_MODEL = ROOT / "models" / "web_model.json"
TEMPLATE = ROOT / "frontend" / "prompt.html"   # merged: ghost + inline chips

# Loopback by default (a local demo must not expose an LLM proxy to the LAN), but
# overridable: inside a container 127.0.0.1 binds the *container's* loopback, so
# nothing outside it — Docker, Traefik — can reach the port. Deployments set
# HOST=0.0.0.0 and rely on the reverse proxy for exposure.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

LLM = LLMNext.from_env()   # reads OPENROUTER_API_KEY / OPENROUTER_MODEL


@functools.lru_cache(maxsize=1)
def build_page() -> bytes:
    # Cached: the baked-in web model is ~12 MB, so re-reading it and re-running the
    # replace on every GET burns ~25 MB of I/O and a large string copy per page view.
    # The artifacts are immutable for the life of the process — read them once.
    if not WEB_MODEL.exists():
        return b"<h3>No web model. Build it: python -m src.export_web</h3>"
    model_json = WEB_MODEL.read_text(encoding="utf-8")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__MODEL__*/", model_json)
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass          # client hung up early (navigated away / aborted fetch)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, build_page(), "text/html; charset=utf-8")
        elif self.path == "/health":
            # Cheap liveness probe for the orchestrator: it must not build the 12 MB
            # page, and it must not depend on the LLM (a missing key is a degraded
            # mode — offline ghost still works — not an unhealthy container).
            ok = WEB_MODEL.exists()
            body = json.dumps({"ok": ok, "llm": bool(LLM.api_key)}).encode()
            self._send(200 if ok else 503, body)
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/next":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            text = payload.get("text", "")
        except (ValueError, json.JSONDecodeError):
            self._send(400, b'{"ok":false,"reason":"bad_request"}')
            return

        # Server-side gate too: never spend a call the gate would skip, even if a
        # client forgets to check (mid-word / too little context).
        if not should_suggest(text):
            self._send(200, json.dumps({"ok": False, "reason": "gated"}).encode())
            return

        res = LLM.predict(text)
        out = {
            "ok": res.ok,
            "suggestion": res.suggestion,
            "reason": res.reason,
            "ttft_ms": round(res.time_to_first_token * 1000) if res.time_to_first_token else None,
            "total_ms": round(res.total_time * 1000) if res.total_time else None,
            "model": res.model,
        }
        self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"))

    def log_message(self, *a):        # keep the console quiet
        pass


if __name__ == "__main__":
    key = "set" if LLM.api_key else "MISSING (offline ghost only)"
    print(f"RU autocomplete demo → http://{HOST}:{PORT}")
    print(f"  model={LLM.model}  OPENROUTER_API_KEY={key}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
