# The runtime needs ZERO third-party packages: scripts/serve.py and src/llm_next.py
# import only the standard library (http.server, json, urllib). streamlit is for the
# app.py demo and pymorphy3 is build-time only — neither belongs in the image. So
# there is no pip install here at all, which is why this stays a ~150 MB image and
# builds in seconds.
FROM python:3.12-slim

# Fail fast and log straight through to Docker (no buffered stdout on crash).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# The service runs the LLM next-tag layer (src/llm_next.py → OpenRouter), which
# is pure stdlib and needs NO model files. The only artifact copied is
# web_model.json (12 MB), used by the optional self-contained demo page at `/`.
# The 40 MB of Completer models (ngram/context/vocab/morph) are deliberately
# left out: word-endings run client-side for zero latency, so the server never
# loads them. Add them back only if you introduce a server-side /complete route.
COPY src/                   ./src/
COPY scripts/               ./scripts/
COPY frontend/              ./frontend/
COPY models/web_model.json  ./models/web_model.json

# Run unprivileged: the process only ever reads its own artifacts and makes an
# outbound HTTPS call, so it has no reason to be root inside the container.
RUN useradd --system --uid 1001 app && chown -R app:app /app
USER app

EXPOSE 8000

# urllib, not curl — the slim image has no curl and adding it would double the
# image's attack surface for one probe. /health is deliberately cheap: it does not
# build the 12 MB page and does not require the LLM key.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "scripts/serve.py"]
