"""Export a compact model the browser can run entirely client-side.

The Streamlit demo does completion *in the browser* (zero round-trips → instant
ghost text). This script bakes the needed data down to a small JSON:

    models/web_model.json  =  { words:[[w,freq]...], ctx:{...}, cfg:{...} }

The heavy CharNGram (OOV generalisation) is intentionally left out of the web
build — for ghost text we complete *known* words ranked by frequency + context
agreement, which the trie + context model cover. The full Python `Completer`
(with n-gram) remains the integration API; this is only the demo's fast path.

    python -m src.export_web
"""
from __future__ import annotations

import json
from pathlib import Path

from .completer import AGR_W
from .context_model import STOPWORDS, ContextModel

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

TOP_WORDS = 80_000      # covers the vast majority of real typing, ~1.5 MB JSON
CTX_MIN_COUNT = 4       # prune rare suffix-bigrams to shrink the payload
WEB_WORD_CAP = 10       # successors per prev word kept in the *web* word-bigram
ALPHA = 1.6             # frequency weight   (mirrors completer.py)
GAMMA = 9.0             # agreement weight    (mirrors completer.py)


# ---- compact grammeme codes for the browser agreement model --------------
# Each word -> 4-char code  pos[N/V/A/O] number[s/p/-] gender[m/f/n/-] nomn[1/0].
# The agreement tables are re-keyed to the same codes so the JS never needs the
# full pymorphy grammeme strings. gn-key (adj/noun agreement) = gender+number.
def _pos_code(pos: str) -> str:
    from .morph import ADJISH, NOUNISH
    if pos in NOUNISH:
        return "N"
    if pos in ADJISH:
        return "A"
    return "V" if pos == "VERB" else "O"


def _num_code(n: str) -> str:
    return {"sing": "s", "plur": "p"}.get(n, "-")


def _gen_code(g: str) -> str:
    return {"masc": "m", "femn": "f", "neut": "n"}.get(g, "-")


def _feat_code(ft) -> str:
    return (_pos_code(ft.pos) + _num_code(ft.number) + _gen_code(ft.gender)
            + ("1" if ft.case in ("nomn", "") else "0"))


def _gn_web(key: str) -> str:            # "femn|sing" -> "fs"
    g, _, n = key.partition("|")
    return _gen_code(g) + _num_code(n)


def main() -> None:
    # words (already sorted by freq desc in vocab.tsv)
    from .domain_lexicon import domain_forms
    domain = domain_forms()
    words = []
    seen = set()
    with (MODELS / "vocab.tsv").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            w, c = line.rstrip("\n").split("\t")
            if i < TOP_WORDS:
                words.append([w, int(c)])
                seen.add(w)
            elif w in domain and w not in seen:   # always keep prompt vocabulary
                words.append([w, int(c)])
                seen.add(w)
    # re-sort so the trie's per-node top stays frequency-ordered
    words.sort(key=lambda x: -x[1])

    # windowed context model, pruned for the web
    ctx = ContextModel.load(MODELS / "context.json")
    ctx.prune(CTX_MIN_COUNT)

    # lexical word-bigram, trimmed for the web: keep only successors the browser
    # can actually complete (present in `words`) and cap per prev word.
    web_wadj = {}
    for a, tbl in ctx.wadj.items():
        kept = {b: c for b, c in tbl.items() if b in seen}
        if len(kept) > WEB_WORD_CAP:
            kept = dict(sorted(kept.items(), key=lambda kv: -kv[1])[:WEB_WORD_CAP])
        if kept:
            web_wadj[a] = kept

    ctx_data = {
        "suf_len": ctx.suf_len, "window": ctx.window, "decay": ctx.decay,
        "word_w": ctx.word_w,
        "adj": {a: t for a, t in ctx.adj.items()},
        "adj_total": dict(ctx.adj_total),
        "tri": {a: t for a, t in ctx.tri.items()},
        "tri_total": dict(ctx.tri_total),
        "wadj": web_wadj,
        "wadj_total": {a: sum(t.values()) for a, t in web_wadj.items()},
        "bigram": {str(d): {a: t for a, t in ctx.bigram[d].items()} for d in ctx.bigram},
        "prev_total": {str(d): dict(ctx.prev_total[d]) for d in ctx.prev_total},
        "uni": dict(ctx.uni), "uni_total": ctx.uni_total,
    }

    # morphology + factored agreement (optional — only if built)
    morph_web: dict[str, str] = {}
    agr_web = None
    morph_path = MODELS / "morph.tsv.gz"
    agr_path = MODELS / "agreement.json"
    if morph_path.exists() and agr_path.exists():
        from .agreement import BACKOFF_ALPHA, GOV_WINDOW, AgreementModel
        from .morph import MorphTable

        morph = MorphTable.load(morph_path)
        for w in seen:                       # only words the browser can rank
            ft = morph.get(w)
            if ft is None:
                continue
            code = _feat_code(ft)
            if code[0] != "O":               # skip words with no agreement role
                morph_web[w] = code
        agr = AgreementModel.load(agr_path)
        agr_web = {
            "alpha": BACKOFF_ALPHA, "gov_window": GOV_WINDOW,
            # re-keyed to compact codes: number for vnum, gender+number for gn
            "vnum": {_num_code(k): {_num_code(k2): c for k2, c in row.items()}
                     for k, row in agr.vnum.items()},
            "adj_gn": {_gn_web(k): {_gn_web(k2): c for k2, c in row.items()}
                       for k, row in agr.adj_gn.items()},
            "noun_gn": {_gn_web(k): {_gn_web(k2): c for k2, c in row.items()}
                        for k, row in agr.noun_gn.items()},
        }

    out = {
        "words": words,
        "ctx": ctx_data,
        "morph": morph_web,
        "agr": agr_web,
        "stop": sorted(STOPWORDS),   # so the browser filters function words too
        "cfg": {"alpha": ALPHA, "gamma": GAMMA, "suf_len": ctx.suf_len,
                "word_w": ctx.word_w, "agr_w": AGR_W},
    }
    path = MODELS / "web_model.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    print(f"web model: {len(words)} words, {len(morph_web)} morph, "
          f"agr={'yes' if agr_web else 'no'} -> {path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
