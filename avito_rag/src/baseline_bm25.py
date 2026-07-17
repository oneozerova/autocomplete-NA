"""Baseline: BM25 (title с бустом + body), оценка MAP@10 на calibration."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from rank_bm25 import BM25Okapi
from common import clean_html, tokenize, map_at_10

DATA = pathlib.Path(__file__).parent.parent / "candidate_data"

articles = pd.read_feather(DATA / "articles.f")
calib = pd.read_feather(DATA / "calibration.f")

TITLE_BOOST = 3  # заголовок повторяется несколько раз — простой способ буста в BM25

print("cleaning html + lemmatizing corpus...")
corpus_tokens = []
for _, row in articles.iterrows():
    text = (row.title + " ") * TITLE_BOOST + clean_html(row.body)
    corpus_tokens.append(tokenize(text))

bm25 = BM25Okapi(corpus_tokens)
ids = articles.article_id.tolist()

def rank(query: str, k: int = 10) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    order = scores.argsort()[::-1][:k]
    return [ids[i] for i in order]

print("ranking calibration queries...")
preds = {row.query_id: rank(row.query_text) for _, row in calib.iterrows()}
print(f"BM25 baseline MAP@10 = {map_at_10(preds, calib):.4f}")
