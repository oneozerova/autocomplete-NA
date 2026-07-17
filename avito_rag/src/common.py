"""Общий код: чистка HTML, токенизация/лемматизация, метрика MAP@10."""
import re
import functools
import pandas as pd
from bs4 import BeautifulSoup
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_token_re = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def clean_html(html: str) -> str:
    """HTML -> плоский текст: убираем разметку, скрипты, схлопываем пробелы."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


@functools.lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    return _morph.parse(word)[0].normal_form


def tokenize(text: str, lemmatize: bool = True) -> list[str]:
    tokens = _token_re.findall(text.lower().replace("ё", "е"))
    if lemmatize:
        tokens = [_lemma(t) for t in tokens]
    return tokens


def ap_at_10(pred: list[int], gt: set[int]) -> float:
    hits = 0
    score = 0.0
    for i, p in enumerate(pred[:10], start=1):
        if p in gt:
            hits += 1
            score += hits / i
    return score / min(len(gt), 10)


def map_at_10(preds: dict[int, list[int]], calib: pd.DataFrame) -> float:
    total = 0.0
    for _, row in calib.iterrows():
        gt = set(int(x) for x in row.ground_truth.split())
        total += ap_at_10(preds[row.query_id], gt)
    return total / len(calib)
