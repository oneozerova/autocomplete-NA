# -*- coding: utf-8 -*-
"""
Поиск статей справки Авито по запросу пользователя (этап retrieval для RAG).

Пайплайн:
  1. Чистка HTML статей (BeautifulSoup), чанкинг длинных статей.
  2. Пять источников релевантности:
     - BM25 по леммам (pymorphy3),
     - char n-gram TF-IDF (устойчив к опечаткам),
     - dense-эмбеддинги intfloat/multilingual-e5-large (560M параметров, < 1B),
     - kNN по размеченным калибровочным запросам (голоса за их ground truth),
     - prior популярности статьи в calibration.
  3. Смешивание: веса подбираются координатным подъёмом по MAP@10 на calibration
     (kNN считается в leave-one-out режиме, чтобы не было утечки).
  4. Реранкинг топ-40 кандидатов кросс-энкодером BAAI/bge-reranker-v2-m3 (568M, < 1B),
     финальный скор — смесь stage-1 и реранкера.

Все модели open-source, скачиваются с HuggingFace и работают локально в ноутбуке.
Запросы пользователей никуда не отправляются.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymorphy3", "rank_bm25"], check=True)


def _cuda_works():
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch; assert torch.cuda.is_available(); "
         "a = torch.randn(64, 64, device='cuda'); print('ok', (a @ a).sum().item())"],
        capture_output=True, text=True)
    return r.returncode == 0 and "ok" in r.stdout


# Kaggle может выдать P100 (sm_60), который свежие сборки torch не поддерживают.
# В этом случае ставим сборку под CUDA 11.8 с поддержкой Pascal.
if not _cuda_works():
    print("preinstalled torch can't use this GPU; installing cu118 build...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.1+cu118",
                    "--index-url", "https://download.pytorch.org/whl/cu118",
                    "--extra-index-url", "https://pypi.org/simple"], check=False)
    # предустановленные torchvision/torchaudio собраны под другой torch и
    # ломают import transformers — убираем (они не нужны)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torchvision", "torchaudio", "torchcodec"], check=False)
USE_CUDA = _cuda_works()
print("USE_CUDA =", USE_CUDA)

if not USE_CUDA:
    # проверяем, что torch хотя бы импортируется (для CPU-фолбэка);
    # если реинсталл его сломал — возвращаем обычную сборку с PyPI
    r = subprocess.run([sys.executable, "-c", "import torch"], capture_output=True)
    if r.returncode != 0:
        print("torch import broken, restoring PyPI build...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--force-reinstall", "torch"], check=False)
        USE_CUDA = _cuda_works()
        print("USE_CUDA after restore =", USE_CUDA)

import re, functools, math, os, random
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
import pymorphy3
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

import glob
hits = glob.glob("/kaggle/input/**/articles.f", recursive=True)
print("input files:", glob.glob("/kaggle/input/**/*", recursive=True)[:20])
DATA = os.path.dirname(hits[0])
articles = pd.read_feather(f"{DATA}/articles.f").reset_index(drop=True)
calib = pd.read_feather(f"{DATA}/calibration.f").reset_index(drop=True)
test = pd.read_feather(f"{DATA}/test.f").reset_index(drop=True)
N_ART = len(articles)
ART_IDS = articles.article_id.to_numpy()
print(f"articles={N_ART}, calib={len(calib)}, test={len(test)}")

# ---------------------------------------------------------------- 1. HTML -> text
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

articles["text"] = articles.body.map(clean_html)

# ---------------------------------------------------------------- 2a. BM25 (леммы)
morph = pymorphy3.MorphAnalyzer()
tok_re = re.compile(r"[а-яёa-z0-9]+")

@functools.lru_cache(maxsize=500_000)
def lemma(w):  # кэш ускоряет pymorphy на порядок
    return morph.parse(w)[0].normal_form

def lemmas(text):
    return [lemma(t) for t in tok_re.findall(text.lower().replace("ё", "е"))]

TITLE_BOOST = 3
corpus_tok = [lemmas((r.title + " ") * TITLE_BOOST + r.text) for r in articles.itertuples()]
bm25 = BM25Okapi(corpus_tok)

def bm25_scores(queries):
    return np.stack([bm25.get_scores(lemmas(q)) for q in queries])

print("bm25 ready")

# ---------------------------------------------------------------- 2b. char TF-IDF
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000)
art_char = char_vec.fit_transform(((articles.title + " ") * TITLE_BOOST + articles.text).str.lower())
art_char = sk_normalize(art_char)

def char_scores(queries):
    q = sk_normalize(char_vec.transform([s.lower() for s in queries]))
    return (q @ art_char.T).toarray()

print("char tf-idf ready")

# ---------------------------------------------------------------- 2c. dense e5
from sentence_transformers import SentenceTransformer
device = "cuda" if USE_CUDA else "cpu"
# на CPU берём модель поменьше, чтобы уложиться во время сессии
E5_NAME = "intfloat/multilingual-e5-large" if USE_CUDA else "intfloat/multilingual-e5-base"
e5 = SentenceTransformer(E5_NAME, device=device)
e5.max_seq_length = 512

CHUNK, OVERLAP = 1200, 200
chunk_texts, chunk_owner = [], []   # owner: индекс статьи
for i, r in enumerate(articles.itertuples()):
    body = r.text
    pieces = [body[s:s + CHUNK] for s in range(0, max(len(body) - OVERLAP, 1), CHUNK - OVERLAP)] or [""]
    for p in pieces:
        chunk_texts.append(f"{r.title}. {p}")
        chunk_owner.append(i)
    chunk_texts.append(r.title)  # отдельный чанк-заголовок
    chunk_owner.append(i)
chunk_owner = np.array(chunk_owner)
print("chunks:", len(chunk_texts))

emb_chunks = e5.encode([f"passage: {t}" for t in chunk_texts], batch_size=64,
                       normalize_embeddings=True, show_progress_bar=True)

def dense_scores(queries):
    qe = e5.encode([f"query: {q}" for q in queries], batch_size=64, normalize_embeddings=True)
    sim = qe @ emb_chunks.T                      # (Q, n_chunks)
    out = np.full((len(queries), N_ART), -1.0, dtype=np.float32)
    for j in range(N_ART):
        out[:, j] = sim[:, chunk_owner == j].max(axis=1)
    return out

calib_q_emb = e5.encode([f"query: {q}" for q in calib.query_text], batch_size=64, normalize_embeddings=True)
test_q_emb = e5.encode([f"query: {q}" for q in test.query_text], batch_size=64, normalize_embeddings=True)
print("dense ready")

# ---------------------------------------------------------------- 2d. kNN по калибровке
id2idx = {a: i for i, a in enumerate(ART_IDS)}
calib_gt = [[id2idx[int(x)] for x in s.split()] for s in calib.ground_truth]

def knn_scores(q_emb, exclude_self=False, topk=20, beta=3.0):
    """Голоса статей из ground truth похожих калибровочных запросов."""
    sim = q_emb @ calib_q_emb.T                  # (Q, 500)
    out = np.zeros((len(q_emb), N_ART), dtype=np.float32)
    for qi in range(len(q_emb)):
        s = sim[qi].copy()
        if exclude_self:
            s[qi] = -1.0                         # leave-one-out при оценке на calibration
        nn = np.argpartition(-s, topk)[:topk]
        for ci in nn:
            w = max(s[ci], 0.0) ** beta
            for ai in calib_gt[ci]:
                out[qi, ai] += w
    return out

# ---------------------------------------------------------------- 2e. prior популярности
prior = np.zeros(N_ART, dtype=np.float32)
for g in calib_gt:
    for ai in g:
        prior[ai] += 1
prior = prior / prior.max()

# ---------------------------------------------------------------- 3. смешивание
def norm01(mat):
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    return (mat - mn) / np.maximum(mx - mn, 1e-9)

def ap10(ranking, gt_set):
    hits = score = 0.0
    for i, p in enumerate(ranking[:10], 1):
        if p in gt_set:
            hits += 1
            score += hits / i
    return score / min(len(gt_set), 10)

calib_gt_sets = [set(g) for g in calib_gt]

def map10(score_mat):
    top = np.argsort(-score_mat, axis=1)[:, :10]
    return float(np.mean([ap10(list(top[i]), calib_gt_sets[i]) for i in range(len(calib_gt_sets))]))

print("scoring calibration components...")
comp_calib = {
    "bm25": norm01(bm25_scores(calib.query_text.tolist())),
    "char": norm01(char_scores(calib.query_text.tolist())),
    "dense": norm01(dense_scores(calib.query_text.tolist())),
    "knn": norm01(knn_scores(calib_q_emb, exclude_self=True)),
    "prior": np.tile(prior, (len(calib), 1)),
}
for k, v in comp_calib.items():
    print(f"  MAP@10 {k:6s} = {map10(v):.4f}")

COMP = ["bm25", "char", "dense", "knn", "prior"]

def blend(comp, w):
    return sum(w[k] * comp[k] for k in COMP)

# координатный подъём по MAP@10
w = {"bm25": 0.3, "char": 0.3, "dense": 1.0, "knn": 1.0, "prior": 0.1}
best = map10(blend(comp_calib, w))
grid = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
for _ in range(4):
    improved = False
    for k in COMP:
        if k == "dense":
            continue  # фиксируем масштаб
        for g in grid:
            w2 = dict(w); w2[k] = g
            m = map10(blend(comp_calib, w2))
            if m > best + 1e-6:
                best, w, improved = m, w2, True
    if not improved:
        break
print(f"stage-1 weights: {w}\nstage-1 MAP@10 (calib, LOO-knn) = {best:.4f}")

# ---------------------------------------------------------------- 4. реранкер
from transformers import AutoTokenizer, AutoModelForSequenceClassification
RR_NAME = "BAAI/bge-reranker-v2-m3" if USE_CUDA else "BAAI/bge-reranker-base"
rr_tok = AutoTokenizer.from_pretrained(RR_NAME)
rr = AutoModelForSequenceClassification.from_pretrained(
    RR_NAME, torch_dtype=torch.float16 if USE_CUDA else torch.float32).to(device).eval()

TOP_RERANK = 40 if USE_CUDA else 20
art_doc = (articles.title + ". " + articles.text.str.slice(0, 1500)).tolist()

@torch.no_grad()
def rerank_scores(queries, stage1):
    """Возвращает матрицу (-inf вне топ-кандидатов) со скором кросс-энкодера."""
    out = np.full(stage1.shape, -1e9, dtype=np.float32)
    cand = np.argsort(-stage1, axis=1)[:, :TOP_RERANK]
    pairs, where = [], []
    for qi, q in enumerate(queries):
        for ai in cand[qi]:
            pairs.append((q, art_doc[ai]))
            where.append((qi, ai))
    scores = []
    B = 64
    for s in range(0, len(pairs), B):
        batch = pairs[s:s + B]
        inp = rr_tok([p[0] for p in batch], [p[1] for p in batch],
                     padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        scores.extend(rr(**inp).logits.view(-1).float().cpu().numpy())
    for (qi, ai), sc in zip(where, scores):
        out[qi, ai] = sc
    return out

print("reranking calibration...")
stage1_calib = blend(comp_calib, w)
rr_calib = rerank_scores(calib.query_text.tolist(), stage1_calib)

# нормируем реранкер сигмоидой, смешиваем только на топ-кандидатах
def final_scores(stage1, rr_mat, alpha):
    s1 = norm01(stage1)
    rr_n = 1 / (1 + np.exp(-rr_mat))
    rr_n[rr_mat < -1e8] = 0.0
    mask = (rr_mat > -1e8).astype(np.float32)
    return s1 + alpha * rr_n * mask

best_a, best_m = 0.0, best
for a in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
    m = map10(final_scores(stage1_calib, rr_calib, a))
    print(f"  alpha={a:.1f}  MAP@10={m:.4f}")
    if m > best_m:
        best_a, best_m = a, m
print(f"final: alpha={best_a}, MAP@10 (calib) = {best_m:.4f}")

# ---------------------------------------------------------------- 5. предсказание на test
print("scoring test...")
comp_test = {
    "bm25": norm01(bm25_scores(test.query_text.tolist())),
    "char": norm01(char_scores(test.query_text.tolist())),
    "dense": norm01(dense_scores(test.query_text.tolist())),
    "knn": norm01(knn_scores(test_q_emb, exclude_self=False)),
    "prior": np.tile(prior, (len(test), 1)),
}
stage1_test = blend(comp_test, w)
rr_test = rerank_scores(test.query_text.tolist(), stage1_test)
final_test = final_scores(stage1_test, rr_test, best_a)

top10 = np.argsort(-final_test, axis=1)[:, :10]
test["answer"] = [" ".join(str(ART_IDS[ai]) for ai in row) for row in top10]
test[["query_id", "answer"]].to_csv("answer.csv", index=False)
print("answer.csv saved")

# отчёт для воспроизводимости
with open("report.txt", "w") as f:
    f.write(f"weights: {w}\nalpha: {best_a}\n")
    for k, v in comp_calib.items():
        f.write(f"MAP@10 {k} = {map10(v):.4f}\n")
    f.write(f"stage1 MAP@10 = {map10(stage1_calib):.4f}\n")
    f.write(f"final MAP@10 = {best_m:.4f}\n")
print(open("report.txt").read())
