"""Experiment: small char-LSTM vs our char n-gram on the SAME ending-completion
benchmark. Answers "would an LSTM be better here?" empirically.

Not part of the shipped project (needs torch). Trains a lightweight char-level
LSTM language model on the same word forms the n-gram uses, then completes via
the same beam search and is scored on identical held-out targets.

    python experiments/lstm_vs_ngram.py
"""
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import load_vocab                     # noqa: E402
from src.ngram_model import CharNGram, BOS, EOS         # noqa: E402
from src.train import build_ngram                       # noqa: E402
from src.trie import FreqTrie                           # noqa: E402
from src.completer import Completer                     # noqa: E402

SEED = 21
torch.manual_seed(SEED)
random.seed(SEED)

TRAIN_SECONDS = 240      # wall-clock training budget (lightweight, CPU)
BATCH = 512
EMB, HID, LAYERS = 64, 256, 1
N_TARGETS = 1500
PREFIX_LENS = (4, 5)
K = 5


# ---- data ---------------------------------------------------------------
def build_charset(items):
    chars = set()
    for w, _ in items:
        chars.update(w)
    idx = {c: i + 3 for i, c in enumerate(sorted(chars))}
    idx["<pad>"] = 0
    idx[BOS] = 1
    idx[EOS] = 2
    return idx


def encode(word, c2i):
    return [c2i[BOS]] + [c2i[c] for c in word] + [c2i[EOS]]


class CharLSTM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HID, num_layers=LAYERS, batch_first=True)
        self.fc = nn.Linear(HID, vocab)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        return self.fc(o)


def train(items, c2i):
    V = len(c2i)
    model = CharLSTM(V)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    encoded = [encode(w, c2i) for w, _ in items]
    weights = [CharNGram.freq_weight(c) for _, c in items]
    N = len(encoded)
    model.train()
    t0 = time.time()
    step = 0
    while time.time() - t0 < TRAIN_SECONDS:
        idx = random.choices(range(N), weights=weights, k=BATCH)
        seqs = [encoded[i] for i in idx]
        L = max(len(s) for s in seqs)
        batch = torch.zeros(BATCH, L, dtype=torch.long)
        for r, s in enumerate(seqs):
            batch[r, :len(s)] = torch.tensor(s)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(inp)
        loss = lossf(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 200 == 0:
            print(f"   step {step:5d}  loss {loss.item():.3f}  "
                  f"({time.time()-t0:.0f}s)")
    print(f"   trained {step} steps in {time.time()-t0:.0f}s")
    model.eval()
    return model


# ---- LSTM completion via (stateless) beam search ------------------------
@torch.no_grad()
def lstm_complete(model, prefix, c2i, i2c, k=K, beam=16, max_len=12):
    base = encode(prefix, c2i)[:-1]              # ^ + prefix  (no EOS)
    beams = [(0.0, "")]
    finished = {}
    for _ in range(max_len):
        seqs = [base + [c2i[ch] for ch in suf] for _, suf in beams]
        L = max(len(s) for s in seqs)
        batch = torch.zeros(len(seqs), L, dtype=torch.long)
        lens = []
        for r, s in enumerate(seqs):
            batch[r, :len(s)] = torch.tensor(s); lens.append(len(s) - 1)
        logits = model(batch)                    # [B, L, V]
        last = logits[torch.arange(len(seqs)), lens]         # [B, V]
        logp = torch.log_softmax(last, dim=-1)
        cand = []
        topv, topi = logp.topk(beam, dim=-1)
        for b, (lp, suf) in enumerate(beams):
            for j in range(beam):
                cid = topi[b, j].item(); clp = topv[b, j].item()
                if cid == c2i[EOS]:
                    word = prefix + suf
                    finished[word] = max(finished.get(word, -1e9),
                                         (lp + clp) / max(1, len(suf) + 1))
                elif cid > 2:
                    cand.append((lp + clp, suf + i2c[cid]))
        if not cand:
            break
        cand.sort(key=lambda x: -x[0])
        beams = cand[:beam]
    return sorted(finished.items(), key=lambda x: -x[1])[:k]


# ---- evaluation (identical targets for both models) ---------------------
def targets(items):
    rng = random.Random(SEED)
    pool = [(w, c) for w, c in items if len(w) >= 6]
    return rng.choices(pool, weights=[c for _, c in pool], k=N_TARGETS)


def eval_completer(fn, tgts):
    n = h1 = h5 = saved = tot = 0
    for w, _ in tgts:
        for plen in PREFIX_LENS:
            if len(w) <= plen:
                continue
            words = [x[0] for x in fn(w[:plen])]
            n += 1
            if words and words[0] == w:
                h1 += 1; saved += len(w) - plen
            if w in words:
                h5 += 1
            tot += len(w) - plen
    return 100*h1/n, 100*h5/n, 100*saved/tot


def main():
    items = load_vocab()
    c2i = build_charset(items)
    i2c = {i: c for c, i in c2i.items()}
    tgts = targets(items)

    print("training lightweight char-LSTM (same word forms) ...")
    model = train(items, c2i)

    print("building char n-gram (order 6) ...")
    ngram = build_ngram(items)

    print("\nevaluating both on identical held-out targets ...")
    lstm_top1, lstm_top5, lstm_ks = eval_completer(
        lambda p: lstm_complete(model, p, c2i, i2c), tgts)
    ng_top1, ng_top5, ng_ks = eval_completer(
        lambda p: ngram.complete(p, k=K), tgts)

    print(f"\n{'model':<16}{'top1':>8}{'top5':>8}{'ks-save':>9}")
    print(f"{'char n-gram':<16}{ng_top1:7.1f}%{ng_top5:7.1f}%{ng_ks:8.1f}%")
    print(f"{'char LSTM':<16}{lstm_top1:7.1f}%{lstm_top5:7.1f}%{lstm_ks:8.1f}%")

    # size / speed
    params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    for w, _ in tgts[:300]:
        lstm_complete(model, w[:4], c2i, i2c)
    lstm_ms = (time.time()-t0)/300*1000
    t0 = time.time()
    for w, _ in tgts[:300]:
        ngram.complete(w[:4], k=K)
    ng_ms = (time.time()-t0)/300*1000
    print(f"\nLSTM params: {params/1e6:.2f}M   latency/complete: "
          f"LSTM {lstm_ms:.1f} ms vs n-gram {ng_ms:.2f} ms")


if __name__ == "__main__":
    main()
