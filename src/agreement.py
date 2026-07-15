"""Factored grammatical-agreement model (number / gender).

Surface n-grams miss agreement because endings are syncretic and they don't know
*which* word governs the form. This model works over grammemes from `morph.py`
and targets the two systematic errors:

  * subject → verb NUMBER:   ежик[и] (plur, nom) → сто[ят]  (not сто[ит])
  * noun ↔ adjective/pronoun GENDER·NUMBER:
        женщина → котор[ую] (femn) ;  красив[ые] → девушк[и] (plur)

For a candidate we pick its *governor* by a light heuristic (nearest preceding
noun-in-nominative for a verb; nearest preceding noun for an adjective/pronoun,
and vice-versa) and score P(candidate grammemes | governor grammemes) from tables
learned on real text. Tiny (a handful of grammeme values), O(1) lookups, no parser.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .morph import Feat, gn_key, is_adj, is_noun

BACKOFF_ALPHA = 0.5   # Laplace smoothing for P(cand_key | gov_key)
GOV_WINDOW = 4        # how many preceding tokens to search for the governor


class AgreementModel:
    def __init__(self, gov_window: int = GOV_WINDOW):
        self.gov_window = gov_window
        # P(verb number | subject number)
        self.vnum: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # P(adjective/pronoun gender·number | noun gender·number)
        self.adj_gn: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # P(noun gender·number | preceding adjective gender·number)
        self.noun_gn: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ---- governor heuristic (shared train/score) ------------------------
    def _nearest(self, left_feats, pred):
        """Nearest preceding feat (list is NEAREST-FIRST) matching pred, in window."""
        for ft in left_feats[: self.gov_window]:
            if ft is not None and pred(ft):
                return ft
        return None

    @staticmethod
    def _is_subject(ft: Feat) -> bool:
        return is_noun(ft) and ft.case in ("nomn", "")   # nominative (or unmarked)

    # ---- training -------------------------------------------------------
    def observe(self, left_feats, cand: Feat) -> None:
        """Accumulate one (governor, candidate) agreement observation.
        `left_feats` = grammemes of preceding tokens, NEAREST FIRST."""
        if cand.pos == "VERB" and cand.number:
            subj = self._nearest(left_feats, self._is_subject)
            if subj and subj.number:
                self.vnum[subj.number][cand.number] += 1
        elif is_adj(cand):
            noun = self._nearest(left_feats, is_noun)
            if noun:
                self.adj_gn[gn_key(noun)][gn_key(cand)] += 1
        elif is_noun(cand):
            adj = self._nearest(left_feats, is_adj)
            if adj:
                self.noun_gn[gn_key(adj)][gn_key(cand)] += 1

    @classmethod
    def from_sentences(cls, tagged_sentences, gov_window: int = GOV_WINDOW,
                       min_count: int = 3) -> "AgreementModel":
        """tagged_sentences yields lists of Feat|None (one per token, in order)."""
        m = cls(gov_window=gov_window)
        for feats in tagged_sentences:
            for i in range(1, len(feats)):
                cand = feats[i]
                if cand is None:
                    continue
                left = feats[:i][::-1]           # nearest first
                m.observe(left, cand)
        m.prune(min_count)
        return m

    def prune(self, min_count: int) -> None:
        for tbl in (self.vnum, self.adj_gn, self.noun_gn):
            for k in list(tbl.keys()):
                kept = {v: c for v, c in tbl[k].items() if c >= min_count}
                if kept:
                    tbl[k] = defaultdict(int, kept)
                else:
                    del tbl[k]

    # ---- scoring --------------------------------------------------------
    @staticmethod
    def _logp(tbl: dict[str, dict[str, int]], gov: str, cand: str) -> float | None:
        row = tbl.get(gov)
        if not row:
            return None
        total = sum(row.values())
        v = len(row)
        return math.log((row.get(cand, 0) + BACKOFF_ALPHA) / (total + BACKOFF_ALPHA * v))

    def governors(self, left_feats):
        """Resolve the three possible governors ONCE per completion (they are the
        same for every candidate): subject number, nearest-noun gn, nearest-adj gn.
        Returns (subj_number|None, noun_gn|None, adj_gn|None)."""
        subj = self._nearest(left_feats, self._is_subject)
        noun = self._nearest(left_feats, is_noun)
        adj = self._nearest(left_feats, is_adj)
        return (subj.number if subj and subj.number else None,
                gn_key(noun) if noun else None,
                gn_key(adj) if adj else None)

    def score_cand(self, govs, cand: Feat | None) -> float | None:
        """log P(cand grammemes | governor grammemes) using precomputed governors."""
        if cand is None:
            return None
        subj_num, noun_gn, adj_gn = govs
        if cand.pos == "VERB" and cand.number and subj_num:
            return self._logp(self.vnum, subj_num, cand.number)
        if is_adj(cand) and noun_gn:
            return self._logp(self.adj_gn, noun_gn, gn_key(cand))
        if is_noun(cand) and adj_gn:
            return self._logp(self.noun_gn, adj_gn, gn_key(cand))
        return None

    def score(self, left_feats, cand: Feat | None) -> float | None:
        """log P(cand grammemes | governor grammemes), or None if not applicable.
        `left_feats` = preceding tokens' grammemes, NEAREST FIRST. Convenience
        wrapper (resolves governors each call); hot paths use governors()+score_cand()."""
        return self.score_cand(self.governors(left_feats), cand)

    # ---- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "gov_window": self.gov_window,
            "vnum": {k: dict(v) for k, v in self.vnum.items()},
            "adj_gn": {k: dict(v) for k, v in self.adj_gn.items()},
            "noun_gn": {k: dict(v) for k, v in self.noun_gn.items()},
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AgreementModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(gov_window=d.get("gov_window", GOV_WINDOW))
        for name in ("vnum", "adj_gn", "noun_gn"):
            tbl = getattr(m, name)
            for k, row in d.get(name, {}).items():
                tbl[k] = defaultdict(int, row)
        return m
