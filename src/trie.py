"""Prefix index over known word forms with cached top-k per node.

Given a prefix it returns the most frequent real word forms that start with it,
in O(len(prefix) + k). Built once from the vocabulary, serialised as JSON.

Memory is kept modest by caching only the top `cap` completions at each node.
"""
from __future__ import annotations

import json
from pathlib import Path


class FreqTrie:
    __slots__ = ("children", "top")

    def __init__(self):
        self.children: dict[str, "FreqTrie"] = {}
        self.top: list[tuple[str, int]] = []  # (word, count), desc, len<=cap

    @classmethod
    def build(cls, items: list[tuple[str, int]], cap: int = 12) -> "FreqTrie":
        root = cls()
        # insert most-frequent first so each node's `top` fills with the best
        for word, count in sorted(items, key=lambda kv: -kv[1]):
            node = root
            for ch in word:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = node.children[ch] = cls()
                node = nxt
                if len(node.top) < cap:
                    node.top.append((word, count))
            # also cache at the terminal node itself (handled above)
        return root

    def query(self, prefix: str) -> list[tuple[str, int]]:
        node = self
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return []
        return node.top

    # ---- persistence (compact nested arrays) ---------------------------
    def _pack(self):
        return [
            [[w, c] for w, c in self.top],
            {ch: child._pack() for ch, child in self.children.items()},
        ]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self._pack(), ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def _unpack(cls, data) -> "FreqTrie":
        node = cls()
        node.top = [(w, c) for w, c in data[0]]
        node.children = {ch: cls._unpack(ch_data) for ch, ch_data in data[1].items()}
        return node

    @classmethod
    def load(cls, path: Path) -> "FreqTrie":
        return cls._unpack(json.loads(Path(path).read_text(encoding="utf-8")))
