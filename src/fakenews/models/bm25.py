"""Self-contained BM25 keyword retriever (no external dependency).

A pure-Python BM25-Okapi implementation used to retrieve the most relevant
**evidence** snippets for a claim during the agentic fact-check — and a strong,
dependency-free fallback when no dense embedder is available.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of for to in on and or is are be with by we this that as at "
            "from it its their our using use based via can which results show was were "
            "has have had not but they he she his her them you your".split())


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: List[str] = []
        self.docs: List[List[str]] = []
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl = 0.0

    def index(self, docs: Dict[str, str]) -> "BM25Retriever":
        self.ids = list(docs.keys())
        self.docs = [tokenize(docs[i]) for i in self.ids]
        self.df = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        self._tf = [Counter(d) for d in self.docs]
        self._len = [len(d) for d in self.docs]
        return self

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q = tokenize(query)
        scores: List[Tuple[str, float]] = []
        for idx, doc_id in enumerate(self.ids):
            tf = self._tf[idx]
            dl = self._len[idx]
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (freq * (self.k1 + 1)) / (denom or 1)
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


__all__ = ["BM25Retriever", "tokenize"]
