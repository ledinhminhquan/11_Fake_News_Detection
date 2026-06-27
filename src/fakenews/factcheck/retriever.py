"""Evidence retrieval for the agentic fact-check.

Retrieves the most relevant **evidence** snippets for a claim from the evidence
corpus. Uses BM25 always (dependency-free) and a dense bi-encoder
(`all-MiniLM-L6-v2`) when sentence-transformers is available, fused with RRF —
reusing the P08/P09 hybrid-retrieval pattern. Degrades to BM25-only offline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..config import RetrievalConfig
from ..logging_utils import get_logger
from ..models.bm25 import BM25Retriever

logger = get_logger(__name__)


def _rrf(rankings: List[List[Tuple[str, float]]], k: int = 60, top: int = 10) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _s) in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]


class _DenseEvidence:
    def __init__(self, model):
        self.model = model
        self.ids: List[str] = []
        self.emb = None

    def index(self, docs: Dict[str, str]):
        import numpy as np
        self.ids = list(docs)
        self.emb = np.asarray(self.model.encode([docs[i] for i in self.ids], normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=False), dtype="float32")
        return self

    def search(self, query: str, k: int):
        import numpy as np
        q = np.asarray(self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0], dtype="float32")
        sims = self.emb @ q
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]


class EvidenceRetriever:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.bm25 = BM25Retriever()
        self.dense = None

    @classmethod
    def from_corpus(cls, cfg: RetrievalConfig, corpus: Dict[str, Dict[str, str]]) -> "EvidenceRetriever":
        r = cls(cfg)
        r.build(corpus)
        return r

    def build(self, corpus: Dict[str, Dict[str, str]]) -> "EvidenceRetriever":
        self.corpus = corpus
        texts = {cid: c["text"] for cid, c in corpus.items()}
        self.bm25.index(texts)
        if self.cfg.use_dense:
            try:
                from sentence_transformers import SentenceTransformer  # lazy
                self.dense = _DenseEvidence(SentenceTransformer(self.cfg.embedder)).index(texts)
            except Exception as exc:
                logger.info("dense evidence index unavailable (%s); BM25 only.", exc)
                self.dense = None
        return self

    def retrieve(self, claim: str, k: Optional[int] = None) -> List[Dict]:
        k = k or self.cfg.top_k
        bm = self.bm25.search(claim, k * 2)
        runs = [bm]
        if self.dense is not None:
            try:
                runs.append(self.dense.search(claim, k * 2))
            except Exception:
                pass
        fused = _rrf(runs, self.cfg.rrf_k, top=k) if len(runs) > 1 else bm[:k]
        # normalize relevance to [0,1] for the coverage gate
        max_s = max((s for _, s in fused), default=1.0) or 1.0
        out = []
        for cid, s in fused:
            c = self.corpus.get(cid, {})
            out.append({"id": cid, "text": c.get("text", ""), "source": c.get("source", ""),
                        "relevance": round(float(s) / max_s, 4)})
        return out


__all__ = ["EvidenceRetriever"]
