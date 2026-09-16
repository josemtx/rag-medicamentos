"""Retrieval híbrido: fusiona BM25 (léxico) y similitud coseno (semántico).

Cada score se normaliza min-max sobre el corpus completo y se combina con un
peso alpha (0 = solo BM25, 1 = solo vectorial). BM25 sobre ~5k chunks es
suficientemente rápido para recalcular por query, no hace falta índice
invertido persistido aparte.

Uso:
    python src/retrieval/hybrid_search.py "dosis de ibuprofeno en niños" --k 5
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# El modelo y los prefijos se definen en build_index: si la query se codificara con
# otro modelo (o sin prefijo) que los chunks, el retrieval fallaría en silencio.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embeddings"))
from build_index import MODEL_NAME, QUERY_PREFIX

CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "chunks"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridIndex:
    def __init__(self):
        with (CHUNKS_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.embeddings = np.load(CHUNKS_DIR / "embeddings.npy")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[dict]:
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))
        query_vec = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        vector_scores = self.embeddings @ query_vec

        combined = alpha * _normalize(vector_scores) + (1 - alpha) * _normalize(bm25_scores)
        top_idx = np.argsort(-combined)[:k]

        return [{**self.chunks[i], "score": float(combined[i])} for i in top_idx]

    def search_rrf(self, query: str, k: int = 5, k_rrf: int = 60) -> list[dict]:
        """Fusiona por posición en el ranking, no por score.

        Min-max depende de la distribución de cada señal: E5 concentra los cosenos
        en un rango estrecho y BM25 no tiene techo, así que un mismo alpha pondera
        distinto según el modelo. RRF solo mira el orden, así que es inmune a eso.
        """
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))
        query_vec = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        vector_scores = self.embeddings @ query_vec

        combined = _rrf_scores(bm25_scores, k_rrf) + _rrf_scores(vector_scores, k_rrf)
        top_idx = np.argsort(-combined)[:k]
        return [{**self.chunks[i], "score": float(combined[i])} for i in top_idx]


def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def _rrf_scores(scores: np.ndarray, k_rrf: int) -> np.ndarray:
    """1/(k_rrf + posición), con posición 1 para el mejor score."""
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return 1.0 / (k_rrf + ranks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5, help="0=solo BM25, 1=solo vectorial")
    args = parser.parse_args()

    index = HybridIndex()
    for r in index.search(args.query, args.k, args.alpha):
        print(f"[{r['score']:.3f}] {r['medicamento']} - {r['titulo']} ({r['doc_type']})")
        print(f"  {r['text'][:200]}...")


if __name__ == "__main__":
    main()
