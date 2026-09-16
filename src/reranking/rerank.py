"""Reranking de candidatos del retrieval híbrido con un cross-encoder.

El retrieval híbrido (BM25 + vectorial) es rápido pero puntúa query y chunk
por separado. Un cross-encoder lee (query, chunk) juntos y da un score de
relevancia más preciso, a costa de ser caro de calcular a escala: se aplica
solo sobre los top-N candidatos del retrieval, no sobre todo el corpus.

Uso:
    python src/reranking/rerank.py "dosis de ibuprofeno en niños" --k 5 --candidates 20
"""
import argparse
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from hybrid_search import HybridIndex

CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def rerank(query: str, candidates: list[dict], cross_encoder: CrossEncoder, k: int) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    reranked = [{**c, "score_rerank": float(s)} for c, s in zip(candidates, scores)]
    return sorted(reranked, key=lambda c: -c["score_rerank"])[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=20)
    args = parser.parse_args()

    index = HybridIndex()
    candidates = index.search(args.query, k=args.candidates)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    results = rerank(args.query, candidates, cross_encoder, args.k)

    for r in results:
        print(f"[rerank={r['score_rerank']:.3f} hybrid={r['score']:.3f}] {r['medicamento']} - {r['titulo']} ({r['doc_type']})")
        print(f"  {r['text'][:200]}...")


if __name__ == "__main__":
    main()
