"""Compara configuraciones de retrieval sobre el dataset de evaluación.

Mide si el chunk que originó cada pregunta aparece entre los recuperados, y en
qué posición: recall@k (¿está?) y MRR (¿cómo de arriba?). Comparar BM25 solo,
vectorial solo, híbrido e híbrido+rerank es lo que justifica cada componente
del pipeline: sin esta tabla, el reranker es solo complejidad no demostrada.

Uso:
    python src/evaluation/evaluate_retrieval.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "reranking"))
from hybrid_search import HybridIndex
from rerank import CROSS_ENCODER_NAME, rerank

QA_PATH = ROOT / "eval" / "qa_dataset.jsonl"
RESULTS_PATH = ROOT / "eval" / "retrieval_results.json"
K = 10
RERANK_CANDIDATES = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


def gold_rank(results: list[dict], gold_chunk_id: str) -> int | None:
    """Posición (1-indexada) del chunk correcto, o None si no está."""
    for i, r in enumerate(results, 1):
        if r["chunk_id"] == gold_chunk_id:
            return i
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        f"recall@{K}": sum(1 for r in ranks if r and r <= K) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "mrr_ci95": mrr_ci95(ranks),
    }


def paired_diff_ci95(ranks_a, ranks_b, resamples: int = 2000, seed: int = 0) -> tuple:
    """Diferencia de MRR entre dos configuraciones, pareada por pregunta.

    Todas las configuraciones se evalúan sobre las mismas preguntas, así que la
    dificultad de cada pregunta afecta a ambas por igual y se cancela al restar.
    Comparar los IC independientes ignora eso y oculta diferencias reales; lo
    que decide es si el IC de la DIFERENCIA excluye el cero.
    """
    rr = lambda ranks: np.array([1 / r if r else 0.0 for r in ranks])
    diff = rr(ranks_a) - rr(ranks_b)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mrr_ci95(ranks: list[int | None], resamples: int = 2000, seed: int = 0) -> list[float]:
    """Intervalo de confianza del 95% para el MRR por bootstrap.

    Con 50 preguntas, dos configuraciones pueden diferir 0.05 de MRR por puro
    azar del muestreo. Sin el intervalo, es imposible saber si una diferencia
    justifica un cambio de arquitectura o es ruido.
    """
    rr = np.array([1 / r if r else 0.0 for r in ranks])
    rng = np.random.default_rng(seed)
    means = rr[rng.integers(0, len(rr), size=(resamples, len(rr)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    with QA_PATH.open(encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    log.info("Preguntas de evaluación: %d", len(dataset))

    index = HybridIndex()
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    configs = {
        "BM25 solo": lambda q: index.search(q, k=K, alpha=0.0),
        "Vectorial solo": lambda q: index.search(q, k=K, alpha=1.0),
        # Barrido de alpha: min-max depende de la distribución de scores, así que
        # el peso óptimo cambia con el modelo de embeddings.
        "Híbrido a=0.3": lambda q: index.search(q, k=K, alpha=0.3),
        "Híbrido a=0.5": lambda q: index.search(q, k=K, alpha=0.5),
        "Híbrido a=0.7": lambda q: index.search(q, k=K, alpha=0.7),
        "Híbrido RRF": lambda q: index.search_rrf(q, k=K),
        "Híbrido a=0.5 + rerank": lambda q: rerank(
            q, index.search(q, k=RERANK_CANDIDATES, alpha=0.5), cross_encoder, k=K
        ),
        "RRF + rerank": lambda q: rerank(
            q, index.search_rrf(q, k=RERANK_CANDIDATES), cross_encoder, k=K
        ),
    }

    results = {}
    for name, search_fn in configs.items():
        ranks = [gold_rank(search_fn(item["question"]), item["gold_chunk_id"]) for item in dataset]
        results[name] = {**metrics(ranks), "ranks": ranks}
        log.info("%-24s MRR=%.3f", name, results[name]["mrr"])

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    header = f"{'Configuración':<24}{'R@1':>8}{'R@5':>8}{f'R@{K}':>8}{'MRR':>8}{'IC95% MRR':>18}"
    print(f"\n{header}\n{'-' * len(header)}")
    for name, m in results.items():
        lo, hi = m["mrr_ci95"]
        print(f"{name:<24}{m['recall@1']:>8.3f}{m['recall@5']:>8.3f}{m[f'recall@{K}']:>8.3f}"
              f"{m['mrr']:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}")
    print(f"\nn={len(dataset)} preguntas. Diferencias menores que el ancho del IC no son concluyentes.")

    baseline = "BM25 solo"
    print(f"\nDiferencia de MRR frente a '{baseline}' (pareada por pregunta):")
    for name, m in results.items():
        if name == baseline:
            continue
        delta, lo, hi = paired_diff_ci95(m["ranks"], results[baseline]["ranks"])
        veredicto = "significativa" if lo > 0 or hi < 0 else "no concluyente"
        print(f"  {name:<24}{delta:+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]  {veredicto}")
    print(f"\nResultados -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
