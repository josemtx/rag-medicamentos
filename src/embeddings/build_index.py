"""Genera embeddings de los chunks y los guarda como índice vectorial en disco.

Modelo local (sentence-transformers), sin API key. Para el tamaño de este
corpus (unos miles de chunks) una matriz numpy + similitud coseno es
suficiente: no hace falta FAISS ni una base vectorial dedicada.

Salida en data/chunks/:
  - embeddings.npy   matriz float32 (n_chunks, dim)
  - chunk_ids.json   lista de chunk_id en el mismo orden que embeddings.npy

Uso:
    python src/embeddings/build_index.py
"""
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "chunks"
# E5 está entrenado para retrieval asimétrico (pregunta -> pasaje) y a 512 tokens.
# El anterior (paraphrase-mpnet) medía similitud entre frases y truncaba a 128:
# ver eval/retrieval_results_mpnet512.json para la comparativa.
MODEL_NAME = "intfloat/multilingual-e5-base"
# E5 exige estos prefijos; sin ellos, o cruzados, el retrieval se degrada sin dar error.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embeddings")


def load_chunks() -> list[dict]:
    path = CHUNKS_DIR / "chunks.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    log.info("Chunks cargados: %d", len(chunks))

    model = SentenceTransformer(MODEL_NAME)
    texts = [f"{PASSAGE_PREFIX}{c['titulo']}: {c['text']}" for c in chunks]

    log.info("Generando embeddings con %s ...", MODEL_NAME)
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    np.save(CHUNKS_DIR / "embeddings.npy", embeddings)
    (CHUNKS_DIR / "chunk_ids.json").write_text(
        json.dumps([c["chunk_id"] for c in chunks], ensure_ascii=False), encoding="utf-8"
    )

    log.info("Embeddings: %s -> %s", embeddings.shape, CHUNKS_DIR / "embeddings.npy")


if __name__ == "__main__":
    main()
