"""Genera el dataset de evaluación: preguntas sintéticas ancladas a un chunk conocido.

Cada pregunta se genera A PARTIR de un chunk concreto, así que ese chunk es el
ground truth para medir retrieval. El corpus tiene medicamentos casi idénticos
(varias marcas del mismo principio activo), por eso se obliga a que la pregunta
incluya el nombre del medicamento: si no, el chunk "correcto" sería ambiguo y
las métricas penalizarían aciertos que en realidad son válidos.

Salida: eval/qa_dataset.jsonl con {question, gold_chunk_id, nregistro, ...}

Uso:
    python src/evaluation/build_qa_dataset.py --n 50
"""
import argparse
import json
import logging
import random
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
OUT_PATH = ROOT / "eval" / "qa_dataset.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
MIN_CHUNK_CHARS = 400

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qa-dataset")

PROMPT = """A partir del siguiente fragmento de la ficha técnica o prospecto del medicamento \
"{medicamento}" (sección: {titulo}), escribe UNA pregunta que un paciente o profesional \
sanitario podría hacer y que se responda con este fragmento.

Reglas:
- La pregunta debe mencionar el nombre del medicamento "{medicamento}".
- Debe responderse únicamente con la información del fragmento.
- Responde SOLO con la pregunta, sin preámbulo ni comillas.

Fragmento:
{text}"""


def generate_question(chunk: dict) -> str | None:
    prompt = PROMPT.format(
        medicamento=chunk["medicamento"], titulo=chunk["titulo"], text=chunk["text"][:1500]
    )
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0.3},
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=180)
    response.raise_for_status()

    question = response.json()["message"]["content"].strip().strip('"').split("\n")[0].strip()
    if "?" not in question or len(question) <= 20:
        return None
    # El modelo ignora a veces la regla de nombrar el medicamento, y sin el nombre
    # la pregunta encaja en muchas marcas del mismo principio activo: ground truth inválido.
    marca = chunk["medicamento"].split()[0].lower()
    return question if marca in question.lower() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    # Un chunk por medicamento: evita que el dataset se concentre en los pocos
    # medicamentos con fichas largas.
    usable = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]
    random.Random(args.seed).shuffle(usable)
    por_medicamento: dict[str, dict] = {}
    for c in usable:
        por_medicamento.setdefault(c["nregistro"], c)
    candidatos = list(por_medicamento.values())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written, failed = 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, chunk in enumerate(candidatos, 1):
            if written >= args.n:
                break
            question = generate_question(chunk)
            if not question:
                failed += 1
                log.warning("[%d] descartada (no cumple formato o no nombra el medicamento)", i)
                continue
            out.write(json.dumps({
                "question": question,
                "gold_chunk_id": chunk["chunk_id"],
                "nregistro": chunk["nregistro"],
                "medicamento": chunk["medicamento"],
                "doc_type": chunk["doc_type"],
                "titulo": chunk["titulo"],
            }, ensure_ascii=False) + "\n")
            written += 1
            log.info("[%d/%d] %s", written, args.n, question[:90])

    log.info("Dataset: %d preguntas (%d descartadas) -> %s", written, failed, OUT_PATH)


if __name__ == "__main__":
    main()
