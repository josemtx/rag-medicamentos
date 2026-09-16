"""Generación aumentada: recupera contexto (híbrido + rerank) y pide a un LLM
local vía Ollama una respuesta citando fuentes. Requiere Ollama corriendo
(http://localhost:11434) con el modelo MODEL ya descargado (`ollama pull`).

Dominio médico: el prompt obliga a responder solo con el contexto dado y a
decir explícitamente que no hay información si el contexto no cubre la
pregunta, en vez de completar con conocimiento general del modelo.

Uso:
    python src/generation/generate.py "dosis de ibuprofeno en niños"
"""
import argparse
import sys
from pathlib import Path

import requests
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reranking"))
from hybrid_search import HybridIndex
from rerank import CROSS_ENCODER_NAME, rerank

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre medicamentos usando \
EXCLUSIVAMENTE la información de contexto proporcionada (extraída de fichas técnicas y \
prospectos oficiales de la AEMPS). Reglas:
- No uses conocimiento propio ni inventes datos que no estén en el contexto.
- Si el contexto no contiene la respuesta, dilo explícitamente: no la inventes.
- Cita siempre el medicamento y la sección de la que sale cada dato.
- No sustituyes a un profesional sanitario: si la pregunta requiere indicación clínica \
individualizada, recomienda consultar a un médico o farmacéutico."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['medicamento']} · {c['titulo']} ({c['doc_type']})]\n{c['text']}"
        for c in chunks
    )
    return f"Contexto:\n{context}\n\nPregunta: {query}"


def generate_answer(query: str, k: int = 5, candidates: int = 20) -> dict:
    index = HybridIndex()
    hybrid_results = index.search(query, k=candidates)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    top_chunks = rerank(query, hybrid_results, cross_encoder, k=k)

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, top_chunks)},
        ],
    }, timeout=120)
    response.raise_for_status()
    return {"answer": response.json()["message"]["content"], "sources": top_chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    result = generate_answer(args.query, k=args.k)
    print(result["answer"])
    print("\n--- Fuentes ---")
    for s in result["sources"]:
        print(f"- {s['medicamento']} · {s['titulo']} ({s['doc_type']}, nregistro={s['nregistro']})")


if __name__ == "__main__":
    main()
