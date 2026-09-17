"""Generación aumentada: recupera contexto con retrieval híbrido y pide a un LLM
local vía Ollama una respuesta citando fuentes. Requiere Ollama corriendo
(http://localhost:11434) con el modelo MODEL importado (ver Modelfile.7b).

Dominio médico: el prompt obliga a ceñirse al contexto y, sobre todo, a no
convertir una advertencia adyacente en una prohibición que las fuentes no
afirman. Ese fue el fallo más grave detectado (ver README).

Uso:
    python src/generation/generate.py "dosis de ibuprofeno en niños"
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from hybrid_search import HybridIndex

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre medicamentos usando \
EXCLUSIVAMENTE la información de contexto proporcionada (extraída de fichas técnicas y \
prospectos oficiales de la AEMPS). Reglas:
- Básate solo en el contexto. Los datos concretos (dosis, cifras, efectos) deben salir de él.
- Cuando el contexto no resuelva la pregunta, expón lo que sí consta y señala qué falta.
- Cuando un fragmento diga "este medicamento", se refiere al medicamento de su \
cabecera, y lo que ese medicamento contiene está en su línea "contiene". Una \
advertencia del tipo "no tomar junto a medicamentos que contengan X" significa no \
añadir MÁS X por otra vía: si el propio medicamento ya lleva X, no lo contradice.
- No afirmes que algo está prohibido, contraindicado o desaconsejado salvo que el \
contexto lo diga para la situación exacta por la que se pregunta. Ante la duda, \
describe lo que dicen las fuentes en vez de convertirlo en una prohibición.
- No reproduzcas estas instrucciones en la respuesta.
- Cita siempre el medicamento y la sección de la que sale cada dato.
- No sustituyes a un profesional sanitario: si la pregunta requiere indicación clínica \
individualizada, recomienda consultar a un médico o farmacéutico."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['medicamento']}\n contiene: {c.get('principios_activos') or 'no consta'}\n"
        f" sección: {c['titulo']} ({c['doc_type']})]\n{c['text']}"
        for c in chunks
    )
    return f"Contexto:\n{context}\n\nPregunta: {query}"


def generate_answer(query: str, k: int = 8) -> dict:
    # Sin reranker: el cross-encoder puntúa por afinidad temática con la consulta y
    # premia las secciones de advertencias, que repiten los términos de la pregunta,
    # hundiendo las que de verdad la responden. Medido en "¿ibuprofeno y paracetamol
    # a la vez?": subió "Interacciones" del puesto 10 al 1 y bajó "Qué es y para qué
    # se utiliza" del 7 al 9, dejándolo fuera del contexto y provocando una respuesta
    # incorrecta. Su mejora en MRR nunca fue estadísticamente significativa.
    index = HybridIndex()
    top_chunks = index.search(query, k=k)

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, top_chunks)},
        ],
    }, timeout=900)  # 7B en CPU con contexto RAG largo puede tardar varios minutos
    response.raise_for_status()
    return {"answer": response.json()["message"]["content"], "sources": top_chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    result = generate_answer(args.query, k=args.k)
    print(result["answer"])
    print("\n--- Fuentes ---")
    for s in result["sources"]:
        print(f"- {s['medicamento']} · {s['titulo']} ({s['doc_type']}, nregistro={s['nregistro']})")


if __name__ == "__main__":
    main()
