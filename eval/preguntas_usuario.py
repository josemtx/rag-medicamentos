"""Prueba cualitativa: preguntas de usuario real contra el pipeline completo.

Las métricas de retrieval (recall@k, MRR) miden si aparece el chunk que originó la
pregunta, y son ciegas a lo que arruina el producto: idioma inconsistente, datos
inventados o una prohibición afirmada que las fuentes no dicen. Esos tres fallos
salieron de aquí, no de las métricas. La última pregunta está fuera del alcance del
corpus a propósito: el sistema debe admitir que no lo sabe.

Uso:
    python eval/preguntas_usuario.py
"""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))
from hybrid_search import HybridIndex
from generate import MODEL, OLLAMA_URL, SYSTEM_PROMPT, build_prompt

PREGUNTAS = [
    "Cual es la dosis maxima diaria de ibuprofeno para un adulto?",
    "Puedo tomar ibuprofeno si estoy embarazada?",
    "Se puede tomar ibuprofeno y paracetamol a la vez?",
    "Que hago si me tomo dos comprimidos de tramadol por error?",
    "El ibuprofeno sirve para tratar la depresion?",  # fuera de alcance: debe declinar
]


def main():
    index = HybridIndex()
    for i, pregunta in enumerate(PREGUNTAS, 1):
        chunks = index.search(pregunta, k=8)
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(pregunta, chunks)},
            ],
        }, timeout=900)
        response.raise_for_status()

        print(f"\n{'=' * 88}\nPREGUNTA {i}: {pregunta}\n{'-' * 88}")
        print(response.json()["message"]["content"])
        print(f"{'-' * 88}\nCONTEXTO:")
        for c in chunks:
            print(f"  - {c['medicamento'][:45]:<45} | {c['titulo'][:38]}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
