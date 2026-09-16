"""Chunking estructurado de fichas técnicas y prospectos descargados por src/ingest.

Cada documento CIMA ya viene segmentado por secciones (seccion/titulo/contenido).
Usamos esa segmentación como unidad de chunk natural: una sección = un chunk.
Si una sección es más larga que MAX_CHARS, se subdivide por párrafos (y por
frases si un párrafo sigue siendo demasiado largo), preservando la sección y
el título como metadata en cada sub-chunk.

Salida: data/chunks/chunks.jsonl (un chunk JSON por línea).

Uso:
    python src/chunking/chunk_documents.py
"""
import json
import logging
import re
import string
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
MAX_CHARS = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chunking")


class _TextExtractor(HTMLParser):
    """Extrae texto plano de HTML simple (divs/p/span), colapsando espacios."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data):
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", raw)).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Divide por párrafos; si un párrafo excede max_chars, por frases."""
    parts = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            parts.append(para)
        else:
            parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())

    chunks, current = [], ""
    for part in parts:
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document(nregistro: str, doc_type: str, sections: list[dict], med_nombre: str) -> list[dict]:
    chunks = []
    for section in sections:
        text = html_to_text(section.get("contenido", ""))
        if len(text.strip(string.punctuation + string.whitespace)) < 3:
            continue
        pieces = [text] if len(text) <= MAX_CHARS else split_long_text(text, MAX_CHARS)
        for i, piece in enumerate(pieces):
            chunks.append({
                "nregistro": nregistro,
                "medicamento": med_nombre,
                "doc_type": doc_type,
                "seccion": section.get("seccion"),
                "titulo": section.get("titulo"),
                "part": i,
                "text": piece,
            })
    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    medicamentos = json.loads((RAW_DIR / "medicamentos.json").read_text(encoding="utf-8"))

    all_chunks = []
    for med in medicamentos:
        nregistro = med["nregistro"]
        for doc_type, prefix in (("ficha_tecnica", "ft"), ("prospecto", "prospecto")):
            path = RAW_DIR / f"{prefix}_{nregistro}.json"
            if not path.exists():
                continue
            sections = json.loads(path.read_text(encoding="utf-8"))
            all_chunks.extend(chunk_document(nregistro, doc_type, sections, med.get("nombre", "")))

    out_path = CHUNKS_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            chunk["chunk_id"] = f"{chunk['nregistro']}_{chunk['doc_type']}_{chunk['seccion']}_{chunk['part']}"
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info("Medicamentos: %d", len(medicamentos))
    log.info("Chunks generados: %d -> %s", len(all_chunks), out_path)


if __name__ == "__main__":
    main()
