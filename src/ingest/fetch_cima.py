"""Descarga corpus de medicamentos (N02, M01A) desde la API pública de CIMA (AEMPS).

Guarda en data/raw/:
  - medicamentos.json          listado deduplicado usado como corpus
  - meta_{nregistro}.json      metadatos por medicamento
  - ft_{nregistro}.json        ficha técnica segmentada
  - prospecto_{nregistro}.json prospecto segmentado

Uso:
    python src/ingest/fetch_cima.py [--limit 80]
"""
import argparse
import json
import logging
import time
from pathlib import Path

import requests

BASE_URL = "https://cima.aemps.es/cima/rest"
ATC_CODES = ["N02", "M01A"]
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
REQUEST_PAUSE_S = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


def request_json(url: str, params: dict | None = None):
    """GET con reintentos y backoff. Devuelve None si falla tras MAX_RETRIES."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            time.sleep(REQUEST_PAUSE_S)
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Fallo %s (intento %d/%d) %s: %s", url, attempt, MAX_RETRIES, params, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return None


def fetch_atc_listado(atc: str) -> list[dict]:
    """Pagina /medicamentos?atc= hasta cubrir totalFilas, filtrando comercializados."""
    resultados = []
    pagina = 1
    while True:
        data = request_json(f"{BASE_URL}/medicamentos", {"atc": atc, "pagina": pagina})
        if not data or not data.get("resultados"):
            break
        resultados.extend(r for r in data["resultados"] if r.get("comerc"))
        total = data.get("totalFilas", 0)
        tamanio = data.get("tamanioPagina", len(data["resultados"]))
        if pagina * tamanio >= total:
            break
        pagina += 1
    log.info("ATC %s: %d medicamentos comercializados", atc, len(resultados))
    return resultados


def build_corpus(limit: int) -> list[dict]:
    """Une N02 + M01A, deduplica por nregistro, recorta a `limit`."""
    por_nregistro: dict[str, dict] = {}
    for atc in ATC_CODES:
        for med in fetch_atc_listado(atc):
            por_nregistro.setdefault(med["nregistro"], med)
    corpus = list(por_nregistro.values())[:limit]
    log.info("Corpus tras dedupe y recorte a %d: %d medicamentos", limit, len(corpus))
    return corpus


def fetch_and_save(nregistro: str) -> dict:
    """Descarga metadatos, ficha técnica y prospecto de un medicamento. Devuelve resultado por doc."""
    result = {"nregistro": nregistro, "meta": False, "ft": False, "prospecto": False}

    meta = request_json(f"{BASE_URL}/medicamento", {"nregistro": nregistro})
    if meta:
        (RAW_DIR / f"meta_{nregistro}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["meta"] = True

    ft = request_json(f"{BASE_URL}/docSegmentado/contenido/1", {"nregistro": nregistro})
    if ft:
        (RAW_DIR / f"ft_{nregistro}.json").write_text(
            json.dumps(ft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["ft"] = True

    prospecto = request_json(f"{BASE_URL}/docSegmentado/contenido/2", {"nregistro": nregistro})
    if prospecto:
        (RAW_DIR / f"prospecto_{nregistro}.json").write_text(
            json.dumps(prospecto, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["prospecto"] = True

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=80, help="Nº máximo de medicamentos en el corpus")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(args.limit)
    (RAW_DIR / "medicamentos.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {"total": len(corpus), "ok": [], "sin_ft": [], "sin_prospecto": [], "sin_meta": []}
    for i, med in enumerate(corpus, 1):
        nregistro = med["nregistro"]
        log.info("[%d/%d] %s (%s)", i, len(corpus), nregistro, med.get("nombre", ""))
        result = fetch_and_save(nregistro)
        if not result["meta"]:
            summary["sin_meta"].append(nregistro)
        if not result["ft"]:
            summary["sin_ft"].append(nregistro)
        if not result["prospecto"]:
            summary["sin_prospecto"].append(nregistro)
        if result["meta"] and result["ft"] and result["prospecto"]:
            summary["ok"].append(nregistro)

    (RAW_DIR / "ingest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("=== Resumen ===")
    log.info("Procesados: %d", summary["total"])
    log.info("Completos (meta+ft+prospecto): %d", len(summary["ok"]))
    log.info("Sin metadatos: %d %s", len(summary["sin_meta"]), summary["sin_meta"])
    log.info("Sin ficha técnica: %d %s", len(summary["sin_ft"]), summary["sin_ft"])
    log.info("Sin prospecto: %d %s", len(summary["sin_prospecto"]), summary["sin_prospecto"])


if __name__ == "__main__":
    main()
