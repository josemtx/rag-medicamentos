# RAG Medicamentos

Asistente RAG sobre fichas técnicas y prospectos de medicamentos, usando la
API pública de CIMA (AEMPS). Proyecto de portfolio enfocado en demostrar
buenas prácticas de ingeniería RAG (chunking estructurado, retrieval híbrido,
reranking, evaluación con métricas), no solo un demo funcional.

## Fuente de datos

[CIMA (AEMPS)](https://cima.aemps.es/cima/publico/home.html) — API REST
pública, sin autenticación, JSON/UTF-8.

- `GET /medicamentos?atc={codigo}` — listado de medicamentos por código ATC
- `GET /medicamento?nregistro={n}` — metadatos (principio activo, laboratorio, vía, estado)
- `GET /docSegmentado/contenido/1?nregistro={n}` — ficha técnica segmentada por secciones
- `GET /docSegmentado/contenido/2?nregistro={n}` — prospecto segmentado por secciones

Corpus: medicamentos comercializados con ATC N02 (analgésicos) y M01A
(antiinflamatorios no esteroideos), deduplicados, acotado a ~80 medicamentos.

> **Aviso**: proyecto educativo para demostrar ingeniería RAG. No es una fuente de
> información médica. Las respuestas que genera contienen errores demostrados (ver
> *Evaluación y hallazgos*). Para información sobre medicamentos, consulta
> [CIMA](https://cima.aemps.es) directamente o a un profesional sanitario.

## Estado

| Fase | Estado | Detalle |
|---|---|---|
| Ingesta (`src/ingest/`) | ✅ | 80 medicamentos, 0 errores |
| Chunking (`src/chunking/`) | ✅ | 4.822 chunks segmentados por sección |
| Embeddings (`src/embeddings/`) | ✅ | `multilingual-e5-base`, 4822×768 |
| Retrieval (`src/retrieval/`) | ✅ | BM25 + coseno, fusión min-max y RRF |
| Reranking (`src/reranking/`) | ✅ | cross-encoder multilingüe sobre top-20 |
| Generación (`src/generation/`) | ✅ | LLM local vía Ollama, con citación de fuentes |
| Evaluación (`src/evaluation/`) | ✅ | 50 preguntas sintéticas, recall@k y MRR con IC bootstrap |
| Observabilidad (`src/observability/`) | ⬜ | pendiente |
| App (`app/`) | ⬜ | pendiente |

## Estructura

```
rag-medicamentos/
├── data/{raw,processed,chunks}/
├── src/{ingest,chunking,embeddings,retrieval,reranking,generation,evaluation,observability}/
├── eval/
├── app/
├── notebooks/
├── README.md
└── requirements.txt
```

## Uso

```bash
pip install -r requirements.txt
python src/ingest/fetch_cima.py --limit 80      # corpus -> data/raw/
python src/chunking/chunk_documents.py          # chunks -> data/chunks/chunks.jsonl
python src/embeddings/build_index.py            # indice vectorial -> data/chunks/embeddings.npy
python src/retrieval/hybrid_search.py "dosis de ibuprofeno en niños"
python src/reranking/rerank.py "dosis de ibuprofeno en niños"
python src/generation/generate.py "dosis de ibuprofeno en niños"
```

## Evaluación y hallazgos

50 preguntas sintéticas, cada una generada a partir de un chunk conocido que hace
de ground truth (`eval/qa_dataset.jsonl`). Métricas en `eval/retrieval_results.json`.

| Configuración | R@1 | R@5 | R@10 | MRR | IC 95% MRR |
|---|---|---|---|---|---|
| BM25 solo | 0.100 | 0.500 | 0.580 | 0.246 | [0.166, 0.332] |
| Vectorial solo | 0.060 | 0.340 | 0.460 | 0.183 | [0.112, 0.260] |
| Híbrido α=0.5 | 0.120 | 0.520 | 0.600 | 0.262 | [0.178, 0.349] |
| Híbrido RRF | 0.080 | 0.480 | 0.600 | 0.234 | [0.159, 0.308] |
| Híbrido + rerank | 0.140 | 0.460 | 0.640 | 0.294 | [0.200, 0.389] |

Lo que enseñan estos números, más que los números en sí:

1. **Ninguna diferencia es estadísticamente significativa.** Con un test pareado por
   pregunta (el más potente, ya que todas las configuraciones ven las mismas preguntas),
   el intervalo de la diferencia frente a BM25 incluye el cero en todos los casos. Con
   n=50 no se puede afirmar que el vectorial, la fusión híbrida ni el reranking aporten
   nada sobre un BM25 pelado en este corpus. Detectar el efecto del reranking (+0.048)
   exigiría unas 250-300 preguntas.

2. **Dos hipótesis razonables resultaron falsas al medirlas.** Que el modelo de
   embeddings truncaba a 128 tokens era cierto (el 72% de los chunks lo excedía), pero
   ampliar la ventana a 512 *empeoró* las métricas: el modelo estaba entrenado a 128 y
   las secuencias largas le quedan fuera de distribución. Y sustituir la fusión min-max
   por Reciprocal Rank Fusion, que es inmune a la escala de los scores, tampoco mejoró.

3. **Las métricas de retrieval son ciegas a los fallos que importan.** Una prueba
   cualitativa con 5 preguntas de usuario reveló problemas que ningún MRR captura:
   el modelo deriva al portugués a mitad de respuesta, inventa contenido que no está en
   las fuentes, y ante "¿se puede tomar ibuprofeno y paracetamol a la vez?" respondió que
   no — citando como fuente ANTIDOL DUAL, que es justamente un comprimido que combina
   ambos. La evaluación a nivel de respuesta (fidelidad y corrección) es la pieza que falta.

### Limitaciones conocidas

- El corpus es **por producto** (fichas técnicas) pero los usuarios preguntan **por
  principio activo**: "¿dosis máxima de ibuprofeno?" no tiene respuesta única, y el
  sistema elige un producto en silencio y lo presenta como general.
- El retrieval acierta el medicamento el 92% de las veces pero la sección solo el 62%.
- El modelo de generación (3B, CPU) se queda corto en síntesis y consistencia idiomática.

## Uso rápido

### Modelo de generación (Ollama)

La generación usa un LLM local vía [Ollama](https://ollama.com). El modelo se
importa desde un GGUF de HuggingFace en vez de `ollama pull`:

```bash
curl -L -o ~/.ollama/imports/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
ollama create qwen2.5:3b -f Modelfile
```
