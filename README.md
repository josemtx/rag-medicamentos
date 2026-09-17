# RAG Medicamentos

Asistente RAG sobre fichas técnicas y prospectos de medicamentos españoles, con la
API pública de CIMA (AEMPS). El objetivo no era un demo que responda bonito, sino
medir si cada pieza del pipeline aporta algo: chunking estructurado, retrieval
híbrido, reranking y un banco de evaluación con intervalos de confianza.

Buena parte de lo interesante del proyecto está en lo que la evaluación **refutó**.

> **Aviso**: proyecto educativo. No es una fuente de información médica: las respuestas
> que genera contienen errores documentados más abajo. Para información sobre
> medicamentos, consulta [CIMA](https://cima.aemps.es) o a un profesional sanitario.

## Datos

[CIMA (AEMPS)](https://cima.aemps.es/cima/publico/home.html) — API REST pública, sin
autenticación, JSON/UTF-8.

| Endpoint | Devuelve |
|---|---|
| `GET /medicamentos?atc={codigo}` | listado por código ATC (paginado, 200/página) |
| `GET /medicamento?nregistro={n}` | metadatos: principios activos, laboratorio, vía |
| `GET /docSegmentado/contenido/1?nregistro={n}` | ficha técnica por secciones |
| `GET /docSegmentado/contenido/2?nregistro={n}` | prospecto por secciones |

Corpus: 80 medicamentos comercializados con ATC N02 (analgésicos) y M01A (AINEs),
deduplicados → **4.822 chunks**. La ingesta completó los 80 sin errores.

## Pipeline

```
CIMA API ──> ingesta ──> chunking ──> embeddings ──> retrieval ──> generación
             80 meds     4.822        E5 multiling.  BM25 + coseno  LLM local
                         por sección  4822×768       fusión α/RRF   (Ollama)
```

**Chunking**: CIMA ya sirve los documentos segmentados por secciones, así que la
sección es la unidad natural de chunk; las que exceden 2.000 caracteres se parten por
párrafos. Cada chunk arrastra sus principios activos con dosis, por un motivo que
explican los hallazgos.

**Retrieval**: BM25 (léxico) y coseno sobre `multilingual-e5-base` (semántico),
fusionados por min-max normalizado con peso α, o por Reciprocal Rank Fusion.

**Evaluación**: 50 preguntas sintéticas, cada una generada a partir de un chunk
concreto que actúa como ground truth, y forzadas a nombrar el medicamento para que ese
ground truth sea identificable entre marcas casi idénticas.

## Resultados

| Configuración | R@1 | R@5 | R@10 | MRR | IC 95% MRR |
|---|---|---|---|---|---|
| BM25 solo | 0.100 | 0.500 | 0.580 | 0.246 | [0.166, 0.332] |
| Vectorial solo | 0.060 | 0.340 | 0.460 | 0.183 | [0.112, 0.260] |
| Híbrido α=0.5 | 0.120 | 0.520 | 0.600 | 0.262 | [0.178, 0.349] |
| Híbrido RRF | 0.080 | 0.480 | 0.600 | 0.234 | [0.159, 0.308] |
| Híbrido + rerank | 0.140 | 0.460 | 0.640 | 0.294 | [0.200, 0.389] |

## Hallazgos

**1. Con n=50 ninguna configuración se distingue de un BM25 pelado.**
Comparando por bootstrap pareado (cada configuración ve las mismas preguntas, así que
la dificultad de cada pregunta se cancela), el intervalo de la diferencia frente a BM25
incluye el cero en todos los casos: híbrido +0.016 [−0.068, +0.098], híbrido+rerank
+0.048 [−0.067, +0.157]. Detectar un efecto de ese tamaño exigiría 250-300 preguntas.
Sin esos intervalos, las medias invitaban a concluir que el híbrido y el reranking
funcionaban.

**2. El reranker degradaba el sistema, no solo era inútil.**
Ante *"¿se puede tomar ibuprofeno y paracetamol a la vez?"*, el cross-encoder subió la
sección "Interacción con otros medicamentos" del puesto 10 al 1, y hundió del 7 al 9 la
sección "Qué es Antidol Dual y para qué se utiliza" — que contiene la respuesta
(*"ambos principios activos trabajan juntos para reducir el dolor"*). El retrieval
híbrido **sí** había encontrado el fragmento correcto; el reranker lo expulsó del
contexto. Es coherente con su entrenamiento: `mmarco-mMiniLMv2` puntúa afinidad
temática, y las secciones de advertencias repiten los términos de la pregunta. Está
fuera del camino de generación y se conserva solo como configuración a comparar.

**3. Quitar el truncamiento de embeddings empeoró las métricas.**
El modelo inicial (`paraphrase-multilingual-mpnet`) truncaba a 128 tokens y el 72% de
los chunks lo excedía. Ampliar la ventana a 512 hundió el MRR vectorial de 0.130 a
0.054: el modelo está entrenado a 128 y las secuencias largas le quedan fuera de
distribución. La solución fue cambiar a un modelo pensado para retrieval asimétrico
(E5, con prefijos `query:`/`passage:`), que subió el vectorial a 0.183.

**4. Las fichas técnicas son autorreferenciales y el chunking rompe el referente.**
Dicen *"este medicamento no se debe tomar con otros que contengan ibuprofeno"* sin decir
qué contiene. Aislado, ese fragmento parece prohibir la combinación — cuando el propio
medicamento ya la lleva. Por eso cada chunk incluye ahora `PARACETAMOL 500 mg,
IBUPROFENO 200 mg`. No bastó por sí solo: hicieron falta las tres cosas (composición en
el contexto, quitar el reranker y un modelo de 7B) para que la respuesta fuera correcta.

**5. Las métricas de retrieval son ciegas a los fallos que importan.**
Cinco preguntas de usuario real destaparon lo que ningún MRR ve: deriva del español al
portugués a mitad de respuesta, contenido inventado que no está en las fuentes, y la
respuesta incorrecta del punto 2. El sistema sí rechaza correctamente lo que está fuera
de alcance (*"¿el ibuprofeno sirve para la depresión?"*).

**6. RAG local en CPU no es viable para uso interactivo.**
Medido con `prompt_eval_duration` / `eval_duration` de Ollama, sin caché:

| Modelo | Contexto | Leer contexto | Generar | Total |
|---|---|---|---|---|
| Qwen2.5-7B | 4.680 tok | **339 s** (14 tok/s) | 96 tok / 32 s | **371 s** |
| Qwen2.5-7B | 2.507 tok | 163 s | 303 tok / 98 s | 261 s |
| Qwen2.5-3B | 4.334 tok | 160 s (27 tok/s) | 90 tok / 16 s | 176 s |

Leer el contexto es el ~90% del tiempo. Bajar de 30 s exigiría menos de 800 tokens de
contexto, o sea menos de dos fragmentos: no queda RAG. El retrieval en cambio tarda ~1 s.
El cuello de botella es exclusivamente el LLM, que es la pieza más sustituible: cambiar
`generate.py` a una API daría respuestas en 2-3 s sin tocar nada del resto.

## Estado

| Fase | Estado |
|---|---|
| Ingesta, chunking, embeddings, retrieval, evaluación | funcionando |
| Generación | funciona, pero 3-6 min por pregunta en CPU |
| Reranking | implementado y **descartado** por medición (hallazgo 2) |
| Observabilidad, app | no empezadas |

### Problemas abiertos

- **Latencia**: bloqueante para una demo. Requiere GPU o una API.
- **Potencia estadística**: 50 preguntas no bastan; harían falta ~250.
- **Evaluación a nivel de respuesta**: falta medir fidelidad y corrección, que es donde
  están los fallos graves. Las métricas de retrieval no los capturan.
- **Granularidad del corpus**: es *por producto* y los usuarios preguntan *por principio
  activo*. "¿Dosis máxima de ibuprofeno?" no tiene respuesta única, y el sistema elige
  un producto en silencio y la presenta como general.
- **Retrieval de sección**: acierta el medicamento el 92% de las veces, pero la sección
  correcta solo el 62%.

## Uso

```bash
pip install -r requirements.txt

python src/ingest/fetch_cima.py --limit 80   # corpus      -> data/raw/
python src/chunking/chunk_documents.py       # chunks      -> data/chunks/chunks.jsonl
python src/embeddings/build_index.py         # índice      -> data/chunks/embeddings.npy

python src/retrieval/hybrid_search.py "dosis de ibuprofeno en niños"
python src/generation/generate.py "dosis de ibuprofeno en niños"

python src/evaluation/build_qa_dataset.py --n 50   # dataset -> eval/qa_dataset.jsonl
python src/evaluation/evaluate_retrieval.py       # métricas -> eval/retrieval_results.json
python eval/preguntas_usuario.py                  # prueba cualitativa
```

Tests (sin framework, `assert` puro): `python src/<fase>/test_*.py`.

### Modelo de generación

Requiere [Ollama](https://ollama.com). El modelo se importa desde un GGUF de
HuggingFace en lugar de `ollama pull`, porque el CDN de Ollama
(`r2.cloudflarestorage.com`) resultó inalcanzable desde la red donde se desarrolló esto:

```bash
curl -L -o ~/.ollama/imports/qwen2.5-7b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
ollama create qwen2.5:7b -f Modelfile.7b
```

`Modelfile` hace lo mismo con la variante de 3B (más rápida, pero falla la pregunta del
hallazgo 2). Ambos fijan `num_ctx 8192`: el contexto RAG no cabe en los 4096 por defecto.

## Estructura

```
rag-medicamentos/
├── src/
│   ├── ingest/          descarga de CIMA con reintentos y rate limit
│   ├── chunking/        segmentación por secciones + parser HTML (stdlib)
│   ├── embeddings/      índice vectorial E5
│   ├── retrieval/       BM25 + coseno, fusión min-max y RRF
│   ├── reranking/       cross-encoder (descartado, ver hallazgo 2)
│   ├── generation/      prompt con grounding + Ollama
│   └── evaluation/      dataset sintético y métricas con bootstrap
├── eval/                dataset, resultados y prueba cualitativa
└── data/                corpus descargado (regenerable, fuera de git)
```
