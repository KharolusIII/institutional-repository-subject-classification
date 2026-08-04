# Auditoría del pipeline histórico 2025

## Alcance y criterio de evidencia

Esta auditoría se realizó antes de refactorizar el proyecto. Se inspeccionó recursivamente `Pipe 2025 OR/`, se leyeron todas las celdas (código, Markdown y salidas almacenadas) de las tres notebooks y las ocho páginas del PDF. La carpeta histórica no fue modificada.

El repositorio no contiene los CSV, TXT, JSONL, Parquet ni directorios de resultados a los que apunta el código. Por ello, esta auditoría distingue:

- **Código**: comportamiento observable en las celdas.
- **Salida almacenada**: evidencia de una ejecución anterior, aunque `execution_count` sea nulo en las notebooks exportadas.
- **Paper**: afirmaciones y resultados publicados en el PDF incluido.
- **No verificable**: información dependiente de datos o artefactos externos que no están versionados.

El nombre `2025` se conserva por continuidad con el proyecto y la carpeta. Las salidas de las notebooks y el PDF incluido están fechados o titulados como 2026; el paper cita como antecedente un trabajo BIREDIAL 2025 que no está incluido localmente.

## 1. Archivos encontrados

| Archivo | Contenido y función |
|---|---|
| `00. SEDICI  Clasificación por Materias Texto Completo Descompresion de Dataset GZ 2025.ipynb.ipynb` | Exportación recursiva del Assetstore montado en Google Drive a una carpeta plana de TXT. No descomprime GZ pese al nombre: lee bytes de cada archivo y los decodifica como UTF-8 o Latin-1. |
| `00.01. SEDICI  Clasificador de materias Texto Completo  MUESTREO Y CLASIFICACION 37 Etiquetas.ipynb` | Notebook principal. Contiene una implementación reciente con muestreo previo a la lectura de full text, combinaciones de features y resultados; después conserva dos generaciones de “código anterior”. |
| `00.01. SEDICI  Clasificador de materias Texto Completo  OPCION JSONL Y PARQUET.ipynb` | Variante para exportar TXT desde Google Drive API a shards JSONL/Parquet reanudables y utilizarlos como fallback. Repite gran parte del pipeline principal y dos bloques anteriores. |
| `EN Nusch et al. CACIC_2026_template_Multi-label_subject_classification_in_institutional_repositories.pdf` | Propuesta de 8 páginas para Open Repositories 2026, con descripción del pipeline y anexo de resultados de validación. |

No se encontraron CSV, Parquet, JSONL, scripts `.py`, configuraciones ni resultados independientes dentro del repositorio. El artículo BIREDIAL 2025 citado por el PDF tampoco está incluido.

## 2. Función de cada notebook

### 2.1 Extracción a TXT

Monta Drive, recorre `/Datos_SEDICI/var/` sin seguir enlaces, excluye la carpeta de salida y guarda cada archivo en `SEDICI_FullText_TXT/` con el mismo basename y extensión `.txt`. Usa una carpeta de salida plana, `OVERWRITE=False`, lectura completa en memoria y logging cada 500 archivos.

La salida almacenada muestra una ejecución iniciada el 8 de enero de 2026 y progreso hasta 151.500 archivos. No aparece el mensaje `FIN`, por lo que la notebook no demuestra que esa corrida haya terminado. La notebook principal encontró posteriormente 151.836 TXT.

### 2.2 Muestreo y clasificación de 37 etiquetas

La implementación más reciente y respaldada por salidas está en las celdas 4–17:

1. configuración;
2. mapeo rápido TXT → handle sin leer contenido;
3. escaneo por chunks de metadata, cobertura, labels Top‑37 y muestreo estratificado de 23.000 más ajuste;
4. lectura concurrente sólo de los TXT seleccionados;
5. filtro de full text vacío y segundo muestreo hasta aproximadamente 20.000;
6. split iterativamente estratificado train/validation/test;
7. evaluación de siete feature sets, cuatro representaciones y tres clasificadores;
8. resumen de mejores resultados de validación.

Las celdas 20–28 son una generación anterior que lee full text antes de muestrear y evalúa features por separado. Las celdas 31–42 son aún anteriores: incluyen mapeo global, construcción del dataset, split aleatorio no estratificado y evaluación. Los encabezados de la propia notebook las marcan como `CODIGO ANTERIOR`.

### 2.3 Opción JSONL/Parquet

Autentica contra Google Drive API, resuelve una ruta de Drive a `folder_id`, lista recursivamente TXT y descarga su contenido. Exporta:

- shards JSONL de 5.000 registros;
- Parquet con Snappy, escrito en batches de 2.000 y rotado aproximadamente cada 5.000;
- `state.json` para reanudación;
- `export_log.csv`.

El esquema es `file_id`, `name`, `modifiedTime`, `size`, `text`. La salida muestra 151.836 archivos y numerosos errores HTTP 403 `cannotDownloadAbusiveFile`; no muestra finalización. Una celda posterior implementa fallback TXT → JSONL → Parquet, pero terminó con `KeyboardInterrupt`. El resto duplica las versiones anteriores de la notebook principal.

## 3. Orden y dependencias

Orden histórico inferido:

1. Exportación DSpace externa: metadata, mapping Assetstore/internal ID → handle y archivos del Assetstore.
2. Notebook de extracción a TXT.
3. Notebook JSONL/Parquet, opcional, para evitar limitaciones del mount de Drive.
4. Notebook principal de muestreo y clasificación.

Dependencias externas observadas:

- Google Colab y Google Drive;
- `pandas`, `numpy`, `scipy`, `matplotlib`, `tqdm`;
- `scikit-learn`;
- `iterative-stratification`, instalada en una celda si falta;
- `torch` y `sentence-transformers`;
- Google Drive API y `pyarrow` en la variante JSONL/Parquet.

No hay archivo de dependencias ni versiones fijadas.

## 4. Flujo completo del pipeline histórico

```text
Assetstore
  → conversión byte-a-texto en TXT plano
  → basename del TXT como file_id/internal_id
  → merge con CSV de mapping
  → handle

Metadata CSV
  → extracción de handle desde columnas URI
  → unión de abstracts
  → unión de dc.subject como keywords
  → unión y limpieza básica de cuatro campos sedici.subject como labels

handles metadata ∩ handles con TXT
  → frecuencia global de labels
  → soporte >= 5
  → Top 37
  → filtro de documentos sin label Top-37
  → muestreo multilabel de 23.000
  → lectura de full text sólo para esos handles
  → filtro de full text vacío
  → recorte estratificado a ~20.000
  → 19.995 documentos
  → split multilabel 70/15/15
  → representaciones ajustadas en train
  → clasificadores One-vs-Rest
  → evaluación en validation
```

## 5. Columnas de entrada

El código detecta dinámicamente:

- abstracts: toda columna cuyo nombre contenga `abstract`;
- keywords: toda columna cuyo nombre contenga `dc.subject`;
- handle: una columna `handle` si existe; en su defecto, concatena columnas cuyo nombre contiene `identifier.uri`, es exactamente `handle` o termina en `.uri`, y extrae `10915/\d+`;
- mapping: columna `handle` y primera coincidencia entre `internal_id`, `internalid`, `file_id`, `id`, `bitstream_id`; como fallback, primera columna que contenga `id`.

La salida confirma, entre otras, `dc.description.abstract`, variantes `[]`, `[de]`, `[en]`, `[es]`, y equivalentes de `dc.subject`. No se conserva la procedencia ni el idioma declarado de cada fragmento después de unirlos.

## 6. Construcción de labels y targets

La implementación ejecutada usa, si existen, exactamente:

1. `sedici.subject.materias`
2. `sedici.subject.materias[]`
3. `sedici.subject.materias[es]`
4. `sedici.subject.other[es]`

Cada valor se divide por `||`; de cada parte se conserva lo anterior a `::`; luego se eliminan vacíos y duplicados y se ordena. Las labels de las cuatro fuentes se mezclan sin conservar `source_field`.

Tras contar soportes en el pool con TXT, se filtran labels con soporte al menos 5 y se toman las primeras 37 por frecuencia. Cada documento conserva sólo labels incluidas en esas 37 y se descartan documentos sin ninguna.

Consecuencia importante para 2026: `sedici.subject.other[es]` sí formó parte del target histórico. Reproducir E0 exige mantenerlo; cambiar el esquema debe ser explícito y producir un reporte por campo.

## 7. Mapping Assetstore, bitstreams y handles

- La notebook de extracción aplana el árbol Assetstore y convierte cada basename a `.txt`.
- La notebook principal interpreta `Path.stem` como `file_id`.
- El CSV de mapping enlaza ese ID con `handle`.
- Si un handle tiene varios TXT, concatena sus textos con dos saltos de línea.
- La metadata se enlaza por el handle extraído con regex `10915/\d+`.
- La versión más antigua contiene además una regex más general para `/handle/<prefix>/<suffix>` o `hdl.handle.net/...`; no es la que respalda los resultados finales.

Riesgo: dos archivos de diferentes carpetas con el mismo basename colisionan en la carpeta plana. Con `OVERWRITE=False`, el segundo se omite silenciosamente; el código lo reconoce en un comentario pero no lo cuantifica.

## 8. Cobertura

Salidas de la implementación reciente:

| Etapa | Resultado |
|---|---:|
| TXT encontrados | 151.836 |
| TXT mapeados a handle | 147.192 (96,941437 %) |
| Handles únicos con algún TXT | 139.793 |
| Filas de metadata | 126.080 |
| Filas de metadata con handle | 126.077 |
| Filas/handles de metadata con algún TXT | 103.434 |
| Cobertura TXT sobre metadata con handle | 82,04034 % |
| Pool con label Top‑37 | 95.885 handles |
| Muestra previa | 23.009 handles |
| TXT asociados leídos | 24.525 |
| Handles con full text no vacío | 22.643 |
| Eliminados por full text vacío | 366 |
| Dataset final | 19.995 handles |

La cobertura se mide por presencia de TXT antes de leerlos y por contenido no vacío después. No se conserva aquí un reporte de bytes corruptos, reemplazos de decodificación, colisiones de basename o calidad del texto.

## 9. Sampling

Parámetros ejecutados:

- objetivo: 20.000 handles;
- buffer: 3.000;
- método: `MultilabelStratifiedShuffleSplit`;
- seed: 42.

Por redondeo del estratificador, el primer muestreo produjo 23.009, no 23.000. Tras eliminar textos vacíos quedaron 22.643. El segundo recorte estratificado produjo 19.995, no 20.000. El paper describe correctamente 19.995 como resultado final, aunque también usa “20,000” como tamaño factible aproximado.

## 10. Split train/validation/test

La implementación reciente hace dos splits iterativamente estratificados:

- train: 70 %, 13.974 documentos;
- validation: 15 %, 3.000;
- test: 15 %, 3.021.

Prueba hasta 40 seeds (`42 + attempt`) para obtener presencia de todas las labels en las tres particiones. La salida informa seed 42 y cero labels ausentes. Genera `dataset_splits.csv` y `label_coverage_by_split.csv`.

Una versión anterior usa `train_test_split` aleatorio sin estratificación. No corresponde al resultado final documentado.

## 11. Feature sets y combinación

La corrida final evalúa:

- `abstract`;
- `keywords`;
- `fulltext`;
- `abs+kw`;
- `abs+ft`;
- `kw+ft`;
- `all3`.

`concat_prefixed` antepone nombres como `ABSTRACT:`, `KEYWORDS:` y `FULLTEXT:` y concatena con saltos de línea. No hay procesamiento por campo antes de concatenar.

## 12. Representaciones

### TF-IDF

`TfidfVectorizer(max_features=100000, ngram_range=(1, 2))`. Los demás parámetros quedan en defaults de scikit-learn. Se ajusta sólo con train y transforma validation.

### BM25 histórico

`CountVectorizer(max_features=100000, ngram_range=(1, 2))`, seguido de:

`idf = log((N - df + 0.5) / (df + 0.5) + 1)`

y normalización BM25 con `k1=1.5`, `b=0.75`, longitud basada en suma de conteos y `avgdl` calculado en train. Vocabulario, IDF y `avgdl` se ajustan sólo en train. Esta fórmula debe preservarse sin cambios bajo un nombre legacy para E0.

### SBERT y LaBSE

- SBERT: `distiluse-base-multilingual-cased-v1`.
- LaBSE: `sentence-transformers/LaBSE`.
- Batch size: 32.
- Dispositivo: CUDA si está disponible.

El pipeline envía cada string completo directamente a `SentenceTransformer.encode`. No consulta `model.max_seq_length`, no divide en chunks, no agrega chunks ni cachea embeddings. Dado que el full text se había limitado a 100.000 caracteres, el tokenizador/modelo trunca internamente el contenido que excede su límite. Éste es el comportamiento que 2026 debe denominar `legacy_truncated`.

No hay BoW como representación evaluada independiente; `CountVectorizer` es sólo una etapa interna de BM25.

## 13. Clasificadores y parámetros

Todos usan `OneVsRestClassifier`:

| Nombre | Estimador histórico |
|---|---|
| LogReg | `LogisticRegression(solver="liblinear", max_iter=3000)` |
| LinearSVC | `LinearSVC(max_iter=200000)` |
| SGD | `SGDClassifier(loss="log_loss", max_iter=3000, tol=1e-3)` |

No se configura `class_weight`, `C`, `alpha`, `penalty` ni `random_state`. No se usa `sklearn.clone`; se reutilizan instancias del diccionario y se vuelven a ajustar.

Versiones anteriores usan `max_iter=2000/20000` o `3000/20000`. El baseline respaldado por los resultados combinados es 3000/200000/3000.

## 14. Métricas y alcance de evaluación

Se calculan en validation:

- `accuracy_score`, que en multilabel es exact match/subset accuracy y aparece como `acc`;
- F1 micro;
- F1 macro.

El análisis por label de una versión intermedia usa `classification_report`, soporte, bins y correlaciones, pero no tiene salidas asociadas en el bloque final. No se calculan hamming loss, ranking metrics, average precision, intervalos de confianza ni evaluación por idioma.

El test se construye y persiste, pero el pipeline final no lo evalúa. El paper dice explícitamente que reporta validation. Por tanto, los valores publicados no son resultados de test.

## 15. Artefactos históricos generados

La implementación reciente declara:

- `txt_to_handle.csv`
- `mapping_fast_stats.csv`
- `coverage_pre_fulltext.csv`
- `label_frequencies_pre_fulltext.csv`
- `pool_handles_pre_fulltext.csv`
- `sampled_handles_pre_fulltext.csv`
- `fulltext_by_handle.csv`
- `fulltext_read_stats.csv`
- `dataset_with_fulltext.csv`
- `coverage_post_fulltext.csv`
- `sampled_handles_final.csv`
- `dataset_splits.csv`
- `label_coverage_by_split.csv`
- `results_val.csv`
- gráficos mostrados inline.

La variante adicional declara shards `sedici_fulltext_*.jsonl`, `sedici_fulltext_*.parquet`, `state.json` y `export_log.csv`. Ninguno de estos artefactos está incluido en Git.

## 16. Resultados en notebooks

Resultados principales de validation:

| Feature set | Mejor configuración | Subset accuracy | F1 micro | F1 macro |
|---|---|---:|---:|---:|
| abstract | BM25 + SGD | 0,336333 | 0,522078 | 0,465683 |
| keywords | BM25 + LinearSVC | 0,318000 | 0,471414 | 0,417009 |
| fulltext | BM25 + SGD | 0,588333 | 0,744948 | 0,691262 |
| abstract + keywords | BM25 + SGD | 0,375000 | 0,560584 | 0,501444 |
| abstract + fulltext | BM25 + SGD | 0,591000 | 0,748107 | 0,699222 |
| keywords + fulltext | BM25 + SGD | 0,588667 | 0,747344 | 0,699051 |
| abstract + keywords + fulltext | BM25 + SGD | 0,597333 | 0,756169 | 0,708046 |

Mejor por representación:

| Representación | Feature set y clasificador | F1 micro | F1 macro |
|---|---|---:|---:|
| TF-IDF | keywords + fulltext, LinearSVC | 0,726626 | 0,659163 |
| BM25 | all3, SGD | 0,756169 | 0,708046 |
| SBERT | fulltext, LinearSVC | 0,598105 | 0,533675 |
| LaBSE | keywords + fulltext, LinearSVC | 0,590181 | 0,508965 |

## 17. Resultados y afirmaciones del paper

El paper informa:

- 19.995 documentos, 37 subjects;
- full text truncado a 100.000 caracteres;
- split estratificado sin pérdida de labels;
- evaluación en validation;
- mejor global all3 + BM25 + SGD: macro-F1 0,708 y micro-F1 0,756;
- mejor TF-IDF: keywords + fulltext + LinearSVC, macro-F1 0,659;
- mejores embeddings: SBERT 0,534 y LaBSE 0,509.

Los valores coinciden con las salidas de la notebook. El anexo PDF presenta problemas de maquetación: varias etiquetas de `feature_set` aparecen desplazadas respecto de sus filas. Las salidas tabulares de la notebook son la evidencia más inequívoca para reconstruir esas asociaciones.

El abstract del paper dice “70% effectiveness” y luego lo identifica con macro-F1 0,708; no debe reinterpretarse como subset accuracy.

## 18. Diferencias notebook/paper

| Tema | Notebook | Paper |
|---|---|---|
| Fecha/identidad | Carpeta y nombres refieren a 2025; salidas son de enero de 2026 | Propuesta Open Repositories 2026; cita un antecedente BIREDIAL 2025 no incluido |
| Soporte y Top-K | Filtra soporte `>=5` y después toma las 37 más frecuentes | Dice “37 most frequent subjects (minimum support: 5)”; descripción compatible pero menos precisa |
| Tamaño objetivo | 20.000 produce 19.995 por redondeo del estratificador | Usa 20.000 como ejemplo y 19.995 como dataset final |
| Métrica “Accuracy” | `accuracy_score` multilabel (`acc`) | La llama “Accuracy” sin aclarar que es subset accuracy |
| Tiempo | Sólo fit + predict de cada clasificador | Tabla incluye `time_sec` sin delimitar su alcance |
| Reproducibilidad | Paths personales, sin versiones ni notebook autocontenida | Afirma que se proveerá notebook ejecutable con outputs versionados |
| Artefactos | Se escriben bajo un tag determinístico que puede sobrescribirse | Habla de outputs versionados/reutilizables |
| Test | Se crea, pero no se evalúa | Reporta validation, coherente con la notebook |

## 19. Código duplicado

- La notebook principal contiene tres generaciones completas o parciales del pipeline.
- La notebook JSONL/Parquet copia casi literalmente las dos generaciones antiguas y una variante del pipeline intermedio.
- `BM25Transformer`, `BM25Vectorizer`, construcción de dataset, clasificadores, embeddings y análisis por label aparecen repetidos.
- Los parámetros divergen entre copias (`MAX_FEATURES` 50.000/100.000; `max_iter` 20.000/200.000).
- Hay dos estrategias de split incompatibles: aleatoria e iterativamente estratificada.

Para E0 debe extraerse y congelarse primero la variante de celdas 4–17, sin mezclar accidentalmente decisiones de los bloques anteriores.

## 20. Paths hardcodeados

Todas las notebooks dependían de una ruta personal bajo el montaje de Drive,
representada aquí de forma anonimizada como:

`/content/drive/My Drive/<project-data-root>`

También alternan `My Drive` y `MyDrive`. Nombres hardcodeados:

- `Datos_SEDICI/var`
- `Datos_SEDICI/SEDICI_FullText_TXT`
- `<fulltext-mapping>.csv`
- `<metadata-export>.csv`
- `outputs_fulltext_clf`
- `Datos_SEDICI/exports_jsonl_parquet`

## 21. Posibles problemas metodológicos y técnicos

1. **Idioma ausente**: no se detecta idioma ni se conserva idioma declarado; tampoco hay stopwords language-aware.
2. **Targets mezclados**: cuatro campos, incluido `sedici.subject.other[es]`, se unen sin trazabilidad de fuente.
3. **Selección de labels global**: Top‑37 se decide antes del split usando todo el pool; debe preservarse para E0 y revisarse/documentarse para 2026.
4. **Keywords amplios**: se unen todas las columnas `dc.subject`; no se conserva el campo de origen y la limpieza `::URI` se aplica a targets, no explícitamente al texto de keywords. La salida muestra una keyword `Violencia::http://...`.
5. **Sin normalización Unicode explícita**: sólo se colapsan espacios en algunos pasos.
6. **Truncado doble/implícito**: full text se corta por caracteres y los modelos transformer vuelven a truncar por tokens sin registrar el límite efectivo.
7. **Sin seeds completas**: sampling/split tienen seed; SGD y otros estimadores no reciben `random_state`.
8. **Reutilización de estimadores**: no se clonan los clasificadores.
9. **Run ID sobrescribible**: el tag depende de cuatro parámetros; repetir o cambiar otros parámetros puede sobrescribir artefactos.
10. **Sin manifiesto**: no se guarda configuración resuelta, commit, ambiente ni versiones.
11. **Tiempo incompleto**: `time_sec` empieza después de construir la representación y mide únicamente `fit` + `predict`; no incluye vectorización, embeddings, carga, ni total. La comparación temporal entre sparse y dense no representa costo extremo a extremo.
12. **Resultados sólo validation**: no existe evaluación final test ni aislamiento operativo mediante una etapa de configuración congelada.
13. **Umbral fijo implícito**: se usa `predict` por defecto; no hay optimización ni persistencia de thresholds.
14. **Exportación plana**: riesgo de colisión de basenames del Assetstore.
15. **Conversión no semántica**: la notebook llamada “descompresión GZ” no descomprime ni extrae formatos; decodifica bytes arbitrarios como texto.
16. **Corridas incompletas**: extracción TXT no muestra `FIN`; JSONL/Parquet registra 403 y el fallback termina en `KeyboardInterrupt`.
17. **Dependencias no fijadas**: resultados pueden variar entre versiones.
18. **Outputs incrustados con conteo nulo**: dificulta verificar orden y estado real de ejecución.
19. **Memoria**: algunas versiones leen/concatenan full text global; Parquet concatena todos los shards en RAM. La variante reciente mejora esto leyendo sólo handles muestreados.
20. **Cómputo repetido**: embeddings se recalculan para cada feature set y no se cachean.

No se observó fuga de vocabulario, IDF o BM25 hacia validation en el bloque final: todos se ajustan con train. Tampoco se usa test para selección en el código respaldado por resultados.

## 22. Funciones y comportamientos reutilizables

Reutilizables tras añadir tests y configuración:

- escaneo staged de cobertura sin leer full text;
- detección flexible de columnas de mapping;
- extracción de handle;
- parsing `||` y eliminación de sufijo `::...`;
- agregación deduplicada por handle;
- lectura de full text sólo para handles necesarios;
- muestreo y split multilabel iterativos con reporte por label;
- `concat_prefixed`;
- fórmula exacta de BM25 legacy;
- exportación shardeada y checkpoint atómico de `state.json`;
- métricas históricas como capa de compatibilidad.

No deben reutilizarse sin corregir: paths personales, estado global de notebooks, aplanado con colisiones, lectura arbitraria byte-a-texto, instancias de clasificadores compartidas y encoding transformer directo de documentos largos.

## 23. Diferencias deliberadas del pipeline 2026

E0 conservará explícitamente targets, Top‑37, truncado a 100.000 caracteres, `concat_prefixed`, fórmula BM25, modelos, clasificadores, split y métricas históricas para comprobar reproducibilidad.

Después de E0, 2026 introducirá de forma versionada:

- paquete modular en `src/` y notebook como interfaz;
- configuración YAML, manifests y runs no sobrescribibles;
- target schema explícito y reporte de fuente/soporte;
- idioma declarado separado de idioma detectado;
- detección independiente de abstract/fulltext y best-effort para keywords;
- stopwords por idioma detectado, sólo para sparse y por campo;
- ablation `raw`, `normalized`, `language_stopwords`;
- Unicode NFKC y limpieza de `::URI` en keywords;
- BoW, n-grams word/char configurables;
- transformers con texto natural, modo `legacy_truncated` y modos chunked con pooling;
- cache de embeddings;
- estimadores clonados y seeds centralizadas;
- selección en validation y test final aislado;
- thresholds globales/per-label aprendidos sólo en validation;
- métricas multilabel/ranking, análisis por idioma/label, bootstrap y coocurrencia;
- tiempos separados por etapa;
- soporte para TXT y Parquet sin cargar innecesariamente todo el corpus;
- tests livianos y CI.

Estas mejoras no se presentarán como reproducción del baseline: cada desviación tendrá nombre y configuración propios.

## Conclusión para E0

La referencia histórica reproducible es la secuencia de celdas 4–17 de la notebook principal, complementada por la notebook de extracción a TXT. El mejor resultado documentado es de **validation**, no test: all3 + BM25 legacy + SGD, macro-F1 0,708046 y micro-F1 0,756169. Estos valores son criterios de comparación histórica, no resultados de una futura corrida 2026.

