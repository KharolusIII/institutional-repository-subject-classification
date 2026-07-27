# Plan de implementación 2026

## Regla de trabajo

`Pipe 2025 OR/` es evidencia inmutable. No se moverá, borrará, reformateará ni ejecutará de forma que escriba dentro de ella. Los resultados históricos no se hardcodearán como resultados nuevos. Cada etapa terminará con tests livianos, documentación de decisiones y un proyecto ejecutable.

## Qué se preservará

- Los cuatro campos target históricos para E0, incluido `sedici.subject.other[es]`.
- Parsing por `||`, deduplicación y eliminación de sufijo `::...`.
- Top‑37 con soporte mínimo 5.
- Muestreo objetivo 20.000 + buffer 3.000 y seeds históricas.
- Full text limitado a 100.000 caracteres para E0.
- Split iterative multilabel 70/15/15 con reintentos.
- Los siete feature sets y `concat_prefixed`.
- TF-IDF word (1,2), BM25 `k1=1.5`, `b=0.75` y su fórmula exacta.
- SBERT `distiluse-base-multilingual-cased-v1` y LaBSE.
- One-vs-Rest LogReg, LinearSVC y SGD con parámetros históricos.
- Subset accuracy, F1 micro y F1 macro como métricas de compatibilidad.
- Evaluación histórica sobre validation.

## Qué se reutilizará

- Estrategia staged de mapping/cobertura.
- Lectura del full text después de identificar handles necesarios.
- Agregación de múltiples bitstreams por handle.
- Muestreo y split de `iterative-stratification`.
- Exportación shardeada y estado reanudable, rediseñados para TXT/Parquet.
- Implementación BM25 legacy, aislada y cubierta por tests.

## Qué se refactorizará

- Código de notebooks a módulos cohesivos en `src/ir_subject_classification/`.
- Parámetros globales y paths personales a configuración validada.
- Detección heurística de columnas a schemas explícitos con informes.
- Estado implícito de celdas a etapas idempotentes y artefactos versionados.
- Clasificadores compartidos a factories con `sklearn.clone`.
- Métricas y reporting a funciones testeables.
- Embeddings directos a interfaz con cache y modos legacy/chunked.
- Una notebook Colab delgada que invoque el paquete.

## Riesgos prioritarios

1. Falta de datos históricos y resultados CSV en Git: E0 no podrá comprobarse numéricamente hasta proporcionar las fuentes privadas.
2. El paper BIREDIAL 2025 citado no está en el repositorio.
3. Selección Top‑37 global antes del split y targets mezclados.
4. Posibles colisiones de basename al aplanar el Assetstore.
5. Full text decodificado sin validar tipo/formato.
6. Truncado transformer implícito y no medido.
7. `time_sec` histórico no representa tiempo total.
8. Seeds incompletas y dependencias no fijadas.
9. Runs sobrescribibles y ausencia de manifest.
10. Duplicación con parámetros divergentes.

## Diferencias 2026 deliberadas

- Idioma detectado desde contenido, separado del declarado.
- Abstract y full text detectados independientemente; keywords best-effort o `und`.
- Stopwords por idioma/campo sólo en sparse; transformers sin stopword removal.
- Ablation de preprocessing con documentos y splits idénticos.
- BoW y n-grams adicionales sin alterar E0.
- Chunking según `model.max_seq_length`, overlap y pooling seleccionados en validation.
- Thresholds seleccionados en validation.
- Test aislado hasta congelar configuración.
- Métricas extendidas, ranking, análisis por idioma/label y bootstrap.
- Runs únicos con ambiente, commit, configuración y tiempos por etapa.

## Orden de implementación

### P0 — Reproducción histórica

1. Crear esqueleto mínimo instalable y `baseline_2025.yaml`.
2. Implementar schemas, mapping, metadata, sampling, split, vectorizadores y clasificadores legacy sin cambios metodológicos.
3. Añadir pruebas de caracterización de parsing, BM25 y splitting.
4. Generar los mismos artefactos históricos bajo un `run_id`.
5. Con datos privados disponibles, comparar conteos, splits y validation con la auditoría; registrar tolerancias y discrepancias.
6. No tocar test para selección.

Criterio de cierre: pipeline E0 ejecutable y comparación trazable; si los datos no están disponibles, pruebas sintéticas completas y reproducción marcada como pendiente, nunca afirmada.

### P1 — Modularización y reproducibilidad

1. Completar paquete, CLI, validación de configuración y manifests.
2. Implementar runs no sobrescribibles, cache y reanudación.
3. Añadir README, licencia, citation, data README, `.gitignore`, Colab y CI liviano.
4. Separar tiempos por etapa.

Criterio de cierre: instalación limpia, CLI y notebook llaman a la misma lógica; tests pasan localmente y en CI.

### P2 — Idioma y preprocessing sparse

1. Interfaz `LanguageDetector` con backend configurable y default documentado.
2. Persistir idioma declarado/detectado y score por campo.
3. Implementar normalización técnica y stopwords language-aware por documento/campo.
4. Garantizar `und` sin eliminación y transformers sin stopwords.
5. Ejecutar E1 sobre splits congelados.

Criterio de cierre: tests de desacople declarado/detectado y ablation comparable sin cambio de documentos.

### P3 — Estadísticas y métricas extendidas

1. Dataset/text/language statistics.
2. Métricas multilabel y Precision/Recall@K.
3. Cobertura por split, target schema, coocurrencia y análisis por label.
4. Reporting de validation; conservar test cerrado.

Criterio de cierre: artefactos definidos, denominadores documentados y tests de métricas.

### P4 — Transformers largos

1. Cache de embeddings con clave de modelo/config/texto.
2. `legacy_truncated` como control.
3. Chunking dinámico y pooling mean/length-weighted/max opcional.
4. Selección de pooling sólo en validation.

Criterio de cierre: textos largos cubiertos por tests sin descargar modelos grandes; experimento E3 reanudable.

### P5 — Tuning y thresholds

1. Tuning acotado de las mejores configuraciones.
2. Scores correctamente nombrados (`decision_scores` para LinearSVC).
3. Threshold default/global/per-label con fallback por soporte.
4. Congelar configuración y ejecutar test una sola vez.
5. Bootstrap, per-label y evaluación por idioma en test.

Criterio de cierre: `thresholds.json`, predictions y resultados test trazables, sin selección sobre test.

### P6 — Long tail

1. Configurar Top‑37/60/100 y soporte mínimo.
2. Evaluar class weighting/reweighting sólo después de E0–E5.
3. Informar estabilidad, cobertura y grupos con N insuficiente.

Criterio de cierre: comparaciones con splits y criterios explícitos, sin presentar configuraciones incompatibles como equivalentes.

## Secuencia de experimentos

- E0: reproducción legacy.
- E1: raw vs normalized vs language_stopwords con BoW/TF-IDF/BM25.
- E2: word (1,1)/(1,2)/(1,3) y TF-IDF char (3,5)/(3,6).
- E3: SBERT/LaBSE legacy_truncated vs chunked mean/length-weighted mean.
- E4: tuning acotado en validation.
- E5: thresholds en validation y test final congelado.
- E6: expansión de labels.

## Política de commits y validación

Cada etapa se dividirá en cambios pequeños: interfaces/tests, implementación, integración, documentación. Antes de avanzar:

- ejecutar tests livianos;
- inspeccionar `git diff`;
- comprobar que `Pipe 2025 OR/` sigue intacta;
- documentar cualquier desviación metodológica;
- no afirmar resultados que no provengan de artefactos de una corrida identificable.

