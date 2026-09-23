# YouTube Hate Speech Detector

Detector de mensajes de odio en comentarios de YouTube, desarrollado como solución de consultoría.
Prioriza una solución **práctica e implementable**: en lugar de un booleano, devuelve una probabilidad y una
**banda de decisión** (`permitir` / `revisar` / `ocultar`). El sistema **recomienda**; un humano decide.

> **Estado: niveles Esencial, Medio y Avanzado implementados** (ensemble, análisis por URL, seguimiento en directo,
> red recurrente evaluada, Docker, extensión de navegador). **Desplegado y funcionando** tras un proxy inverso con
> HTTPS, siguiendo la guía de [`deploy/`](deploy/README.md). Experto (transformer, base de datos, MLflow) en el roadmap.

## Resultados (léelos con sentido crítico)

Dataset pequeño (~1000 comentarios, 138 de odio). Estimación honesta por validación cruzada estratificada sobre
train+val, con augmentation solo en el fold de entrenamiento. Modelo servido: **ensemble** (votación ponderada de
regresión logística sobre palabras, regresión logística sobre caracteres y Naive Bayes complementario).

| Métrica | Valor |
|---|---|
| F1-macro (CV) | **≈ 0.71** |
| PR-AUC (CV) | ≈ 0.41 |
| Gap entrenamiento – CV (F1-macro) | 4.6 pp (requisito: < 5 pp) |
| Gap entrenamiento – test | 0.2 pp |
| Latencia API (CPU) | p50 ≈ 5 ms · p95 ≈ 6 ms |

**Frente al baseline** (TF-IDF + regresión logística L1), el ensemble mejora el **ranking** (PR-AUC +1.3 pp, en 16 de
20 particiones, p = 0.04) y no cambia el F1 en el punto de operación. Es una mejora real pero modesta. La
optimización de hiperparámetros con Optuna se hizo con el sobreajuste como **restricción** (sin ella elegía un
ensemble con 17 pp de gap) y se validó con particiones nuevas.

El F1 del conjunto de test tiene solo 28 positivos: **no debe tomarse como cifra de rendimiento**; la CV es la
estimación fiable.

**Limitaciones importantes**
- El «odio» del dataset es ~91 % racismo, sobre un mismo suceso (Ferguson): no generaliza a otros tipos de odio
  ni a otros temas, y puede dar falsas alarmas con menciones neutras a un grupo.
- La precisión máxima alcanzable es baja (~0.4–0.5): por eso `revisar` es la banda principal y `ocultar` es solo una
  recomendación de alta probabilidad, no una decisión fiable por sí sola.
- Las etiquetas originales tienen criterio inconsistente en casos límite (ver el notebook de EDA).

## Quickstart

```bash
git clone https://github.com/jzelada97/youtube-hate-speech && cd youtube-hate-speech
python -m venv .venv && source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -e ".[serve,dev]"
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','omw-1.4')]"

# 1) Datos: descarga youtoxic_english_1000.csv (enlace del cliente) en data/raw/
# 2) Entrenar (genera models/ensemble_v1.joblib + metadata con umbrales de bandas)
python scripts/train.py --model ensemble        # o --model baseline
# 3) API
uvicorn hatedet.api.main:app --port 8000
# 4) Demo (otra terminal)
streamlit run src/hatedet/ui/app.py
```

Los datos y los modelos entrenados **no se versionan** (contienen comentarios reales de usuarios). Si no defines
`HATEDET_MODEL_PATH`, la API sirve `ensemble_v1` solo si existe **y cumple el requisito de gap**; si no, `baseline_v3`.

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado, versión del modelo y umbrales de las bandas |
| POST | `/predict` | `{"text": "..."}` → `{label, score, band, model_version}` |
| POST | `/predict/batch` | `{"texts": [...]}` (máx. 200) → `{predictions: [...]}` |
| POST | `/analyze/video` | `{"url": "...", "max_comments": 100}` (máx. 500) → informe agregado del vídeo |
| POST | `/monitor/start` | `{"url": "...", "poll_seconds": 30}` → `{session_id}`; inicia el seguimiento en directo |
| GET | `/monitor/{id}/events?after=N` | Comentarios nuevos clasificados desde el cursor `N` (`{events, next, active, expires_in}`) |
| DELETE | `/monitor/{id}` | Detiene el seguimiento |
| POST | `/review/queue` | `{"items": [{"text": "..."}]}` → encola para revisión humana lo que cae en `revisar`/`ocultar` |
| GET | `/review/pending?limit=20` | Cola pendiente, los más sospechosos primero, con las etiquetas disponibles |
| POST | `/review/{id}` | `{"is_hatespeech": true, "tags": ["IsRacist"]}` → registra el veredicto del moderador |
| GET | `/review/stats` | Pendientes, revisados y **acuerdo con el modelo** (indicador de deriva) |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"All these people are criminals"}'
curl -X POST localhost:8000/analyze/video -H 'Content-Type: application/json' \
     -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","max_comments":100}'
```

`/analyze/video` descarga los comentarios más recientes y devuelve el número de comentarios por banda, el porcentaje
marcado y los más sospechosos. **No se guardan ni se devuelven nombres de autor** (RGPD). Los comentarios se obtienen
con la API oficial de YouTube si defines `YOUTUBE_API_KEY`, o con un extractor sin clave (`youtube-comment-downloader`,
menos estable y sujeto a las condiciones de servicio de YouTube).

Variables de entorno: `HATEDET_MODEL_PATH`, `HATEDET_API_KEY` (si se define, exige cabecera `X-API-Key`),
`HATEDET_T_LOW` / `HATEDET_T_HIGH` (umbrales), `HATEDET_CORS_REGEX`, `YOUTUBE_API_KEY`, `HATEDET_REVIEW_DB`.
La API **no registra el texto** de los comentarios (única excepción: la cola de revisión, ver abajo).
Documentación interactiva en `/docs`.

## Revisión humana y reentrenamiento

La banda `revisar` significa que **una persona debe mirarlo**, así que hay dónde hacerlo. El ciclo completo es
**marcar → revisar → etiquetar → reentrenar**:

```bash
HATEDET_REVIEW_DB=data/review.db uvicorn hatedet.api.main:app --port 8000   # activa la cola
streamlit run src/hatedet/ui/app.py                                        # pestaña «Revisión humana»
python scripts/retrain.py --dry-run                                        # evalúa sin tocar producción
python scripts/retrain.py                                                  # reentrena y promociona si pasa el filtro
```

El moderador ve primero lo más sospechoso y etiqueta con **las 12 categorías del dataset original**, así que los
datos nuevos son directamente compatibles. `retrain.py` incorpora esos veredictos pero **solo sustituye el modelo
si pasa un filtro**: al menos 20 veredictos nuevos, cumplir el requisito de gap (< 5 pp) y no empeorar el PR-AUC
de forma significativa en una comparación pareada. Si no pasa, el modelo en producción no se toca y el motivo
queda escrito en `reports/retrain_<fecha>.json`. `GET /review/stats` da el **acuerdo entre el humano y el
modelo**: cuando esa cifra baja, toca reentrenar.

> **Privacidad**: esta es la única pieza que almacena el texto de comentarios reales, porque revisar sin ver es
> imposible. Está **apagada por defecto** (solo se activa con `HATEDET_REVIEW_DB`), nunca guarda el nombre del
> autor, tiene política de retención (`purge_older_than`) y su base de datos está fuera del repositorio.

## Seguimiento en directo

YouTube **no ofrece avisos en tiempo real** de comentarios nuevos, así que el seguimiento se hace por **sondeo**: la API
consulta los comentarios recientes cada `poll_seconds` (10–300 s), descarta los ya vistos y clasifica los nuevos. Cada
sesión caduca a los 30 minutos y hay un máximo de sesiones simultáneas (`HATEDET_MAX_MONITORS`, por defecto 5) para no
agotar recursos. La pestaña «En directo» de la demo lo usa; la extensión de navegador clasifica lo que el usuario ya ve.

## Red neuronal: resultado (negativo) y por qué

Se implementó una red recurrente bidireccional (BiLSTM/BiGRU, PyTorch) con parada temprana y regularización, y se evaluó
con la **misma CV pareada** que el resto (10 particiones; `python scripts/eval_lstm.py glove`):

| Modelo | PR-AUC | F1-macro (mejor umbral) | Precisión al 70 % de recall |
|---|---:|---:|---:|
| Baseline (TF-IDF + regresión logística L1) | 0.459 | 0.740 | 0.461 |
| Ensemble, parámetros por defecto (no se sirve: 17 pp de gap) | 0.491 | 0.742 | 0.445 |
| **Ensemble optimizado (el modelo servido)** | **0.472** | 0.746 | 0.445 |
| BiLSTM desde cero | 0.392 | 0.708 | 0.318 |
| BiLSTM + GloVe (embeddings preentrenados, congelados) | 0.445 | 0.717 | 0.359 |

Desde cero es **significativamente peor** que el baseline; con GloVe cierra casi todo el hueco en ranking (PR-AUC
estadísticamente igual) pero **no lo supera**. Conclusión: con ~800 comentarios de entrenamiento una red recurrente no
mejora a ML clásico; el siguiente paso con más potencial es un transformer preentrenado (nivel Experto). El código
sigue disponible (`python scripts/train.py --model lstm`, extra `pip install -e ".[nn]"`), pero no es el modelo servido.

### Aumentación con BERT enmascarado: resultado (negativo)

Se probó usar DistilBERT solo para *generar* variaciones de los comentarios de entrenamiento (rellena palabras
tapadas según el contexto; el modelo no decide etiquetas), con los criterios de aceptación fijados antes de medir.
Con la misma CV pareada, **ninguna variante mejora** a la aumentación clásica (EDA) que ya se usa: entre −0.2 y
−2.9 pp de PR-AUC sobre el ensemble servido, y generar solo variantes de la clase de odio empeora de forma
significativa. El generador queda disponible (`src/hatedet/data/mlm_augment.py`,
`python scripts/eval_mlm_augmentation.py`, extra `[nn]`), pero el modelo servido no lo usa y **no incorpora ningún
modelo preentrenado**.

### Aumentación con sinónimos tóxicos: resultado (negativo, empeora)

Se probó cambiar un término tóxico por otro equivalente (`idiot` ↔ `moron`, `thugs` ↔ `punks`), que conserva la
etiqueta por construcción. Con la misma CV pareada y criterios fijados antes de medir, **todas las variantes empeoran**
a EDA (−1 a −6 pp de PR-AUC), incluso frente a no aumentar nada, y cuantas más copias, peor. Con WordNet, el volteo de
etiquetas y BERT enmascarado son cuatro familias de aumentación léxica sin mejora: el límite es la variedad de los
datos, no su cantidad. El generador queda en `src/hatedet/data/toxic_synonyms.py` y
`python scripts/eval_toxic_synonyms.py`, pero el modelo servido no lo usa.

## Docker

```bash
# 1) entrena el modelo (o copia models/ensemble_v1.joblib al servidor): python scripts/train.py --model ensemble
cp .env.example .env && docker compose --env-file .env up -d --build
curl http://127.0.0.1:8000/health          # API;  demo en http://127.0.0.1:8501
```
Imagen con usuario no root, sistema de ficheros de solo lectura, healthcheck y puertos solo en `127.0.0.1`. El modelo no
va en la imagen: se monta como volumen de solo lectura. Para publicarlo (HTTPS, clave de API, firewall) sigue
[`deploy/README.md`](deploy/README.md).

## Extensión de navegador

[`extension/`](extension/README.md): extensión Manifest V3 que marca en YouTube los comentarios sospechosos usando la API.
Se instala en modo desarrollador. **El texto de los comentarios visibles se envía a la API que configures**: usa tu propio
servidor y HTTPS (ver privacidad en su README).

## Niveles de entrega

| Nivel | Contenido | Estado |
|---|---|---|
| Esencial | Modelo ML + API/interfaz + repo documentado | ✅ |
| Medio | Ensemble + análisis por URL de vídeo + tests + tuning (Optuna) | ✅ |
| Avanzado | LSTM + seguimiento en tiempo real + despliegue público + Docker | ✅ Completo. LSTM **sin** mejora sobre ML clásico (resultado medido y documentado); **desplegado y verificado** con HTTPS y clave de API |
| Experto | Transformer + persistencia en BD + MLflow | Parcial: **persistencia en BD hecha** (cola de revisión); transformer y MLflow pendientes |

## Estructura

```
src/hatedet/  data/ (carga, splits, augmentation) · nlp/ (limpieza, normalización, máscara de grupos)
              models/ (baseline, ensemble, lstm, glove, evaluación, bandas) · youtube/ (URL, fuentes, análisis, monitor)
              api/ (FastAPI) · ui/ (Streamlit)
scripts/      train.py · compare_models.py · tune_optuna.py · eda.py · find_label_issues.py · eval_augmentation.py
notebooks/    01_eda.ipynb
extension/    extensión de navegador (Manifest V3) · deploy/ guía de despliegue en VM · Dockerfile, docker-compose.yml
tests/        pytest (`python -m pytest --cov=src/hatedet`; los que necesitan el dataset se saltan si no está)
```

## Metodología

Cada modelo es un objeto de scikit-learn que recibe **texto crudo** y devuelve probabilidades, así que el que se
entrena es exactamente el que se sirve (sin *skew* entre entrenamiento e inferencia). Las decisiones se validan con
**validación cruzada repetida pareada** (mismas particiones para todas las variantes) y el conjunto de test se
reserva para el informe final. Los umbrales de las bandas se eligen con probabilidades fuera de muestra.
