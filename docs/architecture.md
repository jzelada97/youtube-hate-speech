# Architecture — Detector de mensajes de odio en comentarios de YouTube

## 1. Estilo arquitectónico y justificación

**Monolito modular con núcleo de ML desacoplado del servicio, empaquetado como multi-contenedor.**

Un único paquete Python (`src/hatedet/`) con fronteras internas explícitas entre capas, servido por dos
procesos (API y UI) y acompañado de dos servicios de infraestructura (BD y MLflow) en niveles superiores.

Justificación:

- El cliente pide una solución **implementable**. Microservicios multiplicarían el coste operativo sin aportar
  nada: no hay equipos independientes ni necesidad de escalado diferenciado.
- La frontera que sí importa es **entrenamiento ↔ inferencia**. El entrenamiento es batch, lento y offline;
  la inferencia es online y con presupuesto de latencia. Se separan por un **artefacto versionado** (un
  `Pipeline` de scikit-learn serializado + su metadata), no por una llamada de red.
- La UI (Streamlit) se separa de la API (FastAPI) porque "productivizar el modelo" significa API consumible
  por terceros; Streamlit es el cliente de demo, no el producto.

### Regla de dependencia

```
presentation (api, ui)  →  application (services)  →  domain (modelado, nlp)  →  infrastructure (io, db, youtube)
```

Las capas internas no importan las externas. `domain` no conoce FastAPI ni SQLAlchemy.

### Decisión estructural clave: sin skew train/serve

Todo el preprocesamiento (limpieza regex, normalización, lematización, vectorización) vive **dentro** de un
único `sklearn.pipeline.Pipeline` que se serializa con joblib. La API carga ese objeto y llama a
`predict_proba` sobre texto crudo. No existe código de preprocesamiento duplicado en el servicio. Esta es la
causa número uno de fallos de modelos en producción y se elimina por diseño.

## 2. Estructura del repositorio

```
youtube-hate-detector/
├── README.md
├── pyproject.toml                # deps + config de ruff/black/pytest
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile                      # make setup | data | train | api | ui | test | docker
├── .github/workflows/ci.yml
│
├── conf/
│   ├── config.yaml               # rutas, seeds, umbrales de banda
│   ├── model_baseline.yaml
│   ├── model_ensemble.yaml
│   └── model_transformer.yaml
│
├── data/                         # gitignored salvo .gitkeep
│   ├── raw/                      # dataset original, inmutable
│   ├── interim/                  # limpio, deduplicado, con split
│   ├── processed/                # listo para entrenar
│   └── external/                 # léxicos, datasets de apoyo
│
├── models/                       # artefactos versionados (gitignored)
│   ├── baseline_v1.joblib
│   └── baseline_v1.metadata.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_ablation.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/hatedet/
│   ├── __init__.py
│   ├── config.py                 # carga de conf/ + settings de entorno (pydantic-settings)
│   │
│   ├── data/
│   │   ├── loader.py             # ingesta del dataset crudo
│   │   ├── schema.py             # contrato de columnas, validación
│   │   └── splitter.py           # split estratificado train/val/test + dedup
│   │
│   ├── nlp/
│   │   ├── regex_rules.py        # URLs, menciones, emojis, leetspeak, repeticiones
│   │   ├── cleaner.py            # TextCleaner: BaseEstimator + TransformerMixin
│   │   ├── normalizer.py         # stopwords, stemming (SnowballStemmer), lematización (spaCy)
│   │   ├── vectorizers.py        # BoW, TF-IDF word + char_wb, FeatureUnion
│   │   └── augment.py            # sinónimos, back-translation (solo train)
│   │
│   ├── models/
│   │   ├── registry.py           # name -> constructor de estimador
│   │   ├── baseline.py           # LogReg, LinearSVC, ComplementNB
│   │   ├── ensemble.py           # Voting / Stacking
│   │   ├── neural.py             # LSTM (Keras o PyTorch)
│   │   ├── transformer.py        # fine-tuning HF
│   │   └── calibration.py        # CalibratedClassifierCV + búsqueda de umbrales
│   │
│   ├── training/
│   │   ├── train.py              # CLI de entrenamiento
│   │   ├── evaluate.py           # métricas, matriz de confusión, curvas
│   │   ├── tune.py               # estudio de Optuna
│   │   └── tracking.py           # wrapper de MLflow
│   │
│   ├── inference/
│   │   ├── predictor.py          # carga artefacto, predict, bandas de decisión
│   │   └── cascade.py            # ruta rápida lineal + transformer en la franja dudosa
│   │
│   ├── youtube/
│   │   ├── api_client.py         # YouTube Data API v3 (fuente principal)
│   │   ├── scraper.py            # fallback Scrapy/Playwright
│   │   ├── url_parser.py         # extrae videoId de cualquier forma de URL
│   │   └── watcher.py            # polling incremental con cursor
│   │
│   ├── db/
│   │   ├── models.py             # tablas SQLAlchemy
│   │   ├── session.py
│   │   └── repository.py
│   │
│   ├── services/
│   │   ├── moderation.py         # orquesta predictor + bandas + persistencia
│   │   └── video_analysis.py     # URL -> comentarios -> informe agregado
│   │
│   ├── api/
│   │   ├── main.py               # app FastAPI, lifespan carga el modelo una vez
│   │   ├── routes/               # predict.py, video.py, health.py
│   │   ├── schemas.py            # DTOs pydantic
│   │   └── deps.py
│   │
│   └── ui/
│       ├── app.py                # Streamlit
│       └── pages/                # 1_Comentario, 2_Video, 3_Tiempo_real, 4_Metricas
│
├── tests/
│   ├── unit/                     # nlp, models, url_parser, bandas
│   ├── integration/              # API con TestClient, repositorio con SQLite
│   ├── fixtures/
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── docker-compose.yml
│
└── docs/
    ├── architecture.md
    ├── model_card.md
    ├── api.md
    └── presentacion/
```

## 3. Componentes

| Componente | Responsabilidad | Depende de |
|---|---|---|
| `data.loader` / `data.schema` | Ingesta del dataset crudo, validación de contrato de columnas, informe de calidad | pandas |
| `data.splitter` | Deduplicación (near-dup incluida) y split estratificado train/val/test | `data.schema` |
| `nlp.regex_rules` | Catálogo central de expresiones regulares con tests unitarios propios | — |
| `nlp.cleaner` | Transformer sklearn que aplica el pipeline de limpieza | `nlp.regex_rules` |
| `nlp.normalizer` | Stopwords, stemming y lematización seleccionables por configuración | spaCy, NLTK |
| `nlp.vectorizers` | TF-IDF de palabra (1,2) + TF-IDF de carácter (3,5) en `FeatureUnion` | scikit-learn |
| `nlp.augment` | Aumentación aplicada exclusivamente al fold de entrenamiento | nlpaug / MarianMT |
| `models.*` | Constructores de estimadores por familia | scikit-learn, torch, HF |
| `models.calibration` | Calibración de probabilidades y búsqueda de umbrales sobre curva PR | scikit-learn |
| `training.train` | CLI: config → pipeline entrenado → artefacto + metadata | `data`, `nlp`, `models`, `tracking` |
| `training.evaluate` | Métricas, gap train/test, matriz de confusión, análisis de error | scikit-learn |
| `training.tune` | Estudio Optuna sobre CV estratificada, objetivo F1-macro | Optuna |
| `training.tracking` | Logging de params/métricas/artefactos a MLflow | MLflow |
| `inference.predictor` | Carga perezosa del artefacto, `predict_proba`, asignación de banda | joblib |
| `inference.cascade` | Enruta a transformer solo la franja de incertidumbre | `inference.predictor` |
| `youtube.api_client` | Paginación de `commentThreads.list`, control de cuota, backoff | google-api-python-client |
| `youtube.watcher` | Polling incremental por `publishedAt`, deduplicación por `commentId` | `api_client`, APScheduler |
| `db.repository` | Persistencia de predicciones y metadatos de vídeo | SQLAlchemy |
| `services.moderation` | Caso de uso: texto → predicción → banda → persistencia | `inference`, `db` |
| `services.video_analysis` | Caso de uso: URL → comentarios → informe agregado | `youtube`, `services.moderation` |
| `api.main` | Exposición HTTP, validación, manejo de errores, `/metrics` | `services` |
| `ui.app` | Demo: consume la API por HTTP, nunca importa el modelo | httpx |

## 4. Patrones de diseño aplicados

| Patrón | Dónde | Por qué |
|---|---|---|
| **Pipeline** (sklearn) | `nlp` + `models` compuestos en un único estimador | Elimina el skew train/serve; un solo artefacto que versionar |
| **Strategy** | `models.registry`, `nlp.normalizer` (stem vs lemma) | Intercambiar familia de modelo o normalizador desde configuración, sin tocar código |
| **Repository** | `db.repository` | Aísla SQLAlchemy del dominio; permite SQLite en test y Postgres en producción |
| **Adapter** | `youtube.api_client` / `youtube.scraper` tras una interfaz común | La fuente de comentarios es sustituible sin afectar a `services` |
| **Dependency Injection** | `api.deps` con `Depends` de FastAPI | El predictor se carga una vez en el lifespan y se inyecta; tests inyectan un doble |
| **Chain of Responsibility** | `inference.cascade` | Modelo barato primero, caro solo si hace falta |
| **DTO** | `api.schemas` | Contrato HTTP estable e independiente de las estructuras internas |

## 5. Decisiones tecnológicas

| Decisión | Elección | Alternativas descartadas | Razón |
|---|---|---|---|
| Lenguaje | Python 3.11 | 3.12 | Ecosistema NLP con ruedas estables; evita fricción de instalación |
| Vectorización base | TF-IDF palabra (1,2) **+ carácter `char_wb` (3,5)** | Solo palabra | Los n-gramas de carácter capturan ofuscación (`1d10ta`, `h*ijo*`), faltas y variantes morfológicas: es la mejora más barata y de mayor impacto real |
| Modelo base | Regresión logística con `class_weight='balanced'` | LinearSVC, Naive Bayes | Da probabilidades directamente (necesarias para las bandas), es interpretable por coeficientes y regulariza bien en espacios dispersos |
| Ensemble | `StackingClassifier` (LogReg + LinearSVC calibrado + ComplementNB) con meta LogReg | Random Forest sobre TF-IDF | Los modelos de árbol rinden mal en espacios dispersos de alta dimensión |
| HPO | Optuna con `StratifiedKFold` y pruning | GridSearch | Muchos menos trials para el mismo resultado; integra con MLflow |
| Deep learning | BiLSTM sobre embeddings preentrenados | LSTM desde cero | Entrenar embeddings desde cero con pocos datos casi siempre pierde contra TF-IDF |
| Transformer | Checkpoint pequeño en el idioma del dataset (p. ej. `distilbert-base-multilingual` o un RoBERTa en español), fine-tuneado | Modelo grande, API de LLM | Presupuesto CPU; un modelo destilado cubre la mayor parte de la ganancia |
| Serving | FastAPI + uvicorn | Flask, Django | Validación pydantic, async, OpenAPI automática |
| UI | Streamlit consumiendo la API | Streamlit importando el modelo | Mantiene la API como el producto real y la UI como cliente |
| BD | SQLite en desarrollo → PostgreSQL 16 en Docker | MongoDB | Los datos son relacionales; SQLAlchemy permite el mismo código en ambos |
| Tracking | MLflow local (file store) → servicio en compose | W&B | Autoalojado, sin coste, incluido en el brief |
| Tests | pytest + pytest-cov + `TestClient` | unittest | Fixtures y parametrización |
| Calidad | ruff + black + pre-commit | flake8 + isort | Una herramienta, órdenes de magnitud más rápida |
| CI | GitHub Actions (lint + tests en cada PR) | — | Gratuito en repos públicos, evidencia de proceso |
| Despliegue | API en Render/Fly.io, demo en Hugging Face Spaces | Kubernetes | Coste y complejidad injustificados para el alcance |

## 6. Contrato de la API

```
GET  /health                 -> {status, model_version, uptime_s}
POST /predict                -> {text} => {label, score, band, model_version, latency_ms}
POST /predict/batch          -> {texts[]} => {results[]}
POST /analyze/video          -> {url, max_comments} => {video_id, analyzed, hate_ratio,
                                                       band_counts, top_offenders[], sample[]}
POST /watch/start            -> {url, interval_s} => {job_id}
GET  /watch/{job_id}         -> {job_id, status, new_comments, hate_count, last_poll_at}
DELETE /watch/{job_id}       -> 204
GET  /metrics                -> Prometheus text format
```

Bandas de decisión (configurables en `conf/config.yaml`, no hardcodeadas):

| Banda | Condición | Acción recomendada |
|---|---|---|
| `allow` | `score < t_low` | Ninguna |
| `review` | `t_low ≤ score < t_high` | Cola de revisión humana |
| `remove` | `score ≥ t_high` | Ocultar y notificar |

`t_high` se fija buscando el punto de **precisión ≥ 0.90** sobre la curva PR de validación; `t_low` se fija
buscando **recall ≥ 0.95**. Así el borrado automático es conservador y la cola humana absorbe la ambigüedad.

## 7. Modelo de datos

```
videos(video_id PK, url, title, channel_id, fetched_at)
comments(comment_id PK, video_id FK, author_channel_id, text_hash, text,
         published_at, ingested_at)
predictions(id PK, comment_id FK, model_version, score, label, band,
            threshold_low, threshold_high, predicted_at)
feedback(id PK, prediction_id FK, human_label, reviewer, created_at)   -- bucle de mejora
model_versions(version PK, algorithm, trained_at, f1_macro, f1_hate,
               pr_auc, train_test_gap, mlflow_run_id, artifact_path)
watch_jobs(job_id PK, video_id FK, interval_s, status, cursor_published_at, created_at)
```

`feedback` es lo que convierte el sistema en algo que mejora con el uso: las correcciones humanas de la cola
de revisión son datos de entrenamiento etiquetados y gratuitos para la siguiente iteración.

## 8. Cross-cutting concerns

- **Configuración**: `conf/*.yaml` para lo no sensible, variables de entorno vía `pydantic-settings` para
  secretos. `.env` en `.gitignore`, `.env.example` versionado. Ninguna API key en el código, nunca.
- **Errores**: jerarquía propia (`HateDetError` → `DataError`, `ModelNotLoadedError`, `YouTubeAPIError`,
  `QuotaExceededError`). La API las mapea a códigos HTTP en un único exception handler. Nunca se devuelve
  una traza al cliente.
- **Logging**: `structlog` con salida JSON, `request_id` por petición. Se loguea el **hash** del comentario y
  el score, no el texto completo, salvo que el modo debug esté activo.
- **Seguridad**: rate limiting en `/predict` (slowapi), límite de tamaño del payload, escapado estricto al
  renderizar comentarios en la UI, CORS restringido, imagen Docker con usuario no root.
- **Reproducibilidad**: `random_state` centralizado en configuración, versiones pinneadas en `pyproject.toml`,
  `make train` como único punto de entrada al entrenamiento, hash del dataset registrado en MLflow.
- **Observabilidad**: `/metrics` expone latencia, throughput, distribución de scores y reparto por banda. Un
  desplazamiento en la distribución de scores es la señal temprana de drift del modelo.

## 9. Riesgo arquitectónico principal

El coste y la latencia del transformer en producción. **Mitigación de diseño**: `inference.cascade`. El modelo
lineal puntúa todo; solo los comentarios que caen en la franja de incertidumbre (típicamente 10-20% del
tráfico) pasan por el transformer. Se conserva la mayor parte de la ganancia de calidad a una fracción del
coste, y si el transformer no está disponible el sistema degrada a la ruta lineal en lugar de caer.
