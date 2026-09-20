# YouTube Hate Speech Detector

Detector de mensajes de odio en comentarios de YouTube, desarrollado como solución de consultoría.
Prioriza una solución **práctica e implementable**: en lugar de un booleano, devuelve una probabilidad y una
**banda de decisión** (`permitir` / `revisar` / `ocultar`). El sistema **recomienda**; un humano decide.

> **Estado: niveles Esencial y Medio completos** (modelo ensemble + API + análisis por URL de vídeo + demo + tests).
> Avanzado y Experto en el roadmap.

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
`HATEDET_T_LOW` / `HATEDET_T_HIGH` (umbrales), `HATEDET_CORS_REGEX`, `YOUTUBE_API_KEY`. La API **no registra el
texto** de los comentarios. Documentación interactiva en `/docs`.

## Niveles de entrega

| Nivel | Contenido | Estado |
|---|---|---|
| Esencial | Modelo ML + API/interfaz + repo documentado | ✅ |
| Medio | Ensemble + análisis por URL de vídeo + tests + tuning (Optuna) | ✅ |
| Avanzado | LSTM + seguimiento en tiempo real + despliegue público + Docker | En curso |
| Experto | Transformer + persistencia en BD + MLflow | Pendiente |

## Estructura

```
src/hatedet/  data/ (carga, splits, augmentation) · nlp/ (limpieza, normalización, máscara de grupos)
              models/ (baseline, ensemble, evaluación, bandas) · youtube/ (URL, fuentes, análisis)
              api/ (FastAPI) · ui/ (Streamlit)
scripts/      train.py · compare_models.py · tune_optuna.py · eda.py · find_label_issues.py · eval_augmentation.py
notebooks/    01_eda.ipynb
tests/        pytest (`python -m pytest --cov=src/hatedet`; los que necesitan el dataset se saltan si no está)
```

## Metodología

Cada modelo es un objeto de scikit-learn que recibe **texto crudo** y devuelve probabilidades, así que el que se
entrena es exactamente el que se sirve (sin *skew* entre entrenamiento e inferencia). Las decisiones se validan con
**validación cruzada repetida pareada** (mismas particiones para todas las variantes) y el conjunto de test se
reserva para el informe final. Los umbrales de las bandas se eligen con probabilidades fuera de muestra.
