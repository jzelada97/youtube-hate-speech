# YouTube Hate Speech Detector

Detector de mensajes de odio en comentarios de YouTube, desarrollado como solución de consultoría.
Prioriza una solución **práctica e implementable**: en lugar de un booleano, devuelve una probabilidad y una
**banda de decisión** (`permitir` / `revisar` / `ocultar`). El sistema **recomienda**; un humano decide.

> **Estado: Nivel Esencial completo** (modelo + API + demo + tests). Niveles superiores en el roadmap.

## Resultados (léelos con sentido crítico)

Dataset pequeño (~1000 comentarios, 138 de odio). Estimación honesta por validación cruzada estratificada
sobre train+val, con augmentation solo en el fold de entrenamiento:

| Métrica | Valor |
|---|---|
| F1-macro (CV) | **≈ 0.70** |
| F1 de la clase «odio» (CV) | ≈ 0.51 |
| PR-AUC (CV) | ≈ 0.39 |
| Gap entrenamiento – CV (F1-macro) | 3.2 pp (requisito: < 5 pp) |
| Latencia API (CPU) | p50 ≈ 3 ms · p95 ≈ 4 ms |

El F1 en el conjunto de test (0.775) **no debe tomarse como cifra de rendimiento**: el test tiene solo 28
positivos y resultó favorable; la CV es la estimación fiable.

**Limitaciones importantes**
- El «odio» del dataset es ~91 % racismo, sobre un mismo suceso (Ferguson): no generaliza a otros tipos de odio
  ni a otros temas, y puede dar falsas alarmas con menciones neutras a un grupo.
- Con este modelo la precisión máxima alcanzable es baja (~0.4–0.5): por eso `revisar` es la banda principal y
  `ocultar` es solo una recomendación de alta probabilidad, no una decisión fiable por sí sola.
- Las etiquetas originales tienen criterio inconsistente en casos límite (ver el notebook de EDA).

## Quickstart

```bash
git clone https://github.com/jzelada97/youtube-hate-speech && cd youtube-hate-speech
pip install -e ".[serve,dev]"
python -m spacy download en_core_web_sm          # solo si usas lematización
python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','omw-1.4')]"

# 1) Datos: descarga youtoxic_english_1000.csv (enlace del cliente) en data/raw/
# 2) Entrenar (genera models/baseline_v3.joblib + metadata con umbrales)
python scripts/train_baseline.py

# 3) API
uvicorn hatedet.api.main:app --port 8000
# 4) Demo (otra terminal)
streamlit run src/hatedet/ui/app.py
```

Los datos y los modelos entrenados **no se versionan** (contienen comentarios reales de usuarios).

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado, versión del modelo y umbrales de las bandas |
| POST | `/predict` | `{"text": "..."}` → `{label, score, band, model_version}` |
| POST | `/predict/batch` | `{"texts": [...]}` (máx. 200) → `{predictions: [...]}` |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"All these people are criminals"}'
```

Variables de entorno: `HATEDET_MODEL_PATH`, `HATEDET_API_KEY` (si se define, exige cabecera `X-API-Key`),
`HATEDET_T_LOW` / `HATEDET_T_HIGH` (umbrales), `HATEDET_CORS_REGEX`. La API **no registra el texto** de los
comentarios. Documentación interactiva en `/docs`.

## Despliegue en una VM

1. Copia el repo y el modelo (`models/`) a la VM, o entrena allí.
2. Ejecuta `uvicorn hatedet.api.main:app --host 127.0.0.1 --port 8000` como servicio (systemd).
3. Pon un proxy inverso con **HTTPS** delante (p. ej. Caddy) y abre solo el puerto 443. HTTPS es necesario si
   la API se consume desde páginas HTTPS (p. ej. una extensión de navegador).
4. Define `HATEDET_API_KEY` y restringe `HATEDET_CORS_REGEX` a tus orígenes.

## Niveles de entrega

| Nivel | Contenido | Estado |
|---|---|---|
| Esencial | Modelo ML + API/interfaz + repo documentado | ✅ |
| Medio | Ensemble + análisis por URL de vídeo + tests + tuning (Optuna) | Tests ✅ · resto pendiente |
| Avanzado | LSTM + seguimiento en tiempo real + despliegue público + Docker | Pendiente |
| Experto | Transformer + persistencia en BD + MLflow | Pendiente |

## Estructura

```
src/hatedet/  data/ (carga, splits, augmentation) · nlp/ (limpieza, normalización, máscara de grupos)
              models/ (pipeline, evaluación, bandas) · api/ (FastAPI) · ui/ (Streamlit)
scripts/      train_baseline.py · eda.py · find_label_issues.py · eval_augmentation.py · identity_patterns.py
notebooks/    01_eda.ipynb
tests/        pytest (`python -m pytest --cov=src/hatedet`)
```

## Metodología

Un único `sklearn.Pipeline` (limpieza regex → máscara de términos de identidad → stemming/stopwords → TF-IDF →
regresión logística L1) se entrena y se sirve como un solo objeto, sin *skew* entre entrenamiento e inferencia.
Las decisiones se validan con **validación cruzada repetida pareada** (mismas particiones para todas las
variantes) y el conjunto de test se reserva para el informe final.
