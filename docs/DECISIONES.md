# Decisiones e implementación — Detector de odio en comentarios de YouTube

Documento único y acumulativo: cada sección corresponde a una pieza implementada y las decisiones
tomadas sobre ella, en orden cronológico.

---

## 1. Dataset

- **Fuente**: `youtoxic_english_1000.csv` (Google Drive, proporcionado por el cliente).
- **Tamaño**: 1000 comentarios, 13 vídeos únicos, inglés. Sin nulos, 3 duplicados de texto.
- **Etiquetas disponibles** (multi-label, booleanas): `IsToxic` (46%), `IsAbusive` (35%),
  `IsProvocative` (16%), `IsHatespeech` (14%), `IsRacist` (13%), `IsObscene` (10%), `IsThreat` (2%),
  `IsReligiousHate` (1%), `IsNationalist` (1%), `IsSexist` (0.1%), `IsHomophobic` (0%),
  `IsRadicalism` (0%).
- **Decisión — target principal**: `IsHatespeech`. Es la etiqueta más alineada semánticamente con
  "mensaje de odio" (a diferencia de `IsToxic`/`IsAbusive`, que capturan insultos generales sin
  connotación de odio) y tiene positivos suficientes (~140) para entrenar y evaluar con algo de
  fiabilidad.
- **Decisión — subcategorías no se modelan por separado**: `IsHomophobic`, `IsRadicalism`,
  `IsSexist`, `IsNationalist`, `IsReligiousHate` tienen entre 0 y 12 positivos en todo el dataset.
  No hay señal suficiente para entrenar clasificadores independientes fiables; se usan solo como
  análisis descriptivo/exploratorio, no como targets de modelado.
- **Riesgo abierto**: dataset pequeño (1000 filas, 13 vídeos) frente a datasets de referencia del
  dominio (Jigsaw ~160k). Esto hace crítico el control de overfitting (gap train/test < 5pp exigido
  por el cliente) y motiva el uso de data augmentation en el conjunto de entrenamiento.

---

## 2. Entorno y dependencias

- Python 3.12, sin entorno virtual dedicado (dependencias instaladas a nivel de usuario) — decisión
  pragmática para esta fase de desarrollo local; Docker (nivel Avanzado) será el mecanismo real de
  aislamiento y reproducibilidad para despliegue.
- `pyproject.toml` como fuente única de dependencias: pandas, numpy, scikit-learn, scipy, nltk,
  spacy (+ modelo `en_core_web_sm`), matplotlib, seaborn, joblib; grupo `dev` con pytest y ruff.
- Paquete instalado en modo editable (`pip install -e .`) para poder importar `hatedet` desde
  scripts y tests sin manipular `sys.path`.

---

## 3. EDA (`scripts/eda.py`)

Implementado como **script**, no como notebook, para que sea reproducible con un solo comando
(`python scripts/eda.py`) — coherente con el requisito de reproducibilidad del cliente (un tercero
debe poder clonar el repo y obtener resultados sin depender de qué celdas se ejecutaron y en qué
orden). Genera figuras en `reports/figures/` y estadísticas por stdout.

**Hallazgo de calidad de datos no anticipado**: 16 de 997 filas (1.6%) tienen `VideoId = "#NAME?"`,
un artefacto típico de Excel al abrir/guardar el CSV (interpreta el ID como fórmula rota). El texto
y las etiquetas de esas filas son válidos; solo el ID de vídeo está corrupto. No afecta al modelo de
clasificación (no usa `VideoId` como feature), pero sí a cualquier análisis o feature agregado por
vídeo — se documenta aquí para no perder la traza si aparece un bug de agrupación más adelante.

**Hallazgos de EDA**:
- 997 comentarios tras deduplicar (3 duplicados exactos eliminados en `load_raw_comments`).
- 13 vídeos únicos, muy desigualmente representados: de 8 a 274 comentarios por vídeo. El vídeo con
  más comentarios (`9pr1oE34bIM`, 274) tiene una tasa de odio del 11.7%; el de mayor tasa
  (`04kJtp6pVXI`, 172 comentarios) llega al 23.3%. Confirma que el tema del vídeo correlaciona con
  la tasa de odio — justifica la decisión de la sección 4 (split estratificado, no por vídeo).
- Longitud de texto muy asimétrica: mediana 102 caracteres, media 186, máximo 4421 (outlier). Se
  revisará en el preprocesamiento si truncar textos extremos o dejar que el vectorizador los maneje
  tal cual (TF-IDF es robusto a esto por diseño, no se trunca de partida).

---

## 4. Preprocesamiento de texto (`src/hatedet/nlp/`)

- **`regex_rules.py`**: funciones puras (str -> str) para poder testear cada regla por separado.
  Cubre: tags HTML, entidades HTML, URLs, menciones `@`, leetspeak (`h4t3` -> `hate`), caracteres
  repetidos (`loooove` -> `loove`, conserva la duplicación como señal de énfasis en vez de
  eliminarla del todo) y caracteres no alfanuméricos.
  - El strip de tags HTML se añadió por robustez de cara a los niveles Medio/Avanzado (comentarios
    obtenidos en vivo vía scraping/API), aunque el dataset actual no contiene tags reales.
- **`cleaner.py`**: compone las reglas anteriores en `clean_text()` y las envuelve en
  `TextCleaner`, un transformer de scikit-learn sin estado (`fit` es no-op). Vive dentro del mismo
  `Pipeline` que el vectorizador y el modelo — ver "sin skew train/serve" en `docs/architecture.md`.
- **`normalizer.py`**: stopwords (NLTK) + stemming (`SnowballStemmer`) o lematización (spaCy,
  con `parser`/`ner` desactivados por velocidad), seleccionables por parámetro (`method="stem"` /
  `"lemma"` / `None`). Se implementan ambas técnicas a propósito para compararlas por ablación
  (pendiente: `notebooks/02_preprocessing_ablation`) y quedarse solo con la que mejor generalice en
  el modelo baseline, no con ambas en producción.

**Tests**: 20 tests unitarios (`tests/test_regex_rules.py`, `test_cleaner.py`, `test_normalizer.py`,
`test_data.py`), 99% de cobertura sobre `src/hatedet/`. Cubren cada regla de limpieza por separado,
el comportamiento de `TextCleaner` como transformer, ambos métodos de normalización, la validación
de esquema y que los splits no tengan solapamiento de IDs entre particiones.
