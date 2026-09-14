# Enriched Prompt — Detector de mensajes de odio en comentarios de YouTube

## Summary

El cliente (YouTube) necesita un sistema automático que clasifique comentarios como "odio" / "no odio" con
suficiente fiabilidad para alimentar acciones de moderación (ocultar comentario, encolar para revisión humana,
señalar usuario reincidente). La prioridad explícita del cliente es una solución **práctica e implementable**
por encima de la precisión máxima: el entregable debe ser un sistema end-to-end que se pueda desplegar,
consultar y demostrar en vivo, no un notebook con una métrica alta. El proyecto se organiza en cuatro niveles
incrementales (Esencial → Medio → Avanzado → Experto) donde cada nivel es un incremento desplegable, no una
fase de un waterfall.

Decisión de producto derivada de lo anterior: la salida del sistema **no es un booleano** sino una
probabilidad calibrada más una banda de decisión (`permitir` / `revisar` / `eliminar`) con umbrales
configurables. Esto convierte el trade-off precision/recall en una palanca de negocio del cliente en lugar de
una decisión enterrada en el modelo, y es lo que hace la solución realmente operable.

## Functional Requirements

- **[FR-1]** Ingerir y explorar el dataset de comentarios proporcionado (Google Drive); documentar esquema,
  volumen, idioma, distribución de clases y calidad de etiquetas.
- **[FR-2]** Pipeline de preprocesamiento de texto reproducible: limpieza con regex (URLs, menciones `@`,
  timestamps, emojis, HTML entities, caracteres repetidos, leetspeak), normalización, stopwords, stemming y
  lematización (comparados como ablación, no ambos en producción).
- **[FR-3]** Vectorización clásica: Bag of Words y TF-IDF, con n-gramas de palabra y de carácter.
- **[FR-4]** Modelo de clasificación binaria (odio / no odio) entrenado, evaluado y serializado como artefacto
  único (vectorizador + modelo en un solo `Pipeline` de scikit-learn).
- **[FR-5]** Evaluación con métricas apropiadas a clases desbalanceadas: F1-macro, F1 de la clase minoritaria,
  PR-AUC, matriz de confusión, precision@recall fijo.
- **[FR-6]** Optimización de hiperparámetros con Optuna sobre validación cruzada estratificada.
- **[FR-7]** Técnicas de data augmentation en texto (reemplazo por sinónimos, back-translation) aplicadas
  **solo al conjunto de entrenamiento**.
- **[FR-8]** API de inferencia: `POST /predict` (un comentario), `POST /predict/batch`, `GET /health`.
- **[FR-9]** Interfaz de demo (Streamlit) que consume la API y permite probar texto libre.
- **[FR-10]** Análisis de un vídeo dado su URL: extraer sus comentarios y devolver un informe agregado
  (% de odio, top comentarios por score, usuarios reincidentes).
- **[FR-11]** Seguimiento cuasi-tiempo-real: polling de nuevos comentarios de un vídeo y clasificación continua.
- **[FR-12]** Persistencia de predicciones en base de datos (comentario, score, banda, modelo y versión, timestamp).
- **[FR-13]** Tracking de experimentos con MLflow (params, métricas, artefactos, registro de modelos).
- **[FR-14]** Modelos avanzados: ensemble (voting/stacking), red neuronal recurrente (LSTM) y transformer
  fine-tuneado, todos evaluados contra el mismo split de test.

## Non-Functional Requirements

- **[NFR-1]** Gap de generalización `train − test` en la métrica principal (F1-macro) **< 5 puntos porcentuales**.
- **[NFR-2]** Latencia p95 de `POST /predict` < 300 ms en CPU para el modelo servido por defecto.
- **[NFR-3]** Reproducibilidad: semillas fijadas, versiones pinneadas, entrenamiento ejecutable con un solo comando.
- **[NFR-4]** Sin skew entrenamiento/servicio: el mismo objeto de preprocesamiento que se entrena es el que se sirve.
- **[NFR-5]** Cobertura de tests ≥ 80% líneas sobre `src/` a partir del Nivel Medio.
- **[NFR-6]** Todo el sistema debe levantarse con `docker compose up` a partir del Nivel Avanzado.
- **[NFR-7]** Documentación: README con quickstart, docstrings en módulos públicos, model card del modelo servido.
- **[NFR-8]** Repositorio con ramas organizadas, commits limpios y PRs revisables.

## Technical Context

- **Language/Framework**: Python 3.11; scikit-learn, pandas, spaCy, NLTK, Optuna, Hugging Face Transformers,
  FastAPI, Streamlit, SQLAlchemy, MLflow, pytest.
- **Existing patterns**: ninguno — proyecto greenfield, sin repositorio previo. Se define convención desde cero.
- **Dependencies**: dataset vía Google Drive (formato y tamaño por confirmar); YouTube Data API v3 para
  extracción de comentarios (requiere API key y cuota).
- **Target environment**: desarrollo local (Windows/macOS/Linux) → contenedores Docker → despliegue público
  en PaaS de bajo coste (API en Render/Fly.io, demo en Hugging Face Spaces) o VPS.

## Constraints

- El cliente prioriza implementabilidad y demo funcionando sobre SOTA en métricas.
- Presupuesto de infraestructura asumido bajo: inferencia en CPU, sin GPU en producción.
- La YouTube Data API v3 tiene cuota diaria por defecto de 10.000 unidades y **no ofrece webhooks de
  comentarios**: el "tiempo real" solo puede implementarse por polling.
- Entregables obligatorios no negociables: repo GitHub documentado, demo en vivo, presentación técnica y
  tablero Kanban.

## Acceptance Criteria

- **[AC-1]** Existe un modelo entrenado que clasifica odio con F1-macro reportado sobre un test set held-out
  nunca usado para selección de hiperparámetros.
- **[AC-2]** `|F1_train − F1_test| < 5 pp`, evidenciado en el informe de evaluación y en curvas de aprendizaje.
- **[AC-3]** Un tercero puede clonar el repo, seguir el README y obtener una predicción en < 10 minutos.
- **[AC-4]** La API responde a un comentario arbitrario con `{label, score, band, model_version}`.
- **[AC-5]** Dada una URL de vídeo, el sistema devuelve el análisis agregado de sus comentarios.
- **[AC-6]** El historial de ramas y commits muestra trabajo incremental con Conventional Commits y PRs.
- **[AC-7]** La suite de tests pasa en CI y la cobertura cumple NFR-5.
- **[AC-8]** El tablero Kanban refleja el estado real del proyecto en todo momento.

## Security Considerations

- **Sensitivity level: medium-high.** No hay credenciales financieras ni PII de alto riesgo, pero los
  comentarios de YouTube son **datos personales** (autor, canal, texto) bajo GDPR, y el sistema toma decisiones
  automatizadas que afectan a la expresión de personas reales.
- **Data types involved**: texto de comentarios, IDs de autor y canal, IDs de vídeo, timestamps, API keys de
  Google Cloud.
- Riesgos principales: (a) filtración de la API key en el repo; (b) inyección de contenido en la UI al
  renderizar comentarios sin escapar; (c) sesgo del modelo contra dialectos, minorías y términos reapropiados,
  generando censura desproporcionada; (d) retención indefinida de comentarios en la BD.
- Mitigaciones a definir en fase Security: secretos solo por variables de entorno, `.env` en `.gitignore`,
  escapado estricto en la UI, auditoría de sesgo sobre subconjuntos con términos identitarios, política de
  retención y almacenamiento de hash + ID en lugar de texto crudo cuando sea posible.

## Out of Scope

- Moderación multimodal (vídeo, audio, miniaturas).
- Detección de spam, bots o estafas (problema distinto al de odio).
- Acciones reales sobre la plataforma (borrar comentarios, banear usuarios): el sistema **recomienda**, no ejecuta.
- Anotación manual a gran escala de un nuevo corpus.
- Entrenamiento de modelos desde cero a escala de fundación; solo fine-tuning de checkpoints preentrenados.
- Soporte multilingüe garantizado más allá del idioma dominante del dataset.

## Assumptions to confirm with the client

1. Idioma dominante del dataset (español, inglés o mezcla) — condiciona la elección de spaCy, stopwords y checkpoint de transformer.
2. El dataset **trae etiquetas de odio**; si no, se activa el plan B descrito en la sección de riesgos.
3. El cliente puede proporcionar una API key de YouTube Data API v3 y, si hace falta, una ampliación de cuota.
4. Duración objetivo asumida: 4 sprints de 1 semana. A ajustar.
5. Equipo asumido: 2-4 personas.
