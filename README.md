# YouTube Hate Speech Detector

Sistema de detección automática de mensajes de odio en comentarios de YouTube, desarrollado como
solución de consultoría para YouTube. Prioriza una solución **práctica e implementable** por encima
de la precisión máxima: salida como probabilidad calibrada + banda de decisión (`permitir` /
`revisar` / `eliminar`), no un booleano.

## Estado del proyecto

Fase actual: **planificación y exploración de datos**. Ver [`docs/architecture.md`](docs/architecture.md)
y [`docs/enriched-prompt.md`](docs/enriched-prompt.md) para el plan técnico completo.

## Dataset

`data/raw/youtoxic_english_1000.csv` (no versionado en git — ver `.gitignore`).

- 1000 comentarios en inglés, 13 vídeos únicos.
- Multi-etiqueta: `IsToxic`, `IsAbusive`, `IsThreat`, `IsProvocative`, `IsObscene`, `IsHatespeech`,
  `IsRacist`, `IsNationalist`, `IsSexist`, `IsHomophobic`, `IsReligiousHate`, `IsRadicalism`.
- Target principal elegido: `IsHatespeech` (14% positivos). Varias subcategorías (`IsHomophobic`,
  `IsRadicalism`, `IsSexist`, `IsNationalist`, `IsReligiousHate`) tienen demasiado pocos positivos
  para entrenarse de forma independiente — se tratan como análisis exploratorio, no como targets.
- Dataset pequeño: el control de overfitting y la data augmentation son críticos, no opcionales.

## Niveles de entrega

| Nivel | Contenido |
|---|---|
| Esencial | Modelo ML + API/interfaz de consulta + repo documentado |
| Medio | Ensemble + análisis por URL de vídeo + tests + tuning de hiperparámetros |
| Avanzado | LSTM + seguimiento cuasi-tiempo-real + despliegue público + Docker |
| Experto | Transformer + persistencia en BD + tracking con MLflow |

## Quickstart

_Pendiente — se documentará en cuanto exista el pipeline de entrenamiento (Nivel Esencial)._

## Estructura del repo

Ver [`docs/architecture.md`](docs/architecture.md) para la justificación de cada carpeta.
