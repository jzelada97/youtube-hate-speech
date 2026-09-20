# Imagen unica para la API y la demo (el comando lo decide docker-compose).
# Sin torch: el modelo servido es scikit-learn (ligero). El modelo entrenado NO va en la imagen: se monta en /app/models.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]" \
    && python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4 \
    && python -m spacy download en_core_web_sm

# /app/review existe y pertenece a appuser para que el volumen con nombre herede ese propietario:
# el contenedor corre como no root y el sistema de ficheros es de solo lectura, asi que es el unico
# sitio escribible (cola de revision, apagada salvo que se defina HATEDET_REVIEW_DB).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/conf /app/review \
    && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "hatedet.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
