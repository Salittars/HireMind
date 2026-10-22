# Small CPU image. Builds the warehouse from the committed sample so the container answers
# immediately with no network and no API key. Mount data/raw (or run `ingest`) for live data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-embed.txt ./
ARG WITH_EMBEDDINGS=false
RUN pip install -r requirements.txt \
 && if [ "$WITH_EMBEDDINGS" = "true" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch \
      && pip install -r requirements-embed.txt; \
    fi

COPY jobrag ./jobrag
COPY models ./models
COPY seeds ./seeds
COPY evals ./evals
COPY data/sample ./data/sample

ENV JOBRAG_RAW_DIR=/app/data/sample JOBRAG_EMBEDDER=auto
RUN python -m jobrag build

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "jobrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
