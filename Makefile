.PHONY: install install-embed ingest build index all demo ask match report chart eval sample serve test lint docker clean

PY ?= python
SAMPLE = JOBRAG_RAW_DIR=data/sample JOBRAG_DB=data/sample.duckdb JOBRAG_EMBEDDER=tfidf

install:            ## core deps (enough for everything, TF-IDF retrieval)
	pip install -r requirements.txt

install-embed:      ## optional local embeddings (CPU torch + sentence-transformers)
	pip install --index-url https://download.pytorch.org/whl/cpu torch
	pip install -r requirements-embed.txt

ingest:             ## fetch live postings from the public APIs
	$(PY) -m jobrag ingest

build:              ## clean + load DuckDB + run SQL models
	$(PY) -m jobrag build

index:              ## embed all postings (resumable)
	$(PY) -m jobrag index

all: ingest build index chart eval

demo:               ## offline: committed sample, TF-IDF, no key, no network
	$(SAMPLE) $(PY) -m jobrag build
	$(SAMPLE) $(PY) -m jobrag ask "Which jobs ask for dbt and Snowflake?"

ask:
	$(PY) -m jobrag ask "$(Q)"

match:
	$(PY) -m jobrag match $(CV)

report:
	$(PY) -m jobrag report

chart:
	$(PY) -m jobrag chart

eval:
	$(PY) -m jobrag eval --out evals/results.json

sample:
	$(PY) -m jobrag sample

serve:
	uvicorn jobrag.api:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest

lint:
	ruff check .

docker:
	docker build -t hiremind .

clean:
	rm -rf data/*.duckdb data/index .pytest_cache .ruff_cache
