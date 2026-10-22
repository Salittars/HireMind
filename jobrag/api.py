"""FastAPI service: /health, /ask, /match.  Run: uvicorn jobrag.api:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import rag
from .match import match_cv
from .retrieval import get_retriever

app = FastAPI(title="HireMind", version="0.1.0",
              description="Ask questions about real data/AI job postings; match a CV to postings.")


@lru_cache(maxsize=1)
def retriever():
    return get_retriever()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Which jobs ask for dbt and Snowflake?"])
    k: int = Field(5, ge=1, le=20)


class MatchRequest(BaseModel):
    cv_text: str = Field(..., min_length=20)
    k: int = Field(10, ge=1, le=30)


@app.get("/health")
def health() -> dict:
    r = retriever()
    return {"status": "ok", "postings": len(r.docs), "retriever": r.name, "generator": rag.generator_mode()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return rag.ask(req.question, retriever(), k=req.k).to_dict()


@app.post("/match")
def match(req: MatchRequest) -> dict:
    return match_cv(req.cv_text, retriever(), k=req.k)
