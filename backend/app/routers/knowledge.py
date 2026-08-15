"""Knowledge-base endpoints — retrieval introspection and text-only Q&A.

Exposing the raw retrieval results matters for a thesis: it makes the RAG stage
inspectable, so a reviewer can see exactly which passages grounded an answer.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from ..orchestrator import answer_text_question
from ..rag.store import get_store
from ..schemas import DocumentInfo, KnowledgeAnswer, RetrievedPassage

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get(
    "/documents",
    response_model=list[DocumentInfo],
    summary="List knowledge-base documents",
)
async def list_documents() -> list[DocumentInfo]:
    return get_store().describe()


@router.get(
    "/search",
    response_model=list[RetrievedPassage],
    summary="Inspect raw retrieval results",
)
async def search(
    q: str = Query(..., min_length=2, description="Retrieval query"),
    top_k: int = Query(6, ge=1, le=20),
) -> list[RetrievedPassage]:
    return get_store().search(q, top_k=top_k)


@router.post("/ask", response_model=KnowledgeAnswer, summary="Ask a grounded question")
async def ask(
    q: str = Query(..., min_length=3, description="Question about LSD"),
) -> KnowledgeAnswer:
    return await run_in_threadpool(answer_text_question, q)
